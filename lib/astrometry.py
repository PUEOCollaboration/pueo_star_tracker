# Standard imports
from contextlib import suppress
import logging
import traceback
# import sys
import time
# from time import perf_counter as precision_timestamp
import os
from pathlib import Path
# External imports
# import matplotlib.pyplot as plt
import cv2
import numpy as np
# from scipy.optimize import linear_sum_assignment
# from skimage.color.rgb_colors import silver

# from tqdm import tqdm

# Custom imports
from lib.config import Config
from lib.common import current_timestamp, logit
from lib.utils import read_image_grayscale, read_image_BGR, display_overlay_info, timed_function, print_img_info
from lib.source_finder import global_background_estimation, source_finder
# from lib.source_finder import global_background_estimation, local_levels_background_estimation, \
#     median_background_estimation, sextractor_background_estimation, find_sources, find_sources_photutils, \
#     select_top_sources, select_top_sources_photutils, source_finder

from lib.compute_star_centroids import compute_centroids_from_still, compute_centroids_from_trail, \
    compute_centroids_photutils, compute_centroids_from_trails_ellipse_method
from lib.utils import get_files, split_path
from lib.tetra3 import Tetra3
from lib.cedar_solve import Tetra3 as Tetra3Cedar  # , cedar_detect_client
from lib.cedar import Cedar
from lib.astrometry_net import  AstrometryNet
from lib.compute import Compute


class Astrometry:
    """
    Note:
        solver: 'solver1' ~ genuine tetra3 | 'solver2' ~ cedar tetra3
    Params:
        database_name

    """
    _t3 = None
    test_data = {}

    def __init__(self, database_name=None, cfg=None, log=None):
        self.log = log or logging.getLogger('pueo')
        self.log.info('Initializing Astrometry Object')
        self.test = False
        self.database_name = database_name
        self.cfg = cfg
        self.solver = 'solver2'  # Default solver

        self.compute = Compute(log)

        # apply distortion calibration
        if database_name is None:
            db_file = '../data/default_database.npz'
            self.database_name = db_file if os.path.exists(db_file) else 'data/default_database.npz'
        self.database_path = Path(self.database_name).resolve()

        # Create instance
        self.t3_genuine = Tetra3(load_database=self.database_path)
        # Load a database
        self.log.info(f'Database for Genuine Tetra3: {self.database_path}')
        # self.t3_genuine.load_database(path=self.database_path)
        # else
        self.t3_cedar = Tetra3Cedar(load_database=self.database_path)
        # TODO: Remove cedar_detect if not used
        # with suppress(AttributeError, Exception):
        #     self.cedar_detect = cedar_detect_client.CedarDetectClient()
        self.log.info(f'Database for Cedar Tetra3: {self.database_path}')
        # self.t3_cedar.load_database(path=self.database_path)

        self.cedar = Cedar(database_name, cfg)
        self.cedar.test = self.test


    @staticmethod
    def _fit_ellipses_from_mask(mask_u8, min_area=10, min_major=12):
        """
        Fit ellipses to ALL connected components in the binary mask (no AR gating here).
        Returns a list of dicts with: cx, cy, a, b, L, ang, area, ar.
        - 'L' is the major-axis length in pixels (2*a).
        - 'ang' is orientation in degrees, normalized to [0, 180).
        """
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ell = []
        for c in cnts:
            A = cv2.contourArea(c)
            if A < min_area:
                continue
            if len(c) >= 5:
                (cx, cy), (W, H), theta = cv2.fitEllipse(c)
            else:
                (cx, cy), (W, H), theta = cv2.minAreaRect(c)
            a = max(W, H) * 0.5
            b = max(1e-6, min(W, H) * 0.5)
            L = 2.0 * a
            if L < min_major:
                continue
            # normalize orientation so major axis is the reference, range [0,180)
            ang = theta if W >= H else (theta + 90.0)
            ang = (ang + 180.0) % 180.0
            ell.append({
                "cx": float(cx), "cy": float(cy),
                "a": float(a), "b": float(b),
                "L": float(L), "ang": float(ang),
                "area": float(A), "ar": float(a / b)
            })
        return ell

    @staticmethod
    def _ellipse_metrics_ar(ell, ar_elong=1.70):
        """
        Split blobs by AR threshold and compute elongated-only metrics:
          - L_med_elong: median major-axis length among AR>=ar_elong
          - R_star: double-angle orientation coherence in [0,1]
        Returns dict: N, N_elong, f_elong, L_med_elong, R_star
        """
        N = len(ell)
        if N == 0:
            return dict(N=0, N_elong=0, f_elong=0.0, L_med_elong=0.0, R_star=0.0)

        ar = np.array([e["ar"] for e in ell], float)
        L  = np.array([e["L"]  for e in ell], float)
        # accept either 'ang' or 'theta_deg' field for orientation
        th = np.array([ (e["ang"] if "ang" in e else e["theta_deg"]) for e in ell ], float)

        idx = ar >= ar_elong
        N_elong = int(np.count_nonzero(idx))
        f_elong = float(N_elong / N)

        if N_elong:
            L_med_elong = float(np.median(L[idx]))
            th2 = np.deg2rad(2.0 * th[idx])
            R_star = float(np.hypot(np.sum(np.cos(th2)), np.sum(np.sin(th2))) / N_elong)
        else:
            L_med_elong, R_star = 0.0, 0.0

        return dict(
            N=N,
            N_elong=N_elong,
            f_elong=f_elong,
            L_med_elong=L_med_elong,
            R_star=R_star
        )


    def classify_frame_still_biased(self,
                                    ell,
                                    ar_elong=1.70,
                                    min_elong=5,
                                    min_L=7.0,
                                    min_R=0.50,
                                    min_frac=0.60,
                                    min_conf=0.65):   # NEW: confidence gate
        M = self._ellipse_metrics_ar(ell, ar_elong=ar_elong)
        if M["N"] == 0:
            return "empty", 0.0, M

        is_streaked = (M["N_elong"] >= min_elong and
                       M["L_med_elong"] >= min_L and
                       M["R_star"] >= min_R and
                       M["f_elong"] >= min_frac)
        label = "streaked" if is_streaked else "still"

        s_N    = min(1.0, M["N_elong"]    / max(min_elong, 1))
        s_L    = min(1.0, M["L_med_elong"]/ max(min_L, 1e-6))
        s_R    = min(1.0, M["R_star"]     / max(min_R, 1e-6))
        s_frac = min(1.0, M["f_elong"]    / max(min_frac, 1e-6))

        if label == "streaked":
            conf = float((s_N + s_L + s_R + s_frac) / 4.0)
        else:
            conf = float(((1-s_N) + (1-s_L) + (1-s_frac)) / 4.0)

        # NEW: hard gate on minimum confidence
        if label == "streaked" and conf < float(min_conf):
            label = "still"

        M["coherence_R_star"] = M["R_star"]
        M["min_conf"] = float(min_conf)
        return label, conf, M

    @staticmethod
    def _estimate_omega_from_ellipses_plate_scale(
        ell,
        plate_scale_arcsec_per_px,
        exposure_time_s,
        image_shape,
        ar_min=1.7,
        L_min_px=5.0,
        area_min_px=0.0,
    ):
        """
        Estimate camera angular velocity from a list of ellipse dicts produced by
        _fit_ellipses_from_mask, using plate scale and exposure time.

        Parameters
        ----------
        ell : list of dict
            Each element with keys: cx, cy, L, ang, area, ar.
        plate_scale_arcsec_per_px : float
            Plate scale [arcsec / pixel].
        exposure_time_s : float
            Exposure time [s] for this frame.
        image_shape : (H, W)
            Image height and width (same frame ell was fit on).
        ar_min : float
            Minimum aspect ratio (a/b) for a blob to be treated as a streak.
        L_min_px : float
            Minimum major-axis length [px].
        area_min_px : float
            Minimum area [px^2].

        Returns
        -------
        omega_deg_s : np.ndarray | None
            [wx, wy, wz] in deg/s (camera frame), or None if not solvable.
        diag : dict
            Diagnostics: N_used, rank, resid_rms_deg_s.
        """
        if plate_scale_arcsec_per_px <= 0.0 or exposure_time_s <= 0.0:
            return None, {
                "N_used": 0,
                "rank": 0,
                "resid_rms_deg_s": float("nan"),
            }

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) < 2:
            raise ValueError("image_shape must be (H, W)")

        H, W = int(image_shape[0]), int(image_shape[1])
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5

        # Plate scale [rad / pixel]
        s = plate_scale_arcsec_per_px * (np.pi / 180.0) / 3600.0

        A_rows = []
        b_rows = []
        n_used = 0

        for e in ell:
            A_e = float(e.get("area", 0.0))
            ar_e = float(e.get("ar", 0.0))
            L_px = float(e.get("L", 0.0))

            if A_e < area_min_px:
                continue
            if ar_e < ar_min:
                continue
            if L_px < L_min_px:
                continue

            cx_e = float(e["cx"])
            cy_e = float(e["cy"])
            ang_deg = float(e["ang"])

            # Position in tangent plane [rad]
            x_ang = (cx_e - cx) * s
            y_ang = (cy_e - cy) * s

            # Pixel flow along major axis
            theta = np.deg2rad(ang_deg)
            du_dt_px = (L_px * np.cos(theta)) / exposure_time_s
            dv_dt_px = (L_px * np.sin(theta)) / exposure_time_s

            # Flow in angular units [rad/s]
            u_ang = du_dt_px * s
            v_ang = dv_dt_px * s

            # Optical flow model for pure rotation:
            # dx/dt = x*y*wx + (-1 - x^2)*wy + y*wz
            # dy/dt = (1 + y^2)*wx + (-x*y)*wy + (-x)*wz
            A_rows.append([x_ang * y_ang, -(1.0 + x_ang * x_ang), y_ang])
            b_rows.append(u_ang)
            A_rows.append([1.0 + y_ang * y_ang, -x_ang * y_ang, -x_ang])
            b_rows.append(v_ang)

            n_used += 1

        if len(A_rows) < 3:
            return None, {
                "N_used": n_used,
                "rank": 0,
                "resid_rms_deg_s": float("nan"),
            }

        A = np.asarray(A_rows, dtype=np.float64)
        b = np.asarray(b_rows, dtype=np.float64)

        omega_rad_s, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
        omega_deg_s = np.degrees(omega_rad_s)

        # residuals in (rad/s)^2 → RMS rad/s → deg/s
        if residuals.size > 0 and A.shape[0] > 0:
            resid_rms_rad_s = float(np.sqrt(residuals[0] / A.shape[0]))
            resid_rms_deg_s = float(np.degrees(resid_rms_rad_s))
        else:
            resid_rms_deg_s = float("nan")

        diag = {
            "N_used": int(n_used),
            "rank": int(rank),
            "resid_rms_deg_s": resid_rms_deg_s,
        }
        return omega_deg_s, diag

    @staticmethod
    def _blob_reason(area, ar, min_area_px, aspect_ratio_min):
        """
        Return ('OK' or reason code, color BGR) for this blob’s pass/fail.
        """
        if area < min_area_px:
            return "AREA", (0, 0, 255)      # red
        if ar < aspect_ratio_min:
            return "AR", (0, 0, 255)        # red
        return "OK", (0, 200, 0)            # green

    @staticmethod
    def _pt(x, y):
        """Return a cv2-friendly (int, int) point from any numpy/float input."""
        xi = int(np.rint(np.asarray(x).squeeze()))
        yi = int(np.rint(np.asarray(y).squeeze()))
        return (xi, yi)

    @staticmethod
    def _draw_trail_overlay_like_still(
        image,
        mask=None,
        centers_xy=None,
        lengths=None,
        angles_deg=None,
        draw_ids=False,
        ids=None,
        contour_color=(0, 255, 255),   # contours (yellow-ish)
        centroid_color=(0, 0, 255),    # centroids (red)
        axis_color=(0, 255, 0),        # major-axis line (green)
        thickness=1,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.4,
        font_thickness=1
    ):
        """
        Build a contours_img analog to 'compute_centroids_from_still':
        - draws external contours from `mask`
        - draws centroid red dots
        - draws optional ID labels
        - draws short major-axis lines for visual orientation
        """
        overlay = image.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        # 1) Draw contours first (underlay)
        if mask is not None:
            cnts, _ = cv2.findContours((mask > 0).astype('uint8'),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) > 0:
                cv2.drawContours(overlay, cnts, -1, contour_color, thickness)

        # 2) Draw centroids + (optional) IDs + short major-axis lines
        if centers_xy is None or lengths is None or angles_deg is None:
            return overlay

        for k, ((x, y), L, ang_deg) in enumerate(zip(centers_xy, lengths, angles_deg)):
            # centroid (red dot)
            cv2.circle(overlay, (int(round(x)), int(round(y))), 2, centroid_color, -1, cv2.LINE_AA)

            # ID (optional)
            if draw_ids:
                label = str(ids[k] if ids is not None else k)
                cv2.putText(overlay, label, (int(x)+5, int(y)-5),
                            font, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

            # major-axis short line (visual hint)
            half = max(6.0, 0.15 * float(L))
            # Windel use of math
            # theta = math.radians(float(ang_deg))
            # dx, dy = half * math.cos(theta), half * math.sin(theta)

            # Convert degrees to radians and store in theta
            theta = np.radians(float(ang_deg))
            # Calculate dx and dy using cosine and sine of the radian angle
            dx, dy = half * np.cos(theta), half * np.sin(theta)

            p1 = (int(round(x - dx)), int(round(y - dy)))
            p2 = (int(round(x + dx)), int(round(y + dy)))
            cv2.line(overlay, p1, p2, axis_color, thickness, cv2.LINE_AA)

        return overlay

    def _trail_ellipse_adapter(
        self,
        masked_image,
        sources_mask,
        original_image,
        min_area_px=20,
        aspect_ratio_min=2.0,
        use_uniform_length=True,
        uniform_length_mode="median",
        return_partial_images=False,
        partial_results_path=None,
        frame_tag=None,
    ):
        """
        Returns EXACTLY what the old trail path expected:
          (precomputed_star_centroids, contours_img)
        When return_partial_images=True, saves a still-style overlay and mask to partial_results_path.
        """
        ids, xs, ys, fluxes, lengths, angles = compute_centroids_from_trails_ellipse_method(
            image = masked_image,
            sources_mask=sources_mask,
            min_area_px=min_area_px,
            aspect_ratio_min=aspect_ratio_min,
            use_uniform_length=use_uniform_length,
            uniform_length_mode=uniform_length_mode,
        )
        # Ensure the solver gets the same order we saved: mean-intensity descending
        # np.argsort()
        order = np.argsort(-np.asarray(fluxes, float))  # fluxes == mean intensity from ellipse∩mask
        xs = np.asarray(xs, float)[order]
        ys = np.asarray(ys, float)[order]
        fluxes = np.asarray(fluxes, float)[order]
        lengths = np.asarray(lengths, float)[order]
        angles = np.asarray(angles, float)[order]

        # --- Build precomputed_star_centroids (N×5: [y, x, flux, std, diameter]) ---
        if len(ids) == 0: # sames as if ids:
            precomputed_star_centroids = np.empty((0, 5), dtype=float)
            contours_img = self._draw_trail_overlay_like_still(
                image=original_image,
                mask=sources_mask,
                centers_xy=[],
                lengths=[],
                angles_deg=[]
            )
            # Save partial images if requested
            if return_partial_images and partial_results_path:
                tag = frame_tag or "frame"
                cv2.imwrite(f"{partial_results_path}/trail_overlay_{tag}.png", contours_img)
                cv2.imwrite(f"{partial_results_path}/trail_mask_{tag}.png",
                            (sources_mask > 0).astype('uint8') * 255)
            return precomputed_star_centroids, contours_img

        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        fluxes = np.asarray(fluxes, dtype=float)
        lengths = np.asarray(lengths, dtype=float)
        angles = np.asarray(angles, dtype=float)

        std = np.full_like(fluxes, np.nan, dtype=float)   # preserve shape
        diameter = lengths.astype(float)                   # use major axis as diameter proxy
        precomputed_star_centroids = np.stack([ys, xs, fluxes, std, diameter], axis=1)

        # --- Build still-style overlay (contours + centroids + axes + ellipse outlines) ---
        overlay = self._draw_trail_overlay_like_still(
            image=original_image,
            mask=sources_mask,
            centers_xy=list(zip(xs, ys)),
            lengths=lengths,
            angles_deg=angles,
            draw_ids=True,
        )

        # --- Draw per-contour diagnostics and ellipses for ACCEPTED blobs -----------
        # Build array of accepted centers to match contours to accepted results
        acc_centers = np.column_stack([xs, ys]) if len(xs) else np.zeros((0, 2), float)

        # Indexing helper to find nearest accepted center to a contour-fit center
        def nearest_acc_idx(cx, cy, tol_px=6.0):
            if acc_centers.shape[0] == 0:
                return -1
            dx = acc_centers[:, 0] - cx
            dy = acc_centers[:, 1] - cy
            j = int(np.argmin(dx * dx + dy * dy))
            return j if (dx[j] * dx[j] + dy[j] * dy[j]) <= (tol_px * tol_px) else -1

        # Walk all contours from the MASK (fit ellipses to mask geometry)
        cnts, _ = cv2.findContours((sources_mask > 0).astype('uint8'),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Numbering for on-image labels
        blob_idx = 0
        for c in cnts:
            if len(c) < 5:
                # Not enough points for ellipse fit; annotate anyway with area
                area = cv2.contourArea(c)
                reason, color = self._blob_reason(area, ar=999.0, min_area_px=min_area_px, aspect_ratio_min=aspect_ratio_min)
                M = cv2.moments(c)
                cx = cy = np.nan
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                # Fallback to geometric center if moments are not finite
                if not (np.isfinite(cx) and np.isfinite(cy)):
                    cx = float(c[:, 0, 0].mean())
                    cy = float(c[:, 0, 1].mean())
                # Only annotate if we have finite numbers
                if np.isfinite(cx) and np.isfinite(cy):
                    cv2.putText(overlay, f"{blob_idx}:{reason}", (int(round(cx)) + 4, int(round(cy)) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                blob_idx += 1
                continue

            # Try to fit an ellipse; if it fails, annotate FITFAIL and continue
            try:
                (cx, cy), (MA, ma), ang = cv2.fitEllipse(c)
                if MA < ma:
                    MA, ma = ma, MA  # ensure MA = major axis, ma = minor axis
                    ang = (ang + 90.0) % 180.0
            except Exception:
                reason, color = "FITFAIL", (0, 0, 255)
                # geometric center as fallback to place the label
                M = cv2.moments(c)
                gx = gy = np.nan
                if M["m00"] > 0:
                    gx = M["m10"] / M["m00"]
                    gy = M["m01"] / M["m00"]
                if not (np.isfinite(gx) and np.isfinite(gy)):
                    gx = float(c[:, 0, 0].mean())
                    gy = float(c[:, 0, 1].mean())
                if np.isfinite(gx) and np.isfinite(gy):
                    cv2.circle(overlay, (int(round(gx)), int(round(gy))), 2, color, -1, cv2.LINE_AA)
                    cv2.putText(overlay, f"{blob_idx}:{reason}", (int(round(gx)) + 6, int(round(gy)) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                blob_idx += 1
                continue

            area = cv2.contourArea(c)
            ar = (MA / ma) if ma > 0 else 999.0

            # Pass/fail reason & color
            reason, color = self._blob_reason(area, ar, min_area_px, aspect_ratio_min)


            #_x = int(np.rint(np.asarray(cx).squeeze()))
            #_y = int(np.rint(np.asarray(cy).squeeze()))
            #cv2.putText(
            #    overlay, f"{blob_idx}:{reason}",
            #    (int(_x) + 4, int(_y) - 6),
            #    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            #)

            # Draw a small dot at the contour-fit center
            #cv2.circle(overlay, (int(round(cx)), int(round(cy))), 2, color, -1, cv2.LINE_AA)

            # If accepted (OK), draw the ellipse outline and also label its accepted ID (nearest centroid)
            if reason == "OK":
                j = nearest_acc_idx(cx, cy, tol_px=6.0)
                if j < 0:
                    # Base checks passed, but no nearby accepted centroid → draw ellipse anyway in red
                    reason_nomatch = "NOMATCH"
                    cv2.putText(
                        overlay, f"{blob_idx}:{reason_nomatch}",
                        (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA
                    )

                    center = (int(round(cx)), int(round(cy)))
                    axes = (int(round(MA / 2.0)), int(round(ma / 2.0)))
                    cv2.ellipse(overlay, center, axes, float(ang), 0, 360, (0, 0, 255), 1, cv2.LINE_AA)  # red outline

                    blob_idx += 1
                    continue

                # ellipse outline (accepted & matched)
                center = (int(round(cx)), int(round(cy)))
                axes = (int(round(MA / 2.0)), int(round(ma / 2.0)))
                cv2.ellipse(overlay, center, axes, float(ang), 0, 360, (0, 180, 255), 1, cv2.LINE_AA)

                # Optional: show matched accepted ID next to the centroid
                if j >= 0:
                    cv2.putText(overlay, f"ID{j}", (int(cx) + 6, int(cy) + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Save partial images if requested (mirror still path behavior)
        if return_partial_images:
            tag = frame_tag or "frame"
            cv2.imwrite(f"{partial_results_path}/trail_overlay_{tag}.png", overlay)
            cv2.imwrite(f"{partial_results_path}/trail_mask_{tag}.png",
                        (sources_mask > 0).astype('uint8') * 255)

        return precomputed_star_centroids, overlay

    def classify_streaks_axis_axis(
        mask_u8,
        theta_axes_thr=12.0,      # Δθ between major axes
        dir_thr_col=15.0,         # center→center vs each axis
        dist_w_factor=100.0,      # d_thr = W / dist_w_factor
        perp_k=0.5,               # perpendicular gate factor
        perp_min=3.0,             # min perp threshold (px)
    ):
        """
        Input:
          mask_u8: uint8 binary mask (0/255), already background-subtracted & thresholded (your pipeline)
        Output:
          label: "streaked" | "not-streaked"
          metrics: dict with N, R_star_w, TL_norm (and others)
          overlay_bgr: optional visualization (BGR) with merged ellipses; None if no blobs
        """

        # --- helpers ---
        def angle_diff_deg_180(a, b):
            d = abs(a - b) % 180.0
            return min(d, 180.0 - d)

        # TODO: Remove - Windell prototype code
        # def angle_to_axis_deg_with_math(vx, vy, theta_deg):
        #     ux = math.cos(math.radians(theta_deg))
        #     uy = math.sin(math.radians(theta_deg))
        #     dot = abs(vx*ux + vy*uy)
        #     dot = max(-1.0, min(1.0, dot))
        #     return math.degrees(math.acos(dot))

        def angle_to_axis_deg(vx, vy, theta_deg):
            # Convert degrees to radians for trigonometric functions, numpy implementation
            theta_rad = np.radians(theta_deg)

            # Calculate unit vector (ux, uy)
            ux = np.cos(theta_rad)
            uy = np.sin(theta_rad)

            # Calculate the absolute value of the dot product (scalar projection)
            dot = np.abs(vx * ux + vy * uy)

            # Clip the dot product to the valid range [-1.0, 1.0] to prevent math domain errors
            # (NumPy's clip function is ideal for this)
            dot = np.clip(dot, -1.0, 1.0)

            # Calculate the angle (in radians) and convert the final result back to degrees
            return np.degrees(np.arccos(dot))

        def fit_ellipse_from_contour(cnt):
            if len(cnt) < 5:
                return None
            (cx, cy), (MA, mi), ang = cv2.fitEllipse(cnt)
            a, b = (MA/2.0, mi/2.0) if MA >= mi else (mi/2.0, MA/2.0)
            if mi > MA:  # swapped
                ang = (ang + 90.0) % 180.0
            return float(cx), float(cy), float(a), float(b), float(ang)

        def should_merge_axis_axis(bi, bj, W):
            dx = bj["cx"] - bi["cx"]
            dy = bj["cy"] - bi["cy"]
            d = float(np.hypot(dx, dy))
            if d > W / dist_w_factor:
                return False
            if d < 1e-6:
                return True

            # axes nearly parallel
            dtheta = angle_diff_deg_180(bi["ang"], bj["ang"])
            if dtheta > theta_axes_thr:
                return False

            # collinearity vs each axis
            vx, vy = dx/d, dy/d
            a1 = angle_to_axis_deg(vx, vy, bi["ang"])
            a2 = angle_to_axis_deg(vx, vy, bj["ang"])
            if (a1 > dir_thr_col) or (a2 > dir_thr_col):
                return False

            # perpendicular offset gate (relative to bi axis)
            # TODO: Remove -  Windell math implementation
            # c, s = math.cos(math.radians(bi["ang"])), math.sin(math.radians(bi["ang"]))
            # Numpy implementation
            # Calculate cosine (c) and sine (s) of the angle stored in bi["ang"]
            c, s = np.cos(np.radians(bi["ang"])), np.sin(np.radians(bi["ang"]))

            dyp = -s*dx + c*dy
            perp_thr = max(perp_k * 0.5 * (bi["b"] + bj["b"]), perp_min)
            if abs(dyp) > perp_thr:
                return False

            return True

        def union_find(n):
            parent = list(range(n))
            rank = [0]*n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return
                if rank[rx] < rank[ry]:
                    parent[rx] = ry
                elif rank[rx] > rank[ry]:
                    parent[ry] = rx
                else:
                    parent[ry] = rx; rank[rx] += 1
            return find, union

        # --- 1) Extract ellipses from the mask (your gates) ---
        AREA_MIN, MAJOR_MIN, MINOR_MIN, ASPECT_MIN = 10, 12.0, 2.0, 1.7
        mask01 = (mask_u8 > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        blobs = []
        for cnt in contours:
            if len(cnt) < 5:
                continue
            area = float(cv2.contourArea(cnt))
            if area < AREA_MIN:
                continue
            fe = fit_ellipse_from_contour(cnt)
            if fe is None:
                continue
            cx, cy, a, b, ang = fe
            if a*2.0 < MAJOR_MIN or b*2.0 < MINOR_MIN:
                continue
            aspect = a / max(b, 1e-6)
            if aspect < ASPECT_MIN:
                continue
            blobs.append({"cx":cx,"cy":cy,"a":a,"b":b,"ang":ang,"area":area,"contour":cnt})

        H, W = mask01.shape
        if not blobs:
            return "not-streaked", {"N": 0, "R_star_w": 0.0, "L_med_norm": 0.0}, None

        # --- 2) Merge (axis–axis + collinearity + perpendicular) ---
        find, union = union_find(len(blobs))
        for i in range(len(blobs)):
            for j in range(i+1, len(blobs)):
                if should_merge_axis_axis(blobs[i], blobs[j], W):
                    union(i, j)

        groups = {}
        for i in range(len(blobs)):
            r = find(i)
            groups.setdefault(r, []).append(i)

        merged = []
        for idxs in groups.values():
            pts = np.vstack([blobs[i]["contour"] for i in idxs]).reshape(-1,1,2)
            fe = fit_ellipse_from_contour(pts)
            if fe is None:
                cx = float(np.mean([blobs[i]["cx"] for i in idxs]))
                cy = float(np.mean([blobs[i]["cy"] for i in idxs]))
                a  = float(np.median([blobs[i]["a"] for i in idxs]))
                b  = float(np.median([blobs[i]["b"] for i in idxs]))
                ang= float(np.median([blobs[i]["ang"] for i in idxs]))
            else:
                cx, cy, a, b, ang = fe
            area = float(sum(blobs[i]["area"] for i in idxs))
            merged.append({"cx":cx,"cy":cy,"a":a,"b":b,"ang":ang,"area":area,"members":idxs})

        # Optional: 3×MAD outlier reject on major axis
        a_arr = np.array([b["a"] for b in merged], float)
        med = float(np.median(a_arr))
        mad = float(1.4826*np.median(np.abs(a_arr - med)) + 1e-9)
        merged = [b for b in merged if abs(b["a"] - med) <= 3.0*mad]
        if not merged:
            return "not-streaked", {"N": 0, "R_star_w": 0.0, "L_med_norm": 0.0}, None

        a_arr = np.array([b["a"] for b in merged], dtype=float)  # semi-major (px)
        th_arr = np.array([b["ang"] for b in merged], dtype=float)  # angle (deg)
        N = len(merged)

        # Length-weighted double-angle coherence (R*_w)
        th2 = np.deg2rad(2.0 * th_arr)
        C = float(np.sum(a_arr * np.cos(th2)))
        S = float(np.sum(a_arr * np.sin(th2)))
        R_star_w = float(np.hypot(C, S) / (np.sum(a_arr) + 1e-12))

        # Aliases used below
        w = a_arr
        theta = th_arr

        # --- 3) Metrics ---
        # --- 4) Length gate that does NOT scale with N ---
        # Dominant direction φ (deg) from length-weighted double-angle vector
        # Build arrays for metrics
        N = len(merged)
        th_arr = np.array([b["ang"] for b in merged], dtype=float)   # angle (deg)

        # Length-weighted double-angle coherence
        th2 = np.deg2rad(2.0 * th_arr)
        C = float(np.sum(a_arr * np.cos(th2)))
        S = float(np.sum(a_arr * np.sin(th2)))
        R_star_w = float(np.hypot(C, S) / (np.sum(a_arr) + 1e-12))

        # Aliases used below
        w = a_arr
        theta = th_arr

        phi = 0.5 * np.degrees(np.arctan2(
            np.sum(w * np.sin(np.deg2rad(2.0 * theta))),
            np.sum(w * np.cos(np.deg2rad(2.0 * theta)))
        )) % 180.0

        # Project only if orientation is at least mildly coherent (R* ≥ 0.30) and we have enough blobs
        use_proj = (N >= 3 and R_star_w >= 0.30)
        a_eff = w * np.abs(np.cos(np.deg2rad(theta - phi))) if use_proj else w

        # Per-streak size gate (median of effective **major diameter**, normalized by width)
        # a_eff are semi-majors (px); convert to major diameters (2a) then normalize by W
        L_med_major_px = float(np.median(2.0 * a_eff))
        L_med_norm = L_med_major_px / float(W)

        # Decision thresholds (resolution-independent): 12 px major diameter at this width
        N_MIN = 3
        RSTARW_THR = 0.50
        L_MED_NORM_THR = (12.0 / float(W))

        is_streaked = (N >= N_MIN) and (R_star_w >= RSTARW_THR) and (L_med_norm >= L_MED_NORM_THR)
        label = "streaked" if is_streaked else "not-streaked"

        # --- 5) Overlay (unchanged below) ---
        # (keep your existing overlay code here)

        # Metrics to report (replace prior dict)

        metrics = {
            "N": int(N),
            "R_star_w": float(R_star_w),
            "L_med_major_px": float(L_med_major_px),
            "L_med_norm": float(L_med_norm),
            "phi_deg": float(phi),
            "used_projection": bool(use_proj),
        }
        overlay = locals().get("overlay", None)
        return label, metrics, overlay


    @staticmethod
    def get_solver_name(solver_key):
        """Get the human-readable name for a solver key.

        Args:
            solver_key (str): The solver identifier (e.g., 'solver1').

        Returns:
            str: The readable name of the solver.
        """
        solver_map = {
            'solver1': 'ESA Tetra3',
            'solver2': 'Cedar Tetra3',
            'solver3': 'astrometry.net',
        }
        return solver_map.get(solver_key, 'Undefined solver')

    @property
    def solver_name(self):
        # The property now just calls the static method
        return self.get_solver_name(self.solver)

    @property
    def t3(self):
        logit(f'Serving astro solver: {self.solver}', color='cyan')
        return self.t3_genuine if self.solver == 'solver1' else self.t3_cedar

    def tetra3_generate_database(
            self,
            star_catalog: str,
            max_fov: float,
            star_max_magnitude: int,
            output_name: str) -> None:
        """Generate a new star database.

        Args:
            star_catalog (str): Abbreviated name of the star catalog to use. Must be one of 'bsc5', 'hip_main', or 'tyc_main'.
            max_fov (float): Maximum angle (in degrees) allowed between stars in the same pattern. This determines
                             the angular separation for star grouping.
            star_max_magnitude (int): Dimmest apparent magnitude of stars to include in the database. Stars
                                       fainter than this magnitude will be excluded.
            output_name (str or pathlib.Path): The file path or name where the generated star catalog will be saved.
        """
        # Generate and save database
        self.t3.generate_database(
            star_catalog=star_catalog, max_fov=max_fov, star_max_magnitude=star_max_magnitude, save_as=output_name
        )

    def log_trail_classifier_metrics(
            self,
            mask_u8,
            label: str,
            streak_metrics: dict,
            *,
            N_MIN: int = 3,
            RSTARW_THR: float = 0.5,
            L_MED_NORM_PX: float = 12.0,
            streak_conf: float | None = None,
            logger=None,
    ):
        """
        Always prints to screen, even if a logger is present.
        """
        import numpy as np

        H, W = mask_u8.shape[:2]
        L_MED_NORM_THR = L_MED_NORM_PX / float(W)

        N = int(streak_metrics.get("N", -1))
        R_star_w = float(streak_metrics.get("R_star_w", -1.0))
        L_med_norm = float(streak_metrics.get("L_med_norm", -1.0))
        L_med_px = float(streak_metrics.get("L_med_major_px", L_med_norm * float(W)))
        phi_deg = streak_metrics.get("phi_deg", None)
        used_proj = streak_metrics.get("used_projection", None)

        n_init = streak_metrics.get("n_init", "?")
        n_groups = streak_metrics.get("n_groups", "?")
        n_merges = streak_metrics.get("n_merges", "?")

        try:
            mask_fill = float((mask_u8 > 0).sum()) / float(mask_u8.size)
        except Exception:
            mask_fill = float("nan")

        label_str = "STREAKED" if label == "streaked" else "STILL"
        try:
            conf = R_star_w if R_star_w >= 0 else float(streak_conf)
        except Exception:
            conf = float("nan")

        fails = []
        if N < N_MIN:
            fails.append(f"too few streaks: N={N} < {N_MIN}")
        if L_med_norm < L_MED_NORM_THR:
            fails.append(f"streaks too short: L_med≈{L_med_px:.1f}px < {L_MED_NORM_PX:.1f}px")
        if R_star_w < RSTARW_THR:
            fails.append(f"poor orientation coherence: R*_w={R_star_w:.2f} < {RSTARW_THR:.2f}")
        reason_str = "OK (passed all gates)" if not fails else "FAIL → " + " | ".join(fails)

        # Construct message
        msg = (
            f"[Classifier] {label_str} | "
            f"N (streak count)={N} (≥{N_MIN}), L_med_norm={L_med_norm:.3f} (≥{L_MED_NORM_THR:.3f}), "
            f"R*_w={R_star_w:.3f} (≥{RSTARW_THR:.3f}) | "
            f"mask_fill={mask_fill:.3%} | "
            f"conf={conf:.2f} | {reason_str}"
        )

        extras = []
        extras.append(f"L_med_px≈{L_med_px:.1f}")
        if phi_deg is not None:
            extras.append(f"phi≈{float(phi_deg):.1f}°")
        if used_proj is not None:
            extras.append(f"proj={'Y' if used_proj else 'N'}")
        if extras:
            msg += " | " + ", ".join(extras)

        # ✅ Always print to console
        print(msg, flush=True)

        # Log (optional)
        if logger is None and hasattr(self, "log"):
            logger = self.log
        if logger is not None:
            try:
                logger.info(msg)
            except Exception:
                pass

        return {
            "label": label,
            "N": N,
            "R_star_w": R_star_w,
            "L_med_norm": L_med_norm,
            "L_med_major_px": L_med_px,
            "mask_fill": mask_fill,
            "phi_deg": phi_deg,
            "used_projection": used_proj,
            "reasons": fails,
            "message": msg,
        }

    def tetra3_solver(
            self,
            sources_mask: np.ndarray,
            precomputed_star_centroids,
            FOV=None,
            distortion=0,
            min_pattern_checking_stars=15,
            resize_factor=1.0
    ) -> dict:
        """Find the direction in the sky of an image by calculating a "fingerprint" of the star centroids detected in the image and looking for matching fingerprints in
            a pattern catalog loaded into memory.

        Args:
            sources_mask (np.ndarray): A binary mask of detected point sources in the image, indicating the positions of stars.
            precomputed_star_centroids (np.ndarray): An array containing the precomputed centroids of the detected stars,
                                                     which are used for pattern recognition.
            FOV (float, optional): The estimated horizontal field of view of the image in degrees. Default is None.
            distortion (float, optional): The radial distortion factor to correct for lens distortion. Default is 0.
            min_pattern_checking_stars (int, optional): Number of stars used to create possible patterns to look up in database. Default is 15.

        Returns:
            dict: A dictionary containing the results of the sky direction solution, with the following keys:
                - 'RA' (float): Right Ascension in degrees, representing the celestial longitude of the solved image.
                - 'Dec' (float): Declination in degrees, representing the celestial latitude of the solved image.
                - 'Roll' (float): Roll angle in degrees, representing the rotation angle of the camera with respect to a reference direction.
                - 'FOV' (float): Horizontal Field of View in degrees, representing the angular extent of the image in the horizontal direction.
                - 'distortion' (float): Calculated distortion of the provided image
                - 'Cross-Boresight RMSE' (float): Root Mean Square Error of the cross-boresight alignment in arcseconds, indicating the accuracy of alignment.
                - 'Roll RMSE' (float): Root Mean Square Error of the roll angle estimation in arcseconds, indicating the accuracy of roll angle estimation.
                - 'Matches' (int): Number of star matches used for the solution.
                - 'Prob' (float): Probability of the solution, indicating the confidence level of the match.
                - 'T_solve' (float): Time taken to solve the image in seconds.
                - 'matched_centroids' : An Mx2 list with the (y, x) pixel coordinates in the image corresponding to each matched star
                - 'visual' : A PIL image with spots for the given centroids in white, the coarse FOV and distortion estimates in orange,
                the final FOV and distortion estimates in green. Also has circles for the catalogue stars in green or red for successful/unsuccessful match
        """
        height, width = sources_mask.shape
        # number of pattern checking stars used
        # TODO: Done 15 min_pattern_checking_stars; Implemented min_pattern_checking_stars as input param on functions: tetra3_solver, optimize_calibration_parameters, do_astrometry, config.ini
        pattern_checking_stars = min(min_pattern_checking_stars, len(precomputed_star_centroids))
        logit(f"Solving: pattern_checking_stars: {min_pattern_checking_stars}, precomputed_star_centroids: {len(precomputed_star_centroids)}", color='cyan')
        # Solving using centroids
        result = self.t3.solve_from_centroids(
            star_centroids=precomputed_star_centroids,
            size=(int(height/resize_factor), int(width/resize_factor)),
            fov_estimate=FOV,
            # fov_max_error=FOV * 0.10,  # Default None
            pattern_checking_stars=pattern_checking_stars,
            # match_radius=0.01,  # Default 0.01
            # match_threshold=1e-3,
            solve_timeout=self.cfg.solve_timeout, # 5000.0, # Default None milliseconds
            # distortion=distortion,
            return_matches=True,
            # return_visual=False,
        )
        #         star_centroids,
        #         size,
        #         fov_estimate=None,
        #         fov_max_error=None,
        #         pattern_checking_stars=8,
        #         match_radius=0.01,
        #         match_threshold=1e-3,
        #         solve_timeout=None,
        #         target_pixel=None,
        #         distortion=0,
        #         return_matches=False,
        #         return_visual=False,

        logit(f"solve_from_centroids : done")
        # plt.imshow(result['visual'])
        # plt.show(block=True)
        return result

    def optimize_calibration_parameters(
            self,
            is_trail,
            number_sources,
            bkg_threshold,
            min_size,
            max_size,
            calibration_images_dir,
            log_file_path="log/calibration_log.txt",
            calibration_params_path="calibration_params.txt",
            update_calibration=False,
            min_pattern_checking_stars=15,
            local_sigma_cell_size=36,
            sigma_clipped_sigma=3.0,
            leveling_filter_downscale_factor=4,
            src_kernal_size_x=3,
            src_kernal_size_y=3,
            src_sigma_x=1,
            src_dst=1,
            dilate_mask_iterations=1,
            scale_factors=(8, 8),
            resize_mode='downscale',
            level_filter: int = 9,
            ring_filter_type = 'mean'
    ):
        """Optimize camera calibration parameters based on astrometry analysis of
        calibration images.

        This function computes and updates the optimal calibration parameters for a camera system
        using a set of calibration images. It performs astrometry on the images to derive
        parameters like the Field of View (FOV) and distortion coefficients, and assesses
        the resulting parameters against previously stored best parameters.

        Args:
            is_trail (bool): Indicates if the sources in the image are trails (True) or points (False).
            number_sources (int): The maximum number of sources to detect in the images.
            bkg_threshold (float): The threshold value used to identify sources. Pixels whose values exceed
                `local background + threshold * local noise` will be marked as source pixels.
            min_size (int): Minimum size of detected sources (in pixels) to consider for calibration.
            max_size (int): Maximum size of detected sources (in pixels) to consider for calibration.
            calibration_images_dir (str): Directory containing calibration images for processing.
            log_file_path (str, optional): Path to the log file for recording calibration logs. Defaults to "log/calibration_log.txt".
            calibration_params_path (str, optional): Path to the file for saving calibration parameters. Defaults to "calibration_params.txt".
            update_calibration (bool, optional): If True, updates the existing calibration parameters with new images. Defaults to False.
            min_pattern_checking_stars (int, optional): Number of stars used to create possible patterns to look up in database. Default is 15.
            local_sigma_cell_size (int, optional): The size of the cells (in pixels) used for estimating local noise
                levels. Defaults to 36.
            sigma_clipped_sigma (float, optional): The sigma value to use for sigma-clipping when calculating the
                background statistics (mean, median, standard deviation). It controls how aggressive the clipping is.
                Higher values mean less clipping (more tolerance for outliers). Defaults to 3.0.
            leveling_filter_downscale_factor (int, optional): The downscaling factor to apply when creating the
                downsampled image used for local level estimation. Defaults to 4.
            src_kernal_size_x (int, optional): The width of the Gaussian kernel used for smoothing the image. Defaults to 3.
            src_kernal_size_y (int, optional): The height of the Gaussian kernel used for smoothing the image. Defaults to 3.
            src_sigma_x (float, optional): The standard deviation in the X direction for the Gaussian kernel. Defaults to 1.
            src_dst (int, optional): The depth of the output image. Defaults to 1.
            dilate_mask_iterations (int, optional): The number of iterations for dilating the mask to merge nearby
                sources. A higher value merges more pixels. Defaults to 1.
            scale_factors=(float, float optional): Downscaling factors
            resize_mode=(str, optional): resize mode
            level_filter (int, optional): level_filter size
            ring_filter_type (str, optional):  Source Ring Background Estimation Type: mean|median

        Returns:
            dict: A dictionary containing the optimized calibration parameters, including:
                - 'FOV': The average field of view determined from the calibration images.
                - 'distortion': A list containing distortion coefficients calculated from the images.
                - 'RMSE': The root mean square error associated with the new calibration parameters.
        """
        # list of the default best params
        default_best_params = {"FOV": None, "distortion": [-0.1, 0.1]}

        curr_params = {"FOV": [], "distortion": []}
        prev_params = {"FOV": [], "distortion": []}
        curr_best_params = {}
        prev_best_params = {}

        # read previous list of params from file
        if update_calibration:
            try:
                with open(calibration_params_path, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        line = line.strip()
                        if line:
                            key, value = line.split(":")
                            prev_params[key.strip()] = [float(v.strip()) for v in value.split(",")]
            except FileNotFoundError:
                logit("Error: Distortion calibration file not found.")

            # update current calibration with new images
            curr_params = prev_params

        # read best distortions calibration parameters
        try:
            with open("calibration_params_best.txt", "r") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        key, value = line.split(":")
                        prev_best_params[key.strip()] = float(value.strip())
                        default_best_params = prev_best_params
        except FileNotFoundError:
            logit("Error: Distortion calibration file not found.")

        # Check if the directory is empty
        if not os.listdir(calibration_images_dir):
            logit(f"The distortion calibration images directory {calibration_images_dir} is empty.")
            if prev_best_params != {}:
                logit(f"returning previous calibration params.")
                return prev_best_params
            else:
                logit(f"returning default calibration params.")
                return default_best_params
        else:
            # get calibration params from the new set of images
            for filename in os.listdir(calibration_images_dir):
                # calibration image path
                image_path = os.path.join(calibration_images_dir, filename)
                # do astrometry
                astrometry, _, contours_img = self.do_astrometry(
                    img=image_path,
                    is_array=False,
                    is_trail=is_trail,
                    use_photutils=False,  # TODO: NEW
                    subtract_global_bkg=False,  # TODO: NEW
                    fast=False,  # TODO: NEW
                    number_sources=number_sources,
                    bkg_threshold=bkg_threshold,
                    min_size=min_size,
                    max_size=max_size,
                    distortion_calibration_params=default_best_params,
                    log_file_path=log_file_path,
                    min_pattern_checking_stars=min_pattern_checking_stars,
                    local_sigma_cell_size=local_sigma_cell_size,
                    sigma_clipped_sigma=sigma_clipped_sigma,
                    leveling_filter_downscale_factor=leveling_filter_downscale_factor,
                    src_kernal_size_x=src_kernal_size_x,
                    src_kernal_size_y=src_kernal_size_y,
                    src_sigma_x=src_sigma_x,
                    src_dst=src_dst,
                    return_partial_images=False,
                    dilate_mask_iterations=dilate_mask_iterations,
                    level_filter=level_filter,
                    ring_filter_type=ring_filter_type
                )
                logit(f"Astrometry : {astrometry}")
                # display overlay image
                # TODO: Done timestamp_string should be the current timestamp; Milan: Implemented current_timestamp() added to utils.py and updated code there
                # timestamp_string = "2023-09-06 10:00:00"
                timestamp_string = current_timestamp()
                omega = (0.0, 0.0, 0.0)
                # TODO: Removed by Windell ???!!!
                # display_overlay_info(img, timestamp_string, astrometry, omega, scale_factors=scale_factors,
                #                      resize_mode=resize_mode)

                for key in curr_params.keys():
                    curr_params[key].append(astrometry[key])

        # Compute mean of current calibration params
        for key, values in curr_params.items():
            if values:
                mean = sum(values) / len(values)
                curr_best_params[key] = mean

        # compute mean RMSE using new params
        set_RMSE = []
        for filename in os.listdir(calibration_images_dir):
            # calibration image path
            image_path = os.path.join(calibration_images_dir, filename)
            # do astrometry
            # TODO: The new version of do_astrometry from DEVELOPER has several new params that were not accounted for here
            astrometry, _, _ = self.do_astrometry(
                img=image_path,
                is_array=False,
                is_trail=is_trail,
                use_photutils=False,  # TODO: NEW
                subtract_global_bkg=False,  # TODO: NEW
                fast=False,  # TODO: NEW
                number_sources=number_sources,
                bkg_threshold=bkg_threshold,
                min_size=min_size,
                max_size=max_size,
                distortion_calibration_params=curr_best_params,
                log_file_path=log_file_path,
                min_pattern_checking_stars=min_pattern_checking_stars,
                local_sigma_cell_size=local_sigma_cell_size,
                sigma_clipped_sigma=sigma_clipped_sigma,
                leveling_filter_downscale_factor=leveling_filter_downscale_factor,
                src_kernal_size_x=src_kernal_size_x,
                src_kernal_size_y=src_kernal_size_y,
                src_sigma_x=src_sigma_x,
                src_dst=src_dst,
                return_partial_images=False,
                dilate_mask_iterations=dilate_mask_iterations,
                level_filter=level_filter,
                ring_filter_type=ring_filter_type,
                frame_tag = None,
                export_centroids_mean_path = None,
            )
            set_RMSE.append(astrometry["RMSE"])
        curr_best_params["RMSE"] = sum(set_RMSE) / len(set_RMSE)

        # Assess the RMSE for the new calibration parameters in comparison to the previous ones
        if not update_calibration and prev_best_params != {}:
            if curr_best_params['RMSE'] > prev_best_params['RMSE']:
                return prev_best_params

        # write new calibration params to file
        with open(calibration_params_path, "w") as file:
            # Iterate over the dictionary and write each key-value pair
            for key, values in curr_params.items():
                # Convert the list of values to a string
                values_str = ', '.join(str(value) for value in values)
                # Write the key-value pair to the file
                file.write(f"{key}: {values_str}\n")

        # write best calibration params to file
        with open(calibration_params_path.replace(".txt", "_best.txt"), "w") as file:
            # Iterate over the dictionary and write each key-value pair
            for key, value in curr_best_params.items():
                # Write the key-value pair to the file
                file.write(f"{key}: {value}\n")

        return curr_best_params

    # TODO: Windell: The variables with hard coded values here should be put in the config file.
    # TODO:   Milan: All params when using do_astrometry from pueo_star_camera_operation_code.py pass all vars ...

    def do_astrometry(self, *args, **kwargs):
        """Wrapper function that handles both direct calls and multiprocessing pool calls"""
        # TODO: Set to False for PRODUCTION
        if self.test:
            logit(f"DEBUG: args received: {args}")
            logit(f"DEBUG: kwargs received: {kwargs}")

            # Get the original function's parameter names
            import inspect
            sig = inspect.signature(self._do_astrometry)
            param_names = list(sig.parameters.keys())[1:]  # Skip 'self'

            logit(f"DEBUG: _do_astrometry expects parameters: {param_names}")
            # logit(f"DEBUG: Number of args passed: {len(args)}")
            # logit(f"DEBUG: Number of kwargs passed: {len(kwargs)}")

            logit(f"\n{'-' * 60}")
            logit(f"DO_ASTROMETRY PARAMETERS")
            logit(f"{'-' * 60}")

            max_name_length = 32
            for idx, param_name in enumerate(param_names):
                param_value = args[idx+1] if len(args) > idx+1 else None
                if param_value is None:
                    formatted_value = "None"
                elif isinstance(param_value, str):
                    formatted_value = f"'{param_value}'"
                else:
                    formatted_value = str(param_value)
                logit(f"{param_name:>{max_name_length}} : {formatted_value}")

        return self._do_astrometry(*args, **kwargs)

    def _print_parameters(self, func_name, parameters):
        """Helper method to print parameters"""
        if not parameters:
            return

        max_name_length = max(len(name) for name in parameters.keys())

        logit(f"\n{'-' * 60}")
        logit(f"{func_name.upper()} PARAMETERS")
        logit(f"{'-' * 60}")

        max_name_length = 32
        for param_name, param_value in parameters.items():
            if param_value is None:
                formatted_value = "None"
            elif isinstance(param_value, str):
                formatted_value = f"'{param_value}'"
            else:
                formatted_value = str(param_value)

            logit(f"{param_name:>{max_name_length}} : {formatted_value}")

        logit(f"{'-' * 60}\n")


    def _do_astrometry(
            self,
            img,
            is_array=True,
            is_trail=True,
            use_photutils=False,
            subtract_global_bkg=False,
            fast=False,
            number_sources=20,
            bkg_threshold=3.1,
            min_size=4,
            max_size=200,
            distortion_calibration_params=None,
            log_file_path="log/test_log.txt",
            min_pattern_checking_stars=15,
            local_sigma_cell_size=36,
            sigma_clipped_sigma=3.0,
            leveling_filter_downscale_factor=4,
            src_kernal_size_x=3,
            src_kernal_size_y=3,
            src_sigma_x=1,
            src_dst=1,
            dilate_mask_iterations=1,
            return_partial_images=False,
            partial_results_path="./partial_results",
            solver='solver1',
            level_filter: int = 9,
            ring_filter_type = 'mean'
    ):
        """Perform astrometry on an input image to determine celestial coordinates.

        This function processes an image to identify celestial sources and computes their centroids,
        performing astrometric analysis to find the direction in the sky. It utilizes background estimation
        techniques, source detection, and centroid calculation. If distortion calibration parameters are provided,
        they will be applied during the solving process.

        Args:
            img (np.ndarray or str): The input image as a 2D array (if `is_array` is True) or the
                file path to an image (if `is_array` is False).
            is_array (bool, optional): Flag indicating whether the image is provided as a 2D array (True)
                or as a file path (False). Default is True.
            is_trail (bool, optional): Flag indicating whether the image represents a star trail (True)
                or point sources (False). Default is True.
            use_photutils (bool, optional): Flag indicating whether to use photutils functions. Default is False.
            number_sources (int, optional): Maximum number of sources to detect in the image. Default is 20.
            bkg_threshold (float, optional): Background threshold for source detection. Default is 3.1.
            min_size (int, optional): Minimum size of detected sources (in pixels) to consider for calibration. Default is 4.
            max_size (int, optional): Maximum size of detected sources (in pixels) to consider for calibration. Default is 200.
            distortion_calibration_params (dict, optional): Calibration parameters for distortion, containing
                'FOV' and 'distortion' coefficients. Default is an empty dictionary.
            log_file_path (str, optional): Path to the log file for recording calibration logs.
                Defaults to "log/test_log.txt".
            min_pattern_checking_stars (int, optional): Number of stars used to create possible patterns to look up in database. Default is 15.
            local_sigma_cell_size (int, optional): The size of the cells (in pixels) used for estimating local noise
                levels. Defaults to 36.
            sigma_clipped_sigma (float, optional): The sigma value to use for sigma-clipping when calculating the
                background statistics (mean, median, standard deviation). It controls how aggressive the clipping is.
                Higher values mean less clipping (more tolerance for outliers). Defaults to 3.0.
            leveling_filter_downscale_factor (int, optional): The downscaling factor to apply when creating the
                downsampled image used for local level estimation. Defaults to 4.
            src_kernal_size_x (int, optional): The width of the Gaussian kernel used for smoothing the image. Defaults to 3.
            src_kernal_size_y (int, optional): The height of the Gaussian kernel used for smoothing the image. Defaults to 3.
            src_sigma_x (float, optional): The standard deviation in the X direction for the Gaussian kernel. Defaults to 1.
            src_dst (int, optional): The depth of the output image. Defaults to 1.
            dilate_mask_iterations (int, optional): The number of iterations for dilating the mask to merge nearby
                sources. A higher value merges more pixels. Defaults to 1.
            return_partial_images (bool, optional): Whether to return intermediate partial images for debugging.
                Defaults to False.
            partial_results_path (str, optional): The directory path where intermediate results will be saved
                if `return_partial_images` is True. Defaults to "./partial_results/".
            solver (str, optional): solver1|solver2: genuine or cedar solver.
            level_filter (int): The size of the star level filter, shall be 5..199 and an odd number.
            ring_filter_type (str): Source Ring Background Estimation Type: mean|median
        Returns:
            tuple: A tuple containing:
                - astrometry (dict): A dictionary with astrometric solutions including matched centroids and
                  additional details such as RA, Dec, Roll, etc.
                - precomputed_star_centroids (np.ndarray): An array of computed centroids for detected sources.
                - contours_img (np.ndarray): An image with drawn contours for detected sources.
        """
        cleaned_img = None

        # Defaults for 99.5th percentile values
        p999_original = None
        p999_masked_original = None

        def get_params():
            return {
                'is_array': is_array,
                'is_trail': is_trail,
                'use_photutils': use_photutils,
                'subtract_global_bkg': subtract_global_bkg,
                'fast': fast,
                'number_sources': number_sources,
                'bkg_threshold': bkg_threshold,
                'min_size': min_size,
                'max_size': max_size,
                'distortion_calibration_params': distortion_calibration_params,
                'log_file_path': log_file_path,
                'min_pattern_checking_stars': min_pattern_checking_stars,
                'local_sigma_cell_size': local_sigma_cell_size,
                'sigma_clipped_sigma': sigma_clipped_sigma,
                'leveling_filter_downscale_factor': leveling_filter_downscale_factor,
                'src_kernal_size_x': src_kernal_size_x,
                'src_kernal_size_y': src_kernal_size_y,
                'src_sigma_x': src_sigma_x,
                'src_dst': src_dst,
                'dilate_mask_iterations': dilate_mask_iterations,
                'return_partial_images': return_partial_images,
                'partial_results_path': partial_results_path,
                'cedar': cedar_data,
                'solver': solver
            }

        t0 = time.monotonic()
        solver_exec_time = 0.0
        # Capture local params, but remove image
        cedar_data = {}
        resize_factor = 1.0

        self.solver = solver
        logit(f'solver: {self.solver}')
        if distortion_calibration_params is None:
            distortion_calibration_params = {}
        # read image in array format. From camera
        if is_array:
            logit(f"is_array = TRUE. Creating img_bgr.")
            # Scale the 16-bit values to 8-bit (0-255) range
            # The scale_factor = 2**14 - 1 = 16383.0
            scale_factor = float(2 ** self.cfg.pixel_well_depth) - 1
            scaled_data = ((img / scale_factor) * 255).astype('uint8')
            # Create a BGR image
            img_bgr = cv2.merge([scaled_data, scaled_data, scaled_data])

        # read image from file. For debugging using png/tif
        if not is_array:
            logit(f"is_array = FALSE. converting to grayscale")
            # img_bgr = read_image_BGR(img)
            img = read_image_grayscale(img)
            img_bgr = cv2.merge([img, img, img])

        print_img_info(img)
        print_img_info(img_bgr, 'bgr')

        # Save Partial image. For debugging
        if return_partial_images:
            # Create partial results folder. For debugging
            if not os.path.exists(partial_results_path):
                os.makedirs(partial_results_path)
            # save input image
            cv2.imwrite(os.path.join(partial_results_path, "0 - Input Image-bgr.png"), img_bgr)
            cv2.imwrite(os.path.join(partial_results_path, "0 - Input Image-img.png"), img)

        # Source finder
        # global background estimation
        total_exec_time = 0.0
        if subtract_global_bkg:
            logit(f"--------Subtract global background--------")
            global_cleaned_img, global_exec_time = timed_function(
                global_background_estimation, img, sigma_clipped_sigma, return_partial_images, partial_results_path
            )
            img = global_cleaned_img
            total_exec_time += global_exec_time

        # This is False in config.ini as ast_use_photoutils param
        # TODO: This is forced for now until cedar detect is implemented
        p999_original = None
        p999_masked_original = None
        if self.solver in ['solver1', 'solver3']: #  or True:
            if True:
                ##### Uncomment the following lines to use the source-finding functions independently
                # (cleaned_img, background_img), local_exec_time = timed_function(local_levels_background_estimation, img, log_file_path, leveling_filter_downscale_factor, return_partial_images, partial_results_path)    # cleaned_img, background_img = sextractor_background_estimation(img, return_partial_images)
                # (masked_image, estimated_noise), find_sources_exec_time = timed_function(find_sources, img, background_img,fast, bkg_threshold, local_sigma_cell_size,
                #                            src_kernal_size_x, src_kernal_size_y, src_sigma_x, src_dst, return_partial_images, partial_results_path)
                # (masked_image, sources_mask, sources_contours), top_sources_exec_time = timed_function(select_top_sources, img, masked_image, estimated_noise, fast, number_sources=number_sources,
                #                                                             min_size=min_size, max_size=max_size,
                #                                                             dilate_mask_iterations=dilate_mask_iterations,
                #                                                             return_partial_images=return_partial_images, partial_results_path=partial_results_path)
                # source_finder_exec_time = int(local_exec_time) + int(find_sources_exec_time) + int(top_sources_exec_time)
                #####
                # source finder pipeline
                if True:
                    # Direct call to show exception on source_finder
                    source_finder_exec_time = time.monotonic()
                    masked_image, sources_mask, sources_contours, p999_original, p999_masked_original = source_finder(
                        self.cfg,
                        img,
                        log_file_path,
                        leveling_filter_downscale_factor,
                        fast,
                        bkg_threshold,
                        local_sigma_cell_size,
                        src_kernal_size_x,
                        src_kernal_size_y,
                        src_sigma_x,
                        src_dst,
                        number_sources,
                        min_size,
                        max_size,
                        dilate_mask_iterations,
                        is_trail,
                        return_partial_images,
                        partial_results_path,
                        level_filter,
                        ring_filter_type,
                        # NEW: pass config
                        noise_pair_sep_full=int(self.cfg.noise_pair_sep_full),
                        simple_threshold_k=float(self.cfg.simple_threshold_k),
                        hyst_k_high=float(self.cfg.hyst_k_high),
                        hyst_k_low=float(self.cfg.hyst_k_low),
                        hyst_min_area=int(self.cfg.hyst_min_area),
                        hyst_close_kernel=int(self.cfg.hyst_close_kernel),
                        hyst_sigma_gauss=float(self.cfg.hyst_sigma_gauss),
                        merge_min_area=int(self.cfg.merge_min_area),
                        merge_gap_along=int(self.cfg.merge_gap_along),
                        merge_gap_cross=int(self.cfg.merge_gap_cross),
                        merge_ang_tol=int(self.cfg.merge_ang_tol_deg),
                    )
                    # TODO: Windell figure out the image naming (cleaned_img/masked_image)
                    source_finder_exec_time = time.monotonic() - source_finder_exec_time

######################
                # --- Classify using EXISTING mask (no re-threshold) + allow override ---
                logit(f"--------Running classifier.--------", color='cyan')
                mode = self.cfg.ast_centroid_mode

                # in pueo_star_camera_operation_code.py after constructing Astrometry:

                if mode == "auto":
                    # ---- Build ellipse list from the mask (no merging here) ----
                    mask_u8 = (sources_mask > 0).astype(np.uint8) * 255
                    ell = self._fit_ellipses_from_mask(
                        mask_u8,
                        min_area=self.cfg.ellipse_min_area_px,
                        min_major=self.cfg.ellipse_major_min_px,
                    )
                    logit(f"[Classifier] contours={len(sources_contours)}  ell={len(ell)}")

                    # ---- Classify frame as still vs streaked ----
                    label, conf, M = self.classify_frame_still_biased(
                        ell,
                        ar_elong=self.cfg.ellipse_aspect_ratio_min,
                        min_elong=self.cfg.min_elongated_count,
                        min_L=self.cfg.min_median_major_px,
                        min_R=self.cfg.min_orientation_coherence,
                        min_frac=self.cfg.min_elongated_fraction,
                        min_conf=self.cfg.min_confidence,
                    )

                    logit(f"[Classifier] label={label}  conf={conf:.2f}  metrics={M}", color='cyan')
                    logit(f"label: {label}")

                    is_trail = (label == "streaked")
                    suggest_is_trail = (label == "streaked")
                    mode_str = f"auto (classifier suggests is_trail = {suggest_is_trail})"
                    logit(f"[Astrometry] Centroid mode = {mode} → {mode_str}", color='cyan')

                    # --- Estimate angular velocity from classifier ellipses (deg/s) ---
                    if label == "streaked":
                        try:
                            plate_scale_arcsec_per_px = self.cfg.plate_scale_arcsec_per_px
                            # TODO: This should not be in cfg!
                            exposure_time_s = self.cfg.exposure_time_s

                            if (
                                    plate_scale_arcsec_per_px > 0.0
                                    and exposure_time_s > 0.0
                                    and len(ell) > 0
                            ):
                                omega_deg_s, diag_omega = self._estimate_omega_from_ellipses_plate_scale(
                                    ell,
                                    plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                                    exposure_time_s=exposure_time_s,
                                    image_shape=img.shape[:2],
                                    ar_min=float(self.cfg.ellipse_aspect_ratio_min),
                                    L_min_px=float(self.cfg.min_median_major_px),
                                    area_min_px=float(self.cfg.ellipse_min_area_px),
                                )
                                if omega_deg_s is not None:
                                    wx_deg, wy_deg, wz_deg = omega_deg_s
                                    w_mag = float((wx_deg ** 2 + wy_deg ** 2 + wz_deg ** 2) ** 0.5)
                                    logit(
                                        f"\n[Trail Omega] (Auto mode) "
                                        f"plate_scale={plate_scale_arcsec_per_px:.6f} arcsec/px, "
                                        f"texp={exposure_time_s:.4f} s, "
                                        f"wx={wx_deg:.4f} deg/s, "
                                        f"wy={wy_deg:.4f} deg/s, "
                                        f"wz={wz_deg:.4f} deg/s, "
                                        f"N={diag_omega.get('N_used', 0)}, "
                                        f"rms={diag_omega.get('resid_rms_deg_s', float('nan')):.4g} deg/s "
                                        f"w_mag_deg_s : {w_mag:.4f}\n"
                                    )
                                    with open(log_file_path, "a") as file:
                                        file.write(f"\n[Trail Omega] (Auto mode)\n")
                                        file.write(f"plate_scale={plate_scale_arcsec_per_px:.6f} arcsec/px\n")
                                        file.write(f"texp={exposure_time_s:.3f} s\n")
                                        file.write(f"wx={wx_deg:.4f} deg/s\n")
                                        file.write(f"wy={wy_deg:.4f} deg/s\n")
                                        file.write(f"wz={wz_deg:.4f} deg/s\n")
                                        file.write(f"N={diag_omega.get('N_used', 0)}\n")
                                        file.write(f"rms={diag_omega.get('resid_rms_deg_s', float('nan')):.4g} deg/s\n")
                                        file.write(f"w_mag_deg_s : {w_mag:.4f}\n")
                                else:
                                    logit("[Trail Omega] Not enough valid ellipses to solve for angular velocity")
                            else:
                                logit(
                                    "[Trail Omega] Skipped: plate_scale_arcsec_per_px or exposure_time_s "
                                    "not set (>0) in cfg or no ellipses"
                                )
                        except Exception as e:
                            logit(f"[Trail Omega] ERROR estimating angular velocity: {e}")

                    # ---- Choose centroid path based on classifier ----
                    if suggest_is_trail:
                        (precomputed_star_centroids, contours_img), centroids_exec_time = timed_function(
                            self._trail_ellipse_adapter,
                            masked_image=masked_image,
                            sources_mask=sources_mask,
                            original_image=img_bgr,
                            min_area_px=int(self.cfg.ellipse_min_area_px),
                            aspect_ratio_min=float(self.cfg.ellipse_aspect_ratio_min),
                            use_uniform_length=bool(self.cfg.ellipse_use_uniform_length),
                            uniform_length_mode=str(self.cfg.ellipse_uniform_length_mode),
                            return_partial_images=return_partial_images,
                            partial_results_path=partial_results_path,
                        )
                    else:
                        (precomputed_star_centroids, contours_img), centroids_exec_time = timed_function(
                            compute_centroids_from_still,
                            masked_image=masked_image,
                            sources_contours=sources_contours,
                            img=img_bgr,
                            log_file_path=log_file_path,
                            return_partial_images=return_partial_images,
                            partial_results_path=partial_results_path,
                        )
                elif mode == "trail":
                    # NO classifier, NO metrics printing outside auto
                    suggest_is_trail = True
                    mode_str = "override to trail mode"
                    logit(f"[Astrometry] Centroid mode = {mode} → {mode_str}")
                    # --- Estimate angular velocity from ellipses (forced trail mode) ---
                    try:
                        # Build ellipse list from the current mask (same as auto branch)
                        mask_u8 = (sources_mask > 0).astype(np.uint8) * 255
                        ell = self._fit_ellipses_from_mask(
                            mask_u8,
                            min_area=int(self.cfg.ellipse_min_area_px),
                            min_major=int(self.cfg.ellipse_major_min_px),
                        )
                        plate_scale_arcsec_per_px = self.cfg.plate_scale_arcsec_per_px
                        exposure_time_s = self.cfg.exposure_time_s

                        if (
                                plate_scale_arcsec_per_px > 0.0
                                and exposure_time_s > 0.0
                                and len(ell) > 0
                        ):
                            omega_deg_s, diag_omega = self._estimate_omega_from_ellipses_plate_scale(
                                ell,
                                plate_scale_arcsec_per_px=plate_scale_arcsec_per_px,
                                exposure_time_s=exposure_time_s,
                                image_shape=img_bgr.shape[:2],
                                ar_min=float(self.cfg.ellipse_aspect_ratio_min),
                                L_min_px=float(self.cfg.min_median_major_px),
                                area_min_px=float(self.cfg.ellipse_min_area_px),
                            )
                            if omega_deg_s is not None:
                                wx_deg, wy_deg, wz_deg = omega_deg_s
                                w_mag = float((wx_deg ** 2 + wy_deg ** 2 + wz_deg ** 2) ** 0.5)
                                logit(
                                    f"[Trail Omega] (forced mode) "
                                    f"plate_scale={plate_scale_arcsec_per_px:.6f} arcsec/px, "
                                    f"texp={exposure_time_s:.4f} s, "
                                    f"wx={wx_deg:.4f} deg/s, "
                                    f"wy={wy_deg:.4f} deg/s, "
                                    f"wz={wz_deg:.4f} deg/s, "
                                    f"N={diag_omega.get('N_used', 0)}, "
                                    f"rms={diag_omega.get('resid_rms_deg_s', float('nan')):.4g} deg/s "
                                    f"w_mag_deg_s : {w_mag:.4f}\n"
                                )
                                with open(log_file_path, "a") as file:
                                    file.write(f"\n[Trail Omega] (forced mode)\n")
                                    file.write(f"plate_scale={plate_scale_arcsec_per_px:.6f} arcsec/px\n")
                                    file.write(f"texp={exposure_time_s:.3f} s\n")
                                    file.write(f"wx={wx_deg:.4f} deg/s\n")
                                    file.write(f"wy={wy_deg:.4f} deg/s\n")
                                    file.write(f"wz={wz_deg:.4f} deg/s\n")
                                    file.write(f"N={diag_omega.get('N_used', 0)}\n")
                                    file.write(f"rms={diag_omega.get('resid_rms_deg_s', float('nan')):.4g} deg/s\n")
                                    file.write(f"w_mag_deg_s : {w_mag:.4f}\n")
                            else:
                                logit(
                                    "[Trail Omega] (forced mode) "
                                    "Not enough valid ellipses to solve for angular velocity"
                                )
                        else:
                            logit(
                                "[Trail Omega] (forced mode) skipped: "
                                "missing plate_scale_arcsec_per_px/exposure_time_s or no ellipses"
                            )
                    except Exception as e:
                        logit(f"[Trail Omega] (forced mode) ERROR estimating angular velocity: {e}")
                    (precomputed_star_centroids, contours_img), centroids_exec_time = timed_function(
                        self._trail_ellipse_adapter,
                        masked_image=masked_image,
                        sources_mask=sources_mask,
                        original_image=img_bgr,
                        min_area_px=int(self.cfg.ellipse_min_area_px),
                        aspect_ratio_min=float(self.cfg.ellipse_aspect_ratio_min),
                        use_uniform_length=bool(self.cfg.ellipse_use_uniform_length),
                        uniform_length_mode=str(self.cfg.ellipse_uniform_length_mode),
                        return_partial_images=return_partial_images,
                        partial_results_path=partial_results_path,
                    )
                elif mode == "still":
                    # NO classifier, NO metrics logiting outside auto
                    suggest_is_trail = False
                    mode_str = "override to still mode"
                    logit(f"[Astrometry] Centroid mode = {mode} → {mode_str}")
                    (precomputed_star_centroids, contours_img), centroids_exec_time = timed_function(
                        compute_centroids_from_still,
                        masked_image=masked_image,
                        sources_contours=sources_contours,
                        img=img_bgr,
                        log_file_path=log_file_path,
                        return_partial_images=return_partial_images,
                        partial_results_path=partial_results_path,
                    )

                else:
                    # Unknown → behave like still; NO classifier or metrics printing
                    suggest_is_trail = False
                    mode_str = f"unknown '{mode}' → defaulting to still"
                    logit(f"[Astrometry] Centroid mode = {mode} → {mode_str}")

                    (precomputed_star_centroids, contours_img), centroids_exec_time = timed_function(
                        compute_centroids_from_still,
                        masked_image=masked_image,
                        sources_contours=sources_contours,
                        img=img_bgr,
                        log_file_path=log_file_path,
                        return_partial_images=return_partial_images,
                        partial_results_path=partial_results_path,
                    )

                total_exec_time += (
                        float(source_finder_exec_time)
                        + float(centroids_exec_time)
                )

        if self.solver == 'solver2':
            solver2_t0 = time.monotonic()
            # The contours_img does NOT exist in solver2 path therefor we just copy orig image as contours_img
            contours_img = img_bgr

            # TODO: Windell code DOES NOT WORK!!!
            # if contours_img is None or not isinstance(contours_img, np.ndarray):
            #     contours_img = img_bgr

            self.cedar.tetra3_solver = self.tetra3_solver
            self.cedar.solver_name = self.solver_name
            self.cedar.test = self.test
            self.cedar.test_data = self.test_data
            self.cedar.return_partial_images = return_partial_images
            self.cedar.partial_results_path = partial_results_path
            self.cedar.distortion_calibration_params = distortion_calibration_params
            self.cedar.min_pattern_checking_stars = min_pattern_checking_stars
            precomputed_star_centroids, cedar_star_centroids, resize_factor, astrometry, solver_exec_time = self.cedar.get_centroids(img_bgr, img)
            total_exec_time += (time.monotonic() - solver2_t0)
        # Solve image using precomputed centroids
        # TODO: ERROR> precomputed_star_centroids != []:
        # Comparing variable of type np.ndarray to an empty list does not really work!!!
        # To check if list is not empty you can do:
        #   1. if array.size:
        #   2. if array.any():

        # if precomputed_star_centroids != []:
        # Milan: Check if the array is not empty effectively

        # Calculate and add RMS to astrometry result
        height, width = img.shape
        image_size = (int(height / resize_factor), int(width / resize_factor))
        if isinstance(precomputed_star_centroids, np.ndarray) and precomputed_star_centroids.shape[0]:
            logit("--------Get direction in the sky using solver of tetra3/astrometry.net --------")
            logit(f'Image: {img.shape[0]}x{img.shape[1]} {img.dtype} {img.size} distortion: {distortion_calibration_params}')

            max_c = 0 or precomputed_star_centroids.shape[0]
            if distortion_calibration_params and solver == 'solver1':
                astrometry, solver_exec_time = timed_function(
                    self.tetra3_solver,
                    img,
                    precomputed_star_centroids[:max_c, :2],
                    FOV=distortion_calibration_params["FOV"],
                    distortion=distortion_calibration_params["distortion"],
                    min_pattern_checking_stars=min_pattern_checking_stars,
                    resize_factor=resize_factor
                )
            elif solver == 'solver2':
                # Astrometry already created as part of cedar_detect (self.cedar.get_centroids)
                pass
            elif solver == 'solver3':
                # Process the centroids with configuration
                astrometry = {}
                an_solver = AstrometryNet(self.cfg, self.log)
                try:
                    astrometry, solver_exec_time = timed_function(
                        an_solver.process_centroids,
                        precomputed_star_centroids[:max_c],
                        image_size,
                        output_base="6.0 - astrometry.net-solve-field-centroids",
                        output_dir=self.cfg.partial_results_path  # "./astrometry_results"
                    )

                    logit(f"Processing successful: {astrometry['success']}", color='green')
                    if astrometry['success']:
                        self.log.debug("Solution files created:")
                        for name, path in astrometry['solution_files'].items():
                            self.log.debug(f"  {name}: {path}")
                except Exception as e:
                    logit(f"Error processing centroids: {e}", color='red')
                    logit(f"Stack trace:\n{traceback.format_exc()}")
                    self.log.error(f"Error processing centroids: {e}")
                    self.log.error(f"Stack trace:\n{traceback.format_exc()}")
                logit("Done with solver3!")
            else:
                pass

            # Add execution time
            astrometry["precomputed_star_centroids"] = precomputed_star_centroids.shape[0]
            if self.solver == 'solver2':
                astrometry['cedar_detect'] = self.cedar.cd_solutions.copy()
            astrometry["params"] = get_params()
            astrometry["solver_exec_time"] = solver_exec_time
            astrometry['solver'] = self.solver
            astrometry['solver_name'] = self.solver_name
            # This is OK!!!
            astrometry['p999_original'] = p999_original
            astrometry['p999_masked_original'] = p999_masked_original

            logit(f"Astrometry: {str(astrometry)}")

            # Draw valid matched star contours (Green)
            color_green = (0, 255, 0)  # Green
            color_red = (0, 0, 255)  # Red
            color_blue = (255, 0, 0)  # Blue
            color_yellow = (0, 255, 255)  # BGR Yellow (Blue=0, Green=255, Red=255)

            def draw_centroids(centroids, color, radius_div=80):
                """
                radius_div:
                    160 - detected stars
                    120 - candidates stars
                    80 - confirmed stars - matched centroids
                """
                sources_radius = img.shape[0] / radius_div
                # Centroids from astrometry solution are a list, candidates are an np.ndarray
                centroids = centroids[:, :2] if isinstance(centroids, np.ndarray) else centroids
                for idx, (y, x) in enumerate(centroids):
                    # draw contour
                    cv2.circle(contours_img, (int(x * resize_factor), int(y * resize_factor)), int(sources_radius),
                               color, 4)
                    if idx < 5 and radius_div == 160:
                        cv2.circle(contours_img, (int(x * resize_factor), int(y * resize_factor)), int(sources_radius),
                                   color_yellow, 4)

                    # cv2.circle(contours_img, (int(x), int(y)), 1, color_green, -1)

            if self.solver == 'solver2':
                # Draw cedar-detected centroids
                if cedar_star_centroids.shape[0]:
                    draw_centroids(cedar_star_centroids, color_red, 160)

                # Draw cedar-detected centroids filtered
                if precomputed_star_centroids.shape[0]:
                    draw_centroids(precomputed_star_centroids, color_blue, 120)

            # Draw cedar-detected centroids
            if astrometry.get('FOV') is not None:
                draw_centroids(astrometry['matched_centroids'], color_green, 80)

            with suppress(TypeError, ValueError, IndexError):
                #                                              DETECT                      ASTRO RESULTS
                try:
                    rms = self.compute.rms_errors_from_centroids(astrometry, image_size)
                    astrometry = astrometry | rms   # Merge - union two dicts
                except (TypeError, ValueError) as e:
                    # pass
                    self.log.error(f"Failed to compute RMS: {e}")
                    self.log.error(f"Exception type: {type(e).__name__}")
                    self.log.error(f"Stack trace:\n{traceback.format_exc()}")
                    logit(f"Stack trace:\n{traceback.format_exc()}")
        else:
            logit('No centroids found, skipping astrometry solving.', color='red')
            astrometry = {}
            # These are required for the overlay.
            astrometry['solver'] = self.solver
            astrometry['solver_name'] = self.solver_name
            # Regardless of solution we still want to return
            astrometry['p999_original'] = p999_original
            astrometry['p999_masked_original'] = p999_masked_original
            # Example combined:
            # astrometry['p999'] = (p999_original, p999_masked_original)
            # Access: p999_original = astrometry['p999'][0]

        total_exec_time = time.monotonic() - t0
        astrometry["total_exec_time"] = total_exec_time if astrometry else None
        logit(f'  do_astrometry completed in {total_exec_time:.2f}s.')
        return astrometry, precomputed_star_centroids, contours_img

    def test_astrometry(self, img_path, display=False):
        """Do astrometry on existing files.

        Set display=False to only save output images, and no display.
        """
        # params
        cfg = Config()
        cfg.ast_is_array = False

        use_photutils = False
        default_best_params = {"FOV": cfg.lab_fov,
                               "distortion": [cfg.lab_distortion_coefficient_1, cfg.lab_distortion_coefficient_2]}
        fast = False
        subtract_global_bkg = False
        log_file_path = "log/"
        partial_results_path = "./partial_results/"

        # image specific paths
        img = img_path
        if "/" in img:
            img_name = img_path.split("/")[-1].split(".")[0]
        else:
            img_name = img_path.split(".")[0]
        img_log_file_path = f"{os.path.join(log_file_path, 'log_' + img_name)}.txt"
        img_partial_results_path = os.path.join(partial_results_path, img_name)
        # create partial results dir
        if not os.path.exists(img_partial_results_path):
            os.makedirs(img_partial_results_path)

        # perform astrometry
        astrometry, curr_star_centroids, contours_img = self.do_astrometry(
            img,
            cfg.ast_is_array,
            cfg.ast_is_trail,
            use_photutils,
            subtract_global_bkg,
            fast,
            cfg.img_number_sources,
            cfg.img_bkg_threshold,
            cfg.img_min_size,
            cfg.img_max_size,
            default_best_params,
            img_log_file_path,
            cfg.min_pattern_checking_stars,
            cfg.local_sigma_cell_size,
            cfg.sigma_clipped_sigma,
            cfg.leveling_filter_downscale_factor,
            cfg.src_kernal_size_x,
            cfg.src_kernal_size_y,
            cfg.src_sigma_x,
            return_partial_images=cfg.return_partial_images,
            partial_results_path=img_partial_results_path,
            solver='solver2',
            level_filter=9,
            ring_filter_type='mean',
        )
        # display overlay image
        timestamp_string = current_timestamp("%d-%m-%Y-%H-%M-%S")
        omega = (0.0, 0.0, 0.0)
        # display_overlay_info(img, timestamp_string, astrometry, omega, display=True, image_filename=None, downscale_factors=(8, 8)):
        display_overlay_info(
            img,
            timestamp_string,
            astrometry,
            omega,
            display=display,
            image_filename=img_name,
            partial_results_path=img_partial_results_path,
            scale_factors=cfg.scale_factors,
            resize_mode=cfg.resize_mode
        )

    def run_tests(self, dir_path="./test_images", img_path=None, display=False):
        """Run astrometry on a single image or all images in a specified directory."""
        # if no image path is given run tests on all images in a folder
        if img_path is None:
            for file in os.listdir(dir_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    # image specific paths
                    img_path = os.path.join(dir_path, file)
                    logit("#####################################################")
                    logit(f"Processing : {img_path}")
                    logit("#####################################################")
                    self.test_astrometry(img_path, display)
        else:
            if img_path.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                # image specific paths
                self.test_astrometry(img_path, display)


# RUN LOCAL TESTS
if __name__ == "__main__":
    astrometry = Astrometry(database_name=None)
    # run tests
    # run_tests(img_path="./test_images/cloudy_20240422.png", display=True)
    if False:
        astrometry.run_tests(dir_path="../test_images", display=False)

    # Generate optimal calibration params from a set of images
    # optimize_calibration_parameters(is_trail,number_sources, min_size, max_size, calibration_images_dir="../data/calibration_images",log_file_path="log/calibration_log.txt", calibration_params_path="calibration_params.txt", update_calibration=True)

    # Generate database
    # tetra3_generate_database(star_catalog="bsc5", max_fov=14, star_max_magnitude=7, output_name="fov_13_5_bsc5_database.npz")

    # The ‘BSC5’ data is available from <http://tdc-www.harvard.edu/catalogs/bsc5.html> (use byte format file) and
    # ‘hip_main’ and ‘tyc_main’ are available from <https://cdsarc.u-strasbg.fr/ftp/cats/I/239/> (save the appropriate .dat file).
    # The downloaded catalogue must be placed in the tetra3 directory.
    # Windel database
    print('Creating tetra3 Database: ')
    t0 = time.monotonic()
    astrometry.t3_cedar.generate_database(
        max_fov=10.79, ## 2.58,
        # min_fov=2.1,
        save_as="fov_10.79_mag9.0_tyc_v11_cedar_database.npz",
        star_catalog="tyc_main",  # tyc_main
        # pattern_stars_per_fov=10,  # Removed for test, Default: 150
        # verification_stars_per_fov=20, # was 20
        star_max_magnitude=9,
        pattern_max_error=0.005,
        # star_min_separation=0.01,  # (this is in degrees)
        # pattern_max_error=0.005,
        # simplify_pattern=False,
        # range_ra=None,
        # range_dec=None,
        # presort_patterns=True,
        # save_largest_edge=False,
        multiscale_step=1.5
    )
    from common import get_dt

    print(f'Completed in {get_dt(t0)}.')

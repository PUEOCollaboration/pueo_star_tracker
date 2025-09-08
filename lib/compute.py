"""
PUEO Star Tracker Project - Astrometric Computation Library

This module provides the Compute class for performing various astrometric calculations
including angular velocity computation, RMS calculations, and other astronomical
transformations. The library is designed to support the PUEO star tracking system.

Key Features:
- Conversion between astronomical coordinates and rotation matrices
- Angular velocity calculation between successive astrometric solutions
- Future support for RMS error calculations and other astrometric metrics

The implementation follows standard astronomical conventions and uses rigorous
mathematical transformations to ensure accurate results.
"""

# Python Internal Imports
from logging import getLogger

#External Import Libs
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple

# Custom Import Libs

# Constants
DEG2ARCSEC = 3600.0
RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0


class Compute:
    """
    A computation class for astrometric calculations in the PUEO Star Tracker system.

    This class provides methods for converting between astronomical coordinate systems,
    calculating angular velocities between successive orientations, and will include
    additional astrometric computations such as RMS error calculations.

    The implementation follows rigorous mathematical standards and astronomical
    conventions to ensure accurate and reliable results for star tracking applications.

    Attributes:
        log (Logger): Logger instance for tracking computation events and errors
    """

    def __init__(self, log=None):
        """
        Initialize the Compute class with an optional logger.

        Args:
            log (Logger, optional): Logger instance for tracking computation events.
                If not provided, a default logger named 'pueo' will be used.
        """
        self.log = log or getLogger('pueo')

    @staticmethod
    def _radec_roll_to_rotation(ra_deg, dec_deg, roll_deg):
        """
        Convert RA, Dec, and Roll angles to a rotation matrix using standard astronomical transformations.

        This function constructs the rotation matrix that transforms from the International
        Celestial Reference Frame (ICRF) to the camera's body frame. The transformation
        follows the standard astronomical convention:

        1. Rotate by RA around the Z-axis (pointing toward North Celestial Pole)
        2. Rotate by -Dec around the Y-axis (negative because declination increases northward
           from the celestial equator, which corresponds to a negative rotation in the
           right-handed coordinate system where Y points east)
        3. Rotate by Roll around the X-axis (now aligned with the camera boresight)

        The resulting matrix R satisfies: v_camera = R @ v_icrf

        Args:
            ra_deg (float): Right Ascension in degrees
            dec_deg (float): Declination in degrees
            roll_deg (float): Roll angle in degrees

        Returns:
            numpy.ndarray: 3x3 rotation matrix representing the camera orientation
        """
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)
        roll = np.deg2rad(roll_deg)

        # Standard astronomical sequence: R = R_roll(X) @ R_-dec(Y) @ R_ra(Z)
        R_z = Rotation.from_rotvec([0, 0, ra]).as_matrix()  # Rotate by RA around Z
        R_y = Rotation.from_rotvec([0, -dec, 0]).as_matrix()  # Rotate by -Dec around Y
        R_x = Rotation.from_rotvec([roll, 0, 0]).as_matrix()  # Rotate by Roll around X

        # Combine the rotations: First RA, then Dec, then Roll.
        return R_x @ R_y @ R_z

    @staticmethod
    def _decompose_rotation_to_local_axes(R1, R2):
        """
        Decompose the relative rotation between two orientations into camera-axis components.

        This function calculates the components of the angular velocity vector that would
        transform the camera from orientation R1 to orientation R2. The method involves:

        1. Computing the relative rotation matrix: R_rel = R2 @ R1.T
        2. Extracting the rotation vector from R_rel using matrix logarithm
        3. Projecting this rotation vector onto the camera's local axes (X, Y, Z) defined by R1

        The rotation vector represents the integrated angular displacement (ω * Δt) between
        the two orientations. Its projection onto the camera axes gives the components of
        this displacement along each axis.

        Args:
            R1 (numpy.ndarray): 3x3 rotation matrix representing initial camera orientation
            R2 (numpy.ndarray): 3x3 rotation matrix representing final camera orientation

        Returns:
            tuple: Components of the rotation vector along camera axes:
                - d_roll_rad: Component along Z-axis (roll/boresight direction) in radians
                - d_az_rad: Component along X-axis (azimuth/right direction) in radians
                - d_el_rad: Component along Y-axis (elevation/up direction) in radians
        """
        # Calculate the relative rotation matrix from frame1 to frame2
        R_rel = R2 @ R1.T  # This is R_{2<-1} = R2 * R1^{-1}

        # Extract rotation vector from relative rotation matrix
        # This vector represents the axis*angle rotation that transforms R1 to R2
        # Its magnitude is the rotation angle in radians, direction is rotation axis
        delta_theta_vec = Rotation.from_matrix(R_rel).as_rotvec()

        # The rotation vector magnitude should be < π for unambiguous interpretation
        rotation_angle = np.linalg.norm(delta_theta_vec)
        if rotation_angle > np.pi:
            # Handle large rotations by taking the complementary angle
            delta_theta_vec = delta_theta_vec * (1 - 2 * np.pi / rotation_angle)

        # Get the axes of the initial camera frame (R1)
        # The columns of R1 are the world axes expressed in the camera frame
        right = R1[:, 0]  # Camera's X-axis (azimuth/right direction)
        up = R1[:, 1]  # Camera's Y-axis (elevation/up direction)
        forward = R1[:, 2]  # Camera's Z-axis (roll/boresight direction)

        # Project the rotation vector onto the initial camera axes
        # These projections give the components of angular displacement along each axis
        d_roll = np.dot(delta_theta_vec, forward)  # Z-axis component
        d_az = np.dot(delta_theta_vec, right)  # X-axis component
        d_el = np.dot(delta_theta_vec, up)  # Y-axis component

        return d_roll, d_az, d_el

    def angular_velocity(self, astrometry_curr, astrometry_prev, delta_t_sec):
        """
        Compute angular velocity between two astrometry solutions.

        This function calculates the instantaneous angular velocity of the camera
        between two orientations specified by their RA, Dec, and Roll values. The
        calculation involves:

        1. Converting both orientations to rotation matrices using standard
           astronomical transformations
        2. Computing the relative rotation between the orientations
        3. Decomposing this rotation into components along the camera's local axes
        4. Converting these displacement components to angular velocity by dividing
           by the time difference

        The resulting angular velocity vector components represent the camera's
        rotational speed around its own axes (roll, azimuth, elevation) during
        the time interval.

        Args:
            astrometry_curr (dict): Current astrometry solution with keys:
                'RA', 'Dec', 'Roll' (all in degrees)
            astrometry_prev (dict): Previous astrometry solution with keys:
                'RA', 'Dec', 'Roll' (all in degrees)
            delta_t_sec (float): Time difference between solutions in seconds

        Returns:
            dict: Dictionary containing angular velocities in deg/s for:
                - roll_rate: Rotation around camera's Z-axis (boresight/roll)
                - az_rate: Rotation around camera's X-axis (azimuth/right)
                - el_rate: Rotation around camera's Y-axis (elevation/up)

        Raises:
            KeyError: If required keys are missing from input dictionaries
            ValueError: If delta_t_sec is zero or negative
        """

        no_velocity = {
            "roll_rate": float('nan'),
            "az_rate": float('nan'),
            "el_rate": float('nan')
        }

        if delta_t_sec <= 0:
            raise ValueError("delta_t_sec must be positive")

        # Extract orientation parameters
        try:
            ra1, dec1, roll1 = (astrometry_prev['RA'], astrometry_prev['Dec'], astrometry_prev['Roll'])
            ra2, dec2, roll2 = (astrometry_curr['RA'], astrometry_curr['Dec'], astrometry_curr['Roll'])
        except KeyError as e:
            return no_velocity
            # raise KeyError(f"Missing required key in astrometry dict: {e}")

        # Combined validation in one block
        all_values = [ra1, dec1, roll1, ra2, dec2, roll2]

        if any(v is None or np.isnan(v) for v in all_values):
            invalid_type = "None" if any(v is None for v in all_values) else "NaN"
            self.log.warning(f"One or more orientation values are {invalid_type}, returning NaN angular velocity")
            return no_velocity

        # Check for identical orientations to avoid division by numerical noise
        atol = 1e-10
        if (np.isclose(ra1,ra2, atol=atol) and
                np.isclose(dec1, dec2, atol=atol) and
                np.isclose(roll1, roll2, atol=atol)):
            return no_velocity

        # Convert to rotation matrices
        R1 = self._radec_roll_to_rotation(ra1, dec1, roll1)
        R2 = self._radec_roll_to_rotation(ra2, dec2, roll2)

        # Decompose relative rotation into components of the rotation vector
        d_roll_rad, d_az_rad, d_el_rad = self._decompose_rotation_to_local_axes(R1, R2)

        # Convert components of rotation vector to angular velocities (rad/s to deg/s)
        angular_velocity = {
            "roll_rate": np.rad2deg(d_roll_rad) / delta_t_sec,
            "az_rate": np.rad2deg(d_az_rad) / delta_t_sec,
            "el_rate": np.rad2deg(d_el_rad) / delta_t_sec
        }

        return angular_velocity

    @staticmethod
    def _focal_lengths_from_horizontal_fov(fov_x_deg: float, image_size: Tuple[int, int]) -> Tuple[float, float]:
        """Compute fx, fy (pixels) from horizontal FOV and aspect ratio."""
        h, w = image_size
        ax = 0.5 * fov_x_deg * DEG2RAD
        fx = (w / 2.0) / max(np.tan(ax), 1e-15)
        ay = np.arctan((h / max(w, 1.0)) * np.tan(ax))
        fy = (h / 2.0) / max(np.tan(ay), 1e-15)
        return fx, fy

    @staticmethod
    def _inverse_gnomonic(x: np.ndarray, y: np.ndarray, ra0_deg: float, dec0_deg: float):
        """Inverse TAN: (x,y) radians on tangent plane -> (RA,Dec) degrees about (ra0,dec0)."""
        ra0 = ra0_deg * DEG2RAD
        dec0 = dec0_deg * DEG2RAD
        rho = np.hypot(x, y)
        c = np.arctan(rho)
        sinc = np.sin(c)
        cosc = np.cos(c)
        rho = np.where(rho == 0, 1e-16, rho)
        dec = np.arcsin(cosc * np.sin(dec0) + y * sinc * np.cos(dec0) / rho)
        ra = ra0 + np.arctan2(x * sinc, rho * np.cos(dec0) * cosc - y * np.sin(dec0) * sinc)
        return (ra * RAD2DEG) % 360.0, dec * RAD2DEG

    @staticmethod
    def _forward_gnomonic(ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float):
        """Forward TAN: (RA,Dec) deg -> (x,y) radians on tangent plane at (ra0,dec0)."""
        ra = ra_deg * DEG2RAD
        dec = dec_deg * DEG2RAD
        ra0 = ra0_deg * DEG2RAD
        dec0 = dec0_deg * DEG2RAD
        cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
        cosc = np.where(np.abs(cosc) < 1e-15, 1e-15, cosc)
        x = (np.cos(dec) * np.sin(ra - ra0)) / cosc
        y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
        return x, y

    @staticmethod
    def _apply_roll_xy(x: np.ndarray, y: np.ndarray, roll_deg: float, sign: int):
        """Rotate (x,y) by sign*roll (sign=+1 or -1)."""
        th = sign * roll_deg * DEG2RAD
        c, s = np.cos(th), np.sin(th)
        xr = c * x - s * y
        yr = s * x + c * y
        return xr, yr

    @staticmethod
    def _undistort_tan_radial_scalar(xd: np.ndarray, yd: np.ndarray, k: float, iters: int = 6):
        """
        Invert ESA scalar radial distortion applied in TAN space:
          rd = r * (1 + k r^2).
        Given distorted (xd,yd), solve for undistorted (xu,yu) via Newton iteration on r.
        """
        if abs(k) < 1e-20:
            return xd, yd
        rd = np.hypot(xd, yd)
        r = rd.copy()
        for _ in range(iters):
            f = r * (1.0 + k * r * r) - rd
            fp = 1.0 + 3.0 * k * r * r
            r = r - np.where(fp != 0, f / fp, 0.0)
        scale = np.where(rd > 0, r / rd, 1.0)
        xu = xd * scale
        yu = yd * scale
        return xu, yu

    @staticmethod
    def _unit_vec_from_radec(ra_deg: np.ndarray, dec_deg: np.ndarray):
        ra = ra_deg * DEG2RAD
        dec = dec_deg * DEG2RAD
        cosd = np.cos(dec)
        return np.column_stack((cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)))

    @staticmethod
    def _kabsch_rotation(P: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Find rotation R that best maps P->M (3x3 via SVD)."""
        Pc = P - P.mean(axis=0)
        Mc = M - M.mean(axis=0)
        H = Pc.T @ Mc
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        return R

    def rms_errors_from_centroids(self, astrometry: Dict, image_size: Tuple[int,int]) -> Dict[str,float]:
        """
        Sphere-only astrometric residuals with explicit roll bias and roll dispersion.

        Returns dict:
          {
            "RA_RMS": arcsec,
            "Dec_RMS": arcsec,
            "RMSE_sphere": arcsec,
            "Roll_bias_arcsec": arcsec,  # best-fit rotation between predicted and measured
            "Roll_RMS": arcsec    # dispersion of orientation after bias removal
          }
        """
        required = ("RA","Dec","FOV","matched_centroids","matched_stars","solver")
        for k in required:
            if k not in astrometry:
                raise ValueError(f"Missing required key: {k}")

        RA0  = float(astrometry["RA"])
        Dec0 = float(astrometry["Dec"])
        Roll = float(astrometry.get("Roll", 0.0))
        fovx = float(astrometry["FOV"])
        solver_name = str(astrometry["solver"])

        k_dist = 0.0
        if solver_name == "solver1":  # ESA
            k_dist = float(astrometry.get("distortion", 0.0))

        centroids_yx = np.asarray(astrometry["matched_centroids"], dtype=float)  # [y,x]
        stars_radec  = np.asarray(astrometry["matched_stars"], dtype=float)[:, :2]

        xs = centroids_yx[:,1]
        ys = centroids_yx[:,0]

        H, W = image_size
        cx, cy = W/2.0, H/2.0
        fx, fy = self._focal_lengths_from_horizontal_fov(fovx, image_size)

        # Pixels -> camera TAN (y-up)
        x_cam = (xs - cx) / max(fx,1e-15)
        y_cam = -(ys - cy) / max(fy,1e-15)

        # Invert ESA distortion in TAN
        if abs(k_dist) > 0:
            x_cam, y_cam = self._undistort_tan_radial_scalar(x_cam, y_cam, k_dist, iters=6)

        # Rotate by -Roll to ENU TAN
        x_tan, y_tan = self._apply_roll_xy(x_cam, y_cam, Roll, sign=-1)

        # TAN -> sky (measured)
        ra_meas, dec_meas = self._inverse_gnomonic(x_tan, y_tan, RA0, Dec0)

        # Spherical RA/Dec residuals
        ra_cat  = stars_radec[:,0]
        dec_cat = stars_radec[:,1]
        dra = (ra_meas - ra_cat + 180.0) % 360.0 - 180.0
        ra_res_arcsec  = dra * np.cos(np.deg2rad(dec_cat)) * DEG2ARCSEC
        dec_res_arcsec = (dec_meas - dec_cat) * DEG2ARCSEC

        ra_rms  = float(np.sqrt(np.mean(ra_res_arcsec**2)))
        dec_rms = float(np.sqrt(np.mean(dec_res_arcsec**2)))

        # Scalar spherical RMSE via unit vectors
        v_meas = self._unit_vec_from_radec(ra_meas, dec_meas)
        v_cat  = self._unit_vec_from_radec(ra_cat,  dec_cat)
        v_meas /= np.linalg.norm(v_meas, axis=1, keepdims=True)
        v_cat  /= np.linalg.norm(v_cat,  axis=1, keepdims=True)
        dots = np.sum(v_meas * v_cat, axis=1)
        thetas = np.arccos(np.clip(dots, -1.0, 1.0))
        rmse_sphere = float(np.sqrt(np.mean((thetas * RAD2DEG * DEG2ARCSEC)**2)))

        # Roll bias (best-fit rotation between predicted and measured unit vectors)
        R = self._kabsch_rotation(v_cat, v_meas)  # map cat->meas
        angle_rad = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
        roll_bias_arcsec = float(abs(angle_rad) * RAD2DEG * DEG2ARCSEC)

        # Roll RMS (dispersion) on TAN after removing bias:
        # Build TAN coordinates for measured and predicted; apply R to predicted 3D, re-project to TAN,
        # then compute per-star orientation angles about the set centroid and take RMS of differences.
        # 1) Forward TAN for catalog (predicted before bias removal)
        x_cat_tan, y_cat_tan = self._forward_gnomonic(ra_cat, dec_cat, RA0, Dec0)
        # Map catalog unit vectors by R to align to measured
        v_cat_rot = (v_cat @ R)  # Nx3
        # Re-project rotated catalog to TAN
        ra_rot = np.rad2deg(np.arctan2(v_cat_rot[:,1], v_cat_rot[:,0]))  # temporary RA-like for projection
        dec_rot = np.rad2deg(np.arcsin(v_cat_rot[:,2] / np.linalg.norm(v_cat_rot, axis=1)))
        # Use proper inverse: convert unit vecs to RA/Dec rigorously
        ra_rot = (np.rad2deg(np.arctan2(v_cat_rot[:,1], v_cat_rot[:,0])) + 360.0) % 360.0
        dec_rot = np.rad2deg(np.arcsin(np.clip(v_cat_rot[:,2], -1.0, 1.0)))
        xr_tan, yr_tan = self._forward_gnomonic(ra_rot, dec_rot, RA0, Dec0)

        # Center both sets and compute per-star orientation angles
        M = np.column_stack([x_tan, y_tan]) - np.mean(np.column_stack([x_tan, y_tan]), axis=0)
        P = np.column_stack([xr_tan, yr_tan]) - np.mean(np.column_stack([xr_tan, yr_tan]), axis=0)
        ang_M = np.arctan2(M[:,1], M[:,0])
        ang_P = np.arctan2(P[:,1], P[:,0])
        d_ang = (ang_M - ang_P + np.pi) % (2*np.pi) - np.pi
        roll_rms_arcsec = float(np.sqrt(np.mean((d_ang * RAD2DEG * DEG2ARCSEC)**2)))

        return {
            "RA_RMS": ra_rms,    # _sphere
            "Dec_RMS": dec_rms,  # _sphere
            "RMSE_sphere": rmse_sphere, # _sphere
            "Roll_bias_arcsec": roll_bias_arcsec,
            "Roll_RMS": roll_rms_arcsec, # _arcsec
        }


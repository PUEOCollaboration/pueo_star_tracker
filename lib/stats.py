"""
lib/stats.py

Daily statistics aggregation for PUEO Star Tracker operations.

This module ingests per-cycle INFO files (log_*.txt) that are written at the end
of each camera cycle. INFO files are treated as the authoritative trace for a cycle.

Key features
- Nested INFO parsing: sections -> nested dicts; values parsed with ast.literal_eval where possible.
- Daily aggregation: one row per YYYY-MM-DD.
- MERGE startup rebuild: rebuild days that have INFO files; preserve CSV-only days when INFO files are missing.
- Persistence: CSV is authoritative; optional periodic XLSX export; static HTML report.
- Web: creates/maintains symlink in cfg.web_path pointing to cfg.stats_html_path (typically logs/stats.html).

Creates and updates 4 files:
 - web/stats.html symlink to logs/stats.html
 - logs/stats.csv
 - logs/stats.xlsx
 - logs/stats.html

"""

from __future__ import annotations

import ast
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from lib.common import get_dt


# -----------------------------------------------------------------------------
# Online aggregation helpers
# -----------------------------------------------------------------------------

@dataclass
class _P2Median:
    """
    P² (P-square) online median estimator (p=0.5).

    This estimator approximates the median without storing all samples.
    It is fast and memory bounded, appropriate for ~8640 cycles/day on SBCs.
    """
    n: int = 0
    q: Optional[list[float]] = None
    npos: Optional[list[int]] = None
    np_des: Optional[list[float]] = None
    dn: Optional[list[float]] = None
    _init_samples: Optional[list[float]] = None

    def __post_init__(self) -> None:
        self._init_samples = []

    def add(self, x: float) -> None:
        if self.n < 5:
            self._init_samples.append(float(x))
            self.n += 1
            if self.n == 5:
                self._init_samples.sort()
                self.q = self._init_samples[:]  # q1..q5
                self.npos = [1, 2, 3, 4, 5]
                self.np_des = [1.0, 1.0 + 0.5 * 2.0, 1.0 + 0.5 * 4.0, 1.0 + 0.5 * 6.0, 5.0]
                self.dn = [0.0, 0.5, 1.0, 0.5, 0.0]
            return

        assert self.q is not None and self.npos is not None and self.np_des is not None and self.dn is not None

        self.n += 1

        # Find k such that q[k] <= x < q[k+1]
        if x < self.q[0]:
            self.q[0] = float(x)
            k = 0
        elif x < self.q[1]:
            k = 0
        elif x < self.q[2]:
            k = 1
        elif x < self.q[3]:
            k = 2
        elif x <= self.q[4]:
            k = 3
        else:
            self.q[4] = float(x)
            k = 3

        # Increment positions above k
        for i in range(k + 1, 5):
            self.npos[i] += 1

        # Desired positions
        for i in range(5):
            self.np_des[i] += self.dn[i]

        # Adjust heights q2..q4 (indices 1..3)
        for i in (1, 2, 3):
            d = self.np_des[i] - self.npos[i]
            if (d >= 1.0 and (self.npos[i + 1] - self.npos[i]) > 1) or (d <= -1.0 and (self.npos[i - 1] - self.npos[i]) < -1):
                di = int(math.copysign(1, d))
                qip = self._parabolic(i, di)
                if self.q[i - 1] < qip < self.q[i + 1]:
                    self.q[i] = qip
                else:
                    self.q[i] = self._linear(i, di)
                self.npos[i] += di

    def _parabolic(self, i: int, di: int) -> float:
        assert self.q is not None and self.npos is not None
        q = self.q
        n = self.npos
        return q[i] + di / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + di) * (q[i + 1] - q[i]) / (n[i + 1] - n[i]) +
            (n[i + 1] - n[i] - di) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        )

    def _linear(self, i: int, di: int) -> float:
        assert self.q is not None and self.npos is not None
        return self.q[i] + di * (self.q[i + di] - self.q[i]) / (self.npos[i + di] - self.npos[i])

    def value(self) -> float:
        if self.n == 0:
            return float("nan")
        if self.n < 5:
            assert self._init_samples is not None
            s = sorted(self._init_samples)
            return float(np.median(np.asarray(s, dtype=np.float64)))
        assert self.q is not None
        return float(self.q[2])  # median marker


@dataclass
class _MinMaxMedianAgg:
    """Track min/max exactly and median approximately (P²)."""
    vmin: float = float("inf")
    vmax: float = float("-inf")
    med: _P2Median = field(default_factory=_P2Median)
    n: int = 0

    def add(self, x: float) -> None:
        x = float(x)
        self.n += 1
        if x < self.vmin:
            self.vmin = x
        if x > self.vmax:
            self.vmax = x
        self.med.add(x)

    def min(self) -> float:
        return float("nan") if self.n == 0 else float(self.vmin)

    def max(self) -> float:
        return float("nan") if self.n == 0 else float(self.vmax)

    def median(self) -> float:
        return self.med.value()


@dataclass
class _MeanAgg:
    """Track mean exactly with sum/count."""
    s: float = 0.0
    n: int = 0

    def add(self, x: float) -> None:
        self.s += float(x)
        self.n += 1

    def mean(self) -> float:
        return float("nan") if self.n == 0 else float(self.s / self.n)


@dataclass
class _ManyMMM:
    """
    Track min/max exactly and median approximately across MANY samples per update.
    Uses the same mechanics as _MinMaxMedianAgg but provides add_many().
    """
    agg: _MinMaxMedianAgg = field(default_factory=_MinMaxMedianAgg)

    def add_many(self, xs: Iterable[float]) -> None:
        for x in xs:
            self.agg.add(float(x))

    def min(self) -> float:
        return self.agg.min()

    def max(self) -> float:
        return self.agg.max()

    def median(self) -> float:
        return self.agg.median()

# -----------------------------------------------------------------------------
# INFO parsing
# -----------------------------------------------------------------------------

_SECTION_MAP = {
    "=== IMAGE METADATA ===": "IMAGE_METADATA",
    "--- Initial Background Stats ---": "INITIAL_BACKGROUND",
    "--- Final Background Stats ---": "FINAL_BACKGROUND",
    "--- Noise Estimation ---": "NOISE_ESTIMATION",
    "[Trail Omega]": "TRAIL_OMEGA",
    "=== FILES & STORAGE ===": "FILES_AND_STORAGE",
    "=== DETECTIONS": "DETECTIONS",
    "=== ASTROMETRY ===": "ASTROMETRY",
}


def _as_float(x: Any) -> float:
    """
    Best-effort conversion to float for values that may come in as numbers
    or strings with units. This intentionally stays minimal and targeted.
    """
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        # Try direct parse first (e.g., "22.8")
        try:
            return float(s)
        except Exception:
            pass
        # Strip common unit suffixes by splitting on whitespace
        tok = s.split()[0] if s else ""
        try:
            return float(tok)
        except Exception:
            return float("nan")
    return float("nan")


def _safe_literal_eval(raw: str) -> Any:
    """
    Parse a value as a Python literal if possible; otherwise return raw string.

    INFO files are designed to be Python print-equivalent, but some fields include
    units (e.g., "22.8 °C") or formatted strings. Those remain strings here.
    """
    raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw

def load_info_nested(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load an INFO file into a nested dict keyed by section.

    Behavior
    --------
    - New format (has section markers): parse into nested dict.
    - Old legacy format (no section markers): LOG WARNING and RETURN {} (skip), never crash.
    - Any parse error: log warning and keep going.
    """
    log = logging.getLogger(__name__)

    def _safe_eval(raw: str) -> Any:
        raw = raw.strip()
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw

    p = Path(path)

    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        log.warning(f"Stats load_info_nested failed to read file: {p.as_posix()} err: {e}")
        return {}

    # New-format detection (any explicit section marker)
    is_new_format = any(
        any(line.strip().startswith(k) for k in _SECTION_MAP.keys())
        for line in lines[:200]
    )

    if not is_new_format:
        log.warning(f"Stats legacy info format detected; skipping file: {p.as_posix()}")
        # return {}

    info: Dict[str, Dict[str, Any]] = {}
    current_section: str = "IMAGE_METADATA"
    current_subsection: Optional[str] = None

    info.setdefault(current_section, {})

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        # Section header
        for marker, section_name in _SECTION_MAP.items():
            if line.startswith(marker):
                current_section = section_name
                current_subsection = None
                info.setdefault(current_section, {})
                break
        else:
            # Subsection header: "estimated distortion parameters:"
            if line.endswith(":") and ":" not in line[:-1]:
                current_subsection = line[:-1].strip()
                if current_subsection:
                    if current_subsection not in info[current_section]:
                        info[current_section][current_subsection] = {}
                continue

            # Skip non key/value lines
            if ":" not in line and "=" not in line:
                continue

            # Parse "k: v"
            if ":" in line:
                try:
                    key, raw_val = line.split(":", 1)
                    key = key.strip()
                    raw_val = raw_val.strip()
                    val = _safe_eval(raw_val)

                    if current_subsection and isinstance(info[current_section].get(current_subsection), dict):
                        info[current_section][current_subsection][key] = val
                    else:
                        info[current_section][key] = val
                except Exception as e:
                    log.warning(f"Stats load_info_nested parse warning file: {p.name} line: {idx} err: {e} text: {line}")
                continue

            # Parse "k=v" (Trail Omega)
            if "=" in line:
                try:
                    key, raw_val = line.split("=", 1)
                    key = key.strip()
                    raw_val = raw_val.strip()
                    val = _safe_eval(raw_val)

                    if current_subsection and isinstance(info[current_section].get(current_subsection), dict):
                        info[current_section][current_subsection][key] = val
                    else:
                        info[current_section][key] = val
                except Exception as e:
                    log.warning(f"Stats load_info_nested parse warning file: {p.name} line: {idx} err: {e} text: {line}")
                continue

    return info


# -----------------------------------------------------------------------------
# Stats class
# -----------------------------------------------------------------------------

class Stats:
    """
    Daily statistics processor.

    Integration
    -----------
    In PUEO Server __init__:
        self.stats = Stats(self.cfg, self)

    In camera_take_image end-of-cycle:
        self.stats.update(self.info_file)

    Config (self.cfg.*)
    -------------------
    stats_csv_path
    stats_save_every_sec
    stats_xlsx_enabled
    stats_xlsx_save_every_sec
    stats_html_path
    stats_html_days
    web_path
    ssd_path
    """

    # Base metric definitions. These generate output columns with include flags.
    # Each base metric has:
    # - src: callable(info_nested) -> value or NaN
    # - mode: one of {'sum', 'mmm', 'mean', 'last'}
    #
    # 'mmm' => produces *_min, *_max, *_median
    BASE_METRICS: Dict[str, Dict[str, Any]] = {
        # Counters
        "cycle_cnt": {"mode": "sum", "include": True, "src": lambda info: 1},
        "solved_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("ASTROMETRY", {}).get("RA") not in (None, "None") else 0},
        "unsolved_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("ASTROMETRY", {}).get("RA") in (None, "None") else 0},
        "autonomous_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("IMAGE_METADATA", {}).get("capture_mode") == "autonomous" else 0},
        "manual_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("IMAGE_METADATA", {}).get("capture_mode") == "manual" else 0},
        "frame_motion_still_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("ASTROMETRY", {}).get("frame_motion") == "still" else 0},
        "frame_motion_streaked_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("ASTROMETRY", {}).get("frame_motion") == "streaked" else 0},

        # Exposure / camera settings
        "exposure_time_us": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("IMAGE_METADATA", {}).get("exposure_time_us"))},
        "gain_cb": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("IMAGE_METADATA", {}).get("gain (cB)"))},

        # Temperatures
        "detector_temp_c": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("IMAGE_METADATA", {}).get("detector temperature"))},
        "cpu_temp_c": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("IMAGE_METADATA", {}).get("CPU Temperature"))},

        # Pixel statistics (derived from formatted strings if needed)
        "pixel_min": {"mode": "mmm", "include": True, "src": lambda info: _as_float(_parse_minmax_pair(info.get("IMAGE_METADATA", {}).get("min/max pixel value (counts)"))[0])},
        "pixel_max": {"mode": "mmm", "include": True, "src": lambda info: _as_float(_parse_minmax_pair(info.get("IMAGE_METADATA", {}).get("min/max pixel value (counts)"))[1])},
        "pixel_mean": {"mode": "mmm", "include": True, "src": lambda info: _as_float(_parse_minmax_pair(info.get("IMAGE_METADATA", {}).get("mean/median pixel value (counts)"))[0])},
        "pixel_median": {"mode": "mmm", "include": True, "src": lambda info: _as_float(_parse_minmax_pair(info.get("IMAGE_METADATA", {}).get("mean/median pixel value (counts)"))[1])},

        # Background / noise (these keys exist in your sections)
        "bkg_init_mean": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("INITIAL_BACKGROUND", {}).get("Initial background level mean"))},
        "bkg_init_stdev": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("INITIAL_BACKGROUND", {}).get("Initial background level stdev"))},
        "bkg_init_p9995": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("INITIAL_BACKGROUND", {}).get("p99.95 Initial background level"))},
        "bkg_final_mean": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("FINAL_BACKGROUND", {}).get("Final background level mean"))},
        "bkg_final_stdev": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("FINAL_BACKGROUND", {}).get("Final pass background level stdev"))},
        "bkg_final_p9995": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("FINAL_BACKGROUND", {}).get("p99.95 Final background level"))},
        "sigma_estimate": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("NOISE_ESTIMATION", {}).get("sigma_estimate"))},

        # Astrometry scalars
        "solver1_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("ASTROMETRY", {}).get("solver") == "solver1" else 0},
        "solver2_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("ASTROMETRY", {}).get("solver") == "solver2" else 0},
        "solver3_cnt": {"mode": "sum", "include": True, "src": lambda info: 1 if info.get("ASTROMETRY", {}).get("solver") == "solver3" else 0},
        "psc": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("precomputed_star_centroids"))},
        "detected_sources": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("detected_sources"))},
        "filtered_sources": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("filtered_sources"))},
        "n_mask_pixels": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("n_mask_pixels"))},
        "p999_original": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("p999_original"))},
        "p999_masked_original": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("p999_masked_original"))},
        "frame_motion_conf": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("frame_motion_conf"))},
        "mean_centroid_diameter": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("mean_centroid_diameter"))},
        "median_centroid_diameter": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("median_centroid_diameter"))},

        "matches": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("Matches"))},
        "rmse": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("RMSE"))},
        "solve_time_s": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("T_solve"))},
        "solver_exec_time_s": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("solver_exec_time"))},
        "total_exec_time_s": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("total_exec_time"))},
        "plate_scale_arcsec_px": {"mode": "mmm", "include": True, "src": lambda info: _as_float(info.get("ASTROMETRY", {}).get("plate scale"))},

        # Matched stars magnitude stats (per-cycle computed, then daily aggregated)
        "mag": {"mode": "mag", "include": True, "src": lambda info: _cycle_mags(info)},
        "matched_stars_cnt": {"mode": "mmm", "include": True, "src": lambda info: _as_float(len(info.get("ASTROMETRY", {}).get("matched_stars") or []))},

    }

    # Output schema (column -> spec). Built from BASE_METRICS at import time.
    SCHEMA: Dict[str, Dict[str, Any]] = {}

    def __init__(self, cfg, server):
        t0 = time.monotonic()

        self.cfg = cfg
        self.server = server
        self.log = self.server.log or logging.getLogger("pueo")
        self.logit = self.server.logit

        self.csv_path = Path(self.cfg.stats_csv_path)
        self.html_path = Path(self.cfg.stats_html_path)
        self.save_every_sec = int(self.cfg.stats_save_every_sec)
        self.xlsx_enabled = bool(self.cfg.stats_xlsx_enabled)
        self.stats_xlsx_save_every_sec = int(self.cfg.stats_xlsx_save_every_sec)
        self.html_days = int(self.cfg.stats_html_days)
        self.html_reload_sec = int(self.cfg.stats_html_reload_sec)

        self.xlsx_path = self.csv_path.with_suffix(".xlsx")

        # Aggregation state: day -> base_metric -> aggregator object
        self._days: Dict[str, Dict[str, Any]] = {}
        self._last_save_utc: Optional[datetime] = None
        self._last_save_xlsx_utc: Optional[datetime] = None

        # Initialize schema
        if not Stats.SCHEMA:
            Stats.SCHEMA = _build_schema(Stats.BASE_METRICS)

        # Startup load + merge
        t_load0 = time.monotonic()
        self._init_data()
        self.log.info(f"Stats init_data dt_s: {get_dt(t_load0)}")

        # Web symlink
        t_link0 = time.monotonic()
        self._ensure_web_symlink()
        self.log.info(f"Stats ensure_web_symlink dt_s: {get_dt(t_link0)}")

        # Force initial save so downstream systems can load CSV/HTML immediately
        t_save0 =time.monotonic()
        self._save_all(force=True)
        self.log.info(f"Stats initial save dt_s: {get_dt(t_save0)}")

        self.logit(f"Stats initialisation completed in {get_dt(t0)}.", color="green")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, info_filename: str | Path) -> None:
        """
        Update daily stats from a single INFO file path.

        This is intended to be called at end-of-cycle.
        Any metric extraction failure is logged (single-line warning) and skipped,
        so one malformed INFO file can never crash the server loop.
        """
        t0 = time.monotonic()

        info_path = Path(info_filename)

        try:
            nested = load_info_nested(info_path)
        except Exception as e:
            self.log.warning(f"Stats update info_parse_err file: {info_path.as_posix()} err: {e}")
            return

        day = self._extract_day(nested, info_path)

        if day not in self._days:
            self._days[day] = self._new_day_state()

        day_state = self._days[day]

        t_up0 = time.monotonic()
        for base_name, spec in Stats.BASE_METRICS.items():
            try:
                val = spec["src"](nested)
                self._update_base_metric(day_state, base_name, spec["mode"], val)
            except Exception as e:
                # IMPORTANT: log file + metric so you can find/inspect exact source data
                self.log.warning(
                    f"Stats update metric_err day: {day} metric: {base_name} mode: {spec.get('mode')} "
                    f"file: {info_path.as_posix()} err: {e}"
                )
                continue
        self.log.info(f"Stats update base_metrics dt_s: {get_dt(t_up0)} day: {day} file: {info_path.name}")

        t_sv0 = time.monotonic()
        self._maybe_save()
        self.log.info(f"Stats update maybe_save dt_s: {get_dt(t_sv0)} file: {info_path.name}")

        self.logit(f"Stats update completed in {get_dt(t0)} file: {info_path.name}", color="green")

    def get_data(self, days: int = 0) -> Dict[str, Any]:
        """
        Return telemetry-ready JSON-safe dictionary of the last N days.

        Parameters
        ----------
        days : int
            0 => all available days
            1 => current day only
            2 => last two days, etc.
        """
        df = self._to_dataframe()

        if days and days > 0:
            df = df.sort_index(ascending=False).head(days).sort_index()

        payload = []
        for day, row in df.iterrows():
            d = {"date": str(day)}
            for col in df.columns:
                d[col] = _json_safe(row[col])
            payload.append(d)

        return {
            "stats_days": len(df.index),
            "stats": payload,
            "last_update_utc": self._last_save_utc.isoformat() if self._last_save_utc else None,
        }

    @property
    def size(self) -> int:
        """Cycle count for current day (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        df = self._to_dataframe()
        if today in df.index:
            v = df.loc[today].get("cycle_cnt", 0)
            try:
                return int(v)
            except Exception:
                return 0
        return 0

    @property
    def days(self) -> int:
        """Number of days in the dataset."""
        return int(len(self._to_dataframe().index))

    def load_from_info(self, filename: str | Path) -> Dict[str, Dict[str, Any]]:
        """
        Load a single INFO file into nested dictionaries.

        This is useful for ad-hoc inspection or debugging.
        """
        return load_info_nested(filename)

    # ------------------------------------------------------------------
    # Initialization / rebuild / merge
    # ------------------------------------------------------------------

    def _init_data(self) -> None:
        """
        Initialize daily stats from CSV and INFO files (MERGE semantics).

        - Load existing CSV if present.
        - Rebuild any days that have INFO files under cfg.ssd_path/YYYY-MM-DD/log_*.txt.
        - INFO-derived days overwrite CSV-derived days.
        - CSV-only days are preserved if INFO files are missing.
        """
        df_csv = self._load_csv()
        info_days_state = self._rebuild_from_info_tree()

        # Start from CSV
        if not df_csv.empty:
            for day, row in df_csv.iterrows():
                self._days[str(day)] = self._state_from_row(row.to_dict())

        # Overwrite / add INFO-derived days
        for day, day_state in info_days_state.items():
            self._days[day] = day_state

        # Normalize: ensure all days have all base keys
        for day in list(self._days.keys()):
            self._days[day] = self._normalize_day_state(self._days[day])

    def _load_csv(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.csv_path)
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = df["date"].astype(str)
        df = df.set_index("date", drop=True)
        return df


    def _rebuild_from_info_tree(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan cfg.ssd_path/YYYY-MM-DD/log_*.txt and rebuild daily states.

        Robustness rules:
        - Any INFO parse failure => warn + skip file (no crash).
        - Any metric extraction failure => warn + skip that metric for that file (no crash).
        - Includes file path in warnings so you can inspect the offending INFO.
        """
        t0 = time.monotonic()
        ssd_root = Path(self.cfg.ssd_path)

        by_day: Dict[str, list[Path]] = {}
        for day_dir in sorted(ssd_root.iterdir()):
            if not day_dir.is_dir():
                continue
            day = day_dir.name
            files = sorted(day_dir.glob("log_*.txt"))
            if files:
                by_day[day] = files

        rebuilt: Dict[str, Dict[str, Any]] = {}
        for day, files in by_day.items():
            day_state = self._new_day_state()

            for p in files:
                try:
                    nested = load_info_nested(p)
                except Exception as e:
                    self.log.warning(f"Stats rebuild info_parse_err day: {day} file: {p.as_posix()} err: {e}")
                    continue

                for base_name, spec in Stats.BASE_METRICS.items():
                    try:
                        val = spec["src"](nested)
                        self._update_base_metric(day_state, base_name, spec["mode"], val)
                    except Exception as e:
                        # This is the one that fixes your crash (e.g., float(None) in matched_stars mags)
                        self.log.warning(
                            f"Stats rebuild metric_err day: {day} metric: {base_name} mode: {spec.get('mode')} "
                            f"file: {p.as_posix()} err: {e}"
                        )
                        continue

            rebuilt[day] = day_state

        self.log.info(f"Stats rebuild_from_info_tree dt_s: {get_dt(t0)} days: {len(rebuilt)}")
        return rebuilt

    # ------------------------------------------------------------------
    # Aggregation mechanics
    # ------------------------------------------------------------------

    def _new_day_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for base_name, spec in Stats.BASE_METRICS.items():
            mode = spec["mode"]
            if mode == "sum":
                state[base_name] = 0.0
            elif mode == "mag":
                state[base_name] = _ManyMMM()
            elif mode == "mean":
                state[base_name] = _MeanAgg()
            elif mode == "mmm":
                state[base_name] = _MinMaxMedianAgg()
            elif mode == "last":
                state[base_name] = None

        return state

    def _normalize_day_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        norm = self._new_day_state()
        for k, v in state.items():
            norm[k] = v
        return norm

    def _state_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct internal aggregators from stored CSV row.

        Notes
        -----
        - This is necessarily lossy for median/mean online state.
        - We reconstruct MMM aggregators with min/max/median set to the stored values,
          then continue updating online from new cycles.
        """
        state = self._new_day_state()

        for base_name, spec in Stats.BASE_METRICS.items():
            mode = spec["mode"]

            if mode == "sum":
                v = row.get(base_name, 0.0)
                state[base_name] = float(v) if _is_number(v) else 0.0
                continue

            if mode == "last":
                state[base_name] = row.get(base_name, None)
                continue

            if mode == "mean":
                # reconstruct with current mean as a single sample (best-effort)
                m = row.get(f"{base_name}_mean", row.get(base_name, float("nan")))
                agg = _MeanAgg()
                if _is_number(m):
                    agg.add(float(m))
                state[base_name] = agg
                continue

            if mode == "mmm":
                agg = _MinMaxMedianAgg()
                vmin = row.get(f"{base_name}_min", float("nan"))
                vmax = row.get(f"{base_name}_max", float("nan"))
                vmed = row.get(f"{base_name}_median", float("nan"))
                if _is_number(vmin):
                    agg.vmin = float(vmin)
                    agg.n = 1
                if _is_number(vmax):
                    agg.vmax = float(vmax)
                    agg.n = max(agg.n, 1)
                # Seed median estimator with 5 identical samples if we have a value
                if _is_number(vmed):
                    agg.med = _P2Median()
                    for _ in range(5):
                        agg.med.add(float(vmed))
                    agg.n = max(agg.n, 5)
                state[base_name] = agg
                continue

        return state

    def _update_base_metric(self, day_state: Dict[str, Any], base_name: str, mode: str, val: Any) -> None:
        if mode == "sum":
            day_state[base_name] += float(val)
            return

        if mode == "last":
            if val is not None and val != "" and not (isinstance(val, float) and math.isnan(val)):
                day_state[base_name] = val
            return

        if mode == "mag":
            mags = val or []
            if mags:
                day_state[base_name].add_many(mags)
            return

        if mode == "mean":
            x = _as_float(val)
            if not math.isnan(x):
                day_state[base_name].add(x)
            return

        if mode == "mmm":
            x = _as_float(val)
            if not math.isnan(x):
                day_state[base_name].add(x)
            return

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _maybe_save(self) -> None:
        now = datetime.now(timezone.utc)
        if self._last_save_utc is None:
            self._save_all(force=False)
            return
        dt = (now - self._last_save_utc).total_seconds()
        if dt >= self.save_every_sec:
            self._save_all(force=False)

    def _save_all(self, force: bool) -> None:
        t0 = time.monotonic()

        df = self._to_dataframe()

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, index=True)

        self._write_html(df)

        if self.xlsx_enabled:
            self._maybe_export_xlsx(df)

        self._last_save_utc = datetime.now(timezone.utc)

        self.logit(f"Stats saved in {get_dt(t0)} rows: {len(df.index)}.", color="green")

    def _maybe_export_xlsx(self, df: pd.DataFrame) -> None:
        """
        Export XLSX periodically based on cfg.stats_xlsx_save_every_sec.
        """
        now = datetime.now(timezone.utc)

        if not self.xlsx_path.exists() or self._last_save_xlsx_utc is None:
            df.to_excel(self.xlsx_path, index=True)
            self._last_save_xlsx_utc = datetime.now(timezone.utc)
            self.log.info(f"Stats xlsx exported file: {self.xlsx_path.as_posix()}")
            return

        dt = (now - self._last_save_xlsx_utc).total_seconds()
        if dt >= self.stats_xlsx_save_every_sec:
            self._last_save_xlsx_utc = datetime.now(timezone.utc)
            df.to_excel(self.xlsx_path, index=True)
            self.log.info(f"Stats xlsx exported file: {self.xlsx_path.as_posix()}")

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def _write_html(self, df: pd.DataFrame) -> None:
        """
        Generate logs/stats.html with:
        - Summary (row-oriented, last N days)
        - Full transposed (metrics as rows, days as columns)
        """
        t0 = time.time()

        df_sorted = df.sort_index(ascending=False)
        df_last = df_sorted.head(self.html_days)

        # Summary columns (curated; safe if columns missing)
        summary_cols = [
            "cycle_cnt",
            "solved_cnt",
            "unsolved_cnt",
            "autonomous_cnt",
            "manual_cnt",
            "frame_motion_streaked_cnt",
            "frame_motion_still_cnt",
            "matched_stars_cnt_median",
            "mag_min",
            "mag_median",
            "mag_max",
            "p999_original_max",
            "sigma_estimate_median",
            "total_exec_time_s_median",
        ]
        summary_cols = [c for c in summary_cols if c in df_last.columns]

        # Transposed: choose key metrics (grouped)
        transposed_metrics = [
            "cycle_cnt",
            "solved_cnt",
            "solver1_cnt",
            "solver2_cnt",
            "solver3_cnt",
            "unsolved_cnt",
            "autonomous_cnt",
            "manual_cnt",
            "frame_motion_streaked_cnt",
            "frame_motion_still_cnt",
            "exposure_time_us_median",
            "gain_cb_median",
            "detector_temp_c_median",
            "cpu_temp_c_median",
            "detected_sources_median",
            "filtered_sources_median",
            "matched_stars_cnt_median",
            "mag_min",
            "mag_median",
            "mag_max",
            "p999_original_max",
            "p999_masked_original_max",
            "n_mask_pixels_median",
            "sigma_estimate_median",
            "solver_exec_time_s_median",
            "solve_time_s_median",
            "total_exec_time_s_median",
        ]
        transposed_metrics = [m for m in transposed_metrics if m in df_last.columns]

        # Build HTML
        last_update = self._last_save_utc.isoformat() if self._last_save_utc else None
        csv_rel = os.path.relpath(self.csv_path, start=self.html_path.parent)
        xlsx_rel = os.path.relpath(self.xlsx_path, start=self.html_path.parent)

        parts = []
        parts.append("<html><head><meta charset='utf-8'>")
        parts.append("<title>PUEO Stats</title>")
        parts.append("<script>setTimeout(function(){location.reload();}, %d);</script>" % (max(self.html_reload_sec, 1)*1000))
        parts.append(
            "<style>"
            "body{font-family:Arial,Helvetica,sans-serif;margin:16px;}"
            "table{border-collapse:collapse;margin:8px 0; font-size: 12px;}"
            "th,td{border:1px solid #ccc;padding:4px 6px;white-space:nowrap;}"
            "th{background:#eee;}"
            ".small{font-size:12px;color:#444;}"
            ".section{margin-top:18px;}"
            "details summary{cursor:pointer;font-weight:bold;margin:6px 0;}"
            "</style>"
        )
        parts.append("</head><body>")

        parts.append("<h2>PUEO Star Tracker Daily Stats</h2>")
        parts.append(f"<div class='small'>Last update UTC: {last_update}</div>")
        parts.append("<div class='small'>Artifacts: ")
        parts.append(f"<a href='{csv_rel}'>stats.csv</a>")
        if self.xlsx_enabled:
            parts.append(f" | <a href='{xlsx_rel}'>stats.xlsx</a>")
        parts.append("</div>")

        # Summary table (row-oriented)
        parts.append("<div class='section'>")
        parts.append(f"<h3>Summary (last {len(df_last.index)} days)</h3>")
        parts.append("<table>")
        parts.append("<tr><th>date</th>" + "".join(f"<th>{c}</th>" for c in summary_cols) + "</tr>")
        for day, row in df_last.iterrows():
            parts.append("<tr>")
            parts.append(f"<td>{day}</td>")
            for c in summary_cols:
                parts.append(f"<td>{_fmt_cell(row.get(c))}</td>")
            parts.append("</tr>")
        parts.append("</table>")
        parts.append("</div>")

        # Full history (transposed, metric-oriented)
        parts.append("<div class='section'>")
        parts.append(f"<h3>Full (transposed) (last {len(df_last.index)} days)</h3>")
        parts.append("<details open><summary>Show/Hide</summary>")
        parts.append("<table>")
        days = list(df_last.index)
        parts.append("<tr><th>metric</th>" + "".join(f"<th>{d}</th>" for d in days) + "</tr>")
        for metric in transposed_metrics:
            parts.append("<tr>")
            parts.append(f"<td>{metric}</td>")
            for d in days:
                parts.append(f"<td>{_fmt_cell(df_last.loc[d].get(metric))}</td>")
            parts.append("</tr>")
        parts.append("</table>")
        parts.append("</details>")
        parts.append("</div>")

        parts.append("</body></html>")

        self.html_path.parent.mkdir(parents=True, exist_ok=True)
        self.html_path.write_text("\n".join(parts), encoding="utf-8")

        self.log.info(f"Stats write_html dt_s: {time.time() - t0:.3f}")

    # ------------------------------------------------------------------
    # DataFrame / schema expansion
    # ------------------------------------------------------------------

    def _to_dataframe(self) -> pd.DataFrame:
        """
        Convert internal day-state aggregators to a DataFrame with SCHEMA columns.
        """
        rows = []
        for day in sorted(self._days.keys()):
            base_state = self._days[day]
            row = {"date": day}
            # Expand each base metric into output columns
            for base_name, spec in Stats.BASE_METRICS.items():
                mode = spec["mode"]
                if mode == "sum":
                    row[base_name] = int(base_state[base_name])
                elif mode == "last":
                    row[base_name] = base_state[base_name]
                elif mode == "mag":
                    agg = base_state[base_name]  # _ManyMMM
                    row["mag_min"] = agg.min()
                    row["mag_max"] = agg.max()
                    row["mag_median"] = agg.median()
                elif mode == "mean":
                    row[f"{base_name}_mean"] = base_state[base_name].mean()
                elif mode == "mmm":
                    agg: _MinMaxMedianAgg = base_state[base_name]
                    row[f"{base_name}_min"] = agg.min()
                    row[f"{base_name}_max"] = agg.max()
                    row[f"{base_name}_median"] = agg.median()

            # Apply include flags (columns may be dropped)
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["date"])
        df = df.set_index("date", drop=True)

        # Drop disabled columns
        for col, colspec in Stats.SCHEMA.items():
            if not colspec.get("include", True) and col in df.columns:
                df = df.drop(columns=[col])

        # Ensure stable column order: SCHEMA first, then any extras
        ordered = [c for c in Stats.SCHEMA.keys() if c in df.columns]
        extras = [c for c in df.columns if c not in ordered]
        df = df[ordered + extras]

        return df

    # ------------------------------------------------------------------
    # Web symlink
    # ------------------------------------------------------------------

    def _ensure_web_symlink(self) -> None:
        """
        Ensure the web symlink exists and points to the correct target.

        Behavior
        --------
        - If cfg.web_path/<stats_html_filename> exists as a symlink and resolves to cfg.stats_html_path: do nothing.
        - Otherwise, create/update the symlink using server.utils.create_symlink(...).
        - On Windows dev hosts, do nothing (symlinks not used there).
        """
        if os.name == "nt":
            self.log.debug(f"Stats web symlink skipped on Windows web_path: {self.cfg.web_path} target: {self.html_path.as_posix()}")
            return

        web_root = Path(self.cfg.web_path)
        link_name = self.html_path.name
        link_path = web_root / link_name
        target = self.html_path.resolve()

        if link_path.is_symlink() and link_path.resolve() == target:
            return

        ok = self.server.utils.create_symlink(
            path=str(web_root),
            filename=str(target),
            symlink_name=link_name,
            use_relative_path=True,
        )
        self.log.info(f"Stats web symlink ensured ok: {ok} link: {link_path.as_posix()} target: {target.as_posix()}")

    # ------------------------------------------------------------------
    # Date extraction
    # ------------------------------------------------------------------

    def _extract_day(self, nested: Dict[str, Dict[str, Any]], info_path: Path) -> str:
        """
        Prefer IMAGE_METADATA.capture_start_utc, fallback to parent directory name.
        """
        s = nested.get("IMAGE_METADATA", {}).get("capture_start_utc")
        if isinstance(s, str) and s:
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                return dt.date().isoformat()
            except Exception:
                pass
        return info_path.parent.name


# -----------------------------------------------------------------------------
# Derived parsers / helpers
# -----------------------------------------------------------------------------

def _parse_minmax_pair(v: Any) -> tuple[Any, Any]:
    """
    Parse formatted strings like "3084 / 65535" into (3084, 65535).
    """
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return v[0], v[1]
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/", 1)
        return a.strip(), b.strip()
    return float("nan"), float("nan")


def _cycle_mag_stats(info: Dict[str, Dict[str, Any]]) -> tuple[float, float, float]:
    """
    Compute per-cycle magnitude min/max/median from ASTROMETRY.matched_stars.
    """
    stars = info.get("ASTROMETRY", {}).get("matched_stars") or []
    if not stars:
        return float("nan"), float("nan"), float("nan")
    mags = [float(s[2]) for s in stars if isinstance(s, (list, tuple)) and len(s) >= 3]
    if not mags:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(mags, dtype=np.float64)
    return float(np.min(arr)), float(np.max(arr)), float(np.median(arr))


def _cycle_mags(info: Dict[str, Dict[str, Any]]) -> list[float]:
    """
    Return the list of magnitudes from ASTROMETRY.matched_stars for this cycle.

    Robust to occasional malformed entries (e.g., magnitude None).
    """
    stars = info.get("ASTROMETRY", {}).get("matched_stars") or []
    if not stars:
        return []

    mags: list[float] = []
    for s in stars:
        if not isinstance(s, (list, tuple)) or len(s) < 3:
            continue
        m = s[2]
        if m is None:
            continue
        try:
            mags.append(float(m))
        except Exception:
            continue

    return mags


def _build_schema(base_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build per-column schema with include flags.

    Users can disable any individual output column by toggling Stats.SCHEMA[col]["include"].
    """
    schema: Dict[str, Dict[str, Any]] = {}

    for base_name, spec in base_metrics.items():
        mode = spec["mode"]
        if mode == "sum" or mode == "last":
            schema[base_name] = {"include": bool(spec.get("include", True)), "base": base_name, "field": mode}
        elif mode == "mag":
            schema["mag_min"] = {"include": bool(spec.get("include", True)), "base": base_name, "field": "min"}
            schema["mag_max"] = {"include": bool(spec.get("include", True)), "base": base_name, "field": "max"}
            schema["mag_median"] = {"include": bool(spec.get("include", True)), "base": base_name, "field": "median"}
        elif mode == "mean":
            schema[f"{base_name}_mean"] = {"include": bool(spec.get("include", True)), "base": base_name, "field": "mean"}
        elif mode == "mmm":
            schema[f"{base_name}_min"] = {"include": bool(spec.get("include", True)), "base": base_name, "field": "min"}
            schema[f"{base_name}_max"] = {"include": bool(spec.get("include", True)), "base": base_name, "field": "max"}
            schema[f"{base_name}_median"] = {"include": bool(spec.get("include", True)), "base": base_name, "field": "median"}

    # Put core counters first
    ordered = [
        "cycle_cnt", "solved_cnt", "unsolved_cnt", "autonomous_cnt", "manual_cnt",
        "frame_motion_still_cnt", "frame_motion_streaked_cnt",
    ]
    out: Dict[str, Dict[str, Any]] = {}
    for k in ordered:
        if k in schema:
            out[k] = schema[k]
    for k in sorted(schema.keys()):
        if k not in out:
            out[k] = schema[k]
    return out


def _is_number(x: Any) -> bool:
    try:
        v = float(x)
        return not math.isnan(v)
    except Exception:
        return False


def _fmt_cell(v: Any) -> str:
    """Format a cell value for HTML tables.

    - None/NaN render as empty.
    - Integer-like floats (e.g., 12.0) render as integers to avoid ".000" noise in *_cnt fields.
    - Other floats render with human-friendly precision.
    """
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        # If it's effectively an integer, render as int.
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        if abs(v) >= 1000:
            return f"{v:.1f}"
        if abs(v) >= 10:
            return f"{v:.3f}"
        return f"{v:.4f}"
    return str(v)


def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
        return None if math.isnan(v) else v
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, (int, str, bool)):
        return v
    return str(v)

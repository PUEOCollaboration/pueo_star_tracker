"""
Filesystem Monitor (FSMonitor)

This module provides the FSMonitor class, which can monitor multiple filesystem
paths on Linux (Ubuntu) and Windows. The primary use case is monitoring
internal, external, and removable storage (e.g., SSD, SD cards) used for
storing images. The monitor tracks capacity, free space, and file/folder
growth, and provides warnings or critical alerts when thresholds are exceeded.
It also produces usage trends and forecasts based on recent growth rates.

Key Features:
- Cross-platform (Linux/Windows) path monitoring
- Compact, memory-efficient history buffer using ring buffers
- Tracks free/used disk space, folder counts, file counts, and size growth
- Trend reporting for multiple configurable windows (e.g., 1h, 6h, 24h, week, month)
- Forecasting for when warning/critical thresholds will be reached
- Background thread monitoring with graceful shutdown
- JSON export of current status for logging or API integration

Design Notes:
- Timestamps are stored as UTC epoch seconds (`datetime.now(timezone.utc)`).
- For memory efficiency, history uses typed arrays (`array.array`) rather
  than Python dicts, reducing footprint from MBs to ~KBs per monitored path.
- Short-term trends (1h, 6h, 24h) use full-resolution samples, while longer
  windows (week, month, all) rely on aggregated anchors.
- External interface remains compatible with status dict and JSON output.

Author: Milan Štubljar of Štubljar d.o.o. (info@stubljar.com), assisted by ChatGPT (GPT-5)
Created: 2025-09-26
"""


from __future__ import annotations

__author__ = "Milan Štubljar of Štubljar d.o.o. (info@stubljar.com), assisted by ChatGPT (GPT-5)"
__created__ = "2025-09-26"
__version__ = "1.0.0"

import os
import re
import shutil
import threading
import time
import configparser
import logging
import json
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Type
from array import array
import bisect


# ------------------ Configuration dataclass ---------------------------------
@dataclass(frozen=True)
class MonitorCfg:
    """
    Configuration container for FSMonitor with attached logger.

    Attributes
    ----------
    enabled: bool
        Whether monitoring is active (default True). If False, FSMonitor.run()
        should skip all checks.
    check_interval: float
        Seconds between automatic checks (must be >0 if enabled).
    trend_durations: List[str]
        Trend windows, e.g., ["1h", "6h", "24h", "week", "month", "all"].
    scan_max_depth: int
        Maximum folder depth to scan for files (>=0).
    high_res_windows: List[str]
        Subset of trend_durations to keep at high resolution.
    retain_margin: float
        Fractional margin for retention (>0).
    max_retain_cap: int
        Maximum number of samples to retain.
    warning_pct: float
        Percent free space to trigger warning.
    critical_pct: float
        Percent free space to trigger critical alert.
    log: logging.Logger
        Logger used internally for warnings and info.
    """

    enabled: bool = True
    check_interval: float = 60.0
    trend_durations: List[str] = field(default_factory=lambda: ["1h", "6h", "24h", "week", "month", "all"])
    scan_max_depth: int = 2
    high_res_windows: List[str] = field(default_factory=lambda: ["1h", "6h", "24h"])
    retain_margin: float = 0.2
    max_retain_cap: int = 200_000
    warning_pct: float = 5.0
    critical_pct: float = 1.0
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("pueo"))

    @classmethod
    def from_any(cls, cfg: Any, section: str = "FILESYSTEM_MONITOR") -> "MonitorCfg":
        """
        Build a MonitorCfg from any input.

        - If `cfg` is already a MonitorCfg, return it as-is.
        - If cfg is a ConfigParser, read the section.
        - Otherwise, return defaults.
        """
        if isinstance(cfg, cls):
            # Already a MonitorCfg instance → return as-is
            return cfg

        # Create a logger first
        log = logging.getLogger("pueo")

        # Default instance
        instance = cls(log=log)
        warnings: list[str] = []

        # Resolve ConfigParser if present
        cp = None
        if isinstance(cfg, configparser.ConfigParser):
            cp = cfg
        elif hasattr(cfg, "_config") and isinstance(getattr(cfg, "_config"), configparser.ConfigParser):
            cp = getattr(cfg, "_config")

        if cp is None or not cp.has_section(section):
            log.debug("No configuration section found, using defaults")
            return instance

        def _get_float(opt: str, default: float) -> float:
            raw = cp.get(section, opt, fallback=None)
            if raw is None:
                return default
            try:
                return float(raw)
            except Exception:
                log.warning("Invalid float for %s.%s: %r (using default %s)", section, opt, raw, default)
                return default

        def _get_int(opt: str, default: int) -> int:
            raw = cp.get(section, opt, fallback=None)
            if raw is None:
                return default
            try:
                return int(raw)
            except Exception:
                log.warning("Invalid int for %s.%s: %r (using default %s)", section, opt, raw, default)
                return default

        def _get_bool(opt: str, default: bool) -> bool:
            raw = cp.get(section, opt, fallback=None)
            if raw is None:
                return default
            try:
                return cp.getboolean(section, opt, fallback=default)
            except Exception:
                log.warning("Invalid bool for %s.%s: %r (using default %s)", section, opt, raw, default)
                return default

        def _get_list(opt: str, default: list[str]) -> list[str]:
            raw = cp.get(section, opt, fallback=None)
            if raw is None:
                return default
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if not parts:
                log.warning("Empty list for %s.%s: %r (using default)", section, opt, raw)
                return default
            return parts

        # Parse all values
        enabled = _get_bool("enabled", instance.enabled)
        check_interval = _get_float("check_interval", instance.check_interval)
        trend_durations = _get_list("trend_durations", instance.trend_durations)
        scan_max_depth = _get_int("scan_max_depth", instance.scan_max_depth)
        high_res_windows = _get_list("high_res_windows", instance.high_res_windows)
        retain_margin = _get_float("retain_margin", instance.retain_margin)
        max_retain_cap = _get_int("max_retain_cap", instance.max_retain_cap)
        warning_pct = _get_float("warning_pct", instance.warning_pct)
        critical_pct = _get_float("critical_pct", instance.critical_pct)

        cfg_instance = cls(
            enabled=enabled,
            check_interval=check_interval,
            trend_durations=trend_durations,
            scan_max_depth=scan_max_depth,
            high_res_windows=high_res_windows,
            retain_margin=retain_margin,
            max_retain_cap=max_retain_cap,
            warning_pct=warning_pct,
            critical_pct=critical_pct,
            log=log,
        )

        # Validate semantic rules
        cfg_instance._validate()

        return cfg_instance

    def _validate(self):
        """Ensure semantic correctness; raises ValueError if invalid."""
        errors = []
        if self.enabled and self.check_interval <= 0:
            errors.append("check_interval must be >0 when enabled")
        if self.scan_max_depth < 0:
            errors.append("scan_max_depth must be >=0")
        if self.retain_margin < 0:
            errors.append("retain_margin must be >=0")
        if self.max_retain_cap <= 0:
            errors.append("max_retain_cap must be >0")
        if not (0.0 <= self.warning_pct <= 100.0):
            errors.append("warning_pct must be between 0 and 100")
        if not (0.0 <= self.critical_pct <= 100.0):
            errors.append("critical_pct must be between 0 and 100")
        if self.warning_pct >= self.critical_pct:
            errors.append("warning_pct must be greater than critical_pct")
        if errors:
            raise ValueError("MonitorCfg validation errors: " + "; ".join(errors))


# ------------------ Compact history buffer ----------------------------------
class HistoryBuffer:
    """
    Compact ring buffer storing time-series samples in typed arrays.

    Stored per-sample fields (parallel arrays):
      - ts: int64 epoch seconds (UTC)
      - folders: int32
      - files: int32
      - size_mb: float64
      - used_mb: float64
      - total_mb: float64

    Advantages: very small per-sample memory compared to dicts and fast binary searches.
    """

    def __init__(self, retain: int):
        # Ensure minimum retain
        retain = max(int(retain), 8)
        self.capacity = retain
        self._len = 0
        self._pos = 0  # next write index
        # typed arrays
        self.ts = array("q", [0]) * retain  # int64
        self.folders = array("i", [0]) * retain  # int32
        self.files = array("i", [0]) * retain
        self.size_mb = array("d", [0.0]) * retain  # double
        self.used_mb = array("d", [0.0]) * retain
        self.total_mb = array("d", [0.0]) * retain

    def append(self, ts_epoch: int, folders: int, files: int, size_mb: float, used_mb: float, total_mb: float) -> None:
        """Append a sample into the ring buffer (overwrites oldest when full)."""
        idx = self._pos
        self.ts[idx] = int(ts_epoch)
        self.folders[idx] = int(folders)
        self.files[idx] = int(files)
        self.size_mb[idx] = float(size_mb)
        self.used_mb[idx] = float(used_mb)
        self.total_mb[idx] = float(total_mb)
        self._pos = (idx + 1) % self.capacity
        if self._len < self.capacity:
            self._len += 1

    def __len__(self) -> int:
        """Return current number of samples stored."""
        return self._len

    def _logical_index(self, i: int) -> int:
        """Convert logical index (0..len-1) to physical array index.
        0 is oldest, len-1 is newest sample."""
        if i < 0 or i >= self._len:
            raise IndexError("index out of range")
        start = (self._pos - self._len) % self.capacity
        return (start + i) % self.capacity

    def latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent sample or None if empty."""
        if self._len == 0:
            return None
        idx = self._logical_index(self._len - 1)
        return {
            "ts": int(self.ts[idx]),
            "folders": int(self.folders[idx]),
            "files": int(self.files[idx]),
            "size_mb": float(self.size_mb[idx]),
            "used_mb": float(self.used_mb[idx]),
            "total_mb": float(self.total_mb[idx]),
        }

    def earliest_after(self, cutoff_epoch: int) -> Optional[Dict[str, Any]]:
        """
        Find earliest sample with ts >= cutoff_epoch using binary search on the logical array slice.
        Returns sample dict or None.
        """
        if self._len == 0:
            return None
        # build a temporary list of timestamps for binary search
        # because capacity is small (a few thousand) converting to list is cheap
        ts_list = [self.ts[self._logical_index(i)] for i in range(self._len)]
        pos = bisect.bisect_left(ts_list, cutoff_epoch)
        if pos >= len(ts_list):
            return None
        idx = self._logical_index(pos)
        return {
            "ts": int(self.ts[idx]),
            "folders": int(self.folders[idx]),
            "files": int(self.files[idx]),
            "size_mb": float(self.size_mb[idx]),
            "used_mb": float(self.used_mb[idx]),
            "total_mb": float(self.total_mb[idx]),
        }

    def as_list(self) -> List[Dict[str, Any]]:
        """Return all samples as a list of dicts (oldest..newest)."""
        return [self._sample_at(i) for i in range(self._len)]

    def _sample_at(self, i: int) -> Dict[str, Any]:
        """Return a dict representing sample at logical index i."""
        idx = self._logical_index(i)
        return {
            "ts": int(self.ts[idx]),
            "folders": int(self.folders[idx]),
            "files": int(self.files[idx]),
            "size_mb": float(self.size_mb[idx]),
            "used_mb": float(self.used_mb[idx]),
            "total_mb": float(self.total_mb[idx]),
        }


# ------------------ FSMonitor ------------------------------------------------
class FSMonitor:
    """
    Filesystem monitor for multiple paths. Memory-efficient history storage.

    Public API highlights:
        - add_path(name, path, warning_pct, critical_pct, fs_type)
        - run() / stop()
        - check()
        - status() -> dict (same structure as earlier prototype)
        - status_json(indent=None) -> str  (serializes dict to JSON; name chosen intentionally)
    """

    def __init__(self, cfg: MonitorCfg, logger: Optional[logging.Logger] = None):
        self.cfg: MonitorCfg = MonitorCfg.from_any(cfg)  # single-line conversion (validated)
        self.log = logger or logging.getLogger("pueo")
        self.enabled = self.cfg.enabled
        self.check_interval = float(self.cfg.check_interval)
        self.trend_durations = sorted(self.cfg.trend_durations, key=self._duration_sort_key)
        self.scan_max_depth = self.cfg.scan_max_depth
        self.high_res_windows = set(self.cfg.high_res_windows)
        self.retain_margin = float(self.cfg.retain_margin)
        self.max_retain_cap = int(self.cfg.max_retain_cap)

        # Is any of the monitored paths in critical status?
        self.is_critical = False

        # Number of checks
        self._check_cnt = 0

        # Monitored path metadata
        self._paths: Dict[str, Dict[str, Any]] = {}

        # History buffers per path
        self._history: Dict[str, HistoryBuffer] = {}

        # Aggregates for long windows (daily summaries) {name: deque of (date_iso, folders, files, size_mb)}
        self._daily_aggregates: Dict[str, deque] = {}

        # Threading control
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        self.log.debug("FSMonitor initialized (check_interval=%s, high_res=%s)",
                       self.check_interval, sorted(self.high_res_windows))

    # -------------------- public API ---------------------------------------

    def add_path(self, name: str, path: str, *, warning_pct: float = 80.0, critical_pct: float = 92.0,
                 fs_type: str = "data", max_depth: Optional[int] = None) -> None:
        """
        Register a filesystem path for monitoring.

        This method adds a path to the FSMonitor instance, initializes its history buffer,
        sets retention based on the longest high-resolution window, and prepares daily aggregates.

        Parameters
        ----------
        name : str
            Unique identifier for the monitored path.
        path : str
            Absolute or relative filesystem path to monitor.
        warning_pct : float, optional
            Threshold percentage of used space to trigger a warning (default 80.0).
        critical_pct : float, optional
            Threshold percentage of used space to trigger a critical alert (default 92.0).
        fs_type : str, optional
            Type of filesystem (default "data"). Can be used for metadata classification.
        max_depth : int, optional
            Maximum folder depth to scan for files (default: FSMonitor.scan_max_depth).

        Effects / Attributes Updated
        ----------------------------
        self._paths : dict
            Stores metadata about monitored paths.
        self._history : dict
            Initializes a HistoryBuffer instance for the path to store time-series data.
        self._daily_aggregates : dict
            Initializes a deque to store daily aggregates of folder/file/size metrics.

        Notes
        -----
        - Retention size of the HistoryBuffer is computed based on the longest high-resolution trend window
          and `check_interval`, capped by `self.max_retain_cap`.
        - Thread-safe: acquires `self._lock` to update shared structures.
        """

        if max_depth is None:
            max_depth = self.scan_max_depth
        normalized = os.path.abspath(path)
        with self._lock:
            self._paths[name] = {
                "path": normalized,
                "warning_pct": float(warning_pct),
                "critical_pct": float(critical_pct),
                "fs_type": fs_type,
                "max_depth": int(max_depth),
            }
            # determine retention based on longest high-res window
            longest_seconds = max((self._parse_duration(w).total_seconds() for w in self.high_res_windows), default=3600)
            retain = int((longest_seconds / max(1.0, self.check_interval)) * (1.0 + self.retain_margin))
            retain = min(retain, self.max_retain_cap)
            # create history buffer
            self._history[name] = HistoryBuffer(retain)
            self._daily_aggregates.setdefault(name, deque(maxlen=90))  # store ~90 days of daily aggregates
            self.log.info("Added path %r -> %s (warning=%.1f%% critical=%.1f%%, retain=%d)",
                          name, normalized, warning_pct, critical_pct, retain)

    def remove_path(self, name: str) -> None:
        """Remove a path from monitor by name."""
        with self._lock:
            if name in self._paths:
                del self._paths[name]
                self._history.pop(name, None)
                self._daily_aggregates.pop(name, None)
                self.log.info("Removed path %r", name)

    def run(self) -> None:
        """
        Start the FSMonitor background thread.

        If already running, does nothing. If monitoring is disabled
        (self.cfg.enabled is False), logs a warning and does not start.
        """
        with self._lock:
            if not self.cfg.enabled:
                self.log.warning("FSMonitor.run() called but monitoring is disabled in configuration")
                return

            if self._thread and self._thread.is_alive():
                self.log.debug("run() called but already running")
                return

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="FSMonitorThread", daemon=True)
            self._thread.start()
            self.log.info("FSMonitor thread started")

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        """Start the FSMonitor background thread."""
        self.log.debug("Stopping FSMonitor")
        self._stop_event.set()
        t = None
        with self._lock:
            t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)
        self.log.info("FSMonitor stopped")

    def check(self) -> None:
        """Perform an immediate check for all registered paths and store compact sample."""
        self._check_cnt += 1
        self.log.debug(f'Running FSMonitor Check() #{self._check_cnt}')
        now_dt = datetime.now(timezone.utc)
        now_epoch = int(now_dt.timestamp())
        with self._lock:
            for name, meta in list(self._paths.items()):
                try:
                    snap = self._check_single(name, meta, now=now_dt)
                    # store compact sample
                    hb = self._history.get(name)
                    if hb is not None:
                        hb.append(now_epoch,
                                  snap.get("folders") or 0,
                                  snap.get("files") or 0,
                                  snap.get("size_mb") or 0.0,
                                  snap.get("used_mb") or 0.0,
                                  snap.get("total_mb") or 0.0)
                        # daily aggregate update
                        self._update_daily_aggregate(name, now_dt.date(), snap)
                except Exception:
                    self.log.exception("Error checking %s", name)

            self.log_compact_status()  # Updates self.is_critical and log compact status

    def log_compact_status(self, status: dict = None):
        """Log compact status summary."""
        status = status or self.status()
        for name, st in status.items():
            cur = st["current"]
            used = cur.get("used_pct", "<n/a>")
            folders = cur.get("folders", "<n/a>")
            files = cur.get("files", "<n/a>")
            self.log.info("Status %s: level=%s used=%%%.2f folders=%s files=%s forecast=%s",
                     name, st["status"], used if isinstance(used, float) else 0.0, folders, files, st["forecast"])

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Return the status mapping for all paths (same structure as original prototype)."""
        result: Dict[str, Dict[str, Any]] = {}
        now_dt = datetime.now(timezone.utc)
        now_epoch = int(now_dt.timestamp())
        is_critical_all = False
        with self._lock:
            for name, meta in self._paths.items():
                hb = self._history.get(name)
                current = hb.latest() if hb else None
                status_level = "normal"
                is_critical = False
                if current:
                    used_pct = ((current.get("used_mb", 0.0) / current.get("total_mb", 1.0) * 100.0)
                                if current.get("total_mb") else 0.0)
                    if used_pct >= meta["critical_pct"]:
                        status_level = "critical"
                        is_critical = True
                    elif used_pct >= meta["warning_pct"]:
                        status_level = "warning"
                    # augment current with percent fields for compatibility
                    current = dict(current)
                    current["used_pct"] = round(used_pct, 2)
                    current["free_pct"] = round(100.0 - used_pct, 2)
                is_critical_all = is_critical_all or is_critical
                # build trend map
                trend = {}
                for dstr in self.trend_durations:
                    trend[dstr] = self._compute_trend(name, dstr, now_epoch)
                forecast = self._compute_forecast(name, meta)
                result[name] = {
                    "status": status_level,
                    "is_critical": is_critical,
                    "levels": {"warning": meta["warning_pct"], "critical": meta["critical_pct"]},
                    "current": current or {},
                    "trend": trend,
                    "forecast": forecast,
                }
        self.is_critical = is_critical_all # set to True if at least one of the monitored paths becomes critical.
        return result

    def status_list(self) -> dict:
        """Get status for monitored items
        Example output:
            { 'root': 'normal', 'ssd': 'warning', 'sd_card': 'critical' }
        """
        result = {}
        status = self.status()

        for fs, fs_status in status.items():
            result[fs] = fs_status.get('status', '')
        return result

    def status_json(self, *, indent: Optional[int] = 2, fmt: Optional[str] = 'iso') -> str:
        """
        Return the same information as status() but serialized to JSON.

        Rationale/name: `status_json` is explicit and convenient for returning via
        HTTP APIs or writing to files. It ensures that timestamps like start_ts
        and end_ts and ts are optionally converted to ISO-8601 strings if fmt='iso'.
        """
        st = self.status()

        def _convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    # Convert start_ts/end_ts if fmt is 'iso' and value is int
                    if fmt == "iso" and k in ("start_ts", "end_ts", "ts") and isinstance(v, int):
                        # out[k] = datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
                        out[k] = datetime.fromtimestamp(v, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                    else:
                        out[k] = _convert(v)
                return out
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            # primitive
            return obj

        serializable = _convert(st)
        return json.dumps(serializable, indent=indent, default=str)

    # -------------------- internal methods ---------------------------------

    def _loop(self) -> None:
        """
        Background monitoring loop for FSMonitor.

        This method is intended to run in a dedicated daemon thread. It repeatedly
        performs filesystem checks at intervals specified by `self.check_interval`
        until `self._stop_event` is set.

        Behavior
        --------
        - Performs an immediate check before entering the loop.
        - Waits for `self.check_interval` seconds between checks.
        - Catches all exceptions to prevent the thread from terminating unexpectedly,
          logging any errors encountered.

        Thread Safety
        -------------
        - Uses internal methods (`self.check()`) which acquire necessary locks.
        - Stops gracefully when `self._stop_event` is set via `FSMonitor.stop()`.

        Returns
        -------
        None
        """

        try:
            # do an immediate check
            self.check()
            while not self._stop_event.wait(self.check_interval):
                self.check()
        except Exception:
            self.log.exception("Unhandled exception in FSMonitor loop")

    def _check_single(self, name: str, meta: Dict[str, Any], *, now: Optional[datetime] = None) -> Dict[str, Any]:
        if now is None:
            now = datetime.now(timezone.utc)
        path = meta["path"]
        max_depth = int(meta.get("max_depth", self.scan_max_depth))

        # filesystem usage
        try:
            du = shutil.disk_usage(path)
            total_mb = du.total / (1024 * 1024)
            used_mb = (du.total - du.free) / (1024 * 1024)
            free_mb = du.free / (1024 * 1024)
            used_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0
            free_pct = 100.0 - used_pct
        except FileNotFoundError:
            raise
        except Exception:
            self.log.exception("Error getting disk usage for %s", path)
            total_mb = used_mb = free_mb = used_pct = free_pct = 0.0

        snapshot: Dict[str, Any] = {
            "ts": now.isoformat(),
            "path": path,
            "size_mb": None,
            "folders": None,
            "files": None,
            "used_mb": round(used_mb, 3),
            "free_mb": round(free_mb, 3),
            "total_mb": round(total_mb, 3),
            "used_pct": round(used_pct, 2),
            "free_pct": round(free_pct, 2),
        }

        # root path shortcut
        if self._is_root_path(path):
            snapshot["note"] = "root"
            return snapshot

        # count day-folders and files
        folder_count = 0
        file_count = 0
        size_bytes = 0
        try:
            with os.scandir(path) as it:
                day_dirs = []
                for entry in it:
                    if entry.is_dir(follow_symlinks=False) and self._looks_like_date_dir(entry.name):
                        day_dirs.append(entry.path)
                if not day_dirs:
                    with os.scandir(path) as it2:
                        for entry in it2:
                            if entry.is_dir(follow_symlinks=False):
                                day_dirs.append(entry.path)

                for dpath in day_dirs:
                    folder_count += 1
                    for root, dirs, files in os.walk(dpath, topdown=True):
                        depth = self._path_depth_from(path, root)
                        if depth is None:
                            continue
                        if depth > max_depth:
                            dirs[:] = []
                            continue
                        file_count += len(files)
                        for fn in files:
                            try:
                                fp = os.path.join(root, fn)
                                st = os.lstat(fp)
                                size_bytes += st.st_size
                            except FileNotFoundError:
                                continue
                            except PermissionError:
                                continue
        except FileNotFoundError:
            self.log.warning("Path %s not found during scan", path)
        except PermissionError:
            self.log.warning("Permission denied scanning %s", path)
        except Exception:
            self.log.exception("Error scanning path %s", path)

        snapshot["folders"] = folder_count
        snapshot["files"] = file_count
        snapshot["size_mb"] = round(size_bytes / (1024 * 1024), 3)
        return snapshot

    def _update_daily_aggregate(self, name: str, date_obj: Any, snapshot: Dict[str, Any]) -> None:
        """Maintain a small daily aggregate (one entry per date) used for week/month windows."""
        dq = self._daily_aggregates.setdefault(name, deque(maxlen=90))
        date_iso = date_obj.isoformat()
        if dq and dq[-1][0] == date_iso:
            # update existing day aggregate by replacing last tuple
            prev = dq[-1]
            # store cumulative metrics for that date (folders, files, size_mb)
            new = (date_iso,
                   snapshot.get("folders") or prev[1],
                   snapshot.get("files") or prev[2],
                   snapshot.get("size_mb") or prev[3])
            dq[-1] = new
        else:
            dq.append((date_iso,
                       snapshot.get("folders") or 0,
                       snapshot.get("files") or 0,
                       snapshot.get("size_mb") or 0.0))

    # ----------------- trend & forecast -----------------------------------
    def _compute_trend(self, name: str, duration_str: str, now_epoch: int) -> Dict[str, Any]:
        """
        Compute growth trend for a path over a given duration.
        Only positive increments are counted, ignoring deletions.
        """
        hb = self._history.get(name)
        if not hb or len(hb) < 2:
            return {"folders": 0, "files": 0, "size_mb": 0.0, "start_ts": None, "end_ts": None}

        if duration_str == "all":
            start_idx = 0
        else:
            dur_td = self._parse_duration(duration_str)
            cutoff_epoch = now_epoch - int(dur_td.total_seconds())
            # find first sample at or after cutoff
            start_sample = hb.earliest_after(cutoff_epoch)
            if start_sample:
                start_idx = next(i for i in range(len(hb)) if hb._sample_at(i)["ts"] == start_sample["ts"])
            else:
                start_idx = 0

        folders_growth = 0
        files_growth = 0
        size_growth = 0.0

        prev = hb._sample_at(start_idx)
        for i in range(start_idx + 1, len(hb)):
            curr = hb._sample_at(i)
            folders_growth += max(0, curr["folders"] - prev["folders"])
            files_growth += max(0, curr["files"] - prev["files"])
            size_growth += max(0.0, (curr["size_mb"] or 0.0) - (prev["size_mb"] or 0.0))
            prev = curr

        start = hb._sample_at(start_idx)
        end = hb.latest()
        return {
            "folders": folders_growth,
            "files": files_growth,
            "size_mb": round(size_growth, 3),
            "start_ts": start.get("ts") if start else None,
            "end_ts": end.get("ts") if end else None
        }

    def _compute_forecast(self, name: str, meta: Dict[str, Any]) -> Dict[str, Optional[str]]:
        hb = self._history.get(name)
        if not hb or len(hb) == 0:
            return {"critical": None, "warning": None}
        now_snap = hb.latest()
        total_mb = now_snap.get("total_mb") or 0.0
        current_used_mb = now_snap.get("used_mb") or 0.0
        if total_mb <= 0:
            return {"critical": None, "warning": None}

        # prefer 24h, then 6h, then 1h, then all
        for choice in ("24h", "6h", "1h", "all"):
            tr = self._compute_trend(name, choice, int(now_snap.get("ts")))
            if tr and tr.get("size_mb") and tr.get("start_ts"):
                # compute rate
                start_ts = tr.get("start_ts")
                end_ts = tr.get("end_ts")
                dt = max(1.0, float(end_ts - start_ts))
                rate_mb_per_s = float(tr.get("size_mb")) / dt
                if rate_mb_per_s != 0.0:
                    break
        else:
            rate_mb_per_s = 0.0

        def secs_to_threshold(target_pct: float) -> Optional[str]:
            if rate_mb_per_s <= 0:
                return None
            target_used_mb = total_mb * (target_pct / 100.0)
            remaining_mb = target_used_mb - current_used_mb
            if remaining_mb <= 0:
                return "0d 00:00:00"
            seconds = remaining_mb / rate_mb_per_s
            if seconds <= 0:
                return None
            td = timedelta(seconds=int(seconds))
            days = td.days
            hh_mm_ss = str(td - timedelta(days=days))
            return f"{days}d {hh_mm_ss}"

        return {"critical": secs_to_threshold(meta["critical_pct"]), "warning": secs_to_threshold(meta["warning_pct"])}

    # ----------------- utility -------------------------------------------
    @staticmethod
    def _looks_like_date_dir(name: str) -> bool:
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", name))

    @staticmethod
    def _is_root_path(path: str) -> bool:
        path = os.path.abspath(path)
        if os.name == "nt":
            if re.match(r"^[A-Za-z]:\\?$", path):
                return True
            drive = os.path.splitdrive(path)[0] + os.sep
            return path.rstrip("\\/") == drive.rstrip("\\/")
        else:
            return os.path.normpath(path) == os.path.sep

    @staticmethod
    def _path_depth_from(base: str, path: str) -> Optional[int]:
        try:
            base_norm = os.path.normpath(os.path.abspath(base))
            path_norm = os.path.normpath(os.path.abspath(path))
            if not path_norm.startswith(base_norm):
                return None
            if base_norm == path_norm:
                return 0
            rel = os.path.relpath(path_norm, base_norm)
            parts = rel.split(os.sep)
            return len(parts)
        except Exception:
            return None

    @staticmethod
    def _parse_duration(s: str) -> timedelta:
        s = s.strip().lower()
        if s == "all":
            return timedelta(days=365 * 100)
        m = re.match(r"^(\d+)\s*([smhdw])?$", s)
        if m:
            val = int(m.group(1))
            unit = m.group(2) or "s"
            if unit == "s":
                return timedelta(seconds=val)
            if unit == "m":
                return timedelta(minutes=val)
            if unit == "h":
                return timedelta(hours=val)
            if unit == "d":
                return timedelta(days=val)
            if unit == "w":
                return timedelta(weeks=val)
        if "week" in s:
            num = int(re.match(r"^(\d+)", s).group(1)) if re.match(r"^\d+", s) else 1
            return timedelta(weeks=num)
        if "month" in s:
            num = int(re.match(r"^(\d+)", s).group(1)) if re.match(r"^\d+", s) else 1
            return timedelta(days=30 * num)
        if "hour" in s or s == "h":
            num = int(re.match(r"^(\d+)", s).group(1)) if re.match(r"^\d+", s) else 1
            return timedelta(hours=num)
        return timedelta(days=1)

    @staticmethod
    def _duration_sort_key(s: str) -> int:
        """Sort key for trend durations (e.g., '1h', '24h', 'week')."""
        if s == "all":
            return 10 ** 9
        td = FSMonitor._parse_duration(s)
        return int(td.total_seconds())

    def __del__(self):
        try:
            self.stop(timeout=1.0)
        except Exception:
            pass


# ---------------- example main / demo --------------------------------------
def main(run_seconds: int = 120) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("fsmonitor.example")

    cfg = MonitorCfg(enabled=True, check_interval=10.0, trend_durations=["1m", "5m", "24h", "all"], scan_max_depth=2)
    mon = FSMonitor(cfg)

    # Use the requested path 'output' pointing to ../output
    mon.add_path("output", os.path.join(os.getcwd(), "..", "output"), warning_pct=82.0, critical_pct=95.0, fs_type="data")

    # Add root for disk-only checks as an example
    root = os.path.abspath(os.sep) if os.name != "nt" else os.path.splitdrive(os.getcwd())[0] + os.sep
    mon.add_path("root", root, warning_pct=85.0, critical_pct=95.0, fs_type="internal")

    # Prepare demo: require an existing example folder ../output/2025-09-11
    base_output = os.path.abspath(os.path.join(os.getcwd(), "..", "output"))
    sample_day = os.path.join(base_output, "2025-09-11")
    if not os.path.isdir(sample_day):
        log.warning("Sample day folder %s missing. Creating a tiny demo folder.", sample_day)
        try:
            os.makedirs(sample_day, exist_ok=True)
            # create a couple of small files to simulate images
            for i in range(3):
                with open(os.path.join(sample_day, f"img_{i}.dat"), "wb") as f:
                    f.write(os.urandom(1024))
        except Exception:
            log.exception("Failed to create sample day folder")

    mon.run()

    start = time.time()
    day_counter = 1
    try:
        while (time.time() - start) < run_seconds:
            # Before each snapshot, copy sample_day to a new earlier date to simulate incoming days
            # produce dates like 2025-01-01, 2025-01-02, ... each cycle
            target_date = f"2025-01-{day_counter:02d}"
            target_path = os.path.join(base_output, target_date)
            if not os.path.exists(target_path):
                try:
                    shutil.copytree(sample_day, target_path)
                    log.info("Copied %s -> %s for demo", sample_day, target_path)
                except Exception:
                    log.exception("Failed to copy sample day to %s", target_path)
            else:
                log.debug("Demo target already exists: %s", target_path)

            time.sleep(cfg.check_interval / 2.0)
            # do a manual check (to show we can trigger it on demand)
            mon.check()

            # time.sleep(cfg.check_interval)
            mon.log_compact_status()

            # print status JSON compact
            sj = mon.status_json(indent=4)
            log.info("Status JSON (len=%d)", len(sj))
            # log.info(f"Status JSON: {sj}")

            day_counter += 1
            # rotate day_counter to avoid generating too many copies in long demo runs
            if day_counter > 20:
                day_counter = 1

            # sleep remainder of interval
            time.sleep(cfg.check_interval / 2.0)
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        log.info("Shutting down monitor")
        mon.stop()
        log.info("Exited")


if __name__ == "__main__":
    main(120)

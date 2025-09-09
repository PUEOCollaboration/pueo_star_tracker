#!/usr/bin/env python3
# analyze_telemetry.py

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
from termcolor import cprint
from tqdm import tqdm


# --- Configuration / defaults ---
TARGET_CADENCE = 1.0  # sec per entry (target)
MAX_LOG_SIZE_MB = 16  # MB per file
MAX_LOG_BACKUP_FILES = 64
APPROX_BYTES_PER_LINE = 512  # average bytes per log line (approx)


class AnalyzeTelemetry:
    def __init__(
        self,
        log_path: Optional[str] = None,
        target_cadence: float = TARGET_CADENCE,
        max_log_size_mb: int = MAX_LOG_SIZE_MB,
        max_log_backup_files: int = MAX_LOG_BACKUP_FILES,
    ):
        if log_path:
            self.log_dir = os.path.abspath(log_path)
        else:
            if os.path.isdir("./logs"):
                self.log_dir = os.path.abspath("./logs")
            else:
                self.log_dir = os.path.abspath(".")

        self.target_cadence = float(target_cadence)
        self.max_log_size_mb = int(max_log_size_mb)
        self.max_log_backup_files = int(max_log_backup_files)

        self.header: Optional[List[str]] = None
        self.rows: List[List[float]] = []
        self.timestamps: List[datetime] = []
        self.pre_header_buffer: List[tuple[datetime, List[Optional[float]]]] = []

        self.count_files = 0
        self.count_data_lines = 0
        self.count_header_lines = 0
        self.count_other_lines = 0

        self.re_header = re.compile(r"Telemetry header:\s*(.*)")
        self.re_data = re.compile(r"Telemetry data:\s*(.*)")
        self.re_first_number = re.compile(r"-?\d+(?:\.\d+)?")

    def _find_log_files(self) -> List[str]:
        pattern = os.path.join(self.log_dir, "telemetry.log*")
        matches = glob.glob(pattern)
        if not matches:
            cprint(f"[ERROR] No telemetry logs found in '{self.log_dir}'", "red", attrs=["bold"])
            sys.exit(1)
        return sorted(matches, key=os.path.getmtime)[: self.max_log_backup_files]

    def _parse_header_from_line(self, line: str) -> Optional[List[str]]:
        m = self.re_header.search(line)
        if not m:
            return None
        return [h.strip() for h in m.group(1).strip().split(",")]

    def _parse_data_from_line(self, line: str):
        m = self.re_data.search(line)
        if not m:
            return None, None
        parts = [p.strip() for p in m.group(1).strip().split(",")]
        if not parts:
            return None, None

        ts = None
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                ts = datetime.strptime(parts[0], fmt)
                break
            except ValueError:
                continue
        if ts is None:
            return None, None

        values: List[Optional[float]] = []
        for token in parts[1:]:
            mnum = self.re_first_number.search(token)
            if mnum:
                try:
                    values.append(float(mnum.group(0)))
                except ValueError:
                    values.append(float("nan"))
            else:
                values.append(float("nan"))
        return ts, values

    def _ensure_header_and_expand(self, new_header: List[str]):
        if self.header is None:
            self.header = list(new_header)
            for ts, vals in self.pre_header_buffer:
                self.rows.append(self._build_row_aligned(vals, len(self.header)))
                self.timestamps.append(ts)
                self.count_data_lines += 1
            self.pre_header_buffer.clear()
            self.count_header_lines += 1
        else:
            if new_header != self.header and len(new_header) > len(self.header):
                diff = len(new_header) - len(self.header)
                for row in self.rows:
                    row.extend([float("nan")] * diff)
                self.header = list(new_header)
                self.count_header_lines += 1
                cprint(f"[WARNING] Header expanded to {len(new_header)} fields.", "yellow")

    def _build_row_aligned(self, values: List[Optional[float]], target_len: int) -> List[float]:
        row = [float("nan")] * target_len
        n = len(values)
        if n <= target_len:
            start = target_len - n
            for i, v in enumerate(values):
                row[start + i] = v if v is not None else float("nan")
        else:
            row = [v if v is not None else float("nan") for v in values]
        return row

    def run(self):
        files = self._find_log_files()

        cprint("\nPUEO Star Tracker - Telemetry Analysis", "cyan", attrs=["bold"])
        for filepath in files:
            self.count_files += 1
            fname = os.path.basename(filepath)
            cprint(f"\nAnalyzing file: {fname} ...", "cyan")
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    for line in tqdm(fh, desc=f"Processing {fname}", unit=" lines"):
                        line = line.rstrip("\n")
                        if "Telemetry header:" in line:
                            hdr = self._parse_header_from_line(line)
                            if hdr:
                                self._ensure_header_and_expand(hdr)
                            continue
                        if "Telemetry data:" in line:
                            ts, vals = self._parse_data_from_line(line)
                            if ts is None:
                                self.count_other_lines += 1
                                continue
                            if self.header is None:
                                self.pre_header_buffer.append((ts, vals))
                                continue
                            row = self._build_row_aligned(vals, len(self.header))
                            self.rows.append(row)
                            self.timestamps.append(ts)
                            self.count_data_lines += 1
                            continue
                        self.count_other_lines += 1
            except Exception as exc:
                cprint(f"[ERROR] Cannot process {filepath}: {exc}", "red")

        if self.header is None:
            cprint("[ERROR] No telemetry header found in any logs.", "red", attrs=["bold"])
            sys.exit(1)

        if self.pre_header_buffer:
            for ts, vals in self.pre_header_buffer:
                self.rows.append(self._build_row_aligned(vals, len(self.header)))
                self.timestamps.append(ts)
                self.count_data_lines += 1
            self.pre_header_buffer.clear()

    # --------------------
    # statistics & printing
    # --------------------
    def _human_duration(self, seconds: float) -> str:
        seconds = int(round(seconds))
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, sec = divmod(rem, 60)
        return f"{int(days)}d {int(hours)}h {int(minutes)}m"

    def summarize_and_print(self):
        # basic summary
        cprint("\nPUEO Star Tracker - Telemetry Analysis", "cyan", attrs=["bold"])
        cprint(f"Processed files: {self.count_files}", "green")
        cprint(f"Telemetry header lines: {self.count_header_lines}", "green")
        cprint(f"Telemetry data lines processed: {self.count_data_lines}", "green")
        cprint(f"Other lines (ignored): {self.count_other_lines}", "green")

        if self.timestamps:
            first_ts = min(self.timestamps)
            last_ts = max(self.timestamps)
            cprint(f"First timestamp: {first_ts}", "magenta")
            cprint(f"Last  timestamp: {last_ts}", "magenta")
            span = (last_ts - first_ts).total_seconds()
            cprint(f"Time span: {span / 3600.0:.2f} hours", "magenta")
        else:
            first_ts = last_ts = None
            span = 0.0

        # compute observed cadence (sec per entry) using rows count
        n_rows = len(self.rows)
        observed_cadence = None
        if n_rows >= 2 and self.timestamps:
            # Using chronological order of timestamps
            sorted_ts = sorted(self.timestamps)
            observed_cadence = (sorted_ts[-1] - sorted_ts[0]).total_seconds() / max(1, (n_rows - 1))
            cprint(f"Target cadence : {self.target_cadence:.2f} sec/entry", "cyan")
            cprint(f"Observed cadence: {observed_cadence:.2f} sec/entry", "cyan")
        else:
            cprint(f"Target cadence : {self.target_cadence:.2f} sec/entry", "cyan")
            cprint("Observed cadence: N/A (not enough data)", "yellow")

        # build numpy matrix (n_rows x n_cols) for stats
        n_cols = len(self.header)
        if n_rows == 0:
            cprint("[WARNING] No data rows to compute numeric statistics.", "yellow")
            return

        mat = np.full((n_rows, n_cols), np.nan, dtype=float)
        for i, row in enumerate(self.rows):
            # pad/truncate row to n_cols
            if len(row) < n_cols:
                row_p = list(row) + [float("nan")] * (n_cols - len(row))
            else:
                row_p = row[:n_cols]
            mat[i, :] = np.array(row_p, dtype=float)

        # print transposed stats table (Parameter rows) with width param_col=43, values 12
        param_col_width = 43
        val_w = 12
        headers = ["Parameter", "Entries", "Min", "Max", "Avg", "Median", "Std", "Var"]
        header_line = f"{headers[0]:<{param_col_width}}" + "".join(h.rjust(val_w) for h in headers[1:])
        cprint("\nTelemetry numeric statistics:", "blue", attrs=["bold"])
        cprint(header_line, "yellow")
        cprint("-" * (param_col_width + val_w * (len(headers) - 1)), "yellow")

        for col_idx, col_name in enumerate(self.header):
            col_arr = mat[:, col_idx]
            # count non-NaN entries
            valid_mask = ~np.isnan(col_arr)
            entries = int(np.count_nonzero(valid_mask))
            if entries == 0:
                # skip empty columns
                continue
            col_vals = col_arr[valid_mask]
            min_v = np.nanmin(col_vals)
            max_v = np.nanmax(col_vals)
            mean_v = np.nanmean(col_vals)
            median_v = float(np.nanmedian(col_vals))
            std_v = float(np.nanstd(col_vals))
            var_v = float(np.nanvar(col_vals))
            line = (
                f"{col_name:<{param_col_width}}"
                f"{str(entries).rjust(val_w)}"
                f"{min_v:>{val_w}.2f}"
                f"{max_v:>{val_w}.2f}"
                f"{mean_v:>{val_w}.2f}"
                f"{median_v:>{val_w}.2f}"
                f"{std_v:>{val_w}.2f}"
                f"{var_v:>{val_w}.2f}"
            )
            cprint(line, "white")

        # estimation using bytes and cadence
        total_bytes = self.max_log_backup_files * self.max_log_size_mb * 1024 * 1024
        estimated_lines = total_bytes // APPROX_BYTES_PER_LINE

        # target-based time
        seconds_target = estimated_lines * self.target_cadence
        human_target = self._human_duration(seconds_target)

        cprint(f"\nEstimated max 'Telemetry data:' lines: {estimated_lines}", "magenta")
        cprint(f"Target conditions: log_cadence={self.target_cadence:.2f} row/sec, "
               f"max_log_size={self.max_log_size_mb}MB, max_backup_files={self.max_log_backup_files}", "magenta")
        cprint(f"Approximate total capture time (target cadence): {human_target}", "magenta")

        # observed-based time
        if observed_cadence and observed_cadence > 0:
            seconds_obs = estimated_lines * observed_cadence
            human_obs = self._human_duration(seconds_obs)
            cprint(f"Approximate total capture time (observed cadence {observed_cadence:.2f}s/row): {human_obs}",
                   "magenta")

        # done
        cprint("\nDone.", "cyan")


def main():
    parser = argparse.ArgumentParser(description="Analyze PUEO Star Tracker telemetry logs.")
    parser.add_argument("log_path", nargs="?", help="Path to folder with telemetry.log* files", default=None)
    args = parser.parse_args()

    analyzer = AnalyzeTelemetry(log_path=args.log_path)
    analyzer.run()
    analyzer.summarize_and_print()


if __name__ == "__main__":
    main()

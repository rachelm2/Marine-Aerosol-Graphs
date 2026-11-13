"""Plot aerosol size distributions and total particle concentration over time."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMESTAMP_COLUMN = "Start datetime (PC)"
CONCENTRATION_COLUMN = "N tot (1/cm^3)"
DIAMETER_COLUMN = "Dp (nm)"
DEFAULT_INPUTS = tuple(
    Path(__file__).with_name(f"Spider_MAGIC_{index}.csv")
    for index in range(1, 5)
)
DEFAULT_OUTPUT = Path(__file__).with_name("Spider_MAGIC_aerosol_size_distribution.png")


def load_aerosol_data(
    paths: Sequence[Path],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Load and combine CSV files containing aerosol scans."""
    if not paths:
        raise ValueError("Provide at least one aerosol CSV file.")

    frames: list[pd.DataFrame] = []
    diameter_columns: list[str] | None = None

    for path in paths:
        frame = pd.read_csv(path)
        required_columns = {TIMESTAMP_COLUMN, CONCENTRATION_COLUMN, DIAMETER_COLUMN}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required column(s): {missing}")

        diameter_start = frame.columns.get_loc(DIAMETER_COLUMN) + 1
        current_diameter_columns = list(frame.columns[diameter_start:])
        if not current_diameter_columns:
            raise ValueError(f"{path} does not contain any diameter-bin columns.")
        if diameter_columns is None:
            diameter_columns = current_diameter_columns
        elif current_diameter_columns != diameter_columns:
            raise ValueError(f"{path} has diameter bins that do not match the other files.")

        scan = frame[[TIMESTAMP_COLUMN, CONCENTRATION_COLUMN, *current_diameter_columns]].copy()
        scan[TIMESTAMP_COLUMN] = pd.to_datetime(scan[TIMESTAMP_COLUMN], errors="coerce")
        if scan[TIMESTAMP_COLUMN].isna().any():
            raise ValueError(f"{path} contains an invalid or missing scan timestamp.")

        numeric_columns = [CONCENTRATION_COLUMN, *current_diameter_columns]
        scan[numeric_columns] = scan[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if scan[CONCENTRATION_COLUMN].isna().any():
            raise ValueError(f"{path} contains a missing or non-numeric total concentration.")
        if (scan[numeric_columns] < 0).any().any():
            raise ValueError(f"{path} contains negative particle concentrations.")
        frames.append(scan)

    assert diameter_columns is not None
    try:
        diameters = np.asarray(diameter_columns, dtype=float)
    except ValueError as error:
        raise ValueError("Diameter-bin column names must be numeric particle diameters.") from error
    if not np.isfinite(diameters).all() or (diameters <= 0).any():
        raise ValueError("Diameter bins must be finite, positive values.")
    if not np.all(np.diff(diameters) > 0):
        raise ValueError("Diameter bins must be strictly increasing.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby(TIMESTAMP_COLUMN, as_index=False, sort=True).mean(
        numeric_only=True
    )
    times = pd.DatetimeIndex(combined[TIMESTAMP_COLUMN])
    total_concentration = combined[CONCENTRATION_COLUMN].to_numpy(dtype=float)
    distribution = combined[diameter_columns].to_numpy(dtype=float)
    return times, diameters, distribution, total_concentration


def _time_edges(times: pd.DatetimeIndex) -> np.ndarray:
    """Calculate time-cell edges, including for a single scan."""
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    if len(time_ns) == 1:
        half_minute = 30 * 1_000_000_000
        edge_ns = np.array([time_ns[0] - half_minute, time_ns[0] + half_minute])
    else:
        intervals = np.diff(time_ns)
        edge_ns = np.empty(len(time_ns) + 1, dtype=np.int64)
        edge_ns[1:-1] = time_ns[:-1] + intervals // 2
        edge_ns[0] = time_ns[0] - intervals[0] // 2
        edge_ns[-1] = time_ns[-1] + intervals[-1] // 2
    return mdates.date2num(pd.to_datetime(edge_ns).to_pydatetime())


def _diameter_edges(diameters: np.ndarray) -> np.ndarray:
    """Calculate bin edges for log-spaced diameter-bin centers."""
    log_diameters = np.log10(diameters)
    log_edges = np.empty(len(diameters) + 1, dtype=float)
    log_edges[1:-1] = (log_diameters[:-1] + log_diameters[1:]) / 2
    log_edges[0] = log_diameters[0] - (log_edges[1] - log_diameters[0])
    log_edges[-1] = log_diameters[-1] + (log_diameters[-1] - log_edges[-2])
    return np.power(10, log_edges)


def plot_aerosol_data(
    times: pd.DatetimeIndex,
    diameters: np.ndarray,
    distribution: np.ndarray,
    total_concentration: np.ndarray,
    output: Path,
) -> None:
    """Create and save a shared-time heatmap and total-concentration plot."""
    figure, (size_axis, total_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )
    mesh = size_axis.pcolormesh(
        _time_edges(times),
        _diameter_edges(diameters),
        np.ma.masked_invalid(distribution.T),
        shading="flat",
        cmap="viridis",
        vmin=0,
    )
    size_axis.set_yscale("log")
    size_axis.set_ylabel("Particle diameter (nm)")
    size_axis.set_title("Aerosol particle size distribution")
    figure.colorbar(mesh, ax=size_axis, label="dN/dlog10Dp (cm$^{-3}$)")

    total_axis.plot(times, total_concentration, color="tab:blue", linewidth=0.8)
    total_axis.set_ylabel("N total (cm$^{-3}$)")
    total_axis.set_xlabel("Time")
    total_axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    total_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    total_axis.grid(True, alpha=0.25)
    figure.suptitle("Spider MAGIC aerosol measurements")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot aerosol size distributions and total number concentration."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="Input CSV files (default: Spider_MAGIC_1.csv through Spider_MAGIC_4.csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT.name}).",
    )
    args = parser.parse_args()

    input_paths = args.csv_files or list(DEFAULT_INPUTS)
    times, diameters, distribution, total_concentration = load_aerosol_data(input_paths)
    plot_aerosol_data(times, diameters, distribution, total_concentration, args.output)
    print(f"Saved aerosol plot to {args.output}")


if __name__ == "__main__":
    main()
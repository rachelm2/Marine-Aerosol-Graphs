"""Plot SpiderMAGIC environmental and instrument measurements over time."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


TIMESTAMP_COLUMN = "Start datetime (PC)"
DEFAULT_INPUTS = tuple(
    Path(__file__).with_name(f"Spider_MAGIC_{index}.csv")
    for index in range(1, 5)
)
DEFAULT_OUTPUT = Path(__file__).with_name("Spider_MAGIC_environmental_conditions.png")

PANELS = (
    (
        "Relative humidity (%)",
        (("RH (%)", "RH (%)", "tab:blue", "-"),
         ("Wick sensor (%)", "Wick sensor (%)", "tab:orange", "--")),
    ),
    (
        "Temperature (°C)",
        (("T (deg C)", "Temperature", "tab:red", "-"),
         ("Dew point (deg C)", "Dew point", "tab:cyan", "--")),
    ),
    ("Pressure (mbar)", (("P (mbar)", "Pressure", "tab:green", "-"),)),
    (
        "Flow (ccm)",
        (("Qsh (ccm)", "Qsh", "tab:purple", "-"),
         ("Qa (ccm)", "Qa", "tab:brown", "--"),
         ("Qcpc (ccm)", "Qcpc", "tab:pink", ":")),
    ),
    ("Pulse height (mV)", (("Pulse height (mV)", "Pulse height", "tab:gray", "-"),)),
)

REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        [TIMESTAMP_COLUMN]
        + [column for _, series in PANELS for column, _, _, _ in series]
    )
)
NUMERIC_COLUMNS = tuple(column for column in REQUIRED_COLUMNS if column != TIMESTAMP_COLUMN)


def load_measurements(paths: Sequence[Path]) -> pd.DataFrame:
    """Load selected measurements from CSV files and sort them by scan time."""
    if not paths:
        raise ValueError("Provide at least one SpiderMAGIC CSV file.")

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        missing_columns = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required column(s): {missing}")

        measurements = frame[list(REQUIRED_COLUMNS)].copy()
        measurements[TIMESTAMP_COLUMN] = pd.to_datetime(
            measurements[TIMESTAMP_COLUMN], errors="coerce"
        )
        if measurements[TIMESTAMP_COLUMN].isna().any():
            raise ValueError(f"{path} contains an invalid or missing scan timestamp.")
        measurements[list(NUMERIC_COLUMNS)] = measurements[list(NUMERIC_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )
        if measurements[list(NUMERIC_COLUMNS)].isna().any().any():
            raise ValueError(f"{path} contains missing or non-numeric measurements.")
        frames.append(measurements)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )


def plot_measurements(measurements: pd.DataFrame, output: Path) -> None:
    """Create and save the five-panel shared-time figure."""
    figure, axes = plt.subplots(
        len(PANELS),
        1,
        sharex=True,
        figsize=(14, 12),
        layout="constrained",
    )

    for axis, (y_label, series) in zip(axes, PANELS):
        for column, label, color, line_style in series:
            axis.plot(
                measurements[TIMESTAMP_COLUMN],
                measurements[column],
                label=label,
                color=color,
                linestyle=line_style,
                linewidth=0.8,
            )
        axis.set_ylabel(y_label)
        axis.grid(axis="y", color="0.85", linewidth=0.6)
        if len(series) > 1:
            axis.legend(loc="upper right", frameon=False, ncol=len(series))

    axes[-1].set_xlabel("Start datetime (PC)")
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    figure.suptitle(
        "SpiderMAGIC Environmental and Instrument Conditions During Indoor Sampling"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SpiderMAGIC environmental and instrument conditions over time."
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

    measurements = load_measurements(args.csv_files or DEFAULT_INPUTS)
    plot_measurements(measurements, args.output)
    print(f"Saved SpiderMAGIC conditions plot to {args.output}")


if __name__ == "__main__":
    main()

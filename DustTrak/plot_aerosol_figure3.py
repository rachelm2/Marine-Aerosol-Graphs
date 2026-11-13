import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt


DATA_HEADER_ROW = 37
CONCENTRATION_COLUMN = "PM2.5 [mg/m3]"
TIME_TICK_INTERVAL_SECONDS = 900
SMOOTHING_WINDOW = 5


def moving_average(values, window_size):
    smoothed = []
    half_window = window_size // 2
    for index in range(len(values)):
        start = max(0, index - half_window)
        end = min(len(values), index + half_window + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def read_test(path):
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        metadata = {}
        for _ in range(DATA_HEADER_ROW - 1):
            row = next(reader, None)
            if row and len(row) > 1 and row[0].strip():
                metadata[row[0].strip()] = row[1].strip()

        headers = next(reader, None)
        if headers is None:
            raise ValueError(f"{path.name} is missing the row {DATA_HEADER_ROW} header.")

        required_columns = ["Elapsed Time [s]", CONCENTRATION_COLUMN]
        missing_columns = [column for column in required_columns if column not in headers]
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing required columns: {', '.join(missing_columns)}"
            )

        try:
            start_time = datetime.strptime(
                f"{metadata['Test Start Date']} {metadata['Test Start Time']}",
                "%m/%d/%Y %I:%M:%S %p",
            )
        except KeyError as error:
            raise ValueError(f"{path.name} is missing metadata: {error.args[0]}") from error
        except ValueError as error:
            raise ValueError(f"{path.name} has an invalid test start date or time.") from error

        samples = []
        rows = csv.DictReader(csv_file, fieldnames=headers)
        for row_number, row in enumerate(rows, start=DATA_HEADER_ROW + 1):
            if not row or not any(value and value.strip() for value in row.values()):
                continue
            try:
                elapsed_seconds = float(row["Elapsed Time [s]"])
                concentration = float(row[CONCENTRATION_COLUMN])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path.name} has invalid measurement data on row {row_number}."
                ) from error
            samples.append(
                (
                    start_time + timedelta(seconds=elapsed_seconds),
                    concentration,
                )
            )

    return start_time, samples


def main():
    data_directory = Path(__file__).resolve().parent
    csv_paths = sorted(data_directory.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_directory}.")

    tests = [read_test(path) for path in csv_paths]
    tests.sort(key=lambda test: test[0])

    plot_times = []
    concentrations = []
    tick_positions = []
    tick_labels = []
    next_plot_time = 0.0
    for _, samples in tests:
        if not samples:
            continue

        first_timestamp = samples[0][0]
        test_times = []
        test_concentrations = []
        for timestamp, concentration in samples:
            elapsed_seconds = (timestamp - first_timestamp).total_seconds()
            if concentration == 0:
                continue
            test_times.append(next_plot_time + elapsed_seconds)
            test_concentrations.append(concentration)

        plot_times.extend(test_times)
        concentrations.extend(
            moving_average(test_concentrations, SMOOTHING_WINDOW)
        )

        test_duration = (samples[-1][0] - first_timestamp).total_seconds()
        tick_elapsed = 0
        while tick_elapsed <= test_duration:
            if test_duration - tick_elapsed >= TIME_TICK_INTERVAL_SECONDS / 2:
                tick_positions.append(next_plot_time + tick_elapsed)
                tick_labels.append(
                    (first_timestamp + timedelta(seconds=tick_elapsed)).strftime(
                        "%m-%d\n%I:%M %p"
                    )
                )
            tick_elapsed += TIME_TICK_INTERVAL_SECONDS
        next_plot_time += test_duration + 1

    if not plot_times:
        raise ValueError("No non-zero PM2.5 measurements were found in the CSV files.")

    figure, axes = plt.subplots(figsize=(12, 8))
    axes.plot(plot_times, concentrations, color="tab:blue", linewidth=1, label="PM2.5")
    axes.set_title("PM2.5 Concentration vs. Real Time")
    axes.set_xlabel("Real Time (gaps between tests removed)")
    axes.set_ylabel(
        f"Concentration (mg/m³, {SMOOTHING_WINDOW}-sample moving average)"
    )
    axes.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    axes.set_xlim(plot_times[0], plot_times[-1])
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    output_path = data_directory / "aerosol_concentrations_figure3.png"
    figure.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
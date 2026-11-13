import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt


DATA_HEADER_ROW = 37
TIME_TICK_INTERVAL_SECONDS = 900
SMOOTHING_WINDOW = 5
METRICS = {
    "PM2.5 [mg/m3]": "PM2.5",
    "PM1 [mg/m3]": "PM1",
    "PM4 [mg/m3]": "PM4",
}
COLORS = {
    "PM2.5 [mg/m3]": "tab:orange",
    "PM1 [mg/m3]": "tab:purple",
    "PM4 [mg/m3]": "tab:blue",
}
LINESTYLES = {
    "PM2.5 [mg/m3]": "-",
    "PM1 [mg/m3]": "--",
    "PM4 [mg/m3]": ":",
}


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

        required_columns = ["Elapsed Time [s]", *METRICS]
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
                concentrations = {
                    column: float(row[column]) for column in METRICS
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path.name} has invalid measurement data on row {row_number}."
                ) from error
            samples.append(
                (start_time + timedelta(seconds=elapsed_seconds), concentrations)
            )

    return start_time, samples


def main():
    data_directory = Path(__file__).resolve().parent
    csv_paths = sorted(data_directory.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_directory}.")

    tests = [read_test(path) for path in csv_paths]
    tests.sort(key=lambda test: test[0])

    series_times = {column: [] for column in METRICS}
    series_values = {column: [] for column in METRICS}
    tick_positions = []
    tick_labels = []
    next_plot_time = 0.0
    for _, samples in tests:
        if not samples:
            continue

        first_timestamp = samples[0][0]
        test_times = {column: [] for column in METRICS}
        test_values = {column: [] for column in METRICS}
        for timestamp, concentrations in samples:
            elapsed_seconds = (timestamp - first_timestamp).total_seconds()
            for column in METRICS:
                concentration = concentrations[column]
                if concentration == 0:
                    continue
                test_times[column].append(next_plot_time + elapsed_seconds)
                test_values[column].append(concentration)

        for column in METRICS:
            series_times[column].extend(test_times[column])
            series_values[column].extend(
                moving_average(test_values[column], SMOOTHING_WINDOW)
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

    if not any(series_times.values()):
        raise ValueError("No non-zero particle measurements were found in the CSV files.")

    figure, axes = plt.subplots(figsize=(12, 10))
    for column, label in METRICS.items():
        axes.plot(
            series_times[column],
            series_values[column],
            color=COLORS[column],
            linestyle=LINESTYLES[column],
            linewidth=1,
            label=label,
        )

    axes.set_title("Aerosol Concentration vs. Real Time")
    axes.set_xlabel("Real Time (gaps between tests removed)")
    axes.set_ylabel(
        f"Concentration (mg/m³, {SMOOTHING_WINDOW}-sample moving average)"
    )
    axes.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
    axes.set_xlim(
        min(times[0] for times in series_times.values() if times),
        max(times[-1] for times in series_times.values() if times),
    )
    axes.grid(True, alpha=0.3)
    axes.legend(title="Particle size")
    figure.tight_layout()
    output_path = data_directory / "aerosol_concentrations.png"
    figure.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = Path(__file__).resolve().parent / "POPS_Aerosol.csv"
SOURCE_TIME_ZONE = "UTC"  # Change if the instrument records local time

df = pd.read_csv(CSV_FILE)

# Get the date from Time Stamp, then add Julian Time in seconds.
date = pd.to_datetime(
	df["Time Stamp"],
	format="%Y%m%dT%H%M%S",
	errors="coerce",
).dt.normalize()

seconds = pd.to_numeric(df["Julian Time"], errors="coerce")
df["DateTime"] = date + pd.to_timedelta(seconds, unit="s")

# Interpret the readings in their source timezone, then convert to UTC.
df["DateTime"] = (
	df["DateTime"]
	.dt.tz_localize(SOURCE_TIME_ZONE)
	.dt.tz_convert("UTC")
)

data_columns = ("Particle Conc", "Ambient Pressure", "Laser Temperature", "Flow")
for source_column in data_columns:
	df[source_column] = pd.to_numeric(df[source_column], errors="coerce")

df = df.dropna(subset=["DateTime"]).sort_values("DateTime")
particle_df = df.dropna(subset=["Particle Conc"])

plt.plot(particle_df["DateTime"], particle_df["Particle Conc"])
plt.xlabel("Date/time (UTC)")
plt.ylabel("Particle concentration (particles cm⁻³)")
plt.title("Particle concentration over time")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.tight_layout()

fig, pressure_axis = plt.subplots()
temperature_axis = pressure_axis.twinx()
flow_axis = pressure_axis.twinx()
flow_axis.spines["right"].set_position(("outward", 60))

pressure_line, = pressure_axis.plot(
	df["DateTime"], df["Ambient Pressure"], color="tab:blue", label="Pressure"
)
temperature_line, = temperature_axis.plot(
	df["DateTime"], df["Laser Temperature"], color="tab:orange",
	label="Laser temperature"
)
flow_line, = flow_axis.plot(
	df["DateTime"], df["Flow"], color="tab:green", label="Flow"
)

pressure_axis.set_xlabel("Date/time (UTC)")
pressure_axis.set_ylabel("Pressure (mbar)", color="tab:blue")
temperature_axis.set_ylabel("Laser temperature (°C)", color="tab:orange")
flow_axis.set_ylabel("Flow (cc/min)", color="tab:green")
pressure_axis.set_title("Pressure, laser temperature, and flow over time")
pressure_axis.tick_params(axis="y", labelcolor="tab:blue")
temperature_axis.tick_params(axis="y", labelcolor="tab:orange")
flow_axis.tick_params(axis="y", labelcolor="tab:green")
pressure_axis.grid(True)
pressure_axis.legend(
	[pressure_line, temperature_line, flow_line],
	[line.get_label() for line in (pressure_line, temperature_line, flow_line)],
	loc="upper left",
)
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

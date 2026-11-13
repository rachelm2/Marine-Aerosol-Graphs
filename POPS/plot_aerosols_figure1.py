from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = Path(__file__).resolve().parent / "POPS_Aerosol.csv"
SOURCE_TIME_ZONE = "UTC"  # Change if the instrument records local time
SMOOTHING_WINDOW = 5

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

df["Particle Conc"] = pd.to_numeric(df["Particle Conc"], errors="coerce")
df = df.dropna(subset=["DateTime", "Particle Conc"]).sort_values("DateTime")
df["Particle Conc Smoothed"] = (
    df["Particle Conc"]
    .rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1)
    .mean()
)

plt.plot(df["DateTime"], df["Particle Conc Smoothed"])
plt.xlabel("Date/time (UTC)")
plt.ylabel(
    f"Particle concentration (particles cm⁻³, {SMOOTHING_WINDOW}-sample moving average)"
)
plt.title("Particle concentration over time")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.show()
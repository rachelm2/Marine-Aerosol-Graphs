import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_FILE = Path(__file__).with_name("POPS_Aerosol.csv")

header = pd.read_csv(CSV_FILE, nrows=0).columns
bin_columns = sorted(
	(column for column in header if re.fullmatch(r"b\d+_\d+nm", column)),
	key=lambda column: int(re.search(r"_(\d+)nm$", column).group(1)),
)
if len(bin_columns) < 2:
	raise ValueError("The CSV must contain at least two particle-size bin columns.")

df = pd.read_csv(CSV_FILE, usecols=bin_columns)

for column in bin_columns:
	df[column] = pd.to_numeric(df[column], errors="coerce")
df = df.dropna(subset=bin_columns)
if df.empty:
	raise ValueError("The CSV contains no rows with valid particle-size bin counts.")

diameters_nm = np.array(
	[int(re.search(r"_(\d+)nm$", column).group(1)) for column in bin_columns],
	dtype=float,
)
log_diameters = np.log10(diameters_nm)
log_edges = np.empty(len(log_diameters) + 1)
log_edges[1:-1] = (log_diameters[:-1] + log_diameters[1:]) / 2
log_edges[0] = log_diameters[0] - (log_edges[1] - log_diameters[0])
log_edges[-1] = log_diameters[-1] + (log_diameters[-1] - log_edges[-2])

# Estimate bin boundaries halfway between the listed diameter centers in log space.
counts_per_bin = df[bin_columns].to_numpy(dtype=float)
dndlogdp = counts_per_bin / np.diff(log_edges)
mean_dndlogdp = dndlogdp.mean(axis=0)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(diameters_nm, mean_dndlogdp, marker="o", color="#0072B2")
ax.set_xscale("log")
ax.set_xlabel("Dp (nm)")
ax.set_ylabel("Mean dN/dlog₁₀Dp (raw counts per log₁₀ interval)")
ax.set_title("Mean particle size distribution")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.show()
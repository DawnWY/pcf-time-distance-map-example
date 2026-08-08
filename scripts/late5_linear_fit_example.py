import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Change this to the event folder you want to process.
folder = Path("/Volumes/Ying_Disk/research/PCF/T_Dmap/20121110")
fit_points_file = folder / "selected_fit_points_reviewed.csv"

# Read manually selected points: time_hours, height_arcsec.
arr = np.loadtxt(fit_points_file, delimiter=",", skiprows=1)
arr = np.atleast_2d(arr)

# Sort by time.
arr = arr[np.argsort(arr[:, 0])]

# Use the last 5 selected points. If fewer than 5 exist, use all available points.
n_use = min(5, arr.shape[0])
late = arr[-n_use:]

# Unit conversion.
t_sec = late[:, 0] * 3600.0      # hours -> seconds
h_km = late[:, 1] * 725.0        # arcsec -> km

# Linear fit: h(t) = v * (t - t_ref) + h0.
t_ref = t_sec[0]
x = t_sec - t_ref
v_late_km_s, h0_late_km = np.polyfit(x, h_km, 1)

print(f"Using last {n_use} points")
print(f"Late linear velocity = {v_late_km_s:.2f} km/s")
print(f"Late linear velocity = {v_late_km_s * 1000:.2f} m/s")

# Plot the last selected points and the linear fit.
x_dense = np.linspace(x.min(), x.max(), 200)
t_dense = x_dense + t_ref
h_dense = v_late_km_s * x_dense + h0_late_km

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(t_sec / 3600, h_km / 1000, s=20, label="Last 5 points")
ax.plot(t_dense / 3600, h_dense / 1000, "r-", label="Late linear fit")

ax.set_xlabel("Time since start [hours]")
ax.set_ylabel("Height [Mm]")
ax.legend()
ax.grid(alpha=0.3)

ax.text(
    0.05,
    0.92,
    fr"$v_{{late}}={v_late_km_s:.1f}$ km s$^{{-1}}$",
    transform=ax.transAxes,
    fontsize=10,
    va="top",
    ha="left",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
)

plt.savefig(folder / "late5_linear_fit.png", dpi=300, bbox_inches="tight")
plt.savefig(folder / "late5_linear_fit.pdf", bbox_inches="tight")
plt.show()

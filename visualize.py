"""Create validation depth profiles and a map of selected sample locations."""

import matplotlib.pyplot as plt
import numpy as np

from evaluate import evaluate
from model import DEPTHS, load_or_generate_dummy_data, train_model


def main():
    data = load_or_generate_dummy_data()
    model, (valid_x, valid_y), stats = train_model(data)
    predicted, actual, _ = evaluate(model, valid_x, valid_y, stats[2], stats[3])
    sample_ids = np.linspace(0, len(valid_x) - 1, 3, dtype=int)
    lat_ids = np.array([2, len(data["lat"]) // 2, len(data["lat"]) - 3])
    lon_ids = np.array([2, len(data["lon"]) // 2, len(data["lon"]) - 3])

    figure, axis = plt.subplots(figsize=(7, 5))
    for sample_id, lat_id, lon_id in zip(sample_ids, lat_ids, lon_ids):
        axis.plot(actual[sample_id, :, lat_id, lon_id], DEPTHS, "o-",
                  label=f"Actual ({data['lat'][lat_id]:.2f}, {data['lon'][lon_id]:.2f})")
        axis.plot(predicted[sample_id, :, lat_id, lon_id], DEPTHS, "x--", color=axis.lines[-1].get_color())
    axis.invert_yaxis()
    axis.set(xlabel="Temperature (deg C)", ylabel="Depth (m)", title="OceanEmbed depth profiles")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig("depth_profiles.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(data["lon"][lon_ids], data["lat"][lat_ids], s=70, c=[0, 1, 2], cmap="viridis", edgecolor="black")
    for number, lat_id, lon_id in zip(range(1, 4), lat_ids, lon_ids):
        axis.annotate(str(number), (data["lon"][lon_id], data["lat"][lat_id]), xytext=(5, 5), textcoords="offset points")
    axis.set(xlabel="Longitude", ylabel="Latitude", title="Selected validation locations")
    axis.set_xlim(data["lon"].min(), data["lon"].max())
    axis.set_ylim(data["lat"].min(), data["lat"].max())
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig("region_map.png", dpi=160)
    plt.close(figure)
    print("Saved depth_profiles.png and region_map.png")


if __name__ == "__main__":
    main()
"""OceanEmbed: compact learned embedding model and training entry point."""

from pathlib import Path
import pickle

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEPTHS = np.array([0, 50, 100, 500, 1000], dtype=float)
SURFACE_KEYS = ("sst", "ssh", "sss", "u_current", "v_current", "u_wind", "v_wind")


def load_or_generate_dummy_data(path="ocean_data.npz", seed=7, n_days=64,
                                n_lat=16, n_lon=20):
    """Load the shared NPZ contract, or create a deterministic local substitute."""
    path = Path(path)
    if path.exists():
        with np.load(path) as archive:
            required = set(SURFACE_KEYS) | {"subsurface_temp", "lat", "lon"}
            missing = required.difference(archive.files)
            if missing:
                raise KeyError("ocean_data.npz is missing: " + ", ".join(sorted(missing)))
            return {key: archive[key].astype(np.float32) for key in required}

    rng = np.random.default_rng(seed)
    lat = np.linspace(-2.5, 2.5, n_lat, dtype=np.float32)
    lon = np.linspace(145.0, 150.0, n_lon, dtype=np.float32)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    days = np.arange(n_days, dtype=np.float32)[:, None, None]
    spatial = np.sin(xx / 2.2) + 0.6 * np.cos(yy * 1.7)
    phase = np.sin(days / 8.0)
    noise = lambda scale: rng.normal(0, scale, (n_days, n_lat, n_lon)).astype(np.float32)
    sst = (27.0 + 0.5 * spatial + 0.3 * phase + noise(0.08)).astype(np.float32)
    ssh = (0.15 * spatial + 0.04 * phase + noise(0.015)).astype(np.float32)
    sss = (35.0 - 0.15 * spatial + noise(0.025)).astype(np.float32)
    u_current = (0.25 * np.cos(yy) + 0.03 * phase + noise(0.02)).astype(np.float32)
    v_current = (0.18 * np.sin(xx) + noise(0.02)).astype(np.float32)
    u_wind = (5.0 + 0.7 * np.cos(xx) + noise(0.15)).astype(np.float32)
    v_wind = (2.0 + 0.5 * np.sin(yy) + noise(0.15)).astype(np.float32)
    depth_scale = np.exp(-DEPTHS / 850.0).astype(np.float32)[None, :, None, None]
    surface_signal = (sst - 25.0)[:, None] * depth_scale
    subsurface_temp = (10.0 + surface_signal + 0.3 * spatial[None, None]
                       + noise(0.05)[:, None] * depth_scale).astype(np.float32)
    return {"sst": sst, "ssh": ssh, "sss": sss, "u_current": u_current,
            "v_current": v_current, "u_wind": u_wind, "v_wind": v_wind,
            "subsurface_temp": subsurface_temp, "lat": lat, "lon": lon}


def _standardize(train, values):
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = np.maximum(train.std(axis=(0, 2, 3), keepdims=True), 1e-6)
    return (values - mean) / std, mean, std


def prepare_arrays(data, split=0.8):
    surface = np.stack([data[key] for key in SURFACE_KEYS], axis=1).astype(np.float32)
    target = data["subsurface_temp"].astype(np.float32)
    n_train = max(1, int(len(surface) * split))
    train_x, valid_x = surface[:n_train], surface[n_train:]
    train_y, valid_y = target[:n_train], target[n_train:]
    train_x, surface_mean, surface_std = _standardize(train_x, train_x)
    valid_x = (valid_x - surface_mean) / surface_std
    train_y, target_mean, target_std = _standardize(train_y, train_y)
    valid_y = (valid_y - target_mean) / target_std
    return train_x, train_y, valid_x, valid_y, (surface_mean, surface_std, target_mean, target_std)


if TORCH_AVAILABLE:
    class OceanEmbed(nn.Module):
        """Small convolutional encoder-decoder with an explicit embedding."""

        def __init__(self, embedding_dim=16, n_depths=5):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(7, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.to_embedding = nn.Linear(32, embedding_dim)
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, 32 * 4 * 5), nn.ReLU(),
                nn.Unflatten(1, (32, 4, 5)),
                nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(24, 16, 4, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, n_depths, 3, padding=1),
            )

        def forward(self, surface):
            embedding = self.to_embedding(self.encoder(surface).flatten(1))
            decoded = self.decoder(embedding)
            decoded = F.interpolate(decoded, size=surface.shape[-2:], mode="bilinear", align_corners=False)
            return decoded, embedding

    def train_model(data=None, epochs=400, batch_size=8, learning_rate=1e-3, seed=7):
        torch.manual_seed(seed)
        data = data or load_or_generate_dummy_data()
        train_x, train_y, valid_x, valid_y, stats = prepare_arrays(data)
        train_x, train_y = torch.from_numpy(train_x), torch.from_numpy(train_y)
        valid_x, valid_y = torch.from_numpy(valid_x), torch.from_numpy(valid_y)
        model = OceanEmbed()
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                prediction, _ = model(batch_x)
                loss = nn.functional.mse_loss(prediction, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  epoch {epoch + 1:>4d}/{epochs}  train_loss={avg_loss:.6f}")
        print(f"\nTraining complete. Final loss: {loss_history[-1]:.6f}")
        print(f"Loss curve (every 50 epochs): {[f'{loss_history[i]:.4f}' for i in range(0, len(loss_history), 50)]}")
        return model, (valid_x, valid_y), stats

    def save_trained_model(model, stats, path="oceanembed_model.pt"):
        torch.save({"model_state_dict": model.state_dict(), "stats": stats,
                    "depths": DEPTHS, "surface_keys": SURFACE_KEYS}, path)
else:
    from sklearn.neural_network import MLPRegressor

    class OceanEmbed:
        """CPU fallback: learned 16-unit bottleneck MLP for each grid cell."""

        def __init__(self):
            self.regressor = MLPRegressor(hidden_layer_sizes=(16, 32), activation="relu",
                                          solver="adam", max_iter=120, random_state=7,
                                          early_stopping=True, validation_fraction=0.1)

        def predict(self, surface):
            n_days, _, n_lat, n_lon = surface.shape
            features = surface.transpose(0, 2, 3, 1).reshape(-1, 7)
            return self.regressor.predict(features).reshape(n_days, n_lat, n_lon, 5).transpose(0, 3, 1, 2)

    def train_model(data=None, epochs=35, batch_size=8, learning_rate=1e-3, seed=7):
        del epochs, batch_size, learning_rate
        data = data or load_or_generate_dummy_data()
        train_x, train_y, valid_x, valid_y, stats = prepare_arrays(data)
        model = OceanEmbed()
        features = train_x.transpose(0, 2, 3, 1).reshape(-1, 7)
        targets = train_y.transpose(0, 2, 3, 1).reshape(-1, 5)
        model.regressor.random_state = seed
        model.regressor.fit(features, targets)
        return model, (valid_x, valid_y), stats

    def save_trained_model(model, stats, path="oceanembed_model.pkl"):
        with open(path, "wb") as handle:
            pickle.dump({"model": model, "stats": stats, "depths": DEPTHS,
                         "surface_keys": SURFACE_KEYS}, handle)


if __name__ == "__main__":
    model, (_, valid_y), stats = train_model()
    save_trained_model(model, stats)
    backend = "PyTorch CNN" if TORCH_AVAILABLE else "scikit-learn MLP fallback"
    print(f"Trained OceanEmbed ({backend}).")
    print(f"Validation samples: {len(valid_y)}")
    print(f"Saved {'oceanembed_model.pt' if TORCH_AVAILABLE else 'oceanembed_model.pkl'}")

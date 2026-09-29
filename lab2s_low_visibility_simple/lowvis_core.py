"""
lowvis_core.py — shared code of the SIMPLIFIED low-visibility exercise (notebooks 2S-1, 2S-2, 2S-3 and serve_simple.py).

The model predicts the probability that visibility at Milano Malpensa is below 1500 m EXACTLY 3 HOURS from now.
This is a PROXY for Low-Visibility Procedures (LVP): the data has no record of LVP being declared, and real
LVP depend on runway visual range (RVR), cloud ceiling and local procedures.
"""
import json                                   # json: reads and writes the model card (a text file with labelled values)
from pathlib import Path                      # Path: file paths that work on every operating system

import numpy as np                            # numpy: calculations on arrays of numbers
import pandas as pd                           # pandas: tables of data
import torch                                  # PyTorch: neural networks
import torch.nn as nn                         # building blocks of neural networks

# The 8 model inputs, in this exact order
FEATURES = ["vis_m", "spread_c", "rh", "wind_kt", "hour_sin", "hour_cos", "month_sin", "month_cos"]

# Readable names of the inputs, used in plots
FEATURE_NAMES = {
    "vis_m": "visibility now (m)",
    "spread_c": "temperature − dew point (°C)",
    "rh": "humidity (%)",
    "wind_kt": "wind speed (kt)",
    "hour_sin": "hour (sine)",
    "hour_cos": "hour (cosine)",
    "month_sin": "month (sine)",
    "month_cos": "month (cosine)",
}

LOW_VIS_M = 1500                              # visibility threshold of our target (m)
HORIZON_H = 3                                 # how many hours ahead we predict


def load_weather(path):
    raw = pd.read_csv(path, na_values="M", low_memory=False)      # read the archive; "M" means missing
    raw["valid"] = pd.to_datetime(raw["valid"])                   # observation time: text -> date/time
    routine = raw[raw["valid"].dt.minute == 50]                   # keep only the routine hourly reports (HH:50)
    routine = routine.drop_duplicates("valid")                    # remove duplicated times
    routine = routine.set_index("valid")                          # use the time as the row label
    routine = routine.sort_index()                                # sort by time
    obs = routine.asfreq("h")                                     # one row per hour; missing hours become empty rows

    weather = pd.DataFrame(index=obs.index)                       # new table on the hourly timeline
    weather["temperature_c"] = (obs["tmpf"] - 32) * 5 / 9         # temperature: °F -> °C
    weather["dewpoint_c"] = (obs["dwpf"] - 32) * 5 / 9            # dew point: °F -> °C
    weather["humidity_pct"] = obs["relh"]                         # relative humidity (%)
    weather["wind_speed_kt"] = obs["sknt"]                        # wind speed (kt)
    weather["visibility_m"] = obs["vsby"] * 1609.34               # visibility: statute miles -> metres
    return weather                                                # give the table back


def build_features(weather, times):
    f = pd.DataFrame(index=weather.index)                         # new table with the same rows
    f["vis_m"] = weather["visibility_m"].clip(upper=9999)         # visibility now, capped at 9999 m like METAR
    f["spread_c"] = weather["temperature_c"] - weather["dewpoint_c"]   # temperature − dew point (0 = saturated air)
    f["rh"] = weather["humidity_pct"]                             # humidity (%)
    f["wind_kt"] = weather["wind_speed_kt"]                       # wind speed (kt)
    hours = pd.DatetimeIndex(times).hour                          # hour of the day, 0–23
    months = pd.DatetimeIndex(times).month                        # month, 1–12
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24)                # hour on a circle (sine)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24)                # hour on a circle (cosine)
    f["month_sin"] = np.sin(2 * np.pi * months / 12)              # month on a circle (sine)
    f["month_cos"] = np.cos(2 * np.pi * months / 12)              # month on a circle (cosine)
    return f[FEATURES]                                            # the 8 inputs, in the agreed order


def build_target(weather):
    low_now = (weather["visibility_m"] < LOW_VIS_M).astype(float) # 1 if visibility < 1500 m at that hour
    low_now[weather["visibility_m"].isna()] = np.nan              # keep "missing" where visibility is missing
    return low_now.shift(-HORIZON_H)                              # the value 3 hours LATER (future: target only!)


class FogNet(nn.Module):
    def __init__(self, n_inputs=8, hidden=16):
        super().__init__()                                        # standard first line of a PyTorch model
        self.layer1 = nn.Linear(n_inputs, hidden)                 # layer 1: 8 inputs -> 16 neurons
        self.layer2 = nn.Linear(hidden, hidden)                   # layer 2: 16 -> 16 neurons
        self.output = nn.Linear(hidden, 1)                        # output layer: 16 -> 1 score
        self.relu = nn.ReLU()                                     # ReLU: keeps positive values, sets negative ones to 0

    def forward(self, x):
        h1 = self.relu(self.layer1(x))                            # values of the 16 neurons of layer 1
        h2 = self.relu(self.layer2(h1))                           # values of the 16 neurons of layer 2
        score = self.output(h2)                                   # one raw score per example
        return score.squeeze(1)                                   # shape (batch,)


def save_model(model, card, folder):
    folder = Path(folder)                                         # make sure it is a Path
    folder.mkdir(parents=True, exist_ok=True)                     # create the folder if needed
    torch.save(model.state_dict(), folder / "fog_model.pt")       # save the learned weights
    (folder / "model_card.json").write_text(json.dumps(card, indent=2, default=str))   # save the model card


class SavedModel:
    """The trained model loaded from disk, with its scaling values and alert threshold."""

    def __init__(self, folder):
        folder = Path(folder)                                     # make sure it is a Path
        self.card = json.loads((folder / "model_card.json").read_text())   # read the model card
        self.model = FogNet(len(FEATURES), self.card["hidden"])   # same structure as in training
        self.model.load_state_dict(torch.load(folder / "fog_model.pt", map_location="cpu"))   # the learned weights
        self.model.eval()                                         # evaluation mode
        self.means = np.array(self.card["scaler_mean"], dtype="float32")    # training averages
        self.stds = np.array(self.card["scaler_std"], dtype="float32")      # training spreads
        self.threshold = float(self.card["threshold"])            # alert threshold

    def scale(self, features_table):
        X = features_table[FEATURES].values.astype("float32")     # the 8 inputs as numbers
        return (X - self.means) / self.stds                       # scaled with the TRAINING statistics

    def probability(self, features_table):
        X_scaled = self.scale(features_table)                     # scale the inputs
        with torch.no_grad():                                     # prediction only
            scores = self.model(torch.tensor(X_scaled))           # raw scores
        return torch.sigmoid(scores).numpy()                      # scores -> probabilities

"""
lvp_core.py — the code SHARED by the training notebook and by the prediction service (serve.py).

Why a shared file? If training and serving each had their own copy of the data preparation, a small
difference between the two copies would silently change the predictions ("training/serving skew").
With one shared file, both sides are guaranteed to prepare the data in exactly the same way.

TARGET: the model predicts P(visibility < 1500 m within the next 3 hours). This is a PROXY for
Low-Visibility Procedures (LVP): the data has no record of LVP being declared, and real LVP depend on
runway visual range (RVR), cloud ceiling and local procedures.
"""
import json                                   # json: reads and writes the model card (a text file with labelled values)
from pathlib import Path                      # Path: builds file paths that work on every operating system

import numpy as np                            # numpy: calculations on arrays of numbers
import pandas as pd                           # pandas: tables of data
import torch                                  # PyTorch: neural networks
import torch.nn as nn                         # building blocks of neural networks


# The inputs the service receives, in aviation units. This is the "contract" of the service.
RAW_COLUMNS = ["time", "temperature_c", "dewpoint_c", "wind_dir_deg", "wind_speed_kt", "visibility_m", "qnh_hpa"]

# The inputs of the model, in this exact order (the same as in Lab 2, with humidity computed from T and dew point)
FEATURES = ["temp_c", "spread_c", "rh", "wind_east", "wind_north", "qnh_hpa", "vis_m",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]

# Visibility threshold of our target (proxy for LVP), in metres
LOW_VIS_THRESHOLD_M = 1500


# -------------------------------------------------------------------------------------------------
# 1. Read the METAR archive and convert it to aviation units (used for the training data only)
# -------------------------------------------------------------------------------------------------
def load_iem_metar(path):
    raw = pd.read_csv(path, na_values="M", low_memory=False)      # read the file; "M" means missing
    raw["valid"] = pd.to_datetime(raw["valid"])                   # observation time: text -> date/time
    routine = raw[raw["valid"].dt.minute == 50]                   # keep only the routine reports (HH:50)
    routine = routine.drop_duplicates("valid")                    # remove duplicated times
    routine = routine.set_index("valid")                          # use the time as the row label
    routine = routine.sort_index()                                # sort by time
    obs = routine.asfreq("h")                                     # one row per hour; missing hours become empty rows

    out = pd.DataFrame(index=obs.index)                           # new table on the hourly timeline
    out["time"] = obs.index                                       # the observation time as a column
    out["temperature_c"] = (obs["tmpf"] - 32) * 5 / 9             # temperature: °F -> °C
    out["dewpoint_c"] = (obs["dwpf"] - 32) * 5 / 9                # dew point: °F -> °C
    out["wind_dir_deg"] = obs["drct"]                             # wind direction (empty = calm or variable)
    out["wind_speed_kt"] = obs["sknt"]                            # wind speed in knots
    out["visibility_m"] = obs["vsby"] * 1609.34                   # visibility: statute miles -> metres
    qnh = obs["alti"] * 33.8639                                   # pressure: inches of mercury -> hPa
    impossible = (qnh < 950) | (qnh > 1060)                       # a few values are encoding errors
    qnh[impossible] = np.nan                                      # replace them with "missing"
    out["qnh_hpa"] = qnh                                          # store the cleaned pressure
    return out                                                    # give the table back


# -------------------------------------------------------------------------------------------------
# 2. Operational inputs -> model inputs (used by BOTH training and serving)
# -------------------------------------------------------------------------------------------------
def relative_humidity(temp_c, dew_c):
    a = 17.625                                                    # constant of the Magnus formula
    b = 243.04                                                    # constant of the Magnus formula (°C)
    saturation_at_dew = np.exp(a * dew_c / (b + dew_c))           # vapour pressure at the dew point (relative)
    saturation_at_temp = np.exp(a * temp_c / (b + temp_c))        # vapour pressure at saturation (relative)
    return 100 * saturation_at_dew / saturation_at_temp           # relative humidity in %


def build_features(df):
    times = pd.to_datetime(df["time"])                            # the observation times
    f = pd.DataFrame(index=df.index)                              # new table with the same rows
    f["temp_c"] = df["temperature_c"]                             # temperature (°C)
    f["spread_c"] = df["temperature_c"] - df["dewpoint_c"]        # temperature minus dew point (°C)
    rh = relative_humidity(df["temperature_c"], df["dewpoint_c"]) # humidity computed from T and dew point
    f["rh"] = rh.clip(upper=100)                                  # humidity cannot exceed 100 %
    direction = df["wind_dir_deg"].astype(float).fillna(0)        # calm/variable wind -> direction 0 (speed ~0 anyway)
    radians = np.deg2rad(direction)                               # degrees -> radians
    f["wind_east"] = np.sin(radians) * df["wind_speed_kt"]        # east–west wind component
    f["wind_north"] = np.cos(radians) * df["wind_speed_kt"]       # north–south wind component
    f["qnh_hpa"] = df["qnh_hpa"]                                  # pressure (hPa)
    f["vis_m"] = df["visibility_m"].clip(upper=9999)              # visibility, capped at 9999 m like in METAR
    hours = times.dt.hour.values                                  # hour of the day, 0–23
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24)                # hour on a circle (sine)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24)                # hour on a circle (cosine)
    months = times.dt.month.values                                # month, 1–12
    f["month_sin"] = np.sin(2 * np.pi * months / 12)              # month on a circle (sine)
    f["month_cos"] = np.cos(2 * np.pi * months / 12)              # month on a circle (cosine)
    return f[FEATURES]                                            # return the columns in the agreed order


def build_target(df):
    low_vis = (df["visibility_m"] < LOW_VIS_THRESHOLD_M).astype(float)   # 1 if visibility < 1500 m now
    low_vis[df["visibility_m"].isna()] = np.nan                   # keep "missing" where visibility is missing
    in_1h = low_vis.shift(-1)                                     # the flag 1 hour later (FUTURE: target only!)
    in_2h = low_vis.shift(-2)                                     # the flag 2 hours later
    in_3h = low_vis.shift(-3)                                     # the flag 3 hours later
    next_three = pd.concat([in_1h, in_2h, in_3h], axis=1)         # the three future flags side by side
    return next_three.max(axis=1, skipna=False)                   # 1 if any of them is 1 (proxy for LVP)


# -------------------------------------------------------------------------------------------------
# 3. The neural network (the same as in Lab 2)
# -------------------------------------------------------------------------------------------------
class FogMLP(nn.Module):
    def __init__(self, n_inputs, hidden=32, dropout=0.0):
        super().__init__()                                        # standard first line of a PyTorch model
        self.layers = nn.Sequential(                              # a chain of layers:
            nn.Linear(n_inputs, hidden),                          #   inputs -> hidden neurons
            nn.ReLU(),                                            #   non-linearity
            nn.Dropout(dropout),                                  #   dropout (training only)
            nn.Linear(hidden, hidden),                            #   hidden -> hidden
            nn.ReLU(),                                            #   non-linearity
            nn.Dropout(dropout),                                  #   dropout (training only)
            nn.Linear(hidden, 1),                                 #   hidden -> one score
        )                                                         # end of the chain

    def forward(self, x):
        return self.layers(x).squeeze(1)                          # one raw score ("logit") per example


# -------------------------------------------------------------------------------------------------
# 4. Saving and loading: model weights + model card
# -------------------------------------------------------------------------------------------------
def save_artifacts(model, card, folder):
    folder = Path(folder)                                         # make sure `folder` is a Path
    folder.mkdir(parents=True, exist_ok=True)                     # create the folder if it does not exist
    torch.save(model.state_dict(), folder / "model_state.pt")     # save the learned weights
    card_text = json.dumps(card, indent=2, default=str)           # turn the model card into readable text
    (folder / "model_card.json").write_text(card_text)            # save the model card


class LoadedModel:
    """A trained model loaded from disk, together with everything needed to use it."""

    def __init__(self, folder):
        folder = Path(folder)                                     # make sure `folder` is a Path
        self.card = json.loads((folder / "model_card.json").read_text())   # read the model card
        settings = self.card["hyperparameters"]                   # the settings used in training
        self.model = FogMLP(len(self.card["features"]), settings["hidden"], settings["dropout"])  # same structure
        weights = torch.load(folder / "model_state.pt", map_location="cpu")  # read the learned weights
        self.model.load_state_dict(weights)                       # put the weights into the model
        self.model.eval()                                         # evaluation mode (dropout off)
        self.means = np.array(self.card["scaler_mean"], dtype="float32")     # scaling averages from training
        self.spreads = np.array(self.card["scaler_std"], dtype="float32")    # scaling spreads from training
        self.threshold = float(self.card["threshold"])            # the alert threshold chosen in training
        if self.card["features"] != FEATURES:                     # safety check: same inputs as this code?
            raise ValueError("The saved model expects different inputs than this code!")   # stop if not

    def predict_probability(self, raw_table):
        X = build_features(raw_table).values.astype("float32")    # operational inputs -> model inputs
        X_scaled = (X - self.means) / self.spreads                # scale with the TRAINING statistics
        with torch.no_grad():                                     # no learning, only prediction
            scores = self.model(torch.tensor(X_scaled))           # raw scores
            probabilities = torch.sigmoid(scores)                 # scores -> probabilities
        return probabilities.numpy()                              # give them back as a numpy array

"""
traffic_core.py — shared code of Lab 1 Parts 2 and 3 (and the save step at the end of Part 1).

The model forecasts tomorrow's number of IFR movements at one airport from the traffic of the last few days
and the day of the week of the day to forecast. The definitions here are the same as in the Part 1 notebook.
"""
import json                                   # json: reads and writes the model card (a text file with labelled values)
from pathlib import Path                      # Path: file paths that work on every operating system

import numpy as np                            # numpy: calculations on arrays of numbers
import pandas as pd                           # pandas: tables of data
import torch                                  # PyTorch: neural networks
import torch.nn as nn                         # building blocks of neural networks

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]   # order of the weekday switches


def load_series(data_dir, airport="EBBR"):
    tables = []                                                   # one table per year
    for year in [2023, 2024, 2025]:                               # the continuous period used in Lab 1
        tables.append(pd.read_csv(Path(data_dir) / ("airport_traffic_" + str(year) + ".csv")))   # read the file
    traffic = pd.concat(tables, ignore_index=True)                # stack the years
    traffic["FLT_DATE"] = pd.to_datetime(traffic["FLT_DATE"])     # text -> dates
    rows = traffic[traffic["APT_ICAO"] == airport]                # keep one airport
    rows = rows.sort_values("FLT_DATE")                           # sort by date
    return rows.set_index("FLT_DATE")["FLT_TOT_1"].astype("float32")   # daily total movements, labelled by date


def input_names(lookback):
    names = []                                                    # readable names of the inputs, in order
    for k in range(lookback, 0, -1):                              # from the oldest day to yesterday
        if k == 1:                                                # the most recent day...
            names.append("traffic yesterday")                     # ...is "yesterday"
        else:                                                     # the others...
            names.append("traffic " + str(k) + " days before")    # ..."k days before"
    for day in WEEKDAYS:                                          # then the 7 weekday switches
        names.append("is " + day)                                 # e.g. "is Saturday"
    return names                                                  # give the list back


def make_input(past_traffic, target_date):
    weekday_switches = np.zeros(7, dtype="float32")               # seven zeros...
    weekday_switches[pd.Timestamp(target_date).dayofweek] = 1.0   # ...with a 1 at the weekday of the day to forecast
    past = np.array(past_traffic, dtype="float32")                # the past traffic, oldest first
    return np.concatenate([past, weekday_switches])               # one input vector (not scaled yet)


def make_examples(series, lookback):
    values = series.values.astype("float32")                      # the traffic values
    dates = series.index                                          # their dates
    inputs = []                                                   # one input per example
    answers = []                                                  # the true traffic of the day to forecast
    answer_dates = []                                             # the date of the day to forecast
    for t in range(lookback, len(values)):                        # every day with enough history before it
        inputs.append(make_input(values[t - lookback:t], dates[t]))   # past days + weekday of day t
        answers.append(values[t])                                 # the traffic of day t
        answer_dates.append(dates[t])                             # the date of day t
    return np.array(inputs), np.array(answers), pd.DatetimeIndex(answer_dates)   # give everything back


class TrafficMLP(nn.Module):                                      # the same network as in Lab 1, Part 1
    def __init__(self, n_inputs, hidden=32):
        super().__init__()                                        # standard first line of a PyTorch model
        self.layers = nn.Sequential(                              # a chain of layers:
            nn.Linear(n_inputs, hidden),                          #   layer 1
            nn.ReLU(),                                            #   non-linearity
            nn.Linear(hidden, hidden),                            #   layer 2
            nn.ReLU(),                                            #   non-linearity
            nn.Linear(hidden, 1),                                 #   output: one number
        )                                                         # end of the chain

    def forward(self, x):
        return self.layers(x).squeeze(1)                          # one (scaled) forecast per example


def save_model(model, card, folder):
    folder = Path(folder)                                         # make sure it is a Path
    folder.mkdir(parents=True, exist_ok=True)                     # create the folder if needed
    torch.save(model.state_dict(), folder / "traffic_model.pt")   # the learned weights
    (folder / "model_card.json").write_text(json.dumps(card, indent=2, default=str))   # the model card


class SavedTrafficModel:
    """The trained traffic model loaded from disk, with its scaling values."""

    def __init__(self, folder):
        folder = Path(folder)                                     # make sure it is a Path
        self.card = json.loads((folder / "model_card.json").read_text())   # read the model card
        self.lookback = int(self.card["lookback"])                # how many past days the model needs
        self.model = TrafficMLP(self.lookback + 7, int(self.card["hidden"]))   # same structure as in training
        self.model.load_state_dict(torch.load(folder / "traffic_model.pt", map_location="cpu"))   # the weights
        self.model.eval()                                         # evaluation mode
        self.mean = float(self.card["mean"])                      # scaling average (training period)
        self.std = float(self.card["std"])                        # scaling spread (training period)

    def scale(self, X):
        X_scaled = np.array(X, dtype="float32").copy()            # copy the inputs
        X_scaled[:, :self.lookback] = (X_scaled[:, :self.lookback] - self.mean) / self.std   # scale the traffic part
        return X_scaled                                           # the weekday switches stay 0/1

    def forecast_many(self, X):
        with torch.no_grad():                                     # prediction only
            scaled = self.model(torch.tensor(self.scale(X))).numpy()   # scaled forecasts
        return scaled * self.std + self.mean                      # back to flights per day

    def forecast(self, past_traffic, target_date):
        X = make_input(past_traffic, target_date).reshape(1, -1)  # one example as a one-row table
        return float(self.forecast_many(X)[0])                    # the forecast in flights

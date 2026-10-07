"""
serve_simple.py — the simplified Lab 1 network as a small web service (EUROCONTROL AI Lab, training use only).
Start it from this folder with:  python -m uvicorn serve_simple:app --host 127.0.0.1 --port 8000
"""
import json                                                    # reads the saved settings
from datetime import date                                      # the type of the day to forecast

import torch                                                   # PyTorch
import torch.nn as nn                                          # building blocks of networks
from fastapi import FastAPI                                    # creates the web service
from pydantic import BaseModel, Field, conlist, confloat       # describe and check the request


class TrafficNet(nn.Module):                                   # the same network as in Part 1
    def __init__(self, hidden):
        super().__init__()                                     # standard first line of a PyTorch model
        self.layer1 = nn.Linear(14, hidden)                    # hidden layer: 14 inputs -> hidden neurons
        self.output = nn.Linear(hidden, 1)                     # output layer: hidden neurons -> 1 number
        self.relu = nn.ReLU()                                  # negative values -> 0

    def forward(self, x):
        h = self.relu(self.layer1(x))                          # activations of the hidden neurons
        return self.output(h).squeeze(1)                       # one forecast (scaled) per example


settings = json.load(open("artifacts/traffic_net.json"))       # HIDDEN, MEAN and STD saved in Part 1
model = TrafficNet(settings["HIDDEN"])                         # an empty network of the right size...
model.load_state_dict(torch.load("artifacts/traffic_net.pt"))  # ...filled with the learned weights
model.eval()                                                   # evaluation mode: no learning


class Request(BaseModel):                                      # what a client must send (checked automatically: 422 if wrong)
    day: date = Field(..., description="The day to forecast (YYYY-MM-DD)")
    last_7_days: conlist(confloat(ge=0, le=5000), min_length=7, max_length=7) = Field(
        ..., description="Flights of the 7 previous days, oldest first (each between 0 and 5000)")


app = FastAPI(title="Brussels traffic forecast (simplified Lab 1)")   # the service


@app.get("/health")                                            # GET /health -> is the service alive?
def health():
    return {"status": "ok", "hidden_neurons": settings["HIDDEN"]}


@app.post("/forecast")                                         # POST /forecast -> 7 days + day in, forecast out
def forecast(request: Request):
    inputs = []                                                # the 14 inputs, as in Part 1
    for value in request.last_7_days:                          # the 7 traffic numbers...
        inputs.append((value - settings["MEAN"]) / settings["STD"])   # ...scaled
    for k in range(7):                                         # the 7 weekday switches...
        if k == request.day.weekday():                         # ...the weekday of the day to forecast is on
            inputs.append(1.0)
        else:                                                  # ...the others are off
            inputs.append(0.0)
    with torch.no_grad():                                      # only forecasting
        scaled = float(model(torch.tensor([inputs]))[0])       # the network's answer (scaled)
    flights = scaled * settings["STD"] + settings["MEAN"]      # back to flights
    return {"day": request.day, "forecast": round(flights, 1)}

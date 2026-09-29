"""
serve_simple.py — a small prediction service for the SIMPLIFIED low-visibility model, running on your own computer.

It answers: probability that visibility at Malpensa is below 1500 m EXACTLY 3 hours after the given observation.
This is a PROXY for Low-Visibility Procedures (LVP), not a forecast of LVP being declared.

Start it from this folder with:
    python -m uvicorn serve_simple:app --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000/docs (automatic documentation) or http://127.0.0.1:8000/ (a simple web form).
"""
import sys                                                      # sys: access to Python's list of code folders
from datetime import datetime                                   # datetime: the type of the observation time
from pathlib import Path                                        # Path: file paths that work on every system

import pandas as pd                                             # pandas: tables of data
from fastapi import FastAPI                                     # FastAPI: the library that creates the web service
from fastapi.responses import FileResponse                      # FileResponse: sends the web form to the browser
from pydantic import BaseModel, Field                           # pydantic: describes and checks the input data

HERE = Path(__file__).resolve().parent                          # the folder that contains this file
sys.path.insert(0, str(HERE))                                   # let Python find lowvis_core.py
import lowvis_core                                              # noqa: E402  (the shared code of this lab)

MODEL = lowvis_core.SavedModel(HERE / "artifacts")              # load the model saved in Part 1, once, at start-up

app = FastAPI(                                                  # create the web service
    title="Low-visibility nowcast — LIMC (simplified)",         # its name, shown on the /docs page
    description="Probability of visibility < 1500 m exactly 3 hours after the observation. "
                "A PROXY for Low-Visibility Procedures. EUROCONTROL AI Lab — training use only.",   # its description
    version=MODEL.card["version"],                              # its version = the model version
)


class Observation(BaseModel):                                   # what a client must send (checked automatically)
    time: datetime = Field(..., description="Observation time (UTC)")                          # date and time
    visibility_m: float = Field(..., ge=0, le=9999, description="Visibility now (m); 9999 = 10 km or more")   # 0–9999 m
    temperature_c: float = Field(..., ge=-30, le=50, description="Air temperature (°C)")       # -30 to 50 °C
    dewpoint_c: float = Field(..., ge=-40, le=35, description="Dew point (°C)")                # -40 to 35 °C
    humidity_pct: float = Field(..., ge=0, le=100, description="Relative humidity (%)")        # 0 to 100 %
    wind_speed_kt: float = Field(..., ge=0, le=80, description="Wind speed (kt)")              # 0 to 80 kt


class Prediction(BaseModel):                                    # what the service sends back
    probability: float                                          # probability of visibility < 1500 m in 3 hours
    alert: bool                                                 # True if the probability is above the threshold
    threshold: float                                            # the threshold used
    model_version: str                                          # which model answered


@app.get("/health")                                             # GET /health -> is the service alive?
def health():
    return {"status": "ok", "model_version": MODEL.card["version"]}   # a short "I am alive" message


@app.get("/model")                                              # GET /model -> the model card
def model_card():
    return MODEL.card                                           # send the model card


@app.post("/predict", response_model=Prediction)                # POST /predict -> one observation in, one answer out
def predict(observation: Observation):
    table = pd.DataFrame([observation.model_dump()])            # the observation as a one-row table
    features = lowvis_core.build_features(table, table["time"]) # the 8 model inputs, computed with the SHARED code
    probability = float(MODEL.probability(features)[0])         # ask the model
    return Prediction(probability=round(probability, 5),        # the probability, rounded
                      alert=probability >= MODEL.threshold,     # alert yes/no
                      threshold=MODEL.threshold,                # the threshold
                      model_version=MODEL.card["version"])      # the model version


@app.get("/", include_in_schema=False)                          # GET / -> the web form
def web_form():
    return FileResponse(HERE / "static" / "index.html")         # send the HTML page to the browser

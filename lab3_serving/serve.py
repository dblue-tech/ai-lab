"""
serve.py — a small prediction service for the low-visibility model, running on your own computer.

The model predicts P(visibility < 1500 m within 3 h), a PROXY for Low-Visibility Procedures (LVP):
real LVP depend on RVR, cloud ceiling and local procedures, which are not in the training data.

Start it from the lab3_serving folder with:
    python -m uvicorn serve:app --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000/docs (automatic documentation) or http://127.0.0.1:8000/ (a simple web form).
"""
import sys                                                      # sys: access to Python's list of code folders
from datetime import datetime                                   # datetime: the type used for the observation time
from pathlib import Path                                        # Path: file paths that work on every system
from typing import List, Optional                               # List = "a list of ..."; Optional = "may be empty"

import pandas as pd                                             # pandas: tables of data
from fastapi import FastAPI                                     # FastAPI: the library that creates the web service
from fastapi.responses import FileResponse                      # FileResponse: sends a file (our web form) to the browser
from pydantic import BaseModel, Field, model_validator          # pydantic: describes and checks the input data

LAB_DIR = Path(__file__).resolve().parent                       # the folder that contains this file
sys.path.insert(0, str(LAB_DIR))                                # let Python find lvp_core.py in that folder
from lvp_core import LoadedModel                                # noqa: E402  (the shared code of Lab 3)

# ------------------------------------------------------------------------------------------
# 1. Load the model ONCE, when the service starts (not at every request)
# ------------------------------------------------------------------------------------------
MODEL = LoadedModel(LAB_DIR / "artifacts")                      # read the weights and the model card from disk

app = FastAPI(                                                  # create the web service
    title="Low-visibility nowcasting service — LIMC",           # its name (shown on the /docs page)
    description="Probability of visibility < 1500 m within 3 hours, a PROXY for Low-Visibility Procedures "
                "(not a forecast of LVP being declared). EUROCONTROL AI Lab — training use only.",   # its description
    version=MODEL.card["version"],                              # its version = the model version
)


# ------------------------------------------------------------------------------------------
# 2. The input "contract": what a client must send. FastAPI checks every request automatically.
#    The limits come from the training data (see Lab 3 Part 1, section 1).
# ------------------------------------------------------------------------------------------
class Observation(BaseModel):
    time: datetime = Field(..., description="Observation time (UTC)")                      # required date/time
    temperature_c: float = Field(..., ge=-30, le=50, description="Air temperature (°C)")   # required, between -30 and 50
    dewpoint_c: float = Field(..., ge=-40, le=35, description="Dew point (°C)")            # required, between -40 and 35
    wind_dir_deg: Optional[float] = Field(None, ge=0, le=360, description="Wind direction (degrees); empty = calm/variable")  # optional
    wind_speed_kt: float = Field(..., ge=0, le=80, description="Wind speed (kt)")          # required, 0 to 80 kt
    visibility_m: float = Field(..., ge=0, le=9999, description="Visibility (m); 9999 = 10 km or more")   # required
    qnh_hpa: float = Field(..., ge=950, le=1060, description="QNH (hPa)")                  # required, 950 to 1060 hPa

    @model_validator(mode="after")                              # an extra check, run after the checks above
    def dewpoint_not_above_temperature(self):
        if self.dewpoint_c > self.temperature_c + 0.5:          # the dew point cannot be above the temperature
            raise ValueError("dew point cannot be higher than temperature")   # reject the request with this message
        return self                                             # otherwise accept it


class Prediction(BaseModel):                                    # what the service sends back
    probability: float                                          # probability of visibility < 1500 m within 3 h
    alert: bool                                                 # True if the probability is above the threshold
    threshold: float                                            # the threshold used
    model_version: str                                          # which model produced the answer


def make_predictions(observations):
    rows = []                                                   # one table row per observation
    for observation in observations:                            # for each observation received...
        rows.append(observation.model_dump())                   # ...turn it into a plain dictionary
    table = pd.DataFrame(rows)                                  # build a table from the rows
    probabilities = MODEL.predict_probability(table)            # ask the model (shared code in lvp_core.py)

    answers = []                                                # one answer per observation
    for p in probabilities:                                     # for each probability...
        answer = Prediction(                                    # ...build an answer:
            probability=round(float(p), 5),                     #   the probability, rounded to 5 decimals
            alert=bool(p >= MODEL.threshold),                   #   alert yes/no
            threshold=MODEL.threshold,                          #   the threshold used
            model_version=MODEL.card["version"],                #   the model version
        )                                                       # (end of the answer)
        answers.append(answer)                                  # add it to the list
    return answers                                              # give the answers back


# ------------------------------------------------------------------------------------------
# 3. The endpoints: the "addresses" the service answers to
# ------------------------------------------------------------------------------------------
@app.get("/health")                                             # GET /health -> is the service alive?
def health():
    return {"status": "ok", "model_version": MODEL.card["version"]}   # a short "I am alive" message


@app.get("/model")                                              # GET /model -> the model card
def model_card():
    return MODEL.card                                           # send the full model card


@app.post("/predict", response_model=Prediction)                # POST /predict -> one observation in, one answer out
def predict(observation: Observation):
    answers = make_predictions([observation])                   # make a prediction for a list of one observation
    return answers[0]                                           # send back the first (and only) answer


@app.post("/predict_batch", response_model=List[Prediction])    # POST /predict_batch -> many observations at once
def predict_batch(observations: List[Observation]):
    return make_predictions(observations)                       # send back one answer per observation


@app.get("/", include_in_schema=False)                          # GET / -> the web form
def web_form():
    return FileResponse(LAB_DIR / "static" / "index.html")      # send the HTML page to the browser

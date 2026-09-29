"""
serve_traffic.py — a small prediction service for the Lab 1 traffic model, running on your own computer.

It answers: how many IFR movements will the airport have on a given day, knowing the traffic of the days before?

Start it from this folder with:
    python -m uvicorn serve_traffic:app --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000/docs (automatic documentation) or http://127.0.0.1:8000/ (a simple web form).
"""
import sys                                                      # sys: access to Python's list of code folders
from datetime import date                                       # date: the type of the day to forecast
from pathlib import Path                                        # Path: file paths that work on every system
from typing import List, Optional                               # List = "a list of ..."; Optional = "may be empty"

from fastapi import FastAPI                                     # FastAPI: the library that creates the web service
from fastapi.responses import FileResponse                      # FileResponse: sends the web form to the browser
from pydantic import BaseModel, Field, field_validator          # pydantic: describes and checks the input data

HERE = Path(__file__).resolve().parent                          # the folder that contains this file
sys.path.insert(0, str(HERE))                                   # let Python find traffic_core.py
import traffic_core                                             # noqa: E402  (the shared code of Lab 1)

MODEL = traffic_core.SavedTrafficModel(HERE / "artifacts")      # load the model saved in Part 1, once, at start-up
LOOKBACK = MODEL.lookback                                       # how many past days the model needs

app = FastAPI(                                                  # create the web service
    title="Next-day traffic forecast — " + MODEL.card["airport"],   # its name, shown on the /docs page
    description="Forecast of the total IFR movements of one day from the traffic of the previous "
                + str(LOOKBACK) + " days. EUROCONTROL AI Lab — training use only.",   # its description
    version=MODEL.card["version"],                              # its version = the model version
)


class ForecastRequest(BaseModel):                               # what a client must send (checked automatically)
    day: date = Field(..., description="The day to forecast (YYYY-MM-DD)")                    # the target day
    past_traffic: List[float] = Field(..., description="Total IFR movements of the previous days, OLDEST first")   # list

    @field_validator("past_traffic")                            # an extra check on the list of past days
    def check_past_traffic(cls, values):
        if len(values) != LOOKBACK:                             # the model needs exactly LOOKBACK values...
            raise ValueError("exactly " + str(LOOKBACK) + " past days are needed, oldest first")   # ...otherwise refuse
        for v in values:                                        # every value...
            if v < 0 or v > 5000:                               # ...must be a plausible number of daily movements
                raise ValueError("each daily traffic value must be between 0 and 5000 flights")   # otherwise refuse
        return values                                           # accept the list


class ForecastAnswer(BaseModel):                                # what the service sends back
    day: date                                                   # the day forecast
    forecast_flights: float                                     # the model's forecast
    baseline_same_weekday_last_week: Optional[float]            # the simple baseline (empty if fewer than 7 past days)
    model_version: str                                          # which model answered


@app.get("/health")                                             # GET /health -> is the service alive?
def health():
    return {"status": "ok", "model_version": MODEL.card["version"], "lookback": LOOKBACK}   # "I am alive"


@app.get("/model")                                              # GET /model -> the model card
def model_card():
    return MODEL.card                                           # send the model card


@app.post("/forecast", response_model=ForecastAnswer)           # POST /forecast -> past days in, forecast out
def forecast(request: ForecastRequest):
    flights = MODEL.forecast(request.past_traffic, request.day) # the model's forecast (shared code)
    baseline = None                                             # no baseline if the model uses fewer than 7 days...
    if LOOKBACK >= 7:                                           # ...otherwise:
        baseline = request.past_traffic[LOOKBACK - 7]           # the traffic 7 days before the day to forecast
    return ForecastAnswer(day=request.day,                      # the day
                          forecast_flights=round(flights, 1),   # the forecast, rounded
                          baseline_same_weekday_last_week=baseline,   # the baseline
                          model_version=MODEL.card["version"])  # the model version


@app.get("/", include_in_schema=False)                          # GET / -> the web form
def web_form():
    return FileResponse(HERE / "static" / "index.html")         # send the HTML page to the browser

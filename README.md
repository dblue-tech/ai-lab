# EUROCONTROL AI Lab: course package

**Version of 28 September 2026.** If your folder contains other files than those listed under *Contents*, they are from an older version and can be deleted.

Hands-on PyTorch exercises on real aviation data. Everything runs locally: no cloud services, accounts or internet access during the session.

## Contents
```
README.md                         this file
requirements.txt                  Python packages (pinned versions)
check_setup.py                    readiness check for each workstation
data/                             datasets (airport_traffic_2019/2023/2024/2025.csv, LIMC.csv)
docs/                             AI_Lab_Exercises_Description.docx, AI_Lab_IT_Setup_Guide.docx, AI_Lab_Data_Description.docx
lab1_airport_traffic/
    Lab1_airport_traffic.ipynb                   Exercise 1, Part 1: train (run first: section 8.1 saves the model)
    Lab1_part2_inside_the_model.ipynb            Exercise 1, Part 2: open the black box + explainability
    Lab1_part3_serve.ipynb                       Exercise 1, Part 3: the model as a local service
    solutions/Lab1_airport_traffic_SOLUTIONS.ipynb, solutions/Lab1_part2_inside_the_model_SOLUTIONS.ipynb
    traffic_core.py, serve_traffic.py, static/index.html   code used by Parts 2 and 3
lab2_low_visibility/
    Lab2_low_visibility.ipynb                    Exercise 2, participant notebook
    solutions/Lab2_low_visibility_SOLUTIONS.ipynb    same notebook with the optional coding challenges solved
lab3_serving/
    Lab3_part1_train_and_save.ipynb              Exercise 3, part 1
    Lab3_part2_run_the_service.ipynb             Exercise 3, part 2
    lvp_core.py, serve.py, static/index.html     code used by Exercise 3
lab2s_low_visibility_simple/                     SIMPLIFIED TRACK (alternative to Exercises 2 + 3)
    2S-1_train.ipynb                             train the simplified model (run first: it saves the model)
    2S-2_inside_the_model.ipynb                  open the black box + explainability
    2S-3_serve.ipynb                             run the model as a local service
    solutions/2S-1_train_SOLUTIONS.ipynb, solutions/2S-2_inside_the_model_SOLUTIONS.ipynb
    lowvis_core.py, serve_simple.py, static/index.html   code used by the three notebooks
```

| Exercise | Topic | Time |
|---|---|---|
| **1** | Forecasting European airport traffic: time-series regression on EUROCONTROL AIU data (Brussels, EBBR) with hyperparameter experiments · Part 2 inside the model · Part 3 serving | ~2 h + 50 min + 45 min |
| **2** | Nowcasting low visibility at Milano Malpensa from METAR: rare-event classification. The target (visibility < 1 500 m within 3 h) is a *proxy* for LVP, since the data holds no LVP declarations | ~2 h |
| **3** | End-to-end: train, test and save the model with a model card, then run it as a local FastAPI service and test it | ~1 h 40 |
| **2S** | *Simplified alternative to 2 + 3*, same METAR data: **2S-1** train (visibility < 1 500 m in 3 h, 8 inputs, precision/recall, class weight) · **2S-2** inside the model (network drawing, weights, one prediction by hand, explainability task) · **2S-3** local service | ~1 h 15 + 50 min + 45 min |

## How the notebooks work
- Each exercise is a complete, ready-to-run notebook. **No coding is required.** Participants run the cells from top to bottom (Shift + Enter). Every line of code has a plain-language comment, and every plot is followed by a short reading.
- The notebooks are saved with their outputs, so the results can be read even before running them. Participants keep the notebooks.
- **Optional coding challenges (Exercise 1 Parts 1–2, Exercise 2, 2S-1 and 2S-2):** a few cells per notebook marked **🧑‍💻**, for participants who want to write code. They are switched off (every line starts with `#`), contain `___` blanks and hints, and the rest of the notebook does not depend on them. Solutions are in each lab's `solutions/` folder.

## Quick start
Full instructions for Windows, macOS, Linux and conda are in `docs/AI_Lab_IT_Setup_Guide.docx`.

**venv on macOS or Linux** (on Windows, activate with `.venv\Scripts\activate`):
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # macOS: pip install torch==2.14.0
pip install -r requirements.txt
```

**conda on any OS** (Miniforge recommended):
```bash
conda create -n ai-lab -c conda-forge python=3.11 pip -y && conda activate ai-lab
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # macOS: pip install torch==2.14.0
pip install -r requirements.txt
```

**Then, in either case:**
```bash
python check_setup.py        # must print READY
jupyter lab
```

In Exercise 1, run **Part 1** first (section 8.1 saves the model in `lab1_airport_traffic/artifacts/`), then Parts 2 and 3.
In Exercise 3, run **Part 1** first: it creates `lab3_serving/artifacts/` with the saved model. Part 2 then starts the service by itself.
In the simplified track, run **2S-1** first: it creates `lab2s_low_visibility_simple/artifacts/`, used by 2S-2 and 2S-3.

## Data
- `airport_traffic_{2019,2023,2024,2025}.csv` comes from the EUROCONTROL Aviation Intelligence Unit (https://ansperformance.eu/csv/): daily IFR movements per airport.
- `LIMC.csv` comes from the Iowa Environmental Mesonet ASOS/METAR archive (https://mesonet.agron.iastate.edu/request/download.phtml): METAR reports for Milano Malpensa, 2020–2025.

All models are training artefacts, not for operational use.

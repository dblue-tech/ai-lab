# EUROCONTROL AI Lab: course package

**Version of 8 October 2026.** If your folder contains other files than those listed under *Contents*, they are from an older version and can be deleted.

Hands-on PyTorch exercises on real aviation data. The same notebooks run in two ways:
- **Google Colab** (nothing to install): participants only need a web browser, internet access and a Google account. See *Running in Google Colab* below.
- **Locally on the lab PCs** (no cloud, no accounts, no internet during the session): see *Quick start* below.

## Contents
```
README.md                         this file
requirements.txt                  Python packages (pinned versions)
check_setup.py                    readiness check for each workstation (local installation only)
data/                             datasets (airport_traffic_2019/2023/2024/2025.csv, LIMC.csv)
docs/                             AI_Lab_Exercises_Description.docx, AI_Lab_IT_Setup_Guide.docx, AI_Lab_Data_Description.docx,
                                  AI_Lab_Colab_Instructions.docx (participant handout for Google Colab, with all links)
lab1_airport_traffic/
    Lab1_airport_traffic.ipynb                   Exercise 1, Part 1: train (run first: section 8.1 saves the model)
    Lab1_part2_inside_the_model.ipynb            Exercise 1, Part 2: open the black box + explainability
    Lab1_part3_serve.ipynb                       Exercise 1, Part 3: the model as a local service
    solutions/Lab1_airport_traffic_SOLUTIONS.ipynb, solutions/Lab1_part2_inside_the_model_SOLUTIONS.ipynb
    traffic_core.py, serve_traffic.py, static/index.html   code used by Parts 2 and 3
    artifacts/                                   the trained model (re-created by Part 1)
lab2_low_visibility/
    Lab2_low_visibility.ipynb                    Exercise 2, participant notebook
    solutions/Lab2_low_visibility_SOLUTIONS.ipynb    same notebook with the optional coding challenges solved
lab3_serving/
    Lab3_part1_train_and_save.ipynb              Exercise 3, part 1
    Lab3_part2_run_the_service.ipynb             Exercise 3, part 2
    lvp_core.py, serve.py, static/index.html     code used by Exercise 3
    artifacts/                                   the trained model (re-created by Part 1)
lab1_simplified/                                 SIMPLIFIED EXERCISE 1 (alternative to Exercise 1, Parts 1–3)
    1S-1_train.ipynb                             train; one settings cell to change and re-run; table of your attempts
    1S-2_inside_the_model.ipynb                  the weights, one forecast by hand, two what-if questions
    1S-3_serve.ipynb                             the network as a local service
    serve_simple.py                              the whole service in one short file
    artifacts/                                   the trained model (re-created by 1S-1)
lab2s_low_visibility_simple/                     SIMPLIFIED TRACK (alternative to Exercises 2 + 3)
    2S-1_train.ipynb                             train the simplified model (run first: it saves the model)
    2S-2_inside_the_model.ipynb                  open the black box + explainability
    2S-3_serve.ipynb                             run the model as a local service
    solutions/2S-1_train_SOLUTIONS.ipynb, solutions/2S-2_inside_the_model_SOLUTIONS.ipynb
    lowvis_core.py, serve_simple.py, static/index.html   code used by the three notebooks
    artifacts/                                   the trained model (re-created by 2S-1)
```

| Exercise | Topic | Time |
|---|---|---|
| **1** | Forecasting European airport traffic: time-series regression on EUROCONTROL AIU data (Brussels, EBBR) with hyperparameter experiments · Part 2 inside the model (incl. neuron activations) · Part 3 serving | ~2 h + 1 h + 45 min |
| **2** | Nowcasting low visibility at Milano Malpensa from METAR: rare-event classification. The target (visibility < 1 500 m within 3 h) is a *proxy* for LVP, since the data holds no LVP declarations | ~2 h |
| **3** | End-to-end: train, test and save the model with a model card, then run it as a local FastAPI service and test it | ~1 h 40 |
| **1S** | *Simplified alternative to Exercise 1*: last 7 days + weekday → tomorrow's traffic, one hidden layer; everything defined inside the notebooks. **1S-1** train and experiment with 4 settings (training / validation / test errors, table of attempts) · **1S-2** inside the network and what-if · **1S-3** compact local service | ~45 min + 30 min + 30 min |
| **2S** | *Simplified alternative to 2 + 3*, same METAR data: **2S-1** train (visibility < 1 500 m in 3 h, 8 inputs, precision/recall, class weight) · **2S-2** inside the model (network drawing, weights, one prediction by hand, neuron activations, explainability task) · **2S-3** local service | ~1 h 15 + 1 h + 45 min |

## How the notebooks work
- Each exercise is a complete, ready-to-run notebook. **No coding is required.** Participants run the cells from top to bottom (Shift + Enter). Every line of code has a plain-language comment, and every plot is followed by a short reading.
- The notebooks are saved with their outputs, so the results can be read even before running them. Participants keep the notebooks.
- **🧪 Sandbox cells (Exercise 1 Part 1 and 2S-1):** participants can try any hyperparameter values, even strange ones, and see training / validation / test results and their own table of attempts. These runs are kept apart: nothing else in the notebook uses them, so the rest of the notebook (final test, saved model) never changes.
- **Optional coding challenges (Exercise 1 Parts 1–2, Exercise 2, 2S-1 and 2S-2):** a few cells per notebook marked **🧑‍💻**, for participants who want to write code. They are switched off (every line starts with `#`), contain `___` blanks and hints, and the rest of the notebook does not depend on them. Solutions are in each lab's `solutions/` folder.

## Running in Google Colab
Requirements on the classroom PCs: a web browser and internet access to `colab.research.google.com`, `github.com` and `googleusercontent.com`. Each participant signs in with a Google account (provided by the instructor).

1. Open a notebook with its link below (or the **Open in Colab** badge at the top of each notebook). Every link has the form `https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/<path of the notebook>`. A printable handout with all the links is in `docs/AI_Lab_Colab_Instructions.docx`.
2. Run the first code cell: it copies this repository (data, code, trained models) into the Colab machine and moves into the notebook's folder. On a lab PC the same cell does nothing.
3. Run the rest of the notebook from top to bottom, as on a PC.
4. To keep the notebook with your results: *File → Save a copy in Drive*. Without this, changes are lost when the Colab session ends.

Each notebook runs on its own Colab machine, so Parts 2 and 3 cannot see a model trained in Part 1: they use the ready-trained copy in `artifacts/` (the same model Part 1 produces). In the serving notebooks, the service runs on the Colab machine and the notebook shows a special link to open its documentation and web form.

| Notebook | Open | Full link |
|---|---|---|
| 1 · Part 1 — train | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/Lab1_airport_traffic.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/Lab1_airport_traffic.ipynb |
| 1 · Part 2 — inside the model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/Lab1_part2_inside_the_model.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/Lab1_part2_inside_the_model.ipynb |
| 1 · Part 3 — serve | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/Lab1_part3_serve.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/Lab1_part3_serve.ipynb |
| 1S-1 — train (simplified) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_simplified/1S-1_train.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_simplified/1S-1_train.ipynb |
| 1S-2 — inside the network (simplified) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_simplified/1S-2_inside_the_model.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_simplified/1S-2_inside_the_model.ipynb |
| 1S-3 — serve (simplified) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_simplified/1S-3_serve.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_simplified/1S-3_serve.ipynb |
| 2 — low visibility | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2_low_visibility/Lab2_low_visibility.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2_low_visibility/Lab2_low_visibility.ipynb |
| 3 · Part 1 — train and save | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab3_serving/Lab3_part1_train_and_save.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab3_serving/Lab3_part1_train_and_save.ipynb |
| 3 · Part 2 — run the service | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab3_serving/Lab3_part2_run_the_service.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab3_serving/Lab3_part2_run_the_service.ipynb |
| 2S-1 — train | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/2S-1_train.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/2S-1_train.ipynb |
| 2S-2 — inside the model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/2S-2_inside_the_model.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/2S-2_inside_the_model.ipynb |
| 2S-3 — serve | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/2S-3_serve.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/2S-3_serve.ipynb |
| Solutions: 1 · Part 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/solutions/Lab1_airport_traffic_SOLUTIONS.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/solutions/Lab1_airport_traffic_SOLUTIONS.ipynb |
| Solutions: 1 · Part 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/solutions/Lab1_part2_inside_the_model_SOLUTIONS.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab1_airport_traffic/solutions/Lab1_part2_inside_the_model_SOLUTIONS.ipynb |
| Solutions: 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2_low_visibility/solutions/Lab2_low_visibility_SOLUTIONS.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2_low_visibility/solutions/Lab2_low_visibility_SOLUTIONS.ipynb |
| Solutions: 2S-1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/solutions/2S-1_train_SOLUTIONS.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/solutions/2S-1_train_SOLUTIONS.ipynb |
| Solutions: 2S-2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/solutions/2S-2_inside_the_model_SOLUTIONS.ipynb) | https://colab.research.google.com/github/dblue-tech/ai-lab/blob/main/lab2s_low_visibility_simple/solutions/2S-2_inside_the_model_SOLUTIONS.ipynb |

## Quick start (local installation)
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

In Exercise 1, run **Part 1** first (section 8.1 saves the model in `lab1_airport_traffic/artifacts/`), then Parts 2 and 3. A ready-trained copy of each model is included in `artifacts/`, so Parts 2 and 3 also work on their own.
In Exercise 3, run **Part 1** first: it creates `lab3_serving/artifacts/` with the saved model. Part 2 then starts the service by itself.
In the simplified Exercise 1, run **1S-1** first (its §7 saves the model in `lab1_simplified/artifacts/`); a ready-trained copy is included. 1S-1 keeps your attempts in `lab1_simplified/my_attempts.csv`.
In the simplified track, run **2S-1** first: it creates `lab2s_low_visibility_simple/artifacts/`, used by 2S-2 and 2S-3.

## Data
- `airport_traffic_{2019,2023,2024,2025}.csv` comes from the EUROCONTROL Aviation Intelligence Unit (https://ansperformance.eu/csv/): daily IFR movements per airport.
- `LIMC.csv` comes from the Iowa Environmental Mesonet ASOS/METAR archive (https://mesonet.agron.iastate.edu/request/download.phtml): METAR reports for Milano Malpensa, 2020–2025.

All models are training artefacts, not for operational use.

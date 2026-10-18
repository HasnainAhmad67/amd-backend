# ml — the machine-learning code

This folder contains the Python package `telltale_ml`. It **trains** the models and contains the code the server uses to **run** them.

> New to the project? Read [`START_HERE.md`](../START_HERE.md) first.

## The three models

| Model | What it learns | Data | Train it with |
|---|---|---|---|
| **Sensor risk** | Predicts the chance a machine fails, and which way it fails, from temperature, speed, torque and tool wear | AI4I 2020 (0.5 MB, downloaded automatically) | Happens automatically on first start. Retrain: `.\telltale train-sensor` |
| **Sound anomaly** | Learns what a *normal* machine sounds like (CNN + classical spectra, nearest-neighbour scoring) and says how unusual a clip is as a calibrated p-value | MIMII DUE (~1.1 GB per machine type) | `.\telltale train-sound fan` |
| **Bearing faults** | Recognises 6 bearing conditions from sound (healthy, inner race, outer race, ball, looseness, combined), and says "not sure: A or B" when it can't tell | Mendeley bearing sounds (75 MB, downloaded automatically) | `.\telltale train-bearing` |

On Mac/Linux, replace `.\telltale` with `bash telltale.sh`. Trained models are saved into the `models/` folder at the project root.

## Where things are

```
configs/                 settings for training (YAML)  <- tweak numbers here first
├── sensor.yaml
├── acoustic.yaml
└── bearing.yaml
src/telltale_ml/
├── sensor/
│   ├── schema.py            names of the sensor inputs
│   ├── features.py          turns raw readings into model inputs (used by training AND the server)
│   ├── train.py             trains the sensor models
│   ├── predictor.py         runs them (used by the server)
│   └── synthetic.py         makes fake-but-realistic data
├── acoustic/
│   ├── features.py          turns audio into spectrograms (used by training AND the server)
│   ├── model.py             the neural network (PyTorch)
│   ├── train_anomaly.py     trains the sound model and exports it to ONNX
│   ├── detector.py          runs it (used by the server)
│   ├── fault.py             bearing features + classifier
│   └── train_fault.py       trains the bearing classifier
├── datasets/download.py     downloads the datasets
└── metrics.py               how we measure model quality
tests/                   automatic checks (use fake data, no downloads)
```

How we improved them, with every experiment and result (including what didn't work): [`docs/model-improvements.md`](../docs/model-improvements.md).

## Golden rules

1. **Features are computed in one place only** (`features.py`). The server imports the same code, so what the model sees in training is exactly what it sees live.
2. **Never pick thresholds using the test data.** Use the validation data, or the scores will look better than they really are.
3. **Be honest about results.** AI4I is simulated data, so near-perfect scores are expected and don't prove real-world accuracy.

## Training on an AMD GPU

On a Linux machine with an AMD GPU and ROCm drivers (for example the AMD Developer Cloud), run `make rocm-shell` from the project folder, then any `telltale-train-*` command inside it. The training code picks up the GPU automatically. See the root `README.md` → *Train on an AMD GPU*.

## Advanced: running without Docker

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/). From the project folder:

```
uv sync --extra torch
uv run telltale-download list
uv run telltale-train-sensor
uv run pytest ml/tests -q
```

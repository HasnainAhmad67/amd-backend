# apps/api — the server

The server ("backend" or "API") is written in **Python** with **FastAPI**. The website asks it for data. It:

- runs the pretend factory machines and scores them with the sensor model,
- analyses uploaded sound recordings,
- raises alerts and suggests what to inspect.

> New to the project? Read [`START_HERE.md`](../../START_HERE.md) first.

## Try it in your browser

While the app runs, open the **API docs** link that `.\telltale start` prints (e.g. <http://localhost:8000/docs>). Every server URL is listed there, and you can click **Try it out** to call it.

## See your changes live

Run `.\telltale dev` (Mac/Linux: `bash telltale.sh dev`) from the project folder. Save a `.py` file and the server restarts by itself within a couple of seconds. If you made a typo, the error appears in `.\telltale logs`.

## Where things are

```
telltale_api/
├── main.py              starts everything
├── api/routes/          one file per group of URLs
│   ├── machines.py          /api/v1/machines ...
│   ├── sensors.py           /api/v1/sensors/predict, /what-if
│   ├── acoustic.py          /api/v1/acoustic/analyze (sound upload)
│   ├── alerts.py            /api/v1/alerts
│   ├── models.py            /api/v1/models
│   └── health.py            /api/v1/health
├── services/            the actual logic
│   ├── fleet.py             the pretend machines and their faults
│   ├── playbook.py          turns model results into inspection advice
│   ├── alerts.py            the alert list
│   ├── registry.py          loads the trained models
│   ├── bootstrap.py         trains the sensor model on first start
│   └── narrator.py          plain-language summaries (optional LLM)
├── schemas/             the exact shape of every request and response
├── core/config.py       settings (can be changed in the .env file)
└── data/
    ├── playbook.yaml        inspection advice  <- easy to edit, no coding
    └── fleet.yaml           the list of demo machines  <- easy to edit, no coding
```

## Common tasks

| I want to… | Do this |
|---|---|
| Change the inspection advice | Edit `data/playbook.yaml`. Keep the indentation (spaces, not tabs). If a line contains `: `, put the whole line in quotes. |
| Add or rename demo machines | Edit `data/fleet.yaml`. Scenarios: `healthy`, `tool_wear`, `cooling_fault`, `overload`, `power_anomaly`. |
| Add a new URL | Add a function in the right file in `api/routes/`, define its data in `schemas/`, then call it from `apps/web/src/api/client.ts`. |
| Change a setting | Add it to `.env` in the project folder, e.g. `TELLTALE_SIMULATOR_INTERVAL_S=1`. All settings are in `core/config.py`. |

## Tests

The tests live in `tests/` and train tiny models on fake data, so no downloads are needed. To run them you need Python and [uv](https://docs.astral.sh/uv/). From the project folder:

```
uv sync --extra torch
uv run pytest apps/api/tests -q
```

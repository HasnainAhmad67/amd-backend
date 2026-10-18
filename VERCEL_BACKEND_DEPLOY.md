# Deploy Telltale backend on Vercel

This backend package has a root `index.py` FastAPI entrypoint. It includes the API source, ML inference source, and trained sensor, fan-acoustic, and bearing models. It is separate from the Vite frontend project.

## Import settings

Upload the contents of this ZIP's `telltale-vercel-backend/` directory to the root of your backend GitHub repository. The root `pyproject.toml` declares an explicit uv workspace for `apps/api` and `ml`, and `uv.lock` locks the proper package metadata names. In Vercel, import that repo as a new project and use:

- **Root Directory:** `./` if repository root contains `index.py`, `vercel.json`, `pyproject.toml`, `uv.lock`, `ml/`, `apps/`, and `models/`.
- **Framework Preset:** Python or FastAPI, if shown.
- **Install Command:** leave at Vercel's detected/default setting so it uses `uv.lock`; do not set `pip install -r requirements.txt` for this package.
- **Build Command:** leave empty/default.
- **Output Directory:** leave empty.

If you keep the files in a subfolder instead of putting them at repository root, set Root Directory to that exact folder. Do not choose `apps/api` as the root; the root `index.py` needs sibling `ml/` and `models/`.

## Keep the Python function bundle lean

The Vercel dependency set is inference-only. Audio features use the local NumPy/SciPy implementation in `ml/src/telltale_ml/acoustic/audio_ops.py`, so the deployment does not install `librosa`, `numba`, or `llvmlite`; `scikit-learn` is kept in the optional `train` extra and is imported only if the bundled sensor model is missing and auto-training starts. Keep Vercel's default install command so it installs only the locked runtime dependencies. This is intended to bring the function comfortably below the standard 500 MB limit (with margin), but use Vercel's final function-size report as the source of truth.

If Vercel still reports a function above your 450 MB target, first confirm that the project is using this updated `uv.lock` and the repository root is `./`. Only if the final function remains above Vercel's standard limit, the fallback is to enable Vercel's large-functions beta with `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` (and Fluid Compute, where required) or host the ML API on a platform with a larger function allowance.

## Backend environment variables

The root entrypoint defaults to public-demo settings: models under `./models`, temporary data under `/tmp/telltale-data`, offline sensor setup, demo-mode assessment auth, template AI, and permissive CORS (`["*"]`). These defaults are intentionally suitable only for a public demo. Do not use private/sensitive data. If `TELLTALE_AI_PROVIDER=amd` is already set, it still requires a valid AMD API key; the template provider is the no-key default when that variable is unset.

After deployment, verify `https://YOUR-BACKEND.vercel.app/api/v1/health` returns HTTP 200 JSON. Set the frontend's `VITE_API_BASE_URL` to `https://YOUR-BACKEND.vercel.app` without the `/api/v1` suffix, and redeploy the frontend.

Function storage is ephemeral. Persistent fleet/database state and long-running simulator jobs are not guaranteed on serverless functions.

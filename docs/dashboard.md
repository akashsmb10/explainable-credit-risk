# Dashboard and hosting guide

The public browser demo and the full Streamlit app explore the same executed credit-risk artifacts. Both use existing historical cardholder scores, not new applicant underwriting.

## Public demo

The `site/` directory is a static application with plain HTML, CSS and JavaScript. It includes a compact snapshot exported from saved test predictions and exact raw-tree SHAP contributions. No Python server, secret or model retraining is required to view it.

Export it after running the pipeline:

```powershell
.\.venv\Scripts\python.exe -m src.web_demo
```

Preview locally:

```powershell
.\.venv\Scripts\python.exe -m http.server 8000 --directory site --bind 127.0.0.1
```

Open [the local preview](http://127.0.0.1:8000). This loopback URL works only while the local preview process is running. The GitHub README shows the public URL only after successful publication is verified.

The site supports deep links to the decision simulator (`#decisions`), loss scenarios (`#loss`), model performance (`#performance`) and other sections. All metrics, model scores and saved explanations come from the executed artifacts. Slider calculations are retrospective simulations or explicitly labelled scenarios.

## Full Streamlit app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

Open [the local Streamlit app](http://localhost:8501) while the command is running. A fresh clone must first run `python -m src.pipeline`, because large model and prediction artifacts are not committed. The Streamlit app never trains a model on startup.

GitHub Pages cannot execute the Python Streamlit server. The public demo is therefore a separate frozen-results browser explorer, not a hosted Streamlit process. Hosting the full Python app would require an appropriate Python service and its own deployment setup.

## Browser verification

For optional desktop/mobile interaction checks, install the development dependency and run the browser script:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts/browser_check.py
```

On Windows the script uses installed Microsoft Edge if available. Otherwise install Playwright Chromium with `python -m playwright install chromium`. Checks include page rendering, frozen-policy counts, threshold endpoints, exposure validation and horizontal layout on mobile. The check also captures the README preview from the actual rendered results.

## Updating the public snapshot

After changing model/data code, rerun the pipeline, notebooks and final audit, then export the snapshot and verify the browser controls. Commit source changes to `main`. The initial public release uses a separate `gh-pages` branch containing only the `site/` files. This deployment branch is published by GitHub Pages; it is independent of Python CI.

```bash
git subtree split --prefix site -b publish-site
git push origin publish-site:gh-pages
```

Use a fresh local branch name for each update, or manage the existing branch normally. Avoid force-pushing repository history. The publication helper uses Git Credential Manager, keeps credentials in memory and never stores them in source or site files.

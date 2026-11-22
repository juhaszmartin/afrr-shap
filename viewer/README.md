# Results viewer

A local Dash app for browsing the forecast results and the SHAP attributions.

```bash
.venv/bin/pip install -r requirements.txt   # includes dash and plotly
.venv/bin/python viewer/viewer.py                 # http://localhost:8050, PORT=... to change
```

It reads the pipeline outputs in `results/` and `figures/`, plus the model zoo in `experiments/model_zoo/results/` if
present. When the files change, refresh the page to pick up new results.

| Tab | Answers | What's on it |
|---|---|---|
| **Leaderboard** | Which model is best, and where? | Ranking by median MAE skill against naive, % of groups beating naive, wins, mean rank; heatmap of skill per country × target × model (★ = best in row). Click a cell to open it in the explorer. |
| **Country explorer** | How does a model behave in one country? | Skill by horizon; error by hour of day; forecast vs actual for any window, with jumps to the start, the end or the largest naive error. |
| **Drivers (SHAP)** | What does the MLP's forecast depend on? | Category shares per country (click a country); top inputs; beeswarm of signed SHAP coloured by input value; category shares by horizon; seed stability. |
| **About** | | How the metrics are defined, plus a glossary of every model. |

Notes
- Model comparisons use only the country/target groups where every model shown has results. A model that has
  results for fewer than half the groups (one still being trained, for example) is listed in a note instead of
  shrinking the comparison.
- Single-priced countries have no separate short price. Their short-price option is greyed out, and the SHAP tab
  marks it as identical to the long price.
- Forecast averages (`avg_*`) have no saved prediction files. The explorer averages their members' forecasts on the fly.

"""aFRR imbalance forecasts: results viewer.

    .venv/bin/python viewer/viewer.py            # then open http://localhost:8050  (PORT=... to change)

Reads, and re-reads whenever the files change (refresh the page):
- results/{dev,test}/metrics_{CC}.csv            main run: naive, ridge, lightgbm, nn, nn_ensemble
- experiments/model_zoo/results/metrics/{CC}_{model}.csv model zoo and forecast averages (optional)
- results/predictions/{CC}.parquet, experiments/model_zoo/results/predictions/{CC}_{model}.parquet
- figures/shap_{category,mean_abs,raw,test_features,feature_names,stability,info}_*   SHAP of the MLP

Tabs: Leaderboard (which model is best, and where) · Country explorer (skill by horizon, forecast vs actual,
error by hour of day) · Drivers (what the MLP's forecasts depend on) · About.
"""

import json
import os
import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import plotly.graph_objects as go
import polars as pl
import yaml
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
ZOO = ROOT / "experiments" / "model_zoo" / "results"

# ------------------------------------------------------------------------------------------------ vocabulary
KINDS = {"long": "Long price", "short": "Short price", "volume": "Volume"}
UNITS = {"long": "EUR/MWh", "short": "EUR/MWh", "volume": "MWh"}
SERIES = {
    "long": "imbalance_price_long",
    "short": "imbalance_price_short",
    "volume": "imbalance_volume",
}

FAMILIES = {
    "baseline": "Baseline",
    "linear": "Linear & statistical",
    "trees": "Gradient-boosted trees",
    "mlp": "MLP (main run)",
    "seq": "Sequence nets",
    "modern": "Modern nets",
    "avg": "Forecast averages",
}
# key: (label, family, description)
MODELS = {
    "naive": (
        "Naive",
        "baseline",
        "Persistence: the value at t is the forecast for every horizon. Skill is measured against it.",
    ),
    "ar": ("AR", "linear", "Ridge regression on the targets' own history only."),
    "ridge": (
        "Ridge",
        "linear",
        "Ridge regression on every input, with lags and leads.",
    ),
    "arimax": (
        "ARIMAX",
        "linear",
        "Regression on the known-in-advance inputs at t+h plus ARMA errors (Kalman filter), per series.",
    ),
    "lightgbm": (
        "LightGBM",
        "trees",
        "One gradient-boosted tree model per target, pseudo-Huber loss.",
    ),
    "xgboost": (
        "XGBoost",
        "trees",
        "One gradient-boosted tree model per target on the GPU, pseudo-Huber loss.",
    ),
    "nn": ("MLP", "mlp", "Multi-layer perceptron, one seed; the model SHAP explains."),
    "nn_ensemble": ("MLP ensemble", "mlp", "Average of 100 MLP seeds."),
    "gru": (
        "GRU",
        "seq",
        "Recurrent net over a 24 h (48 h hourly) window of every input.",
    ),
    "tcn": ("TCN", "seq", "Dilated causal convolutions over the window."),
    "tide": ("TiDE", "seq", "Dense encoder-decoder (Das et al. 2023)."),
    "patchtst": (
        "PatchTST",
        "seq",
        "Transformer over patches of the window (Nie et al. 2023).",
    ),
    "dlinear": (
        "DLinear",
        "modern",
        "Trend/remainder split, one linear map per series (Zeng et al. 2023).",
    ),
    "nhits": (
        "N-HiTS",
        "modern",
        "MLP stacks with multi-rate pooling and interpolation (Challu et al. 2023).",
    ),
    "tsmixer": (
        "TSMixer",
        "modern",
        "Time- and feature-mixing MLPs with per-step future inputs (Chen et al. 2023).",
    ),
    "itransformer": (
        "iTransformer",
        "modern",
        "Attention across variables, each variable's window is one token (Liu et al. 2024).",
    ),
    "tft": (
        "TFT",
        "modern",
        "Temporal Fusion Transformer: variable selection, LSTM, attention (Lim et al. 2021).",
    ),
    "mamba2": (
        "Mamba-2",
        "modern",
        "Selective state-space model, SSD form (Dao & Gu 2024).",
    ),
    "avg_lgb_xgb": (
        "Avg LightGBM + XGBoost",
        "avg",
        "Equal-weight average of the two forecasts.",
    ),
    "avg_lgb_arimax": (
        "Avg LightGBM + ARIMAX",
        "avg",
        "Equal-weight average of the two forecasts.",
    ),
    "avg_lgb_xgb_arimax": (
        "Avg LightGBM + XGBoost + ARIMAX",
        "avg",
        "Equal-weight average of the three forecasts.",
    ),
    "avg_trees_arimax_ridge_nn": (
        "Avg trees + ARIMAX + ridge + MLP ens.",
        "avg",
        "Equal-weight average of five forecasts.",
    ),
    "avg_lgb_itr": (
        "Avg LightGBM + iTransformer",
        "avg",
        "Equal-weight average of LightGBM and iTransformer ×5; best on dev.",
    ),
    "avg_lgb_arimax_itr": (
        "Avg LightGBM + ARIMAX + iTransformer",
        "avg",
        "Equal-weight average of the three forecasts.",
    ),
    "avg_lgb_xgb_arimax_itr": (
        "Avg LightGBM + XGBoost + ARIMAX + iTransformer",
        "avg",
        "Equal-weight average of four forecasts.",
    ),
    "avg_lgb_arimax_itr_dlin": (
        "Avg LightGBM + ARIMAX + iTransformer + DLinear",
        "avg",
        "Equal-weight average of four forecasts.",
    ),
}
COMBO_MEMBERS = {
    "avg_lgb_xgb": ["lightgbm", "xgboost"],
    "avg_lgb_arimax": ["lightgbm", "arimax"],
    "avg_lgb_xgb_arimax": ["lightgbm", "xgboost", "arimax"],
    "avg_trees_arimax_ridge_nn": [
        "lightgbm",
        "xgboost",
        "arimax",
        "ridge",
        "nn_ensemble",
    ],
    "avg_lgb_itr": ["lightgbm", "itransformer_x5"],
    "avg_lgb_arimax_itr": ["lightgbm", "arimax", "itransformer_x5"],
    "avg_lgb_xgb_arimax_itr": ["lightgbm", "xgboost", "arimax", "itransformer_x5"],
    "avg_lgb_arimax_itr_dlin": ["lightgbm", "arimax", "itransformer_x5", "dlinear_x5"],
}
MAIN_RUN = {"naive", "ridge", "lightgbm", "nn", "nn_ensemble"}
SEEDS_RE = re.compile(r"^(.*)_x(\d+)$")

CATEGORIES = {  # key: (label, colour)
    "target history": ("Target history (own past)", "#f2c14e"),
    "activated_balancing_energy_prices": ("Balancing energy prices", "#f78154"),
    "prices": ("Day-ahead price", "#e05780"),
    "generation": ("Actual generation", "#4d9de0"),
    "generation_forecast": ("Generation forecast", "#9ad1ff"),
    "wind_and_solar_forecast": ("Wind & solar forecast (DA)", "#3bb273"),
    "intraday_wind_and_solar_forecast": ("Wind & solar forecast (ID)", "#9be3a0"),
    "load": ("Actual load", "#9b5de5"),
    "load_forecast": ("Load forecast", "#d0a6ff"),
    "import": ("Cross-border flows", "#2ec4b6"),
    "net_position": ("Net position", "#8d99ae"),
    "weather": ("Weather (ERA5)", "#c9b77d"),
    "calendar": ("Calendar", "#5c677d"),
}
PALETTE = [
    "#4c9aff",
    "#f2c14e",
    "#3bb273",
    "#e05780",
    "#9b5de5",
    "#f78154",
    "#2ec4b6",
    "#c9b77d",
    "#ff9de2",
    "#7cc6fe",
    "#b5e48c",
    "#ffb703",
    "#a0a4b8",
    "#ef476f",
    "#06d6a0",
    "#8338ec",
    "#fb8500",
    "#90be6d",
    "#4cc9f0",
    "#f15bb5",
    "#c77dff",
    "#e9c46a",
    "#83c5be",
    "#ffafcc",
]

BG, CARD, GRID, TEXT, MUTED = "#171923", "#20232f", "#2f3344", "#e8e9f0", "#9aa0b4"


def parse_model(m):
    """'tft_x5' -> ('tft', 5); 'lightgbm' -> ('lightgbm', 1)."""
    hit = SEEDS_RE.match(m)
    return (hit.group(1), int(hit.group(2))) if hit else (m, 1)


def model_label(m):
    base, seeds = parse_model(m)
    label = MODELS.get(base, (base,))[0]
    if base in MODELS and MODELS[base][1] in ("seq", "modern"):
        return f"{label} ×{seeds}" if seeds > 1 else f"{label} (1 seed)"
    return label


def model_family(m):
    return MODELS.get(parse_model(m)[0], (None, "other"))[1]


def model_order(m):
    base, seeds = parse_model(m)
    keys = list(MODELS)
    return (keys.index(base) if base in keys else len(keys), -seeds)


MODEL_COLORS = {
    "naive": "#c8cad6",
    "ar": "#8d99ae",
    "ridge": "#2ec4b6",
    "arimax": "#ef476f",
    "lightgbm": "#4c9aff",
    "xgboost": "#7cc6fe",
    "nn": "#d0a6ff",
    "nn_ensemble": "#9b5de5",
    "gru": "#90be6d",
    "tcn": "#b5e48c",
    "tide": "#3bb273",
    "patchtst": "#83c5be",
    "dlinear": "#ffb703",
    "nhits": "#fb8500",
    "tsmixer": "#e9c46a",
    "itransformer": "#f15bb5",
    "tft": "#ff9de2",
    "mamba2": "#f78154",
    "avg_lgb_xgb": "#4cc9f0",
    "avg_lgb_arimax": "#ffa69e",
    "avg_lgb_xgb_arimax": "#ffd23f",
    "avg_trees_arimax_ridge_nn": "#c9b77d",
    "avg_lgb_itr": "#00f5d4",
    "avg_lgb_arimax_itr": "#ffe066",
    "avg_lgb_xgb_arimax_itr": "#e0c3fc",
    "avg_lgb_arimax_itr_dlin": "#ffcb77",
}


def model_color(m):
    base, _ = parse_model(m)
    if base in MODEL_COLORS:
        return MODEL_COLORS[base]
    keys = list(MODELS)
    i = keys.index(base) if base in keys else abs(hash(base))
    return PALETTE[i % len(PALETTE)]


def cat_label(c):
    return CATEGORIES.get(c, (c.replace("_", " "),))[0]


def cat_color(c):
    return CATEGORIES.get(c, (None, PALETTE[abs(hash(c)) % len(PALETTE)]))[1]


def horizon_minutes(h):
    """'t+15min' -> 15, 't+2h' -> 120."""
    n = int(re.search(r"\d+", h).group())
    return n * 60 if h.endswith("h") else n


def horizon_label(h):
    return f"+{h[2:-3]} min" if h.endswith("min") else f"+{h[2:-1]} h"


def kind_of(series):
    return "volume" if "volume" in series else ("long" if "long" in series else "short")


@lru_cache(None)
def timezones():
    try:
        cfg = yaml.safe_load((ROOT / "configs" / "countries.yaml").read_text())
        return {cc: v.get("timezone", "UTC") for cc, v in cfg.items()}
    except Exception:  # noqa: BLE001
        return {}


# ------------------------------------------------------------------------------------------------ data
_STATE = {"stamp": None}


def _metric_files():
    return sorted(RESULTS.glob("*/metrics_*.csv")) + sorted(ZOO.glob("metrics/*.csv"))


def missing(v):
    """True for None and NaN."""
    return v is None or bool(np.isnan(v))


def data():
    """All metrics, reloaded when any metrics file is added or changed."""
    files = _metric_files()
    stamp = (len(files), max((f.stat().st_mtime for f in files), default=0.0))
    if _STATE["stamp"] == stamp:
        return _STATE
    frames = []
    for f in files:
        try:
            frames.append(
                pl.read_csv(
                    f,
                    columns=[
                        "Country",
                        "Split",
                        "Model",
                        "Series",
                        "Horizon",
                        "Group",
                        "MAE",
                        "RMSE",
                    ],
                    schema_overrides={"MAE": pl.Float64, "RMSE": pl.Float64},
                )
            )
        except Exception:  # noqa: BLE001  (file being written)
            continue
    series = pl.col("Series")
    m = (
        pl.concat(frames, how="vertical_relaxed")
        .filter(~pl.col("Group").str.contains("single"))
        .with_columns(
            pl.col("MAE", "RMSE").fill_nan(None),
            Kind=pl.when(series.str.contains("volume"))
            .then(pl.lit("volume"))
            .when(series.str.contains("long"))
            .then(pl.lit("long"))
            .otherwise(pl.lit("short")),
        )
        .unique(
            ["Country", "Split", "Model", "Series", "Horizon"],
            keep="last",
            maintain_order=True,
        )
    )
    columns = sorted(m.filter(pl.col("MAE").is_not_null())["Model"].unique().to_list())
    horizons = sorted(
        m["Horizon"].unique().to_list(),
        key=lambda h: (h.endswith("h"), horizon_minutes(h)),
    )
    _STATE.update(
        stamp=stamp,
        metrics=m,
        model_columns=columns,
        models=sorted(columns, key=model_order),
        horizons=horizons,
        countries=sorted(m["Country"].unique().to_list()),
        updated=datetime.fromtimestamp(stamp[1], UTC),
    )
    return _STATE


def group_values(split, kinds, metric, horizon="all"):
    """Metric per country/target group (rows: Country, Kind, sorted) and model (one column each, null where a model
    has no result): one horizon, or the mean over a group's horizons."""
    d = data()
    m = d["metrics"].filter(
        (pl.col("Split") == split) & pl.col("Kind").is_in(list(kinds))
    )
    if horizon != "all":
        m = m.filter(pl.col("Horizon") == horizon)
    wide = (
        m.group_by("Country", "Kind", "Model")
        .agg(pl.col(metric).mean())
        .pivot(on="Model", index=["Country", "Kind"], values=metric)
    )
    if "Country" not in wide.columns:  # nothing selected
        wide = pl.DataFrame(schema={"Country": pl.String, "Kind": pl.String})
    wide = wide.with_columns(
        pl.lit(None, pl.Float64).alias(c)
        for c in d["model_columns"]
        if c not in wide.columns
    )
    return wide.select("Country", "Kind", *d["model_columns"]).sort("Country", "Kind")


def kinds_for(cc):
    have = set(
        data()["metrics"].filter(pl.col("Country") == cc)["Kind"].unique().to_list()
    )
    return [k for k in KINDS if k in have]


def horizons_for(cc):
    m = data()["metrics"].filter(pl.col("Country") == cc)
    return sorted(m["Horizon"].unique().to_list(), key=horizon_minutes)


def row_ranks(a):
    """Average ranks (1 = smallest, ties share their mean rank) along each row of a 2-D array."""
    n, k = a.shape
    long = pl.DataFrame({"row": np.repeat(np.arange(n), k), "v": a.ravel()})
    return (
        long.select(pl.col("v").rank("average").over("row"))["v"]
        .to_numpy()
        .reshape(n, k)
    )


def vs_naive(g, cols):
    """Rows of g where naive and every model in cols have a value: (naive, model values, is-volume mask)."""
    complete = g.select("Kind", "naive", *cols).drop_nulls()
    a = complete.select("naive", *cols).to_numpy()
    return a[:, 0], a[:, 1:], (complete["Kind"] == "volume").to_numpy()


def col_median(a, mask):
    return np.median(a[mask], axis=0) if mask.any() else np.full(a.shape[1], np.nan)


def default_models(models, single_seed=False):
    """Every model except the single-seed MLP and, unless asked, single-seed sequence nets that have a multi-seed run."""
    keep = []
    for mdl in models:
        base, seeds = parse_model(mdl)
        if mdl in ("naive", "nn"):
            continue
        if (
            seeds == 1
            and not single_seed
            and any(parse_model(o) == (base, s) for o in models for s in range(2, 101))
        ):
            continue
        keep.append(mdl)
    return keep


def leaderboard(split, kinds, models, metric="MAE", horizon="all"):
    """Per model, over the groups where every model has results: relative skill (median, mean, 10th percentile),
    share of groups beating naive, absolute reduction of the metric vs naive (prices and volumes apart), wins, rank.

    Returns (stats, coverage, groups, scored, naive_error): stats maps model -> row dict, ordered by median skill with
    naive last; coverage maps model -> groups with a result; scored is the number of groups every model has;
    naive_error is the median naive error of the scored price and volume groups."""
    g = group_values(split, kinds, metric, horizon)
    cols = [mdl for mdl in models if mdl in g.columns]
    coverage = {c: g[c].count() for c in cols}
    naive, vals, is_vol = vs_naive(g, cols)
    if not len(naive) or not cols:
        return None, coverage, g.height, len(naive), None
    skill = 1 - vals / naive[:, None]
    reduction = naive[:, None] - vals
    both = np.column_stack([naive, vals])
    best = both.argmin(axis=1)  # 0 = naive
    rank = row_ranks(both)
    stats = pl.DataFrame(
        {
            "model": cols,
            "median": np.median(skill, axis=0),
            "mean": skill.mean(axis=0),
            "p10": np.quantile(skill, 0.1, axis=0),
            "beats": (skill > 0).mean(axis=0) * 100,
            "red_price": col_median(reduction, ~is_vol),
            "red_volume": col_median(reduction, is_vol),
            "wins": np.bincount(best, minlength=len(cols) + 1)[1:],
            "rank": rank[:, 1:].mean(axis=0),
        }
    ).sort("median", descending=True, maintain_order=True)
    rows = {r["model"]: r for r in stats.iter_rows(named=True)}
    rows["naive"] = {
        "model": "naive",
        "median": 0.0,
        "mean": 0.0,
        "p10": 0.0,
        "beats": 0.0,
        "red_price": 0.0,
        "red_volume": 0.0,
        "wins": int((best == 0).sum()),
        "rank": float(rank[:, 0].mean()),
    }
    naive_error = {
        "price": float(np.median(naive[~is_vol])) if (~is_vol).any() else np.nan,
        "volume": float(np.median(naive[is_vol])) if is_vol.any() else np.nan,
    }
    return rows, coverage, g.height, len(naive), naive_error


# ------------------------------------------------------------------------------------------------ predictions
def _schema(path):
    try:
        return pl.read_parquet_schema(path)
    except Exception:  # noqa: BLE001
        return {}


@lru_cache(maxsize=48)
def _pred_column(cc, target, model, stamp):
    """(timestamp, split, value) for one model and target, timestamps as naive UTC."""
    main = RESULTS / "predictions" / f"{cc}.parquet"
    zoo = ZOO / "predictions" / f"{cc}_{model}.parquet"
    col = target if model == "actual" else f"{target}|{model}"
    for path in (main, zoo):
        if col in _schema(path):
            df = pl.read_parquet(path, columns=["timestamp", "split", col])
            ts = pl.col("timestamp")
            if getattr(df.schema["timestamp"], "time_zone", None):
                ts = ts.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
            return df.select(
                ts.cast(pl.Datetime("us")),
                "split",
                pl.col(col).cast(pl.Float64).fill_nan(None).alias("value"),
            )
    if model in COMBO_MEMBERS:
        parts = [_pred_column(cc, target, mm, stamp) for mm in COMBO_MEMBERS[model]]
        if all(p is not None for p in parts):
            out = parts[0].rename({"value": "v0"})
            for i, p in enumerate(parts[1:], 1):
                out = out.join(
                    p.select("timestamp", pl.col("value").alias(f"v{i}")),
                    on="timestamp",
                    how="left",
                    maintain_order="left",
                )
            # missing in any member -> missing average
            total = pl.sum_horizontal(f"v{i}" for i in range(len(parts)))
            any_missing = pl.any_horizontal(
                pl.col(f"v{i}").is_null() for i in range(len(parts))
            )
            return out.select(
                "timestamp",
                "split",
                pl.when(any_missing)
                .then(None)
                .otherwise(total / len(parts))
                .alias("value"),
            )
    return None


def predictions(cc, target, models, split):
    """Frame of target time (local, column 'time'), actual and one column per model with predictions."""
    stamp = data()["stamp"]
    actual = _pred_column(cc, target, "actual", stamp)
    if actual is None:
        return None, []
    df = actual.filter(pl.col("split") == split).select(
        "timestamp", pl.col("value").alias("actual")
    )
    have = []
    for mdl in dict.fromkeys(models):
        p = _pred_column(cc, target, mdl, stamp)
        if p is None:
            continue
        df = df.join(
            p.select("timestamp", pl.col("value").alias(mdl)),
            on="timestamp",
            how="left",
            maintain_order="left",
        )
        have.append(mdl)
    h = horizon_minutes(target.rsplit("_", 1)[1])
    time = (pl.col("timestamp") + pl.duration(minutes=h)).dt.replace_time_zone("UTC")
    df = df.select(
        time.dt.convert_time_zone(timezones().get(cc, "UTC")).alias("time"),
        pl.exclude("timestamp"),
    )
    return df, have


def shift_days(t, days):
    """An aware datetime moved by an exact number of 24 h days, kept in its own time zone."""
    return (t.astimezone(UTC) + timedelta(days=days)).astimezone(t.tzinfo)


# ------------------------------------------------------------------------------------------------ SHAP files
SHAP_MODELS = {  # key: (label, directory, method)
    "nn": ("MLP", FIGURES, "DeepExplainer, 5 seeds"),
    "lightgbm": ("LightGBM", ZOO / "shap" / "lightgbm", "TreeSHAP"),
    "itransformer_x5": (
        "iTransformer ×5",
        ZOO / "shap" / "itransformer_x5",
        "expected gradients, 5 seeds",
    ),
    "dlinear_x5": (
        "DLinear ×5",
        ZOO / "shap" / "dlinear_x5",
        "expected gradients, 5 seeds",
    ),
    "avg_lgb_itr": (
        "Avg LightGBM + iTransformer",
        ZOO / "shap" / "avg_lgb_itr",
        "mean of the two models' SHAP",
    ),
}


def shap_models():
    return [
        m for m, (_, d, _) in SHAP_MODELS.items() if any(d.glob("shap_category_*.csv"))
    ]


def shap_dir(model):
    return SHAP_MODELS.get(model, SHAP_MODELS["nn"])[1]


def shap_countries(model="nn"):
    return sorted(
        p.stem.split("_")[-1] for p in shap_dir(model).glob("shap_category_*.csv")
    )


@lru_cache(maxsize=256)
def _csv(path, mtime):
    return pl.read_csv(path, infer_schema_length=None)


def shap_table(kind, cc, model="nn"):
    p = shap_dir(model) / f"shap_{kind}_{cc}.csv"
    return _csv(p, p.stat().st_mtime) if p.exists() else None


def shap_targets(cc, kind, model="nn"):
    t = shap_table("category", cc, model)
    if t is None:
        return []
    cols = [c for c in t.columns if c.startswith(SERIES[kind] + "_t+")]
    return sorted(cols, key=lambda c: horizon_minutes(c.rsplit("_", 1)[1]))


def shap_raw(cc, target, model="nn"):
    p = shap_dir(model) / f"shap_raw_{cc}_{target.replace('+', '')}.npy"
    return np.load(p) if p.exists() else None


@lru_cache(maxsize=64)
def _features(directory, cc):
    names = (directory / f"shap_feature_names_{cc}.txt").read_text().splitlines()
    vals = np.load(directory / f"shap_test_features_{cc}.npy")
    return names, vals


def shap_features(cc, model="nn"):
    d = shap_dir(model)
    if not (d / f"shap_test_features_{cc}.npy").exists():
        d = FIGURES  # same rows and groups as the MLP's explanation
    return _features(d, cc)


def shap_info(cc, model="nn"):
    p = shap_dir(model) / f"shap_info_{cc}.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------------------------------------ figure helpers
def styled(fig, height=420, **kw):
    layout = {
        "template": "plotly_dark",
        "paper_bgcolor": CARD,
        "plot_bgcolor": CARD,
        "height": height,
        "font": {"color": TEXT, "size": 12, "family": "Inter, system-ui, sans-serif"},
        "margin": {"l": 10, "r": 20, "t": 30, "b": 40},
        "hoverlabel": {"bgcolor": "#2b2f3f", "font_size": 12},
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "orientation": "h",
            "y": -0.22,
            "x": 0,
            "yanchor": "top",
        },
    }
    fig.update_layout(**(layout | kw))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    return fig


def empty(msg, height=300):
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color=MUTED),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return styled(fig, height)


def card(title, caption, *children):
    return html.Div(
        [
            html.Div(
                [html.H3(title), html.P(caption, className="caption")],
                className="card-head",
            ),
            *children,
        ],
        className="card",
    )


def kpi(label, value, sub="", tone=""):
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className=f"kpi-sub {tone}"),
        ],
        className="kpi",
    )


def control(label, component, grow=False):
    return html.Div(
        [html.Label(label), component], className="control grow" if grow else "control"
    )


def radio(id_, options, value):
    return dcc.RadioItems(
        id=id_, options=options, value=value, inline=True, className="pills"
    )


def fmt_skill(v):
    return "–" if missing(v) else f"{v:+.3f}"


SPLITS = [
    {"label": "Test · 2026", "value": "test"},
    {"label": "Dev · 2025", "value": "dev"},
]
SKILL_HELP = (
    "MAE skill = 1 − MAE / MAE(naive), averaged over horizons. Above 0 beats persistence; "
    "0.10 means 10 % lower error than naive."
)


# ------------------------------------------------------------------------------------------------ layout
def horizon_options():
    hs = data()["horizons"]
    fifteen = [h for h in hs if h.endswith("min")]
    hourly = [h for h in hs if h.endswith("h")]
    return (
        [{"label": "All (mean over horizons)", "value": "all"}]
        + [
            {"label": f"{horizon_label(h)} (15-min countries)", "value": h}
            for h in fifteen
        ]
        + [
            {"label": f"{horizon_label(h)} (hourly countries)", "value": h}
            for h in hourly
        ]
    )


METRICS = [{"label": "MAE", "value": "MAE"}, {"label": "RMSE", "value": "RMSE"}]


def layout_leaderboard():
    return html.Div(
        [
            html.Div(
                [
                    control("Evaluation period", radio("lb-split", SPLITS, "test")),
                    control("Error metric", radio("lb-metric", METRICS, "MAE")),
                    control(
                        "Horizon",
                        dcc.Dropdown(
                            id="lb-horizon",
                            options=horizon_options(),
                            value="all",
                            clearable=False,
                            className="dd-wide",
                        ),
                    ),
                    control(
                        "Targets",
                        dcc.Checklist(
                            id="lb-kinds",
                            options=[
                                {"label": v, "value": k} for k, v in KINDS.items()
                            ],
                            value=list(KINDS),
                            inline=True,
                            className="pills",
                        ),
                    ),
                    control(
                        "Model families",
                        dcc.Checklist(
                            id="lb-families",
                            options=[
                                {"label": v, "value": k}
                                for k, v in FAMILIES.items()
                                if k != "baseline"
                            ],
                            value=[k for k in FAMILIES if k != "baseline"],
                            inline=True,
                            className="pills",
                        ),
                        grow=True,
                    ),
                    control(
                        "Runs",
                        dcc.Checklist(
                            id="lb-single",
                            options=[
                                {"label": "Include single-seed nets", "value": "y"}
                            ],
                            value=[],
                            inline=True,
                            className="pills",
                        ),
                    ),
                ],
                className="controls",
            ),
            html.Div(id="lb-kpis", className="kpis"),
            html.Div(id="lb-note", className="note"),
            card(
                "Ranking",
                "",
                html.P(id="lb-caption", className="caption"),
                html.Div(id="lb-table", className="scroll-x"),
            ),
            card(
                "Improvement by horizon, all countries",
                "Median skill over country/target groups at each horizon, for the metric chosen above (targets and "
                "families as selected; the horizon filter does not apply here). The best-ranked model of each family is "
                "shown; click legend entries to add or remove models. Hover for the median absolute reduction.",
                dcc.Graph(id="lb-horizons", config={"displaylogo": False}),
            ),
            card(
                "Skill per country and target",
                "",
                html.P(id="lb-heat-caption", className="caption"),
                dcc.Graph(id="lb-heatmap", config={"displaylogo": False}),
            ),
        ],
        id="page-leaderboard",
    )


def layout_country():
    d = data()
    cc0 = "AT" if "AT" in d["countries"] else d["countries"][0]
    return html.Div(
        [
            html.Div(
                [
                    control(
                        "Country",
                        dcc.Dropdown(
                            id="cx-country",
                            options=d["countries"],
                            value=cc0,
                            clearable=False,
                            className="dd-narrow",
                        ),
                    ),
                    control(
                        "Target",
                        radio(
                            "cx-kind",
                            [{"label": v, "value": k} for k, v in KINDS.items()],
                            "long",
                        ),
                    ),
                    control("Evaluation period", radio("cx-split", SPLITS, "test")),
                    control("Error metric", radio("cx-metric", METRICS, "MAE")),
                    control(
                        "Models (naive is always shown as the reference)",
                        dcc.Dropdown(
                            id="cx-models",
                            multi=True,
                            value=[
                                "lightgbm",
                                "itransformer_x5",
                                "dlinear_x5",
                                "avg_lgb_itr",
                            ],
                            options=[
                                {"label": model_label(x), "value": x}
                                for x in d["models"]
                                if x != "naive"
                            ],
                        ),
                        grow=True,
                    ),
                ],
                className="controls",
            ),
            html.Div(id="cx-kpis", className="kpis"),
            html.Div(
                [
                    card(
                        "By horizon",
                        "Each forecast horizon on its own. Skill = improvement over naive (the 0 line); "
                        "switch to the absolute error to see the size in EUR/MWh or MWh. The table lists every model shown.",
                        radio(
                            "cx-view",
                            [
                                {"label": "Skill vs naive", "value": "skill"},
                                {"label": "Absolute error", "value": "abs"},
                            ],
                            "skill",
                        ),
                        dcc.Graph(id="cx-horizon", config={"displaylogo": False}),
                        html.Div(id="cx-htable", className="scroll-x"),
                    ),
                    card(
                        "Error by hour of day",
                        "Error (metric chosen above) of the forecast-vs-actual horizon by local hour "
                        "of the delivery period, over the whole evaluation period. Shows when in the day forecasts struggle.",
                        dcc.Graph(id="cx-hourly", config={"displaylogo": False}),
                    ),
                ],
                className="grid-2",
            ),
            card(
                "Forecast vs actual",
                "The forecasts made h ahead, plotted at the time they forecast (local time). Drag to zoom, "
                "double-click to reset; click legend entries to hide models.",
                html.Div(
                    [
                        control(
                            "Horizon",
                            dcc.Dropdown(
                                id="cx-horizon-dd",
                                clearable=False,
                                className="dd-narrow",
                            ),
                        ),
                        control(
                            "From",
                            dcc.DatePickerSingle(
                                id="cx-start",
                                display_format="YYYY-MM-DD",
                                first_day_of_week=1,
                            ),
                        ),
                        control(
                            "Window",
                            radio(
                                "cx-window",
                                [
                                    {"label": f"{n} d", "value": n}
                                    for n in (1, 3, 7, 14)
                                ],
                                7,
                            ),
                        ),
                        control(
                            "Jump to",
                            html.Div(
                                [
                                    html.Button(
                                        "Start", id="cx-jump-start", className="btn"
                                    ),
                                    html.Button(
                                        "Largest naive error",
                                        id="cx-jump-spike",
                                        className="btn",
                                    ),
                                    html.Button(
                                        "End", id="cx-jump-end", className="btn"
                                    ),
                                ],
                                className="btn-row",
                            ),
                        ),
                    ],
                    className="controls inner",
                ),
                dcc.Graph(id="cx-series", config={"displaylogo": False}),
            ),
        ],
        id="page-country",
        style={"display": "none"},
    )


def layout_shap():
    cs = shap_countries()
    cc0 = "AT" if "AT" in cs else (cs[0] if cs else None)
    return html.Div(
        [
            html.P(
                [
                    "SHAP attributes each forecast to its inputs; an input's value at t, lags, window steps and leads "
                    "are summed into one input. Mean |SHAP| is averaged over the same 1,000 test rows for every model, "
                    "in standardised target units. The ",
                    html.B("share"),
                    " of an input or category is its part of the "
                    "model's total mean |SHAP| for that target, which makes models comparable. Methods differ: "
                    "DeepExplainer (MLP), TreeSHAP (LightGBM), expected gradients (iTransformer, DLinear); the average's "
                    "SHAP is the mean of its two members'.",
                ],
                className="note",
            ),
            html.Div(
                [
                    control(
                        "Model explained",
                        radio(
                            "sh-model",
                            [
                                {"label": SHAP_MODELS[m][0], "value": m}
                                for m in shap_models()
                            ],
                            "nn",
                        ),
                    ),
                    control(
                        "Target",
                        radio(
                            "sh-kind",
                            [{"label": v, "value": k} for k, v in KINDS.items()],
                            "long",
                        ),
                    ),
                    control(
                        "Horizon (all countries)",
                        radio(
                            "sh-hpos",
                            [
                                {"label": "Average", "value": "avg"},
                                {"label": "Shortest", "value": "first"},
                                {"label": "Longest", "value": "last"},
                            ],
                            "avg",
                        ),
                    ),
                    control(
                        "Sort countries by",
                        dcc.Dropdown(
                            id="sh-sort",
                            clearable=False,
                            value="target history",
                            className="dd-wide",
                            options=[
                                {"label": v[0], "value": k}
                                for k, v in CATEGORIES.items()
                            ],
                        ),
                    ),
                ],
                className="controls",
            ),
            card(
                "What drives the forecast, by country",
                "Share of each input category in the selected model's attributions. Click a country to inspect it below.",
                dcc.Graph(id="sh-countries", config={"displaylogo": False}),
            ),
            html.Div(
                [
                    control(
                        "Country",
                        dcc.Dropdown(
                            id="sh-country",
                            options=cs,
                            value=cc0,
                            clearable=False,
                            className="dd-narrow",
                        ),
                    ),
                    control(
                        "Horizon",
                        dcc.Dropdown(
                            id="sh-horizon", clearable=False, className="dd-narrow"
                        ),
                    ),
                    control(
                        "Features shown",
                        dcc.Slider(
                            id="sh-topn",
                            min=5,
                            max=40,
                            step=5,
                            value=15,
                            marks={n: str(n) for n in (5, 10, 20, 30, 40)},
                        ),
                        grow=True,
                    ),
                    html.Div(id="sh-stability", className="badge"),
                ],
                className="controls",
            ),
            card(
                "Do the top models rely on the same inputs?",
                "Every explained model for the selected country, target and horizon. Left: each input's share of the "
                "model's total mean |SHAP| (rank in the cell), inputs ordered by their average share. Right: category "
                "shares per model. Below: rank agreement between models over all inputs (Spearman) and overlap of "
                "their top-10 inputs.",
                html.Div(
                    [
                        dcc.Graph(id="sh-cmp-inputs", config={"displaylogo": False}),
                        dcc.Graph(id="sh-cmp-cats", config={"displaylogo": False}),
                    ],
                    className="grid-cmp",
                ),
                html.Div(id="sh-cmp-table", className="scroll-x"),
            ),
            html.Div(
                [
                    card(
                        "Top inputs",
                        "Mean |SHAP| per input for the selected model (standardised target units), coloured "
                        "by category.",
                        dcc.Graph(id="sh-features", config={"displaylogo": False}),
                    ),
                    card(
                        "Direction of effect",
                        "One dot per explained test row. Right of 0 pushes the forecast up. Colour is "
                        "the input's value at t (blue low, red high, rank within the 1,000 rows).",
                        dcc.Graph(id="sh-beeswarm", config={"displaylogo": False}),
                    ),
                ],
                className="grid-2",
            ),
            card(
                "How the drivers change with the horizon",
                "Category shares at each horizon for the selected country and model.",
                dcc.Graph(id="sh-profile", config={"displaylogo": False}),
            ),
        ],
        id="page-shap",
        style={"display": "none"},
    )


def layout_about():
    d = data()
    rows = [
        html.Tr(
            [
                html.Td(html.Span(model_label(m), style={"color": model_color(m)})),
                html.Td(FAMILIES.get(model_family(m), "")),
                html.Td(
                    MODELS.get(parse_model(m)[0], ("", "", ""))[2]
                    + (
                        f" Average of {parse_model(m)[1]} seeds."
                        if parse_model(m)[1] > 1
                        else ""
                    )
                ),
            ]
        )
        for m in d["models"]
    ]
    return html.Div(
        [
            card(
                "How to read the numbers",
                "",
                html.Ul(
                    [
                        html.Li(
                            [
                                html.B("Targets: "),
                                "imbalance price (long and short; single-priced countries have one price) "
                                "and imbalance volume, forecast 15–75 min ahead (15-min countries) or 1–2 h ahead (hourly).",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("Periods: "),
                                "models are trained up to 2024, tuned and early-stopped on 2025 (dev) "
                                "and evaluated once on 2026 (test). Prefer models on dev; test shows how they held up.",
                            ]
                        ),
                        html.Li([html.B("MAE skill: "), SKILL_HELP]),
                        html.Li(
                            [
                                html.B("Comparisons "),
                                "between models use only the country/target groups where every model "
                                "shown has results, so all models are scored on the same rows.",
                            ]
                        ),
                        html.Li(
                            [
                                html.B("Data: "),
                                f"{len(d['countries'])} countries, {len(d['models'])} models. "
                                f"Metrics last updated {d['updated']:%Y-%m-%d %H:%M} UTC; refresh the page to reload.",
                            ]
                        ),
                    ]
                ),
            ),
            card(
                "Models",
                "",
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th("Model"),
                                    html.Th("Family"),
                                    html.Th("What it is"),
                                ]
                            )
                        ),
                        html.Tbody(rows),
                    ],
                    className="table",
                ),
            ),
        ],
        id="page-about",
        style={"display": "none"},
    )


def serve_layout():
    d = data()
    return html.Div(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.H1("aFRR imbalance forecasts"),
                            html.Span(
                                f"{len(d['countries'])} countries · {len(d['models'])} models · updated "
                                f"{d['updated']:%Y-%m-%d %H:%M} UTC",
                                className="subtitle",
                            ),
                        ]
                    ),
                    dcc.Tabs(
                        id="tabs",
                        value="leaderboard",
                        className="tabs",
                        children=[
                            dcc.Tab(
                                label=lab,
                                value=val,
                                className="tab",
                                selected_className="tab--selected",
                            )
                            for lab, val in [
                                ("Leaderboard", "leaderboard"),
                                ("Country explorer", "country"),
                                ("Drivers (SHAP)", "shap"),
                                ("About", "about"),
                            ]
                        ],
                    ),
                ],
                className="header",
            ),
            html.Main(
                [layout_leaderboard(), layout_country(), layout_shap(), layout_about()]
            ),
        ]
    )


app = Dash(
    __name__,
    title="aFRR forecasts",
    assets_folder=str(Path(__file__).parent / "assets"),
)
server = app.server
app.layout = serve_layout


@app.callback(
    [Output(f"page-{p}", "style") for p in ("leaderboard", "country", "shap", "about")],
    Input("tabs", "value"),
)
def show_page(tab):
    return [
        {"display": "block" if tab == p else "none"}
        for p in ("leaderboard", "country", "shap", "about")
    ]


# ------------------------------------------------------------------------------------------------ leaderboard
def fmt_red(v, unit):
    return "–" if missing(v) else f"{v:+.1f} {unit}"


def horizon_profile(split, kinds, models, metric):
    """Per horizon, model -> median skill, share beating naive and median absolute reduction, over the groups where
    naive and every model have a result."""
    out = {}
    for h in data()["horizons"]:
        g = group_values(split, kinds, metric, h)
        cols = [m for m in models if m in g.columns]
        naive, vals, vol = vs_naive(g, cols)
        if not len(naive):
            continue
        sk = 1 - vals / naive[:, None]
        red = naive[:, None] - vals
        med, beats = np.median(sk, axis=0), (sk > 0).mean(axis=0) * 100
        rp, rv = col_median(red, ~vol), col_median(red, vol)
        out[h] = {
            m: {
                "skill": med[j],
                "beats": beats[j],
                "red_price": rp[j],
                "red_volume": rv[j],
                "n": len(naive),
            }
            for j, m in enumerate(cols)
        }
    return out


@app.callback(
    Output("lb-kpis", "children"),
    Output("lb-note", "children"),
    Output("lb-caption", "children"),
    Output("lb-table", "children"),
    Output("lb-horizons", "figure"),
    Output("lb-heat-caption", "children"),
    Output("lb-heatmap", "figure"),
    Input("lb-split", "value"),
    Input("lb-kinds", "value"),
    Input("lb-families", "value"),
    Input("lb-single", "value"),
    Input("lb-metric", "value"),
    Input("lb-horizon", "value"),
)
def update_leaderboard(split, kinds, families, single, metric, horizon):
    d = data()
    metric, horizon = metric or "MAE", horizon or "all"
    models = [
        m
        for m in default_models(d["models"], bool(single))
        if model_family(m) in (families or [])
    ]
    if not kinds or not models:
        e = empty("Nothing selected")
        return (
            [],
            "Select at least one target and one model family.",
            "",
            None,
            e,
            "",
            e,
        )
    # models still running (results for under half the groups) would shrink the common set; set them aside
    _, coverage, n_groups, _, _ = leaderboard(split, kinds, models, metric, horizon)
    most = max(coverage.values(), default=0)
    sparse = sorted(
        ((m, c) for m, c in coverage.items() if c < 0.5 * most), key=lambda x: x[1]
    )
    models = [m for m in models if m not in dict(sparse)]
    stats, coverage, n_groups, n_scored, naive_error = leaderboard(
        split, kinds, models, metric, horizon
    )
    if stats is None:
        e = empty("No common groups")
        return (
            [],
            "No country/target group has results for every selected model at this horizon.",
            "",
            None,
            e,
            "",
            e,
        )

    partial = sorted(
        ((m, c) for m, c in coverage.items() if c < n_groups), key=lambda x: x[1]
    )
    note = []
    if n_scored < n_groups:
        note.append(
            f"Scored on the {n_scored} of {n_groups} country/target groups where every model shown has "
            "results"
            + (
                " (missing: "
                + ", ".join(f"{model_label(m)} {int(c)}/{n_groups}" for m, c in partial)
                + ")"
                if partial
                else ""
            )
            + ". "
        )
    if sparse:
        note.append(
            "Not shown yet, too few results: "
            + ", ".join(f"{model_label(m)} ({int(c)}/{n_groups})" for m, c in sparse)
            + "."
        )

    hz_txt = "mean over horizons" if horizon == "all" else horizon_label(horizon)
    learned = {m: r for m, r in stats.items() if m != "naive"}
    top = next(iter(learned))
    most_wins = max(learned, key=lambda m: learned[m]["wins"])
    has_price, has_vol = any(k != "volume" for k in kinds), "volume" in kinds
    red_sub = " · ".join(
        x
        for x in [
            f"{learned[top]['red_price']:+.1f} EUR/MWh on prices" if has_price else "",
            f"{learned[top]['red_volume']:+.1f} MWh on volumes" if has_vol else "",
        ]
        if x
    )
    n_naive = int(stats["naive"]["wins"])
    kpis = [
        kpi(
            f"Best by median {metric} skill",
            model_label(top),
            f"{learned[top]['median'] * 100:+.1f} % {metric} vs naive · {red_sub}",
            "good",
        ),
        kpi(
            "Most often best",
            model_label(most_wins),
            f"lowest {metric} in {int(learned[most_wins]['wins'])} of "
            f"{n_scored} groups",
        ),
        kpi(
            "Naive is best in",
            f"{n_naive} group" + ("" if n_naive == 1 else "s"),
            "no model beats persistence there",
            "bad" if n_naive else "good",
        ),
        kpi(
            "Naive error (median group)",
            " · ".join(
                x
                for x in [
                    f"{naive_error['price']:.0f} EUR/MWh" if has_price else "",
                    f"{naive_error['volume']:.0f} MWh" if has_vol else "",
                ]
                if x
            ),
            f"{metric}, {hz_txt}, {split}",
        ),
    ]
    caption = (
        f"{metric} skill = 1 − {metric} / {metric}(naive) per country/target group ({hz_txt}); "
        f"+10 % means a {metric} 10 % below persistence. ‘Worst 10 %’ is the 10th percentile of skill over groups "
        f"(how badly it does in its weaker groups). ‘Reduction’ is the median of {metric}(naive) − {metric}(model) "
        f"in EUR/MWh (price groups) or MWh (volume groups). ‘Best in’ counts groups where the model has the lowest "
        f"{metric} of those shown."
    )

    vmax = max(0.05, max(abs(r["median"]) for r in learned.values()))
    heads = [
        "#",
        "Model",
        "Family",
        f"Median {metric} skill",
        "Mean",
        "Worst 10 %",
        "Beats naive",
    ]
    heads += ["Reduction, prices"] if has_price else []
    heads += ["Reduction, volumes"] if has_vol else []
    heads += ["Best in", "Mean rank"]
    head = html.Tr(
        [html.Th(h, className="w-bar" if h.startswith("Median") else "") for h in heads]
    )
    body = []
    for i, (m, r) in enumerate(stats.items(), 1):
        w = abs(r["median"]) / vmax * 100
        bar = html.Div(
            [
                html.Div(
                    className="bar " + ("pos" if r["median"] >= 0 else "neg"),
                    style={"width": f"{w * 0.75:.0f}%"},
                ),
                html.Span(f"{r['median'] * 100:+.1f} %"),
            ],
            className="bar-cell",
        )
        is_ref = m == "naive"
        cells = [
            html.Td("" if is_ref else i),
            html.Td(
                html.Span(
                    model_label(m),
                    className="model-name",
                    style={"borderColor": model_color(m)},
                )
            ),
            html.Td(FAMILIES.get(model_family(m), ""), className="muted"),
            html.Td(bar),
            html.Td(f"{r['mean'] * 100:+.1f} %"),
            html.Td(
                "–" if is_ref else f"{r['p10'] * 100:+.1f} %",
                className="" if r["p10"] >= 0 else "neg-text",
            ),
            html.Td("–" if is_ref else f"{r['beats']:.0f} %"),
        ]
        cells += (
            [html.Td("–" if is_ref else fmt_red(r["red_price"], "EUR/MWh"))]
            if has_price
            else []
        )
        cells += (
            [html.Td("–" if is_ref else fmt_red(r["red_volume"], "MWh"))]
            if has_vol
            else []
        )
        cells += [html.Td(int(r["wins"])), html.Td(f"{r['rank']:.1f}")]
        body.append(html.Tr(cells, className="ref" if is_ref else ""))
    table = html.Table([html.Thead(head), html.Tbody(body)], className="table nowrap")

    # skill by horizon across all countries: 15-min horizons left, hourly right
    prof = horizon_profile(split, kinds, list(learned), metric)
    hs15 = [h for h in prof if h.endswith("min")]
    hs60 = [h for h in prof if h.endswith("h")]
    from plotly.subplots import make_subplots

    hfig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.68, 0.32],
        horizontal_spacing=0.04,
        subplot_titles=("15-min countries", "hourly countries"),
    )
    shown = list(
        {model_family(m_): m_ for m_ in list(learned)[::-1]}.values()
    )  # best model of each family
    for mdl in learned:
        vis = True if mdl in shown else "legendonly"
        for col, hs in ((1, hs15), (2, hs60)):
            if not hs:
                continue
            y = [prof[h][mdl]["skill"] if mdl in prof[h] else None for h in hs]
            cd = [
                [prof[h][mdl][k] for k in ("red_price", "red_volume", "beats", "n")]
                if mdl in prof[h]
                else [None] * 4
                for h in hs
            ]
            hfig.add_trace(
                go.Scatter(
                    x=[horizon_label(h) for h in hs],
                    y=y,
                    name=model_label(mdl),
                    legendgroup=mdl,
                    showlegend=col == 1,
                    mode="lines+markers",
                    visible=vis,
                    line=dict(color=model_color(mdl), width=2.4),
                    marker=dict(size=7),
                    customdata=cd,
                    hovertemplate=f"<b>{model_label(mdl)}</b> %{{x}}<br>median {metric} skill %{{y:+.1%}}<br>"
                    "median reduction: %{customdata[0]:+.1f} EUR/MWh (prices), %{customdata[1]:+.1f} MWh "
                    "(volumes)<br>beats naive in %{customdata[2]:.0f} % of %{customdata[3]} groups"
                    "<extra></extra>",
                ),
                row=1,
                col=col,
            )
    for col in (1, 2):
        hfig.add_hline(
            y=0, line=dict(color="#c8cad6", dash="dash", width=1), row=1, col=col
        )
    hfig.update_yaxes(
        tickformat="+.0%", title=f"median {metric} skill vs naive", row=1, col=1
    )
    styled(
        hfig,
        440,
        legend={"orientation": "h", "y": -0.15, "x": 0, "yanchor": "top"},
        hovermode="closest",
        margin={"l": 10, "r": 20, "t": 40, "b": 40},
    )
    hfig.update_annotations(font=dict(size=12, color=MUTED))

    # heatmap over every group (blank where a model has no result), columns in ranking order
    g = group_values(split, kinds, metric, horizon)
    cols = ["naive", *[m for m in learned if m in g.columns]]
    vals = g.select(cols).to_numpy()  # missing results are NaN
    skill = 1 - vals / vals[:, [0]]
    best = [None if np.isnan(r).all() else cols[int(np.nanargmin(r))] for r in vals]
    groups = list(zip(g["Country"], g["Kind"]))
    ylab = [f"{cc} · {KINDS[k]}" for cc, k in groups]
    xlab = [model_label(m) for m in cols]
    text = [
        [
            ("★ " if best[i] == m else "") + ("" if missing(v) else f"{v:.2f}")
            for m, v in zip(cols, row)
        ]
        for i, row in enumerate(skill)
    ]
    custom = [
        [[cc, k, m, vals[i, j], vals[i, 0], UNITS[k]] for j, m in enumerate(cols)]
        for i, (cc, k) in enumerate(groups)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=skill,
            x=xlab,
            y=ylab,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            customdata=custom,
            zmid=0,
            zmin=-0.3,
            zmax=0.3,
            xgap=1,
            ygap=1,
            colorscale=[
                [0, "#c0392b"],
                [0.35, "#7a2f35"],
                [0.5, "#262a38"],
                [0.65, "#22577a"],
                [1, "#4ea8de"],
            ],
            colorbar=dict(
                title=dict(text=f"{metric} skill", side="right"),
                thickness=12,
                len=0.5,
                y=1,
                yanchor="top",
            ),
            hovertemplate=f"<b>%{{customdata[0]}} · %{{y}}</b><br>%{{x}}<br>{metric} skill %{{z:+.1%}}<br>"
            f"{metric} %{{customdata[3]:.2f}} vs naive %{{customdata[4]:.2f}} %{{customdata[5]}}"
            "<extra>click to open</extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top", tickangle=-40)
    styled(
        fig, height=max(420, 22 * len(ylab) + 160), margin=dict(l=10, r=10, t=140, b=10)
    )
    heat_caption = (
        f"Each cell is one model on one country/target, coloured by {metric} skill ({hz_txt}; blue beats "
        "naive, red loses). ★ marks the best model in the row. Click a cell to open that country in the "
        "explorer."
    )
    return kpis, note, caption, table, hfig, heat_caption, fig


@app.callback(
    Output("tabs", "value"),
    Output("cx-country", "value"),
    Output("cx-kind", "value"),
    Output("cx-models", "value"),
    Input("lb-heatmap", "clickData"),
    State("cx-models", "value"),
    prevent_initial_call=True,
)
def open_from_heatmap(click, models):
    if not click:
        return no_update, no_update, no_update, no_update
    cc, kind, model = click["points"][0]["customdata"][:3]
    models = list(models or [])
    if model not in models:
        models = [model, *models]
    return "country", cc, kind, models


# ------------------------------------------------------------------------------------------------ country explorer
@app.callback(
    Output("cx-kind", "options"),
    Output("cx-horizon-dd", "options"),
    Output("cx-horizon-dd", "value"),
    Input("cx-country", "value"),
    State("cx-horizon-dd", "value"),
)
def country_options(cc, horizon):
    kinds = kinds_for(cc)
    kind_opts = [
        {"label": v, "value": k, "disabled": k not in kinds} for k, v in KINDS.items()
    ]
    hs = horizons_for(cc)
    return (
        kind_opts,
        [{"label": horizon_label(h), "value": h} for h in hs],
        horizon if horizon in hs else hs[len(hs) // 2] if hs else None,
    )


@app.callback(
    Output("cx-kpis", "children"),
    Output("cx-horizon", "figure"),
    Output("cx-htable", "children"),
    Input("cx-country", "value"),
    Input("cx-kind", "value"),
    Input("cx-split", "value"),
    Input("cx-models", "value"),
    Input("cx-metric", "value"),
    Input("cx-view", "value"),
)
def country_summary(cc, kind, split, models, metric, view):
    metric = metric or "MAE"
    sel = data()["metrics"].filter(
        (pl.col("Country") == cc)
        & (pl.col("Kind") == kind)
        & (pl.col("Split") == split)
    )
    if sel.is_empty():
        msg = f"No {KINDS[kind].lower()} results for {cc}" + (
            " (single-priced: short = long)" if kind == "short" else ""
        )
        return [], empty(msg), None
    wide = (
        sel.group_by("Horizon", "Model")
        .agg(pl.col(metric).mean())
        .drop_nulls(metric)
        .pivot(on="Model", index="Horizon", values=metric)
    )
    hz = sorted(wide["Horizon"].to_list(), key=horizon_minutes)
    wide = wide.sort(
        pl.col("Horizon").replace_strict(hz, list(range(len(hz))))
    )  # rows in horizon order
    # model -> metric per horizon (NaN where missing), columns in alphabetical order
    val = {
        c: wide[c].to_numpy() for c in sorted(c for c in wide.columns if c != "Horizon")
    }
    skill = {c: 1 - v / val["naive"] for c, v in val.items()}
    avg = {c: float(np.nanmean(v)) for c, v in val.items()}
    best = min((c for c in avg if c != "naive"), key=avg.get)
    shown = [x for x in (models or []) if x in val]
    sel_best = min(shown, key=lambda x: avg[x]) if shown else None
    unit = UNITS[kind]

    def sub(x):
        return f"{metric} {avg[x]:.1f} {unit} · {(1 - avg[x] / avg['naive']) * 100:+.1f} % · {avg['naive'] - avg[x]:+.1f} {unit}"

    kpis = [
        kpi(
            f"Naive {metric}",
            f"{avg['naive']:.1f} {unit}",
            "persistence, mean over horizons",
        ),
        kpi(
            "Best of all models",
            model_label(best),
            sub(best),
            "good" if avg[best] < avg["naive"] else "bad",
        ),
    ]
    if sel_best:
        kpis.append(kpi("Best of those shown", model_label(sel_best), sub(sel_best)))
    kpis.append(
        kpi(
            "Models with results",
            str(len(avg) - 1),
            f"{cc} · {KINDS[kind].lower()} · {split}",
        )
    )

    fig = go.Figure()
    x = [horizon_label(h) for h in hz]
    if view == "abs":
        fig.add_trace(
            go.Scatter(
                x=x,
                y=val["naive"],
                name="Naive",
                mode="lines+markers",
                line=dict(color=model_color("naive"), width=2, dash="dot"),
                hovertemplate=f"%{{x}}: {metric} %{{y:.2f}} {unit}<extra>Naive</extra>",
            )
        )
    else:
        fig.add_hline(
            y=0,
            line=dict(color="#c8cad6", dash="dash", width=1),
            annotation_text="naive",
            annotation_position="bottom right",
            annotation_font_color=MUTED,
        )
    for mdl in shown:
        red = val["naive"] - val[mdl]
        cd = np.column_stack([val[mdl], red, skill[mdl]])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=val[mdl] if view == "abs" else skill[mdl],
                name=model_label(mdl),
                mode="lines+markers",
                line=dict(color=model_color(mdl), width=2.5),
                marker=dict(size=7),
                customdata=cd,
                hovertemplate=f"%{{x}}: {metric} %{{customdata[0]:.2f}} {unit} · %{{customdata[2]:+.1%}} · "
                f"%{{customdata[1]:+.2f}} {unit} vs naive<extra>{model_label(mdl)}</extra>",
            )
        )
    if view == "abs":
        fig.update_yaxes(title=f"{metric} [{unit}]", rangemode="tozero")
    else:
        fig.update_yaxes(title=f"{metric} skill vs naive", tickformat="+.0%")
    fig = styled(
        fig,
        380,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.12, "x": 0, "yanchor": "top"},
    )

    # table: every horizon, metric and change vs naive
    head = html.Tr(
        [html.Th("Model")] + [html.Th(horizon_label(h)) for h in hz] + [html.Th("Mean")]
    )
    rows = [
        html.Tr(
            [html.Td("Naive", className="muted")]
            + [html.Td(f"{v:.1f}", className="muted") for v in val["naive"]]
            + [html.Td(f"{avg['naive']:.1f}", className="muted")],
            className="ref",
        )
    ]
    for mdl in sorted(shown, key=lambda x: avg[x]):
        cells = [
            html.Td(
                html.Span(
                    model_label(mdl),
                    className="model-name",
                    style={"borderColor": model_color(mdl)},
                )
            )
        ]
        for j in [*range(len(hz)), None]:
            v = avg[mdl] if j is None else val[mdl][j]
            n = avg["naive"] if j is None else val["naive"][j]
            sk = 1 - v / n
            cells.append(
                html.Td(
                    [
                        html.Span(f"{v:.1f} "),
                        html.Span(
                            f"{sk * 100:+.1f} %",
                            className="good-text" if sk > 0 else "neg-text",
                        ),
                    ]
                )
            )
        rows.append(html.Tr(cells))
    tbl = html.Div(
        [
            html.P(
                f"{metric} in {unit}, with the change vs naive (green = lower error).",
                className="caption",
            ),
            html.Table([html.Thead(head), html.Tbody(rows)], className="table compact"),
        ]
    )
    return kpis, fig, tbl


def _jump_targets(cc, kind, horizon, split):
    df, _ = predictions(cc, f"{SERIES[kind]}_{horizon}", ["naive"], split)
    return df


@app.callback(
    Output("cx-start", "min_date_allowed"),
    Output("cx-start", "max_date_allowed"),
    Output("cx-start", "date"),
    Input("cx-country", "value"),
    Input("cx-kind", "value"),
    Input("cx-split", "value"),
    Input("cx-horizon-dd", "value"),
    Input("cx-jump-start", "n_clicks"),
    Input("cx-jump-spike", "n_clicks"),
    Input("cx-jump-end", "n_clicks"),
    State("cx-window", "value"),
    State("cx-start", "date"),
)
def series_range(cc, kind, split, horizon, _a, _b, _c, window, date):
    if not horizon:
        return no_update, no_update, no_update
    df = _jump_targets(cc, kind, horizon, split)
    if df is None or df.is_empty():
        return None, None, None
    first, last = df["time"].min(), df["time"].max()
    lo, hi = first.date(), last.date()
    try:
        trig = ctx.triggered_id
    except Exception:  # noqa: BLE001  (called outside a request)
        trig = None
    if trig == "cx-jump-start":
        start = lo
    elif trig == "cx-jump-spike":
        worst = df["time"][(df["naive"] - df["actual"]).abs().arg_max()]
        start = shift_days(worst, -window / 2).date()
    elif trig == "cx-jump-end" or not date:
        start = shift_days(last, -window).date()
    else:
        start = datetime.fromisoformat(date).date()
    start = min(max(start, lo), hi)
    return str(lo), str(hi), str(start)


@app.callback(
    Output("cx-series", "figure"),
    Output("cx-hourly", "figure"),
    Input("cx-country", "value"),
    Input("cx-kind", "value"),
    Input("cx-split", "value"),
    Input("cx-models", "value"),
    Input("cx-horizon-dd", "value"),
    Input("cx-start", "date"),
    Input("cx-window", "value"),
    Input("cx-metric", "value"),
)
def country_series(cc, kind, split, models, horizon, start, window, metric="MAE"):
    if not horizon or kind not in kinds_for(cc):
        msg = f"No {KINDS[kind].lower()} for {cc}"
        return empty(msg, 420), empty(msg, 380)
    target = f"{SERIES[kind]}_{horizon}"
    df, have = predictions(cc, target, ["naive", *(models or [])], split)
    if df is None or df.is_empty():
        return empty("No predictions saved for this country", 420), empty(
            "No predictions", 380
        )
    unit = UNITS[kind]
    tz = timezones().get(cc, "UTC")

    # time series window
    t0 = (
        datetime.fromisoformat(start).replace(tzinfo=ZoneInfo(tz))
        if start
        else shift_days(df["time"].max(), -window)
    )
    w = df.filter((pl.col("time") >= t0) & (pl.col("time") < shift_days(t0, window)))
    x = w["time"].to_list()  # aware datetimes, so the axis shows local time
    ts = go.Figure()
    ts.add_trace(
        go.Scatter(
            x=x,
            y=w["actual"].to_numpy(),
            name="Actual",
            line=dict(color="#ffffff", width=2.2),
        )
    )
    for mdl in have:
        dash_ = "dot" if mdl == "naive" else "solid"
        ts.add_trace(
            go.Scatter(
                x=x,
                y=w[mdl].to_numpy(),
                name=model_label(mdl),
                line=dict(color=model_color(mdl), width=1.6, dash=dash_),
                opacity=0.9,
            )
        )
    ts.update_yaxes(title=f"{KINDS[kind]} [{unit}]")
    ts.update_xaxes(title=None)
    styled(
        ts,
        480,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.2, "x": 0, "yanchor": "top"},
    )
    ts.update_layout(
        title=dict(
            text=f"{cc} · {KINDS[kind]} · forecast made {horizon_label(horizon)} ahead · "
            f"delivery period in local time ({tz})",
            font=dict(size=13, color=MUTED),
            x=0.01,
        )
    )

    # error by hour of day, whole split
    by_hour = df.with_columns(hour=pl.col("time").dt.hour()).group_by("hour")
    hourly = go.Figure()
    for mdl in have:
        err = pl.col(mdl) - pl.col("actual")
        e = by_hour.agg(
            (err.abs().mean() if metric != "RMSE" else (err**2).mean().sqrt()).alias(
                "e"
            )
        ).sort("hour")
        hourly.add_trace(
            go.Scatter(
                x=e["hour"].to_numpy(),
                y=e["e"].to_numpy(),
                name=model_label(mdl),
                mode="lines",
                line=dict(
                    color=model_color(mdl),
                    width=2,
                    dash="dot" if mdl == "naive" else "solid",
                ),
            )
        )
    hourly.update_xaxes(title="hour of day (local)", dtick=3, range=[-0.5, 23.5])
    hourly.update_yaxes(title=f"{metric or 'MAE'} [{unit}]", rangemode="tozero")
    return ts, styled(
        hourly,
        400,
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.3, "x": 0, "yanchor": "top"},
    )


# ------------------------------------------------------------------------------------------------ SHAP
def _shares(cc, kind, pos, model="nn"):
    t = shap_table("category", cc, model)
    cols = shap_targets(cc, kind, model)
    if t is None or not cols:
        return None
    use = cols if pos == "avg" else [cols[0] if pos == "first" else cols[-1]]
    v = t.select("Category", pl.mean_horizontal(use).alias("v"))
    return dict(zip(v["Category"].to_list(), (v["v"] / v["v"].sum()).to_list()))


def union(keys):
    """Keys of several dicts: their common order when they all agree, else sorted."""
    lists = [list(k) for k in keys]
    return (
        lists[0] if all(x == lists[0] for x in lists) else sorted(set().union(*lists))
    )


def ordered_categories(cats):
    return [c for c in CATEGORIES if c in cats] + [
        c for c in cats if c not in CATEGORIES
    ]


@app.callback(
    Output("sh-countries", "figure"),
    Input("sh-kind", "value"),
    Input("sh-hpos", "value"),
    Input("sh-sort", "value"),
    Input("sh-model", "value"),
)
def shap_countries_fig(kind, pos, sort_by, model="nn"):
    shares = {}
    for cc in shap_countries(model):
        if kind not in kinds_for(
            cc
        ):  # single-priced: the short price is the long price
            continue
        s = _shares(cc, kind, pos, model)
        if s is not None:
            shares[cc] = s
    if not shares:
        return empty("No SHAP results")
    ccs = sorted(shares, key=lambda cc: shares[cc].get(sort_by, 0.0))
    fig = go.Figure()
    for c in ordered_categories(union(s.keys() for s in shares.values())):
        fig.add_trace(
            go.Bar(
                y=ccs,
                x=np.array([shares[cc].get(c, 0.0) for cc in ccs]) * 100,
                name=cat_label(c),
                orientation="h",
                marker=dict(color=cat_color(c), line=dict(width=0)),
                customdata=ccs,
                hovertemplate=f"<b>%{{y}}</b> · {cat_label(c)}: %{{x:.1f}} %<extra></extra>",
            )
        )
    fig.update_xaxes(range=[0, 100], ticksuffix=" %")
    pos_txt = {
        "avg": "averaged over horizons",
        "first": "shortest horizon",
        "last": "longest horizon",
    }[pos]
    styled(
        fig,
        max(380, 26 * len(ccs) + 120),
        barmode="stack",
        bargap=0.25,
        legend={
            "orientation": "h",
            "y": -0.06,
            "x": 0,
            "yanchor": "top",
            "traceorder": "normal",
        },
        margin={"l": 10, "r": 20, "t": 30, "b": 20},
    )
    fig.update_layout(
        title=dict(
            text=f"{SHAP_MODELS[model][0]} · {KINDS[kind]} · {pos_txt} · sorted by "
            f"{cat_label(sort_by).lower()}",
            font=dict(size=13, color=MUTED),
            x=0.01,
        )
    )
    return fig


@app.callback(
    Output("sh-country", "value"),
    Input("sh-countries", "clickData"),
    prevent_initial_call=True,
)
def shap_pick_country(click):
    return click["points"][0]["customdata"] if click else no_update


@app.callback(
    Output("sh-horizon", "options"),
    Output("sh-horizon", "value"),
    Input("sh-country", "value"),
    Input("sh-kind", "value"),
    State("sh-horizon", "value"),
)
def shap_horizon_options(cc, kind, current):  # horizons are the same for every model
    cols = shap_targets(cc, kind) if cc else []
    opts = [{"label": "Average of horizons", "value": "avg"}] + [
        {"label": horizon_label(c.rsplit("_", 1)[1]), "value": c.rsplit("_", 1)[1]}
        for c in cols
    ]
    vals = [o["value"] for o in opts]
    return opts, current if current in vals else "avg"


@app.callback(
    Output("sh-features", "figure"),
    Output("sh-beeswarm", "figure"),
    Output("sh-profile", "figure"),
    Output("sh-stability", "children"),
    Input("sh-country", "value"),
    Input("sh-kind", "value"),
    Input("sh-horizon", "value"),
    Input("sh-topn", "value"),
    Input("sh-model", "value"),
)
def shap_detail(cc, kind, horizon, top_n, model="nn"):
    cols = shap_targets(cc, kind, model) if cc else []
    if not cols or not horizon:
        e = empty("No SHAP results for this selection")
        return e, e, e, None
    use = cols if horizon == "avg" else [f"{SERIES[kind]}_{horizon}"]
    hz_txt = "average of horizons" if horizon == "avg" else horizon_label(horizon)
    single = kind == "short" and kind not in kinds_for(cc)
    title = f"{SHAP_MODELS[model][0]} · {cc} · {KINDS[kind]} · {hz_txt}" + (
        " · single-priced (same as long)" if single else ""
    )

    # top features
    top = (
        shap_table("mean_abs", cc, model)
        .select("Feature", "Category", pl.mean_horizontal(use).alias("imp"))
        .sort("imp", descending=True, maintain_order=True)
        .head(top_n)
        .reverse()
    )
    feat = go.Figure()
    for c in dict.fromkeys(top["Category"].to_list()):
        sel = top.filter(pl.col("Category") == c)
        feat.add_trace(
            go.Bar(
                x=sel["imp"].to_numpy(),
                y=sel["Feature"].to_list(),
                orientation="h",
                name=cat_label(c),
                marker_color=cat_color(c),
                hovertemplate="%{y}<br>mean |SHAP| %{x:.4f}<extra>"
                + cat_label(c)
                + "</extra>",
            )
        )
    feat.update_yaxes(categoryorder="array", categoryarray=top["Feature"].to_list())
    feat.update_xaxes(title="mean |SHAP|")
    styled(
        feat,
        max(380, 24 * top.height + 120),
        barmode="stack",
        showlegend=False,
        title=dict(text=title, font=dict(size=13, color=MUTED), x=0.01),
    )

    # beeswarm: signed SHAP per row (mean over the chosen horizons), coloured by the feature's rank at t
    names, values = shap_features(cc, model)
    raws = [r for r in (shap_raw(cc, t, model) for t in use) if r is not None]
    if raws:
        raw = np.mean(raws, axis=0)
        k = min(top_n, 15)
        order = np.argsort(np.abs(raw).mean(0))[::-1][:k][::-1]
        rng = np.random.default_rng(0)
        bees = go.Figure()
        xs, ys, cs, hs = [], [], [], []
        for j, fi in enumerate(order):
            v = values[:, fi]
            s = pl.Series(v, dtype=pl.Float64).fill_nan(None)
            rank = (
                s.rank("average") / s.count()
            ).to_numpy()  # percentile rank, NaN stays NaN
            x = raw[:, fi]
            # jitter scaled by local density so dense regions spread out
            hist, edges = np.histogram(x, bins=40)
            dens = hist[np.clip(np.digitize(x, edges) - 1, 0, len(hist) - 1)] / max(
                hist.max(), 1
            )
            xs.append(x)
            ys.append(j + rng.uniform(-1, 1, len(x)) * 0.38 * dens)
            cs.append(rank)
            hs.append(np.full(len(x), names[fi]))
        bees.add_trace(
            go.Scattergl(
                x=np.concatenate(xs),
                y=np.concatenate(ys),
                mode="markers",
                customdata=np.concatenate(hs),
                marker=dict(
                    size=4,
                    color=np.concatenate(cs),
                    colorscale="RdBu_r",
                    cmin=0,
                    cmax=1,
                    opacity=0.75,
                    colorbar=dict(
                        title=dict(text="input value", side="right"),
                        tickvals=[0, 1],
                        ticktext=["low", "high"],
                        thickness=10,
                        len=0.6,
                    ),
                ),
                hovertemplate="%{customdata}<br>SHAP %{x:+.3f}<extra></extra>",
                showlegend=False,
            )
        )
        bees.add_vline(x=0, line=dict(color=MUTED, width=1))
        bees.update_yaxes(
            tickvals=list(range(len(order))),
            ticktext=[names[i] for i in order],
            range=[-0.6, len(order) - 0.4],
        )
        bees.update_xaxes(title="SHAP value (pushes forecast down ← → up)")
        styled(bees, max(380, 24 * min(top_n, 15) + 120))
    else:
        bees = empty("No per-row SHAP values saved")

    # horizon profile
    t = shap_table("category", cc, model)
    sh = t.select(
        "Category", *(pl.col(c) / pl.col(c).sum() for c in cols)
    )  # share of each horizon's total
    prof = go.Figure()
    x = [horizon_label(c.rsplit("_", 1)[1]) for c in cols]
    rows = {r[0]: np.array(r[1:], dtype=float) for r in sh.iter_rows()}
    for c in ordered_categories(rows):
        prof.add_trace(
            go.Bar(
                x=x,
                y=rows[c] * 100,
                name=cat_label(c),
                marker_color=cat_color(c),
                hovertemplate=f"{cat_label(c)}: %{{y:.1f}} %<extra>%{{x}}</extra>",
            )
        )
    prof.update_yaxes(title="share of mean |SHAP| (%)", ticksuffix=" %", range=[0, 100])
    styled(
        prof,
        420,
        barmode="stack",
        bargap=0.3,
        legend={"orientation": "h", "y": -0.1, "x": 0, "yanchor": "top"},
    )

    stab = shap_table("stability", cc, model)
    info = shap_info(cc, model)
    method = f"{SHAP_MODELS[model][0]}: {SHAP_MODELS[model][2]}, {info.get('explained_rows', '?')} {info.get('split', '')} rows"
    badge = html.Span(method)
    if stab is not None:
        s = (
            stab.filter(pl.col("Target").is_in(use))
            .select("rank_corr_mean", "top10_overlap_mean")
            .mean()
        )
        corr, overlap = (np.nan if missing(v) else v for v in s.row(0))
        badge = html.Span(
            [
                method,
                html.Br(),
                f"Seed stability: rank corr {corr:.2f}, top-10 overlap {overlap * 100:.0f} %",
            ]
        )
    return feat, bees, prof, badge


def ranks(x, descending=False):
    """Average ranks of a 1-D array (1 = smallest, or largest when descending)."""
    return pl.Series(x).rank("average", descending=descending).to_numpy()


@app.callback(
    Output("sh-cmp-inputs", "figure"),
    Output("sh-cmp-cats", "figure"),
    Output("sh-cmp-table", "children"),
    Input("sh-country", "value"),
    Input("sh-kind", "value"),
    Input("sh-horizon", "value"),
    Input("sh-topn", "value"),
)
def shap_compare(cc, kind, horizon, top_n):
    models = [m for m in shap_models() if shap_targets(cc, kind, m)] if cc else []
    if len(models) < 2 or not horizon:
        e = empty("Fewer than two models explained for this country")
        return e, e, None
    shares, cats = {}, {}
    for m in models:
        use = (
            shap_targets(cc, kind, m)
            if horizon == "avg"
            else [f"{SERIES[kind]}_{horizon}"]
        )
        t = (
            shap_table("mean_abs", cc, m)
            .select("Feature", "Category", pl.mean_horizontal(use).alias("v"))
            .with_columns(share=pl.col("v") / pl.col("v").sum())
        )
        shares[m] = dict(zip(t["Feature"].to_list(), t["share"].to_list()))
        cats[m] = dict(
            t.group_by("Category")
            .agg(pl.col("share").sum())
            .sort("Category")
            .iter_rows()
        )
    feats = union(s.keys() for s in shares.values())
    S = np.array(
        [[shares[m].get(f, 0.0) for m in models] for f in feats]
    )  # input x model, 0 where absent
    k = min(top_n or 15, 25)
    order = np.argsort(-S.mean(axis=1), kind="stable")[:k]
    R = np.column_stack([ranks(S[:, j], descending=True) for j in range(len(models))])[
        order
    ]
    labels = [SHAP_MODELS[m][0] for m in models]
    z = S[order] * 100
    text = [[f"#{int(r)}" for r in row] for row in R]
    hz_txt = "average of horizons" if horizon == "avg" else horizon_label(horizon)
    f1 = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=[feats[i] for i in order],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorscale=[[0, "#20232f"], [0.15, "#22577a"], [1, "#4ea8de"]],
            zmin=0,
            xgap=2,
            ygap=2,
            colorbar=dict(
                title=dict(text="share %", side="right"), thickness=10, len=0.7
            ),
            hovertemplate="%{y}<br>%{x}: %{z:.1f} % of total, rank %{text}<extra></extra>",
        )
    )
    f1.update_yaxes(autorange="reversed")
    f1.update_xaxes(side="top")
    styled(
        f1,
        max(380, 24 * len(order) + 110),
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        title=dict(
            text=f"{cc} · {KINDS[kind]} · {hz_txt}",
            font=dict(size=13, color=MUTED),
            x=0.01,
            y=0.99,
        ),
    )

    f2 = go.Figure()
    for c in ordered_categories(union(x.keys() for x in cats.values())):
        f2.add_trace(
            go.Bar(
                y=labels,
                x=np.array([cats[m].get(c, 0.0) for m in models]) * 100,
                name=cat_label(c),
                orientation="h",
                marker_color=cat_color(c),
                hovertemplate=f"%{{y}} · {cat_label(c)}: %{{x:.1f}} %<extra></extra>",
            )
        )
    f2.update_xaxes(range=[0, 100], ticksuffix=" %")
    f2.update_yaxes(autorange="reversed")
    styled(
        f2,
        max(380, 24 * len(order) + 110),
        barmode="stack",
        bargap=0.3,
        legend={"orientation": "h", "y": -0.08, "x": 0, "yanchor": "top"},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )

    head = html.Tr([html.Th("")] + [html.Th(lab) for lab in labels])
    body = []
    top10 = [
        set(np.argsort(-S[:, j], kind="stable")[:10]) for j in range(len(models))
    ]  # ties: first listed
    for a, a_ in enumerate(models):
        cells = [html.Td(SHAP_MODELS[a_][0], className="muted")]
        for b, b_ in enumerate(models):
            if a_ == b_:
                cells.append(html.Td("–", className="muted"))
                continue
            rc = np.corrcoef(ranks(S[:, a]), ranks(S[:, b]))[0, 1]  # Spearman
            ov = len(top10[a] & top10[b])
            cells.append(
                html.Td(
                    f"ρ {rc:.2f} · top-10 {ov}/10",
                    className="good-text"
                    if rc >= 0.8
                    else ("neg-text" if rc < 0.5 else ""),
                )
            )
        body.append(html.Tr(cells))
    tbl = html.Div(
        [
            html.P(
                "Agreement between models (green ρ ≥ 0.8, red ρ < 0.5).",
                className="caption",
            ),
            html.Table([html.Thead(head), html.Tbody(body)], className="table compact"),
        ]
    )
    return f1, f2, tbl


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"aFRR results viewer: http://localhost:{port}")
    app.run(debug=False, host="127.0.0.1", port=port)

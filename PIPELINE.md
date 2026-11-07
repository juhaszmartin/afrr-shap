# aFRR imbalance forecasting with SHAP: what the pipeline does

This document describes the current state of the code: what data goes in, how it is processed, how the model is trained and evaluated, which assumptions are built in, and which issues are known but not yet fixed. File and constant names refer to the code, so it can be checked against it.

## 1. Goal

For each European country in the data, forecast the imbalance price (Long and Short) and the imbalance volume a short time ahead, then use SHAP to explain which inputs drive the forecast. The countries are later compared in groups of similar market design, not one against another.

| Resolution | Countries | Horizons |
|---|---|---|
| 15 min | AT, BE, CH, DE, ES, FI, GR, HR, HU, PL, RO, SK | t+15, +30, +45, +60, +75 min |
| 1 h | BA, FR, LV, PT, RS | t+1 h, t+2 h |

- The hourly countries settled imbalances hourly (FR every 30 min) **until 2024**. Their 15-min inputs are averaged to the hour, except imbalance volumes, which are summed (§4 step 4).
- **TODO: FR, LV and RS settle every 15 min from January 2025, and PT from April 2025**, so for them the test year is no longer hourly. The pipeline still runs them hourly, which has two consequences (§9.4):
  - their 2025 price target is the mean of four quarter-hour settlement prices, which is smoother than the hourly prices they were trained on;
  - RS 2025 volumes look inflated by the summation.
- **LT is not modelled.** ENTSO-E has no LT imbalance prices or volumes after 2024-09-30, so it has no test year.
- **DE** has had a volume target since its volumes were filled from the API (§3.1), so it now has 15 targets like the other 15-min countries.

## 2. How to run

Run everything from the repository root with the project venv.

**One-off data steps.** These only need rerunning to extend or repair the data.
1. `scripts/get_fx_rates.py` downloads ECB daily euro reference rates into `fx_data/ecb_eur_rates.csv`.
2. `scripts/fill_entsoe_gaps.py` (needs `ENTSOE_API_KEY` in `.env`), see §3.1:
   - `fill [CC,..] [category,..]` downloads every gap in the raw ENTSO-E data from the API;
   - `verify [CC,..] [category,..]` compares 3 random stored days per series against the API;
   - `refresh CC category YEAR` replaces one stored year with the current API version.
3. `scripts/era5_download.py` (needs a CDS key in `~/.cdsapirc`) downloads ERA5 and writes `weather_data/` (§3). Without arguments it handles every country whose CSVs are missing; `era5_download.py germany,france` rebuilds the named countries.

Package versions are pinned in `requirements.txt` (Python 3.14.6). Superseded scripts and notebooks are in `legacy/`.

**Pipeline.** [run_pipeline.sh](run_pipeline.sh) runs these steps in order. Steps 1, 2 and 3 accept an optional country list, for example `DE,FR`.
1. `scripts/preprocess_all.py` writes `processed_data/X_{CC}.parquet`, `processed_data/y_{CC}.parquet` and `processed_data/gaps_{CC}.csv`. `scripts/get_entsoe_info.py` then writes [entsoe_availability.md](entsoe_availability.md), the input series each country ends up with.
2. `scripts/train_all_100_seeds.py` writes:
   - `results/NN_100_seeds_Huber_{CC}_data.csv`;
   - `results/all_countries_100_seeds_metrics.txt` (appended to when run on a subset);
   - `models/nn_seed{0-4}_{CC}.pt`, `models/nn_100seeds_best_{CC}.pt` and the scalers.
3. `scripts/run_shap_all.py` writes the SHAP outputs to `figures/` (§7).
4. `scripts/plot_shap_results.py` and `scripts/plot_sample_days.py` write the plots to `figures/` and `samples/`. `plot_results.py` (repository root) writes charts comparing every metric and its naive counterpart across countries and horizons to `figures/performance/`.

## 3. Data sources

**ENTSO-E Transparency Platform.** The raw files are stored as `entsoe_data/{category}/country={CC}/year={YYYY}/data.parquet`. Gap fills sit next to them as `fill.parquet` (§3.1). The categories are:

| Category | Content | Role |
|---|---|---|
| `imbalance_prices` | Long / Short imbalance price | **target** and target history |
| `imbalance_volumes` | net imbalance volume, MWh per settlement period | **target** and target history |
| `activated_balancing_energy_prices` | price per product and direction (aFRR, mFRR, RR, FCR × Up/Down) | feature |
| `contracted_reserve_amount`, `contracted_reserve_prices` (FI, PL), `procured_balancing_capacity` (HU) | reserve capacity | feature (none survives the filters) |
| `prices` | day-ahead price | feature (known in advance) |
| `load`, `load_forecast` | actual and day-ahead load | feature |
| `generation` | actual generation per production type | feature |
| `generation_forecast`, `wind_and_solar_forecast`, `intraday_wind_and_solar_forecast` | forecasts | feature (known in advance) |
| `import` | physical flow per border | feature; the total is rebuilt (§4 step 12) |
| `net_position` | day-ahead net position | feature (known in advance) |
| `generation_import` | a copy of `generation` and `import` | **not used** (`SKIP_CATEGORIES`) |

For DE, the ENTSO-E areas differ by category:
- imbalance prices use `DE`;
- day-ahead prices, imports and net position use `DE_LU`;
- imbalance volume is the sum over the four control areas (50Hertz, Amprion, TenneT, TransnetBW);
- balancing-energy prices are the mean over the control areas that publish them (Amprion publishes none).

**ERA5 weather.** The files are `weather_data/{country}_hourly_*.csv`. Each one is an hourly mean and standard deviation over a lat/lon box around the country, for 2 m temperature, solar radiation (ssrd), and wind speed at 10 m and 100 m. That gives 8 variables.
- The boxes are in `REGIONS` in `era5_download.py`. Neighbouring countries share a download region (DE/FR/BE/CH, central Europe, RO/GR, Iberia, the north, UA). There is one CDS request per region and 4 months, over a box covering the whole region, and each country is cropped from it locally.
- The CDS limits each request to 121,000 fields (variables × days × hours) and also limits how many requests can wait in its queue (each took 2.5–4.5 h in the queue).
- DE and FR were downloaded with this script. The other countries were downloaded earlier, one request per country and month, by `legacy/era5_eu.ipynb`. The script reproduces those CSVs to float32 precision (checked on the BE and CH boxes, which lie inside the DE/FR region: the largest difference is 0.0003 °C). So `weather_data/` can be rebuilt with the script alone.

**Currency.** ENTSO-E passes prices through in the currency the TSO reports in. `LOCAL_CURRENCY` in `preprocess_all.py` converts the following to EUR, using the most recent ECB rate on or before each timestamp. Every other country already reports in EUR.

| Country | Series | Currency | Until |
|---|---|---|---|
| HU | imbalance and balancing-energy prices; procured capacity price | HUF | whole period |
| PL | imbalance and balancing-energy prices; day-ahead price | PLN | whole period; day-ahead until 2019-11-19 23:00 UTC |
| RO | imbalance and balancing-energy prices; day-ahead price | RON | whole period; day-ahead until 2021-06-17 22:00 UTC |
| HR | imbalance prices | HRK | until 2022-12-31 23:00 UTC |

Prices are in EUR/MWh and volumes in MWh.

### 3.1 Gap fill and spot check against the API

The original bulk download of `entsoe_data/` was made by a script that is not in the repository. The data itself is not in the repository either (it is gitignored), and ENTSO-E data is revised after publication (see BA load below), so a fresh download can differ from the stored files.

**Fill (`fill_entsoe_gaps.py fill`).**
- It looks for every gap longer than 2 h in each stored series, from 2019 (or the 15-min cutoff year) to the end of 2025, and downloads it.
- Only timestamps that are missing are added. They go into `fill.parquet`; the original `data.parquet` is never changed. The loader reads both.
- The log is `entsoe_data/fill_log.json`.

What it filled:
- **Dec 31 in every year, all countries and categories.** The original downloader treated the end date as exclusive, so every yearly file stopped at Dec 30 23:45 UTC.
- **DE:** imbalance volumes, balancing-energy prices, imports and net position were missing entirely and are now complete (about 68k–213k rows each). About 10k rows of imbalance prices were also added.
- **GR:** imbalance prices and volumes from 2023 (35k rows). The 734-day target hole is gone; GR now trains on 2022-12 to 2024.
- **FI and PL:** balancing-energy prices for 2025 (about 170k rows each).
- **Smaller holes everywhere** (BA, FR, LT, LV, PT, RS, SK, …).

Not available from ENTSO-E (the API returns nothing):
- BA day-ahead price and net position;
- CH and RS net position;
- GR imbalance volumes 2019–2022;
- most of BA's balancing-price holes;
- **LT imbalance prices and volumes after 2024-09-30**;
- 26 one- or two-day generation-forecast holes, which fail because of an entsoe-py parsing error.

**Spot check (`verify`).** For 3 random days per country and category, every stored cell is compared with the API.
- **Overall, 97.8% of 100,541 cells are identical.**
- Real differences:
  - **BA load 2025 was wrong.** The stored year averaged 949 MW. The API now gives 1,190 MW, which is in line with 2019–2024 (1,100–1,360 MW); every earlier year matches exactly. The stored file was an early version that the TSO later corrected. It was replaced with `refresh BA load 2025`, and the original is kept as `data.parquet.bak` (the loader ignores `.bak`).
  - **BE generation** is split differently between production types, but the totals are unchanged (median change 0.0–0.3%). Not changed.
  - **BE wind/solar forecast 2025:** 8.4% of cells are revised slightly (mean 848.6 vs 848.8 MW). Not changed.
- Mismatches that come from the comparison itself, not from the data:
  - FR and SK balancing prices: several products share one timestamp, type and direction key, so the comparison pairs the wrong rows;
  - PL generation: two header layouts (preprocessing merges them);
  - BA volumes 2019–21: the old column name (preprocessing maps it).

## 4. Preprocessing (`scripts/preprocess_all.py`, `process_country`)

The steps run in this order:

1. **Load ENTSO-E.**
   - Years before `CUTOFF_YEARS` are skipped for 15-min countries, so only the 15-min settlement era is used: ES from 2022, RO 2021, FI 2023, HR 2024, PL 2024.
   - Prices are converted to EUR.
   - **Publication errors are removed:**
     - balancing-energy prices beyond ±15,000 €/MWh (`BALANCING_PRICE_LIMIT`, the harmonised PICASSO/MARI limit) are blanked. This removes the SK aFRR Up prices of −529,189 and −146,147 €/MWh on 2025-09-21;
     - the periods in `BAD_DATA` are blanked (§9.5).
   - Balancing-energy prices are pivoted to one column per product and direction.
   - Header changes between years are merged: PL generation is "Biomass" until 2024 and "Biomass - Actual Aggregated" from 2025, and BA volume is "Imbalance Volume" until 2021.
   - HU procured balancing capacity comes as one (direction, tender id, Price/Volume) column triple per tender, with different tender ids every year. It is aggregated per direction into the total procured volume and its volume-weighted price (`procured_capacity`).
   - A file that cannot be read stops the run with its path, instead of being skipped silently.
2. **Join ERA5** on timestamp.
3. **Training window.** `TRAIN_START` cuts countries whose market changed so much that older data describes a different system:
   - **PL from 2024-06-14 00:00 local time.** PSE's balancing market reform introduced 15-min settlement and new imbalance pricing that day. Before it, the volume averaged +227 MWh, after it −76. PL now runs in 15-min mode, with about 19k training rows (2024-06-14 → 2024-12-31).
   - **RS from 2021-03-01.** Before that date, the published RS volume is almost always of one sign and about 10× larger than afterwards.
4. **Resample** to the country's resolution: truncate to 15 min or 1 h, then take the mean. **Imbalance volumes in hourly countries are summed instead** (`hourly_energy`). They are MWh per settlement period, so an hour is the sum of its periods: the hourly value is mean × 60 / period length. That equals the sum for complete hours and scales up hours that are missing a quarter.
5. **SK single-price fix.** Where Short is 0 and Long is not, Short is set equal to Long.
6. **Regular time grid.** A complete 15-min or 1-h timeline is built from the first to the last timestamp that has targets. The data is left-joined onto it, so one row is always exactly one time step and missing periods become empty rows.
   - A country without targets both before and in 2025 is skipped. Currently that is LT.
7. **Gap report.** Every target gap is written to `processed_data/gaps_{CC}.csv`. Gaps longer than `GAP_FLAG` (1 day) print a warning saying whether they fall in train or test.
8. **Calendar features**, see §5.
9. **Hourly actuals are delayed to the end of their hour** (15-min countries only, `delay_hourly_actuals`). An hourly value labelled hh:00 covers hh:00–hh+1:00. Used from the hh:00 row, it would carry up to 45 minutes of the future, overlapping the t+15 to t+45 targets.
   - It applies to **actual** columns only: generation, load, border flows, balancing prices. Known-in-advance columns (forecasts, day-ahead price, net position) and weather are not delayed. ERA5 temperature and wind are instantaneous, and ssrd accumulates over the preceding hour.
   - A series is hourly on a given UTC day if it reports only on the full hour that day. Its value then appears from the hh:45 row (the hour's last quarter, consistent with the convention in §5) and is carried over hh+1:00–hh+1:30. Days with 15-min values are left as they are, so series that switch resolution (e.g. AT–CZ flows from 2024-07) are handled per day.
   - Number of actual columns with hourly values on at least some days (counted before the column filters): AT 6, BE 17, CH 11, DE 7, ES 27, FI 28, GR 12, HR 2, HU 12, PL 3, RO 19, SK 20. Which series they are is listed in §10.
10. **Blank means zero.** ENTSO-E leaves a value blank when nothing happened: a production type that produced nothing, or a balancing product that was not activated. In `generation` and `activated_balancing_energy_prices`, a blank is set to 0 when at least one other column of the same category reports at that timestamp. Step 9 runs first, so an hourly series in a category whose other columns are 15-min (e.g. AT biomass generation, AT and SK mFRR prices) is carried through its hour rather than set to 0 at :15, :30 and :45.
11. **Bounded forward fill.** Features are forward-filled for at most one hour: 3 steps at 15-min resolution, 1 step at hourly (`FFILL_LIMIT`).
12. **Column filters**, checked separately on the train part (before 2025) and the test part (2025), and only on rows that have targets. A column is dropped if either part has:
    - more than 10% missing (`MAX_NULL_RATIO`);
    - one value covering more than 99% of rows (`MAX_DOMINANT_SHARE`).

    **Import total.** The raw `import - sum` is discarded, because it only added the borders reporting at that timestamp (§9.1). It is rebuilt after the filters as the sum of the kept border columns, only where every border has a value.
13. **Lags and leads**, built by row shift. Because of the regular grid, this is a shift in time. See §5.
14. **Drop incomplete samples.** Any sample with an empty feature or target is removed.

## 5. Features and targets

Feature groups:
- **weather:** the ERA5 columns;
- **known in advance:** any column whose name contains "forecast" or "scheduled", plus the day-ahead price (`prices - value`) and the day-ahead net position;
- **actual:** everything else, including balancing prices, generation, load and imports;
- **target history:** imbalance price Long and Short, and imbalance volume.

| | 15-min countries | Hourly countries |
|---|---|---|
| Value at t of every weather, known-in-advance and actual column | ✓ | ✓ |
| Target history | t, t−15 … t−285 min (20 values) | t, t−1 … t−4 h (5 values) |
| Weather, known-in-advance and actual history | t−15 … t−150 min (10) | t−1 … t−3 h (3) |
| Weather and known-in-advance future values | t+15 … t+150 min (10) | t+1 … t+3 h (3) |
| **Targets** | t+15 … t+75 min (5 horizons × 3 series = 15) | t+1 h, t+2 h (2 × 3 = 6) |

The value of each target at t is a feature (`…_t+0min` or `…_t+0h`). For balancing energy we forecast from t+15, so the price and volume at t are assumed known.

**Time convention.** All timestamps are UTC and mark the **start** of a period: the row labelled t holds the values of the period [t, t+15 min), or [t, t+1 h) in hourly countries.
- "Known at t" therefore means known once that period has ended, at t+15 min (t+1 h). That is the moment the forecast is made.
- So the first horizon, t+15 min (t+1 h), is the period that starts at that moment; t+75 min (t+2 h) is the period starting an hour later.
- This assumes the values of the period just ended are available immediately. In practice, TSOs publish imbalance prices and volumes, and ENTSO-E publishes actuals, with a delay of minutes to hours (§8).
- Hourly series in 15-min countries follow the same rule: an hour's value is used only from the row of its last quarter on (§4 step 9).

**Calendar features** (`TIME_FEATURES`) use the country's **local** clock (`LOCAL_TZ`). The cyclic ones are encoded as sin/cos, so 23:45 sits next to 00:00 and 31 Dec next to 1 Jan:
- `qh_of_day_sin/cos`: quarter-hour of the day (96 per day);
- `qh_of_hour_sin/cos`: position within the hour, since schedules change on the full hour. 15-min countries only;
- `dayofweek_sin/cos` and `is_weekend`;
- `day_of_year_sin/cos`;
- `is_holiday`: from the `holidays` package, using the local date.

**Column naming:**
- Lagged columns have the spaces removed and a suffix: `imbalance_prices-Long_t-15min`.
- The values at t of the other features keep their original names: `generation - Solar - Actual Aggregated`.

**Train and test data per country** (rows after all steps; the test year is always 2025):

| Country | Columns | Train rows | Train from | Test rows |
|---|---|---|---|---|
| AT | 626 | 209,393 | 2019-01-01 | 34,937 |
| BA | 137 | 24,059 | 2020-01-01 | 7,473 |
| BE | 657 | 210,217 | 2019-01-01 | 35,020 |
| CH | 519 | 210,094 | 2019-01-01 | 34,679 |
| DE | 800 | 88,589 | 2022-06-22 | 34,790 |
| ES | 669 | 90,445 | 2022-05-23 | 27,904 |
| FI | 670 | 56,583 | 2023-05-21 | 34,608 |
| FR | 229 | 50,215 | 2019-01-01 | 8,720 |
| GR | 539 | 35,316 | 2022-12-22 | 34,709 |
| HR | 658 | 35,117 | 2024-01-01 | 35,030 |
| HU | 681 | 208,896 | 2019-01-01 | 34,874 |
| LV | 157 | 52,139 | 2019-01-01 | 8,748 |
| PL | 714 | 19,192 | 2024-06-14 | 34,937 |
| PT | 222 | 50,835 | 2019-01-19 | 7,074 |
| RO | 593 | 130,187 | 2021-02-01 | 34,835 |
| RS | 184 | 33,271 | 2021-03-01 | 8,735 |
| SK | 617 | 186,407 | 2019-03-13 | 34,895 |

## 6. Model and training (`scripts/train_all_100_seeds.py`)

- **Split.** By timestamp: everything before 2025-01-01 UTC trains, 2025 tests (`TEST_START`). There is no separate validation set.
- **Scaling.** `StandardScaler` on X and y, fit on train only.
- **Model.** `MultiOutputNN`: Linear(n_features → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → n_targets). One model predicts all targets at once.
- **Training.**
  - Adam, learning rate 1e-4, batch size 1024.
  - Huber loss (δ = 1) on the standardised targets.
  - At most 100 epochs.
  - Early stopping with patience 5 on the **test** loss.
- **Determinism.** The seeds fix the initialisation and batch order, but PyTorch on the GPU is not deterministic (`torch.use_deterministic_algorithms` is not set). Rerunning gives metrics that agree closely but not bit for bit. The same holds for SHAP. Setting `torch.use_deterministic_algorithms(True)` (and `CUBLAS_WORKSPACE_CONFIG=:4096:8`) would make reruns exact on the same hardware, at some cost in speed.
- **Seeds.**
  - Seeds 0–99 are trained. The metrics are the mean and standard deviation over the 100 seeds.
  - Seeds 0–4 are saved for SHAP.
  - The seed with the lowest test loss is saved as the "best" model, used by the sample plots.
- **Metrics per target:** MAE, RMSE, R² and SMAPE, in original units.
- **Naive baseline.** The value at t, carried forward to every horizon (`NAIVE_MAE`, `NAIVE_RMSE`, `NAIVE_R2`).
- **Summaries.** Averages over horizons are reported **per target group**: Long price, Short price and volume. Where Short equals Long in at least 99% of test rows (single pricing), Short is labelled "price Short = Long (single pricing)" and left out of the summary.

## 7. SHAP (`scripts/run_shap_all.py`)

- **Setup.**
  - `shap.DeepExplainer` runs on the models of **seeds 0–4**.
  - The background is 500 random training rows. **1,000 random 2025 rows** are explained.
  - The rows are the same for every seed (random seed 42).
- **Grouping over lags.** SHAP values are additive, so each feature's columns (value at t, lags and leads) are **summed into one group**: `imbalance_prices-Long_t-15min`, `_t-30min`, … all count as `imbalance_prices-Long`.
- **Calendar pairs.** The sin and cos of a cyclic calendar feature are summed into one group in the same way (`qh_of_day`, `qh_of_hour`, `dayofweek`, `day_of_year`), since only together do they describe the position in the cycle. In the summary plots, a pair is coloured by that position (0 = start of the day, week or year, 1 = its end).
- **Averaging over seeds.** The grouped values are averaged over the 5 seeds. Since SHAP is linear in the model output, this explains the seed ensemble.
- **Outputs in `figures/`:**
  - `shap_raw_{CC}_{target}.npy`: grouped, seed-mean SHAP values (samples × groups);
  - `shap_test_features_{CC}.npy`: the scaled value of each group's column nearest to t (for calendar pairs, the position in the cycle), for colouring the plots;
  - `shap_feature_names_{CC}.txt`;
  - `shap_mean_abs_{CC}.csv`: mean |SHAP| per group and target, with a category column (calendar, weather, target history, or the ENTSO-E category);
  - `shap_category_{CC}.csv`: the same, summed per category;
  - `shap_stability_{CC}.csv`: agreement between seeds, given as the mean pairwise Spearman correlation of the feature rankings and the mean overlap of the top-10 features.

## 8. Built-in assumptions

These are deliberate choices. Together they make the results an **upper estimate** of what a deployed forecaster would reach.

| Assumption | Effect |
|---|---|
| The test year is used for early stopping and for picking the best seed | The scores are optimistic |
| The column filters (§4 step 12) also look at the test year: a column that is mostly missing or constant in 2025 is dropped | A small use of test information when choosing the inputs |
| The ±std over seeds only measures training noise | It is not uncertainty about out-of-sample performance |
| ERA5 values at t and in the future stand in for a paid weather forecast of the same or better accuracy | Weather forecast error is ignored |
| Actual generation, load, flows and balancing prices of the period just ended are known at the forecast time (§5, time convention) | ENTSO-E publishes these with a delay; a market participant would see them later. Hourly actuals are only used once their hour is over (§4 step 9) |
| Imbalance price and volume, and balancing-energy prices, are known at t; the forecast starts from t+15 | The value at t is a feature and the basis of the naive baseline |
| Day-ahead price and net position are known for the whole day in advance | They get future values like forecasts |
| Countries are compared only within groups of similar design | Horizons, training windows and pricing rules differ between countries |

## 9. Known issues

### 9.1 Fixed since the previous version

- **Import sum.** The raw `sum` only added the borders reporting at that timestamp. Hourly borders are blank at :15, :30 and :45, so the sum followed a sawtooth (wrong in 73% of RO rows, 63% BE, 58% HU, …). It is now rebuilt from the kept border columns (§4 step 12).
- **`generation_import`** repeated `generation` and `import` column for column, and is no longer loaded.
- **Calendar features** were in UTC, as raw integers, without a quarter-of-the-hour feature, and the holiday flag used the UTC date. All of this is fixed (§5).
- **Day-ahead price and net position** now get future values.
- **Hourly volume** was averaged instead of summed over the hour (§4 step 4).
- **Test split.** The test set was the last 20% of each country's rows, so test periods differed between countries. It is now calendar year 2025 for all of them.
- **December 31** was missing for every year; it is now filled (§3.1).
- **SK aFRR price spike and the ES volume error** are removed (§4 step 1, §9.5).
- **Hourly actuals in 15-min countries leaked up to 45 minutes of the future.** An hourly value labelled hh:00 was forward-filled from the hh:00 row, so at t = hh:00 the model saw an average over the periods it was forecasting. Such values are now used only from hh:45 (§4 step 9).
- **Hourly series in mixed categories were set to 0 at :15, :30 and :45** by the blank-means-zero step (for example AT biomass generation, AT and SK mFRR prices), giving a sawtooth. They are now carried through the hour (§4 step 10).
- **HU procured balancing capacity was skipped silently**, because the loader did not recognise its three-level header and only printed a message. It is now aggregated (§4 step 1), and unreadable files stop the run.
- **Calendar sin/cos pairs** were two separate SHAP features; they are now one group (§7).

### 9.2 Duplicate and near-duplicate features

These pairs correlate at |r| > 0.99, so SHAP splits the credit for one piece of information between two columns:
- **Intraday and day-ahead wind/solar forecasts** (r 0.994–1.0):
  - Solar: BE, ES, FI, GR, HR, HU, RO, SK (identical in SK).
  - Wind Onshore: BA, GR, HR, RO.
- **Up and Down balancing prices are identical:** ES RR and PT RR, RS aFRR, and FR FCR (r 0.9999).
- **FR FCR price ≈ day-ahead price** (r 0.998).
- **Actual load ≈ load forecast** in ES (0.991) and FR (0.994).
- **DE actual solar and wind ≈ their forecasts** (0.994–0.996).
- **PT `import - ES` = `import - sum`**, since PT has only one border.

**Hourly series on the 15-min grid.** Their lags at t−15, t−30 and t−45 repeat the same value, so the lag columns are partly copies of each other. §10 lists these series per country. Summing SHAP over lags (§7) removes most of the credit splitting this causes.

### 9.3 Data gaps that remain

- **LT** is not modelled (no 2025 imbalance data).
- **BA:**
  - no day-ahead price or net position;
  - 69 target gaps longer than one day, 11 of them in 2025 (1–4 days each);
  - only 24k training rows.
- **CH, RS:** no net position.
- **GR:** the targets are missing from 2022-12-28 to 2023-12-31 and there is no volume data for 2019–2022, so GR effectively trains on 2024 only (35k rows). GR net position ends on 2025-09-30, so it is dropped (25% missing in 2025).
- **HR, PL:** one year or less of training data (HR 2024; PL from 2024-06-14). These are the most likely to suffer from distribution shift.
- **PT:** a 2-day target gap starting 2025-04-27 (the Iberian blackout of 28 April). The generation forecast is missing 2025-01-13 → 02-01.
- **SK:** imbalance volume is missing 2021-01-20 → 04-11.
- **Balancing-energy prices missing for all of 2025:** BE (ends 2024-11-30) and LV (ends 2024-10-08), so both are dropped. BA's are 37% missing in training (gaps of 46–87 days in 2023–24) and are dropped too.
- **Reserves:** FI contracted reserve prices are missing for all of 2024 and both FI series end on 2025-08-14. PL has contracted reserves only for 2025-01-01 → 02-02. HU procured capacity covers only a small share of 2023–2025. All of these are dropped.
- **Wind/solar forecasts start late:** BA 2021-08-02, RS 2022-03-22 (both dropped), CH intraday 2025-03-31 (dropped), SK intraday 2019-03-12 (kept).
- **Dropped as more than 10% missing,** per country: BA 10 columns, HU 5 (the SI border and the 4 procured-capacity columns), LV 5, BE 4, FI 4, FR 4, PL 4, RS 3, CH 2, GR 2, PT 2; AT, DE, ES, HR, RO and SK none. The kept series are listed in [entsoe_availability.md](entsoe_availability.md).
- **Weather** is a mean over a lat/lon box, not over the country's shape, and is not weighted by population, load or capacity.

### 9.4 Volume definitions

- Imbalance volumes are in MWh per settlement period; ENTSO-E declares MWH for every country.
- The hourly countries' values are the sum over the hour (§4 step 4).
- **TODO: RS 2025 volumes are probably in the wrong unit.** After the summation they are about 4× the 2024 level. RS settles every 15 min from 2025, and the median |volume| is 14.4 MWh per hour in 2024 but 23.6 per **quarter-hour** in 2025. So the quarter-hour values look like hourly-equivalent rates (MW) rather than MWh per period, and summing four of them inflates the hour. LV, FR and PT, which switched to 15 min at the same time, scale as expected (each quarter-hour is about a quarter of their hourly level). Fix: average RS 2025 quarter-hours instead of summing them (or rescale), after confirming the unit with EMS.
- **TODO: hourly countries in 15-min settlement.** FR, LV and RS settle every 15 min from 2025-01, and PT from 2025-04 (§1). Their 2025 targets are hourly means (prices) or sums (volumes) of quarter-hour values, while training used native hourly (FR: 30-min) settlement. The price target is therefore smoother in the test year than in training. Options: model them at 15 min once enough 15-min history exists, or keep the hourly aggregate and state it as the target definition.

### 9.5 Removed publication errors

Both are removed in the loader. They are documented in the constants `BALANCING_PRICE_LIMIT` and `BAD_DATA` in `preprocess_all.py`.

- **SK aFRR Up price** of −529,189 and −146,147 €/MWh on 2025-09-21 at 21:30 and 21:45 UTC:
  - far outside the PICASSO price limits of ±15,000 €/MWh;
  - the wrong sign for an upward price;
  - while the imbalance price in the same quarter-hours was normal (45–75 €).
- **ES imbalance volume** on the local days **9 February** and **18–24 February 2025** (2025-02-08 23:00 → 02-09 23:00 UTC and 2025-02-17 23:00 → 02-24 23:00 UTC):
  - values swing between −1,119 and +1,809 MWh per quarter-hour, and stay flat for hours (e.g. +730 to +900 for 3 hours on 2025-02-20);
  - in 2022–2024, |volume| never exceeded 715 (99.9th percentile: 374);
  - both periods start and end on local midnight;
  - the imbalance price stays normal throughout (around 0 €/MWh at a solar noon in February, typical for Spain);
  - the API serves the same values, and no event explaining it could be found.

  Whole days reported wrongly points to a publication error, so these 8 days are blanked and drop out of the test set. With them, ES volume scored R² 0.239 against a naive 0.529, because the persistent spike favours carrying the value at t forward. Without them, it scores 0.146 against −0.049.

### 9.6 Other open points

- **The model is untuned.** Huber loss aims at the median while R² and RMSE reward the mean. There is no regularisation, and training stops after a few epochs (100 seeds train in about 1 minute per country). In BA and CH prices this shows up as a better R² than naive but a worse MAE; the RS price beats naive MAE only by 0.1 €/MWh.
- **SMAPE** is reported but is not meaningful for series that cross zero, which includes both prices and volumes.

## 10. Hourly-resolution features in 15-min countries (summary)

A series is counted as hourly if at least 90% of its quarter-hour values (:15, :30, :45) repeat the previous value. This check also flags series that are mostly constant, such as rarely activated mFRR prices.

The **actual** series below (generation, load, flows, balancing prices) are delayed to the end of their hour on their hourly days (§4 step 9). The known-in-advance series and weather are not.

The following hold for almost all 15-min countries:
- the 8 weather variables are hourly throughout;
- `prices - value` and `net_position - value` switch to 15 min in 2025-10.

Per country:
- **AT:** border flows CH, HU, IT_NORD and SI are hourly; CZ is 15-min from 2024-07 and DE_LU from 2021-08. Biomass generation and mFRR prices are hourly.
- **BE:** all generation, the generation forecast, all wind/solar forecasts, and imports from NL and GB are hourly. FR imports are 15-min from 2023-04.
- **CH:** all generation, load, the load forecast, the day-ahead price (still hourly in 2025-12), the AT, FR and IT_NORD flows, and RR prices are hourly. aFRR prices are 15-min from 2022-06.
- **DE:** almost everything is 15-min; Coal-derived gas only from 2023-01.
- **ES:** aFRR prices, Hard-coal generation and PT flows are hourly.
- **FI:** Hard coal, Oil, Other renewable and Waste generation, and EE flows are hourly. NO_4 flows are 15-min from 2025-07 and SE_1/SE_3 from 2025-10.
- **GR:** most generation, all border flows, and the solar forecasts are hourly. Load is 15-min from 2025-11.
- **HR:** mFRR prices, several generation types, BA and RS flows, and solar forecasts are hourly. aFRR prices are 15-min from 2025-10.
- **HU:** HR, RO, RS, SK and UA flows, Oil and Geothermal generation, and net position are hourly. Solar is 15-min from 2024-02.
- **RO:** BG, RS and UA flows are hourly. aFRR prices are 15-min from 2024-07.
- **SK:** all generation, load, the generation forecast, solar forecasts, HU and UA flows, and mFRR prices are hourly. CZ and PL flows and the load forecast are 15-min from 2024-07, and aFRR prices from 2025-03.

## 11. Current results

Averages over all horizons, from the 100-seed mean (see `results/`), on the 2025 test year.
- "Naive" is the value at t carried forward.
- **Bold** marks targets where the model is not better than naive on both R² and MAE.
- "SHAP seed agreement" is the mean Spearman correlation of the feature rankings between the 5 SHAP seeds.
- Prices are in EUR/MWh and volumes in MWh. Single-priced countries list only the Long price.

| Country | Target | R² model | R² naive | MAE model | MAE naive | SHAP seed agreement |
|---|---|---|---|---|---|---|
| AT | price Long | 0.047 | -0.528 | 68.7 | 75.7 | 0.88 |
| AT | volume | 0.153 | -0.054 | 25.2 | 27.4 | 0.90 |
| BA | price Long | **0.307** | 0.029 | **34.7** | 30.8 | 0.88 |
| BA | price Short | **0.149** | -0.184 | **58.7** | 50.8 | 0.89 |
| BA | volume | 0.688 | 0.642 | 200.9 | 202.9 | 0.88 |
| BE | price Long | 0.355 | 0.104 | 53.6 | 56.3 | 0.93 |
| BE | volume | 0.246 | -0.020 | 24.8 | 28.5 | 0.93 |
| CH | price Long | **0.272** | 0.215 | **98.8** | 84.4 | 0.77 |
| CH | price Short | **0.098** | -0.121 | **64.3** | 62.9 | 0.79 |
| CH | volume | 0.445 | 0.411 | 24.8 | 25.2 | 0.83 |
| DE | price Long | 0.235 | -0.160 | 49.2 | 53.7 | 0.86 |
| DE | volume | 0.207 | -0.243 | 79.0 | 97.7 | 0.91 |
| ES | price Long | 0.499 | 0.212 | 22.6 | 25.7 | 0.87 |
| ES | price Short | 0.547 | 0.299 | 23.7 | 26.5 | 0.88 |
| ES | volume | 0.146 | -0.049 | 53.7 | 63.3 | 0.79 |
| FI | price Long | 0.275 | -0.027 | 51.9 | 57.0 | 0.89 |
| FI | volume | 0.622 | 0.603 | 139.0 | 141.6 | 0.91 |
| FR | price Long | 0.247 | -0.270 | 42.0 | 52.7 | 0.84 |
| FR | price Short | 0.285 | -0.210 | 43.9 | 55.0 | 0.85 |
| FR | volume | 0.353 | 0.147 | 501.6 | 570.7 | 0.81 |
| GR | price Long | 0.309 | -0.071 | 55.7 | 56.1 | 0.90 |
| GR | volume | 0.623 | 0.539 | 49.6 | 52.2 | 0.91 |
| HR | price Long | 0.317 | 0.049 | 58.2 | 63.2 | 0.90 |
| HR | volume | 0.389 | 0.322 | 13.0 | 13.4 | 0.90 |
| HU | price Long | 0.219 | -0.227 | 110.3 | 116.4 | 0.94 |
| HU | volume | 0.410 | 0.211 | 17.4 | 19.3 | 0.95 |
| LV | price Long | 0.177 | -0.258 | 107.0 | 126.8 | 0.94 |
| LV | volume | 0.459 | 0.349 | 18.6 | 20.2 | 0.84 |
| PL | price Long | 0.139 | -0.585 | 33.1 | 35.7 | 0.92 |
| PL | volume | **0.381** | 0.505 | **95.2** | 85.5 | 0.93 |
| PT | price Long | 0.581 | 0.440 | 18.5 | 19.7 | 0.88 |
| PT | price Short | 0.170 | -0.569 | 24.0 | 26.4 | 0.85 |
| PT | volume | 0.784 | 0.729 | 148.9 | 170.5 | 0.94 |
| RO | price Long | 0.218 | -0.319 | 161.3 | 201.7 | 0.91 |
| RO | price Short | 0.211 | -0.345 | 155.1 | 196.7 | 0.92 |
| RO | volume | **0.468** | 0.454 | **22.3** | 22.1 | 0.93 |
| RS | price Long | 0.249 | 0.104 | 51.7 | 51.8 | 0.80 |
| RS | volume | 0.068 | -0.636 | 78.4 | 103.5 | 0.76 |
| SK | price Long | 0.051 | -0.390 | 39.7 | 44.1 | 0.91 |
| SK | price Short | 0.011 | -0.575 | 41.8 | 48.2 | 0.90 |
| SK | volume | 0.233 | 0.135 | 7.3 | 7.8 | 0.87 |

**Changes from the previous run** (same split and seeds, before the hourly-actual delay of §4 step 9 and the calendar grouping of §7):
- Countries without delayed columns barely moved (BA, FR, LV, PT, RS: R² within ±0.005, MAE within 0.3), which is the run-to-run noise on the GPU (§6).
- **CH lost the most:** R² −0.02 to −0.04 on all targets, and the Long price MAE rose from 92.4 to 98.8. Its generation and load are hourly and were used before the hour had ended, so the earlier scores were partly a leak.
- **GR volume** fell from 0.643 to 0.623 and **FI volume** from 0.630 to 0.622; BE, GR price and SK lost under 0.01.
- ES (27 delayed columns) and the other 15-min countries are unchanged within noise.
- The RS price is no longer bold: its MAE is now 51.7 against a naive 51.8, a tie in practice.
- Train rows rose slightly in AT, CH, DE, HU, RO (+881) and SK, and test rows in CH, GR, HU, RO and SK changed by up to 18, because the delay changes which rows have missing values (§5 table).
- **SHAP:** in GR the generation share fell from 15% to 10% and the intraday wind/solar forecast rose from 8% to 13%; in BE the intraday forecast rose by 4.5 points. The calendar share fell by 0.2–2 points everywhere, because a sin/cos pair is now one feature and its two halves partly cancel.

**Changes from the first run** (80/20 split, before the fixes in §9.1):
- **SK** went from negative R² on every target (−1.50, −1.07, −1.80) to small positive values, because the −529k €/MWh price spike is removed.
- **PL volume** improved from −1.49 to 0.381 with the 2024-06-14 training window. It is still below naive (0.505), with only 19k training rows.
- **RS volume** improved from −0.30 to 0.068 with the 2021-03 training window, and is now well above naive (−0.64).
- **ES volume** is discussed in §9.5.
- **DE** now has weather, a volume target, balancing prices, imports and net position. Its Long price R² went from 0.229 to 0.235; volume (new) scores 0.207 against a naive −0.243.
- **FR** now has weather. Long price R² went from 0.209 to 0.247, Short from 0.233 to 0.285, and volume from 0.173 to 0.353. FR volume MAE is not comparable with the old run, because FR volumes are now summed over the hour (two 30-min periods) instead of averaged.
- The other countries moved by a few hundredths. This is expected, since the test period moved to calendar year 2025.

**What drives the forecasts (SHAP, share of total mean |SHAP|, averaged over all targets):**
- **Target history** (price and volume at t and before) is the largest category in most countries: 11–45%, highest in BA (45%), LV (31%) and RS (27%).
- **Generation** is 10–25%. It is the largest category in CH (25%, run-of-river hydro, the top CH price driver), DE (23% against 11% target history), HU (20%, solar) and PL (19%). Wind generation is a top-3 volume driver in AT, DE and HR.
- **Wind/solar forecasts** (day-ahead plus intraday) are 15–30% in AT, BE, DE, ES, FI, GR, HR, HU, PL and RO, highest in BE (30%). The intraday solar forecast is the top price driver in BE, DE and HU.
- **The day-ahead price** (`prices - value`) is a top-3 price driver in AT, BE, ES, FR, GR, HR, LV, PL, PT and RS, and the top one in AT, FR, HR and PT.
- **Weather** is 10–20%, **calendar** up to 10% (BA; LV 9%).
- **Seed agreement** (5 seeds) is 0.84–0.95 for most targets. It is lowest in RS (0.76–0.80), CH (0.77–0.83) and ES volume (0.79), whose rankings should be read with care.

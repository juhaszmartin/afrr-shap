# aFRR imbalance forecasting with SHAP: what the pipeline does

This document describes the current state of the code: what data goes in, how it is processed, how the model is trained and evaluated, which assumptions are built in, and which issues are known but not yet fixed. File and constant names refer to the code, so it can be checked against it.

## 1. Goal

For each European country in the data, forecast the imbalance price (long and short) and the imbalance volume a short time ahead, then use SHAP to explain which inputs drive the forecast. The countries are later compared in groups of similar market design, not one against another.

**Targets and their names.** The three target series are called the same everywhere (features, targets, results, figures):

| Name | ENTSO-E column | Meaning |
|---|---|---|
| `imbalance_price_long` | imbalance prices, "Long" | price for a balance responsible party that is long (surplus) |
| `imbalance_price_short` | imbalance prices, "Short" | price for a balance responsible party that is short (deficit) |
| `imbalance_volume` | imbalance volumes, "value" | net imbalance volume, MWh per settlement period; **positive = system long (surplus), negative = short (deficit)** in every country (§4 step 1) |

Under dual pricing the short price is at least the long price; under single pricing the two are equal. A forecast target is the name plus the horizon, e.g. `imbalance_price_long_t+15min`; history columns are e.g. `imbalance_price_long_t-30min`.

**Splits.** Train: everything before 2025. Dev: 2025, used for early stopping and for picking hyperparameters. Test: 2026 (to the latest data, early November), used only for the final metrics. The dates are `data.dev_start` and `data.test_start` in `configs/default.yaml`.

| Resolution | Countries | Horizons |
|---|---|---|
| 15 min | AT, BE, CH, CZ, DE, ES, FI, GR, HR, HU, PL, RO, SI, SK | t+15, +30, +45, +60, +75 min |
| 1 h | EE, FR, LV, PT, RS | t+1 h, t+2 h |

- The hourly countries settled imbalances hourly (FR every 30 min) **until 2024**. Their 15-min inputs are averaged to the hour, except imbalance volumes, which are summed (§4 step 4).
- **TODO: FR, LV and RS settle every 15 min from January 2025, and PT from April 2025**, so for them the dev and test years are no longer hourly. The pipeline still runs them hourly, so their 2025–2026 price target is the mean of four quarter-hour settlement prices, which is smoother than the hourly prices they were trained on (§9.4).
- **LT is not modelled.** ENTSO-E has no LT imbalance prices or volumes after 2024-09-30, so it has no dev or test year.
- **BA is not modelled** (`EXCLUDED_COUNTRIES`, §9.7). Checked against the TSO's (NOSBiH) and the regulator's (SERC) own data, the ENTSO-E BA imbalance volume is not the system imbalance; the prices are administered and published in BAM; and ENTSO-E has no BA data after 2025-12-08, so there is no test year.
- **DE** has had a volume target since its volumes were filled from the API (§3.1), so it now has 15 targets like the other 15-min countries.
- **CZ, SI and EE were added on 2026-11-18** (data: §2 and §3; corrections: §4 steps 1 and 3).
  - **CZ** is modelled from July 2024 only, when OTE moved to 15-min settlement and a single imbalance price (§4 step 3).
  - **SI** is modelled from 2021, its first 15-min year (dual-priced 2021–22, single-priced from 2023).
  - **EE is hourly** (15-min settlement only from April 2025). Its imbalance prices, volumes and balancing prices come from the Baltic TSOs' own dashboard, not ENTSO-E (§3.2). ENTSO-E has no EE volumes from 2024-10-10 to 2025-05-27, an inverted sign in 2025 and preliminary prices from August 2025.

## 2. How to run

Run everything from the repository root with the project venv.

**One-off data steps.** These only need rerunning to extend or repair the data.
1. `scripts/get_fx_rates.py` downloads ECB daily euro reference rates into `data/reference/ecb_eur_rates.csv`.
2. `scripts/fill_entsoe_gaps.py` (needs `ENTSOE_API_KEY` in `.env`), see §3.1:
   - `fill` downloads every gap in the raw ENTSO-E data from the API;
   - `update` refreshes a recent rolling window through the last UTC midnight that is at least one day old;
   - `verify` compares 3 random stored days per series against the API;
   - `refresh --country CC --category CATEGORY --year YEAR` replaces one stored year with the current API version;
   - `download --countries CC,..` downloads every category of a new country, 2019 to the present, as one `data.parquet` per year (used for CZ, SI, EE; log in `data/raw/entsoe/download_log.json`).

   `fill`, `update`, `verify` and `download` accept optional `--countries CC,..` and
   `--categories category,..` filters (`download` requires `--countries`). For example:

   ```bash
   python scripts/fill_entsoe_gaps.py fill --countries HU,DE --categories prices,import
   python scripts/fill_entsoe_gaps.py update --countries DE,FR --categories load,prices
   python scripts/fill_entsoe_gaps.py verify --countries DE --categories load,prices --days 5
   python scripts/fill_entsoe_gaps.py refresh --country BA --category load --year 2025
   python scripts/fill_entsoe_gaps.py download --countries CZ,SI,EE
   ```
3. `scripts/era5_download.py` (needs `CDSAPI_URL` and `CDSAPI_KEY` in `.env`) downloads ERA5 and writes `data/raw/weather/` (§3). Without arguments it handles every country whose CSVs are missing; `era5_download.py germany,france` rebuilds the named countries. It fetches one grid point at a time from the ERA5 point-timeseries dataset; `--grid` uses the gridded per-region requests instead (§3).
4. `scripts/baltic_download.py` downloads the Baltic Transparency Dashboard reports used for EE (§3.2) to `data/raw/baltic/{report}/year=YYYY/data.parquet`. No key is needed; years already on disk are skipped, except the last two months.

Package versions are pinned in `requirements.txt` (Python 3.14.6). Superseded scripts and notebooks are in `legacy/` (gitignored, local only); outputs of runs before 2026-11-17 are in `archive/2026-11-17/`.

**Pipeline.** [run_pipeline.sh](run_pipeline.sh) runs these steps in order. Steps 1, 2 and 3 accept an optional country list, for example `DE,FR`.
1. `scripts/preprocess_all.py` writes `data/processed/X_{CC}.parquet`, `data/processed/y_{CC}.parquet` and `data/processed/gaps_{CC}.csv`. `scripts/get_entsoe_info.py` then writes [entsoe_availability.md](entsoe_availability.md), the input series each country ends up with.
2. `scripts/train_models.py` trains and evaluates every model (§6) and writes:
   - `results/{dev,test}/metrics_{CC}.csv`: one row per model and target (MAE, RMSE, R², SMAPE; the network also with the std over seeds);
   - `results/{dev,test}/summary.txt`: averages over horizons per target group, with the MAE skill against naive, for all countries;
   - `results/predictions/{CC}.parquet`: actuals and every model's forecast on dev and test;
   - `models/{CC}/`: `manifest.json` (feature and target names in model order, rows per split, chosen hyperparameters), the scalers, network seeds 0–4, ridge weights and LightGBM boosters. The directory is emptied before a country is trained, so it never mixes runs.
3. `scripts/run_shap_all.py` writes the SHAP outputs to `figures/` (§7).
4. `scripts/plot_shap_results.py` writes the SHAP summary plots to `figures/`, `scripts/plot_sample_days.py` writes 3-day plots of every model against the actuals to `samples/{dev,test}/` (from the saved predictions), and `plot_results.py` (repository root) writes charts comparing the models per country and horizon to `figures/performance/{dev,test}/`.

The unit tests run with `PYTHONPATH=src .venv/bin/python -m pytest tests`.

## 3. Data sources

**ENTSO-E Transparency Platform.** The raw files are stored as `data/raw/entsoe/{category}/country={CC}/year={YYYY}/data.parquet`. Gap fills sit next to them as `fill.parquet` (§3.1). The categories are:

| Category | Content | Role |
|---|---|---|
| `imbalance_prices` | long / short imbalance price ("Long", "Short") | **target** and target history |
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

**ERA5 weather.** The files are `data/raw/weather/{country}_hourly_*.csv`. Each one is an hourly mean and standard deviation over a lat/lon box around the country, for 2 m temperature, solar radiation (ssrd), and wind speed at 10 m and 100 m. That gives 8 variables.
- The boxes are in `REGIONS` in `era5_download.py`. Neighbouring countries share a download region (DE/FR/BE/CH, central Europe, RO/GR, Iberia, the north, UA). There is one CDS request per region and 4 months, over a box covering the whole region, and each country is cropped from it locally.
- The CDS limits each request to 121,000 fields (variables × days × hours) and also limits how many requests can wait in its queue (each took 2.5–4.5 h in the queue).
- DE and FR were downloaded with this script. The other countries were downloaded earlier, one request per country and month, by `legacy/era5_eu.ipynb`. The script reproduces those CSVs to float32 precision (checked on the BE and CH boxes, which lie inside the DE/FR region: the largest difference is 0.0003 °C). So `data/raw/weather/` can be rebuilt with the script alone.
- **CZ, SI and EE** were downloaded point by point, now the default. The gridded dataset's queue had grown to about 10 h per request, so the script fetched every 0.25° grid point inside the box (CZ 297, SI 78, EE 171) from the ERA5 point-timeseries dataset (`reanalysis-era5-single-levels-timeseries`, about 35 s per point for 2019–2026). It then takes the same box mean and std. The point values equal the gridded ones (largest difference 0.0003 K for temperature, 0 for ssrd, 2·10⁻⁵ m/s for wind; checked on the AT box, whose 341 points match the gridded crop).

**Currency.** ENTSO-E passes prices through in the currency the TSO reports in. `LOCAL_CURRENCY` in `preprocess_all.py` converts the following to EUR, using the most recent ECB rate on or before each timestamp. Every other country already reports in EUR.

| Country | Series | Currency | Until |
|---|---|---|---|
| HU | imbalance and balancing-energy prices; procured capacity price | HUF | whole period |
| PL | imbalance and balancing-energy prices; day-ahead price | PLN | whole period; day-ahead until 2019-11-19 23:00 UTC |
| RO | imbalance and balancing-energy prices; day-ahead price | RON | whole period; day-ahead until 2021-06-17 22:00 UTC |
| HR | imbalance prices | HRK | until 2022-12-31 23:00 UTC |
| CZ | imbalance prices | CZK | whole period (aFRR and mFRR prices were CZK until March 2022, before the CZ training window) |

The currency field of the ENTSO-E XML was checked for every country and for imbalance, balancing-energy and day-ahead prices (2025-06-15): every other series is in EUR, except BA (BAM), which is excluded.

Prices are in EUR/MWh and volumes in MWh.

### 3.1 Gap fill and spot check against the API

The original bulk download of `data/raw/entsoe/` was made by a script that is not in the repository. The data itself is not in the repository either (it is gitignored), and ENTSO-E data is revised after publication (see BA load below), so a fresh download can differ from the stored files.

**Fill (`fill_entsoe_gaps.py fill`).**
- It looks for every gap longer than 2 h in each stored series, from 2019 (or the 15-min cutoff year) to the end of 2025, and downloads it.
- Only timestamps that are missing are added. They go into `fill.parquet`; the original `data.parquet` is never changed. The loader reads both.
- The log is `data/raw/entsoe/fill_log.json`.

**Update (`fill_entsoe_gaps.py update`).**
- It refreshes the last `UPDATE_LOOKBACK_DAYS` days before the delayed UTC cutoff in `scripts/fill_entsoe_gaps.py`.
- The cutoff is the UTC midnight at least one day old, so the latest incomplete or still-changing day is not used.
- New and revised rows are merged directly into `data.parquet`; historical `fill.parquet` files remain separate.
- The behavior for rows inside the rolling window is controlled by `UPDATE_LOOKBACK_POLICY` in `scripts/fill_entsoe_gaps.py`: `override` replaces stored values and `keep` preserves them.
- Series more than `UPDATE_MAX_CATCHUP_YEARS` calendar year behind the cutoff are skipped without an API request and recorded in the update log.
- The log is `data/raw/entsoe/update_log.json`.

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

Rerunning `fill` for CZ, SI and EE (2026-11-18) found nothing new. Their remaining ENTSO-E holes (EE volumes, the SI solar forecasts) are not on the platform.

### 3.2 Baltic Transparency Dashboard (EE)

Elering, AST and Litgrid publish their balancing data on the Baltic Transparency Dashboard (baltic.transparency-dashboard.eu; public API, CC BY 4.0). For **EE**, three categories are read from there instead of ENTSO-E (`BALTIC_SOURCES` in `preprocess_all.py`). Each was compared with ENTSO-E for 2019–2026:

| Category | Dashboard report | Against ENTSO-E |
|---|---|---|
| imbalance volume | `imbalance_volumes_v2` (Estonia) | Identical until October 2024. ENTSO-E then has nothing until 2025-05-27 and is patchy until December. In 2025 ENTSO-E holds −1 × the dashboard value (r −0.87 to −0.98 in May–Oct). In 2026 the values are close (median ratio 1.00) but not identical (r 0.50–0.93), and the dashboard volume tracks the imbalance price more closely in every month (r −0.45 to −0.67 against −0.14 to −0.51). |
| imbalance price | `imbalance_prices`, final (preliminary only where no final exists yet) | Identical until June 2025. From August 2025 ENTSO-E holds Elering's preliminary price, which differs from the final in 20–60% of periods; July 2025 matches neither. EE is single-priced, so long = short. |
| balancing-energy price | `balancing_energy_prices` (reference price, up and down) | ENTSO-E EE mFRR prices end in October 2024 and are blank in hours without activation. Where they exist they equal the reference price (99.6–100%); the reference price is complete for 2019–2026. The columns are named `activated_balancing_energy_prices - reference_Up/Down`. |

- **Resolution:** until 2024 the dashboard is hourly. In January–March 2025 it repeats each hourly value in all four quarter-hours of the hour, and real quarter-hour values start at local midnight on 1 April 2025 (`BALTIC_HOURLY_UNTIL`). Repeated volumes are divided by 4, so the hourly sum equals the hourly value (checked: all 68,069 hourly EE volumes equal the dashboard's).
- With the dashboard sign, the EE volume falls with the imbalance price in every month from 2019 to 2026, so the 2025 sign flip used before is no longer needed.
- **LV** was checked the same way. ENTSO-E LV equals the dashboard in almost every period (r ≥ 0.97 in every month 2023–2026), so LV stays on ENTSO-E.
- The dashboard also has **LT** imbalance prices and volumes after 2024-09-30, which ENTSO-E lacks (§1). LT is not modelled yet.

## 4. Preprocessing (`scripts/preprocess_all.py`, `process_country`)

The steps run in this order:

1. **Load ENTSO-E.**
   - Years before `CUTOFF_YEARS` are skipped for 15-min countries, so only the 15-min settlement era is used: ES from 2022, RO 2021, FI 2023, HR 2024, PL 2024, SI 2021, CZ 2024.
   - Prices are converted to EUR.
   - **Publication errors are removed:**
     - balancing-energy prices beyond ±15,000 €/MWh (`BALANCING_PRICE_LIMIT`, the harmonised PICASSO/MARI limit) are blanked. This removes the SK aFRR Up prices of −529,189 and −146,147 €/MWh on 2025-09-21;
     - the periods in `BAD_DATA` are blanked (§9.5).
   - **HU Long and Short are swapped back until 2021-04-30 22:00 UTC** (local midnight of 1 May 2021, `SWAPPED_LONG_SHORT`). In the published data "Long" is above "Short" in every quarter-hour of 2019–2020, which dual pricing rules out (the long price is at most the short price); from 1 May 2021 HU is single-priced and the two are equal. After the swap, HU 2019–2020 has short ≥ long like every other dual-priced country.
   - **Imbalance volumes get one sign convention: positive = system long (surplus)** (`VOLUME_SIGN_FLIP`), the EB GL 'imbalance' definition and the convention of most TSOs. Both corrections were checked against the TSO's own data:
     - **AT, before 2024-12-01 except May 2020, Nov 2022, Jan 2023 and Jul 2024: × −1.** APG publishes the control-area imbalance in MW with positive = surplus (transparency.apg.at API from 2023, Deltaregelzone archive before). ENTSO-E AT equals −APG/4 (MWh per quarter-hour) in exactly those months and +APG/4 in the others, switching on month boundaries. In May 2020 and Nov 2022 ENTSO-E carries APG's operational rather than settlement values, so the inverted months are probably APG's final-settlement uploads. After the flip, ENTSO-E AT correlates positively with APG in every month 2019–2026 (≥ 0.65; below 0.9 only where APG revised settlement values).
     - **HU, whole period: × −1.** MAVIR publishes 'KE' (imbalance energy), whose sign is opposite to the EB GL imbalance (MAVIR BRP working group, 14 Aug 2025); the HU volume rises with the imbalance price in every month from April 2019, while every other country's falls. HU Jan–Mar 2019 volumes carry no sign at all and are blanked (§9.5). Oct 2026 shows a weaker, mixed relation to the price, probably MAVIR's new settlement (PICASSO/MARI accession); not changed.
     - **EE** needs no flip: its volume comes from the Baltic dashboard (§3.2), whose sign is surplus-positive throughout. ENTSO-E holds the inverted value in 2025.
   - **RS volumes from 2025 are corrected** (`fix_rs_volume`), checked against EMS's own balancing data (aFRR, mFRR and IGCC energy per period, transparency.ems.rs). EMS does not publish the imbalance volume itself, so both corrections are inferred:
     - **Time:** the value labelled t belongs to t + 1 h in CET and t + 2 h in CEST in 2025, and to t + 1 h all year in 2026. Unshifted, the volume does not track EMS's IGCC exchange (corr ≈ −0.07); shifted, it does in every month (−0.24 to −0.55), as the hourly 2021–24 data does unshifted.
     - **Unit:** the quarter-hour values are MW, not MWh, and are divided by 4. After alignment the volume is about 5× larger relative to EMS's IGCC energy than before 2025, while EMS's own energies scale correctly at the switch; with the division, the hourly-equivalent volume is 30–32 MWh in 2025–26 against 19–26 in 2023–24, instead of a 6.5× jump on 2025-01-01.
     - The sign is positive = surplus in both periods (the volume is negative when Serbia imports IGCC energy), so no flip.
     - RS 30–31 Dec 2024, published quarter-hourly while EMS still settled hourly and not alignable, is blanked (§9.5).
   - The target columns get their final names (§1).
   - Balancing-energy prices are pivoted to one column per product and direction.
   - Header changes between years are merged: PL generation is "Biomass" until 2024 and "Biomass - Actual Aggregated" from 2025.
   - HU procured balancing capacity comes as one (direction, tender id, Price/Volume) column triple per tender, with different tender ids every year. It is aggregated per direction into the total procured volume and its volume-weighted price (`procured_capacity`).
   - A file that cannot be read stops the run with its path, instead of being skipped silently.
2. **Join ERA5** on timestamp.
3. **Training window.** `TRAIN_START` cuts countries whose market changed so much that older data describes a different system:
   - **PL from 2024-06-14 00:00 local time.** PSE's balancing market reform introduced 15-min settlement and new imbalance pricing that day. Before it, the volume averaged +227 MWh, after it −76. PL now runs in 15-min mode, with about 19k training rows (2024-06-14 → 2024-12-31).
   - **RS from 2021-03-01.** Before that date, the published RS volume is almost always of one sign and about 10× larger than afterwards.
   - **CZ from 2024-07-01 00:00 local time.** OTE settles every 15 min with a single imbalance price from that day. Before it, CZ was hourly, and ENTSO-E "Long" held the price in the system's direction and "Short" the counter-imbalance price, which is a different target. CZ has about 17.7k training rows (2024-07-01 → 2024-12-31).
   - CZ prices are 0 or negative in 42–59% of surplus quarter-hours (per year). This looks like OTE's price rule for the surplus direction, not an error, and is kept.
4. **Resample** to the country's resolution: truncate to 15 min or 1 h, then take the mean. **Imbalance volumes in hourly countries are summed instead** (`hourly_energy`). They are MWh per settlement period, so an hour is the sum of its periods: the hourly value is mean × 60 / period length. That equals the sum for complete hours and scales up hours that are missing a quarter.
5. **SK single-price fix.** Where the short price is 0 and the long price is not, the short price is set equal to the long price.
6. **Regular time grid.** A complete 15-min or 1-h timeline is built from the first to the last timestamp that has targets. The data is left-joined onto it, so one row is always exactly one time step and missing periods become empty rows.
   - A country without targets both before and in 2025 is skipped. Currently that is LT. Countries in `EXCLUDED_COUNTRIES` (BA) are not processed at all.
7. **Gap report.** Every target gap is written to `data/processed/gaps_{CC}.csv`. Gaps longer than `GAP_FLAG` (1 day) print a warning saying whether they fall in train or test.
8. **Calendar features**, see §5.
9. **Hourly actuals are delayed to the end of their hour** (15-min countries only, `delay_hourly_actuals`). An hourly value labelled hh:00 covers hh:00–hh+1:00. Used from the hh:00 row, it would carry up to 45 minutes of the future, overlapping the t+15 to t+45 targets.
   - It applies to **actual** columns only: generation, load, border flows, balancing prices. Known-in-advance columns (forecasts, day-ahead price, net position) and weather are not delayed. ERA5 temperature and wind are instantaneous, and ssrd accumulates over the preceding hour.
   - A series is hourly on a given UTC day if it reports only on the full hour that day. Its value then appears from the hh:45 row (the hour's last quarter, consistent with the convention in §5) and is carried over hh+1:00–hh+1:30. Days with 15-min values are left as they are, so series that switch resolution (e.g. AT–CZ flows from 2024-07) are handled per day.
   - Number of actual columns with hourly values on at least some days (counted before the column filters): AT 6, BE 17, CH 11, CZ 3, DE 7, ES 27, FI 28, GR 12, HR 2, HU 12, PL 3, RO 19, SI 23, SK 20. Which series they are is listed in §10.
10. **Blank means zero.** ENTSO-E leaves a value blank when nothing happened: a production type that produced nothing, or a balancing product that was not activated. In `generation` and `activated_balancing_energy_prices`, a blank is set to 0 when at least one other column of the same category reports at that timestamp. Step 9 runs first, so an hourly series in a category whose other columns are 15-min (e.g. AT biomass generation, AT and SK mFRR prices) is carried through its hour rather than set to 0 at :15, :30 and :45.
11. **Bounded forward fill.** Features are forward-filled for at most one hour: 3 steps at 15-min resolution, 1 step at hourly (`FFILL_LIMIT`).
12. **Column filters**, checked separately on the train part (before 2025) and the dev part (2025), and only on rows that have targets. A column is dropped if either part has:
    - more than 10% missing (`MAX_NULL_RATIO`);
    - one value covering more than 99% of rows (`MAX_DOMINANT_SHARE`).

    A column is also dropped if it is **more than 10% missing in the test part (2026)**. Only availability is checked there, never the values. Without this check, a series the TSO stopped publishing removed every test row in step 14: ES (no balancing-energy prices in 2026), GR (no generation in 2026) and HU (aFRR prices end on 2026-09-30, when MAVIR joins PICASSO) lost all or part of their test year. Dropped this way: ES RR and aFRR prices, GR the 7 generation series and the generation forecast, HU aFRR prices.

    **Import total.** The raw `import - sum` is discarded, because it only added the borders reporting at that timestamp (§9.1). It is rebuilt after the filters as the sum of the kept border columns, only where every border has a value.
13. **Lags and leads**, built by row shift. Because of the regular grid, this is a shift in time. See §5.
14. **Drop incomplete samples.** Any sample with an empty feature or target is removed. The run prints, per split, how many samples are kept out of those with targets, and warns if more than 20% are lost.
15. **Column order.** The feature columns of `X_{CC}.parquet` are written in sorted order. They used to come from Python sets, whose order changes between processes (string hashing is randomised), so every preprocessing run had a different layout and saved models no longer matched the data.

## 5. Features and targets

Feature groups:
- **weather:** the ERA5 columns;
- **known in advance:** any column whose name contains "forecast" or "scheduled", plus the day-ahead price (`prices - value`) and the day-ahead net position;
- **actual:** everything else, including balancing prices, generation, load and imports;
- **target history:** `imbalance_price_long`, `imbalance_price_short` and `imbalance_volume`.

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
- Target history: `imbalance_price_long_t-15min`, `imbalance_volume_t+0min`, …; forecast targets: `imbalance_price_short_t+30min`, … (§1).
- Other lagged columns have the spaces removed and a suffix: `generation-Solar-ActualAggregated_t-15min`.
- The values at t of the other features keep their original names: `generation - Solar - Actual Aggregated`.

**Data per country** (rows after all steps; train before 2025, dev 2025, test 2026):

| Country | Columns | Train rows | Train from | Dev rows (2025) | Test rows (2026) | Test to |
|---|---|---|---|---|---|---|
| AT | 626 | 209,393 | 2019-01-01 | 34,937 | 29,836 | 2026-11-07 |
| BE | 657 | 210,217 | 2019-01-01 | 35,030 | 29,230 | 2026-11-08 |
| CH | 519 | 210,094 | 2019-01-01 | 34,689 | 29,674 | 2026-11-08 |
| CZ | 639 | 17,653 | 2024-07-01 | 35,040 | 29,942 | 2026-11-08 |
| DE | 800 | 88,589 | 2022-06-22 | 34,800 | 29,729 | 2026-11-08 |
| EE | 167 | 52,127 | 2019-01-01 | 8,582 | 7,360 | 2026-11-08 |
| ES | 625 | 90,540 | 2022-05-23 | 33,737 | 28,213 | 2026-11-05 |
| FI | 670 | 56,583 | 2023-05-21 | 34,608 | 29,952 | 2026-11-09 |
| FR | 229 | 50,215 | 2019-01-01 | 8,723 | 7,405 | 2026-11-08 |
| GR | 452 | 35,331 | 2022-12-22 | 34,709 | 28,968 | 2026-11-01 |
| HR | 658 | 35,117 | 2024-01-01 | 35,040 | 29,520 | 2026-11-07 |
| HU | 659 | 200,486 | 2019-04-01 | 34,874 | 29,700 | 2026-11-07 |
| LV | 157 | 52,139 | 2019-01-01 | 8,751 | 7,462 | 2026-11-09 |
| PL | 714 | 19,192 | 2024-06-14 | 34,937 | 29,952 | 2026-11-09 |
| PT | 222 | 50,835 | 2019-01-19 | 7,077 | 7,258 | 2026-11-07 |
| RO | 593 | 130,187 | 2021-02-01 | 34,845 | 29,616 | 2026-11-09 |
| RS | 184 | 33,221 | 2021-03-01 | 8,727 | 7,410 | 2026-11-07 |
| SI | 476 | 135,691 | 2021-01-01 | 33,551 | 29,466 | 2026-11-08 |
| SK | 617 | 186,407 | 2019-03-13 | 34,895 | 29,223 | 2026-11-07 |

ES has 5.8k more dev rows and 44 fewer columns than before, because its balancing-energy prices, which had gaps in 2025 and are missing in 2026, are now dropped (§4 step 12). GR lost its generation series the same way.

EE has 8,582 of 8,760 dev hours since its imbalance data comes from the Baltic dashboard (§3.2); from ENTSO-E it had 2,293. SI loses its aFRR and RR prices (missing in 2026) and its solar forecasts (missing for 2023–25).

## 6. Models and training (`scripts/train_models.py`)

- **Split.** By timestamp (§1): train before 2025-01-01 UTC, dev 2025, test from 2026-01-01. Every model is fitted on train only. Dev is used for early stopping and hyperparameters; test is only used for the final metrics.
- **Scaling.** `StandardScaler` on X and y, fit on train only (LightGBM gets unscaled inputs but the same standardised targets).
- **Models** (`models` in `configs/default.yaml`), all trained per country:

| Model | What it is | Tuned on dev |
|---|---|---|
| `naive` | the value at t, carried forward to every horizon | – |
| `ridge` | linear regression with an L2 penalty, closed form on the standardised data | one penalty per target from a log grid (`ridge.alphas`); the run warns when the choice is at the edge of the grid |
| `lightgbm` | gradient-boosted trees, one model per target, Huber loss (δ = 1) on the standardised target like the network | number of trees (early stopping on dev MAE, patience 100, at most 2,000); fixed parameters in `lightgbm.params` |
| `nn` | `MultiOutputNN`: Linear(n_features → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → n_targets), one model for all targets; Adam, learning rate 1e-4, batch 1024, Huber loss (δ = 1) on the standardised targets, at most 100 epochs | epochs (early stopping on dev loss, patience 5) |
| `nn_ensemble` | the average prediction of all network seeds | – |

- **Network seeds.** Seeds 0–99 are trained. `nn` reports the mean and standard deviation of each metric over the seeds (the expected quality of one network and its training noise); `nn_ensemble` scores the seed average. Seeds 0–4 are saved for SHAP.
- **Determinism.** `training.deterministic: true` sets `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, so reruns on the same hardware give the same networks. LightGBM runs with `deterministic: true`; ridge is closed form.
- **Metrics per target:** MAE, RMSE, R² and SMAPE, in original units, on dev and on test.
- **Summaries.** Averages over horizons are reported **per target group**: long price, short price and volume, with the MAE skill against naive (1 − MAE / MAE naive). Where the short price equals the long price in at least 99% of a split's rows (single pricing), it is labelled "short price = long price (single pricing)" and left out of the summary. This is decided per split: CH is dual-priced in dev and single-priced in test.
- **Model files.** `models/{CC}/` is emptied before a country is trained. Its `manifest.json` records the feature and target names in model order; `run_shap_all.py` refuses to run when the processed data does not match it. Forecasts are saved in `results/predictions/{CC}.parquet`, so the plots never reload a model.

## 7. SHAP (`scripts/run_shap_all.py`)

- **Setup.**
  - `shap.DeepExplainer` runs on the network seeds 0–4 of the current run (`models/{CC}/`), after checking that the processed data has the feature layout of `manifest.json`.
  - The background is 500 random training rows. **1,000 random test (2026) rows** are explained; a country without test data would fall back to dev (2025). The split used is in `shap_info_{CC}.json`.
  - The rows are the same for every seed (random seed 42).
  - The additivity of the SHAP values (their sum plus the expected value equals the model output) is checked for every seed and the largest relative error is written to `shap_info_{CC}.json`.
  - SHAP is computed for the network only. The LightGBM boosters are saved in `models/{CC}/lightgbm/` and can be explained with `shap.TreeExplainer` if LightGBM is kept.
- **Grouping over lags.** SHAP values are additive, so each feature's columns (value at t, lags and leads) are **summed into one group**: `imbalance_price_long_t-15min`, `_t-30min`, … all count as `imbalance_price_long`.
- **Calendar pairs.** The sin and cos of a cyclic calendar feature are summed into one group in the same way (`qh_of_day`, `qh_of_hour`, `dayofweek`, `day_of_year`), since only together do they describe the position in the cycle. In the summary plots, a pair is coloured by that position (0 = start of the day, week or year, 1 = its end).
- **Averaging over seeds.** The grouped values are averaged over the 5 seeds. Since SHAP is linear in the model output, this explains the seed ensemble.
- **Outputs in `figures/`:**
  - `shap_raw_{CC}_{target}.npy`: grouped, seed-mean SHAP values (samples × groups);
  - `shap_test_features_{CC}.npy`: the scaled value of each group's column nearest to t (for calendar pairs, the position in the cycle), for colouring the plots;
  - `shap_feature_names_{CC}.txt`;
  - `shap_mean_abs_{CC}.csv`: mean |SHAP| per group and target, with a category column (calendar, weather, target history, or the ENTSO-E category);
  - `shap_category_{CC}.csv`: the same, summed per category;
  - `shap_stability_{CC}.csv`: agreement between seeds, given as the mean pairwise Spearman correlation of the feature rankings and the mean overlap of the top-10 features;
  - `shap_info_{CC}.json`: split explained, seeds, feature layout and training time of the models, additivity error.

  Before writing, the country's previous SHAP files and summary plots are removed, so `figures/` never holds two generations.

## 8. Built-in assumptions

These are deliberate choices. Together they make the results an **upper estimate** of what a deployed forecaster would reach.

| Assumption | Effect |
|---|---|
| Early stopping and hyperparameters use dev (2025); test (2026) is only scored | Dev scores are optimistic; test scores are not, apart from the next row |
| The column filters (§4 step 12) look at the dev values, and at whether each input is published in the test year | Inputs are chosen with 2025 data; 2026 data only decides which inputs exist |
| Models are trained on data before 2025 only, and not retrained on 2025 before the 2026 test | Test scores include a full year of ageing; markets changed in 2025 (§11) |
| The ±std over seeds only measures training noise | It is not uncertainty about out-of-sample performance |
| The 2026 test year runs to early November | Test scores miss most of the winter |
| ERA5 values at t and in the future stand in for a paid weather forecast of the same or better accuracy | Weather forecast error is ignored |
| Actual generation, load, flows and balancing prices of the period just ended are known at the forecast time (§5, time convention) | ENTSO-E publishes these with a delay; a market participant would see them later. Hourly actuals are only used once their hour is over (§4 step 9) |
| Imbalance price and volume, and balancing-energy prices, are known at t; the forecast starts from t+15 | The value at t is a feature and the basis of the naive baseline |
| Day-ahead price and net position are known for the whole day in advance | They get future values like forecasts |
| Countries are compared only within groups of similar design | Horizons, training windows and pricing rules differ between countries |

## 9. Known issues

### 9.1 Fixed since the previous version

Fixed on 2026-11-17 (the earlier outputs are in `archive/2026-11-17/`):
- **Feature order changed between preprocessing runs** (§4 step 15), so models, scalers and processed data from different runs did not match. Evaluation re-aligned columns by matching means and variances, which cannot tell apart columns that are identical in training (e.g. `qh_of_hour_sin/cos`, SK intraday and day-ahead solar forecasts), and SHAP did no re-alignment at all. Columns are now sorted, every model directory has a manifest of its feature layout, and SHAP refuses data that does not match it.
- **Stale models.** The "best" model was always seed 0 (the config had `seeds: 1`), while seeds 1–4 were from an earlier run with another column order (their 2025 MAE was 1.5–3.4× that of seed 0 for FR and PT). SHAP averaged them with seed 0. `models/{CC}/` is now emptied before every training run.
- **SHAP files of different runs.** CH to SK still had SHAP tables from a run whose models had been overwritten, and `figures/` held two generations of summary plots (`pricesLong` and `prices__Long`, BA at 15 min). SHAP now removes a country's old outputs before writing.
- **Training stopped after AT**, because it imported `scripts.evaluate_saved_models`, which is not importable when the script is run by the CLI. Dev and test are now scored inside `train_models.py`.
- **Test rows lost silently** in ES, GR and HU (§4 step 12).
- **HU Long and Short swapped until 1 May 2021** (§4 step 1).
- **Names.** The targets are `imbalance_price_long`, `imbalance_price_short` and `imbalance_volume` everywhere (§1).

Fixed later on 2026-11-17, after checking ENTSO-E against the TSOs' own data (the earlier AT, HU and RS results are in `archive/2026-11-17-pre-datafix/`, the BA outputs in `archive/2026-11-17-BA/`):
- **AT volume sign** was inverted in 67 of 72 training months, while dev and test used the other sign (§4 step 1).
- **HU volume sign** was opposite to every other country (§4 step 1); HU Jan–Mar 2019 volumes had no sign (§9.5).
- **RS 2025+ volumes** were 1–2 h early and in MW, so the hourly target was 4× too large and misaligned with the inputs (§4 step 1).
- **BA** is excluded (§9.7). Its prices were also treated as EUR although ENTSO-E publishes them in BAM (1.95583 BAM/EUR).

Fixed before that:
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
  - Wind Onshore: GR, HR, RO.
- **Up and Down balancing prices are identical:** ES RR and PT RR, RS aFRR, and FR FCR (r 0.9999).
- **FR FCR price ≈ day-ahead price** (r 0.998).
- **Actual load ≈ load forecast** in ES (0.991) and FR (0.994).
- **DE actual solar and wind ≈ their forecasts** (0.994–0.996).
- **PT `import - ES` = `import - sum`**, since PT has only one border.

**Hourly series on the 15-min grid.** Their lags at t−15, t−30 and t−45 repeat the same value, so the lag columns are partly copies of each other. §10 lists these series per country. Summing SHAP over lags (§7) removes most of the credit splitting this causes.

### 9.3 Data gaps that remain

- **LT** is not modelled (no 2025 imbalance data); **BA** is excluded (§9.7).
- **CH, RS:** no net position.
- **RS prices in Jul–Sep 2026 are EMS's preliminary values.** EMS publishes final prices from D+8 (transparency.ems.rs), but ENTSO-E still serves the preliminary ones (checked again 2026-11-17), which differ in about 83–95% of those periods. A few earlier months (Jan and Apr 2023, Mar and Jul 2025) differ from both EMS series. Before July 2026 the ENTSO-E RS price equals EMS's final price to within 0.003 €/MWh in most periods.
- **GR:** the targets are missing from 2022-12-28 to 2023-12-31 and there is no volume data for 2019–2022, so GR effectively trains on 2024 only (35k rows). GR net position ends on 2025-09-30, so it is dropped (25% missing in 2025).
- **HR, PL:** one year or less of training data (HR 2024; PL from 2024-06-14). These are the most likely to suffer from distribution shift.
- **PT:** a 2-day target gap starting 2025-04-27 (the Iberian blackout of 28 April). The generation forecast is missing 2025-01-13 → 02-01.
- **SK:** imbalance volume is missing 2021-01-20 → 04-11.
- **Balancing-energy prices missing for all of 2025:** BE (ends 2024-11-30) and LV (ends 2024-10-08), so both are dropped.
- **Reserves:** FI contracted reserve prices are missing for all of 2024 and both FI series end on 2025-08-14. PL has contracted reserves only for 2025-01-01 → 02-02. HU procured capacity covers only a small share of 2023–2025. All of these are dropped.
- **Wind/solar forecasts start late:** RS 2022-03-22 (dropped), CH intraday 2025-03-31 (dropped), SK intraday 2019-03-12 (kept).
- **Dropped as more than 10% missing,** per country: HU 5 (the SI border and the 4 procured-capacity columns), LV 5, BE 4, EE 2 (RU border, day-ahead solar forecast), FI 4, FR 4, PL 4, RS 3, SI 3 (HU border, both solar forecasts), CH 2, GR 2, PT 2; AT, CZ, DE, ES, HR, RO and SK none. The kept series are listed in [entsoe_availability.md](entsoe_availability.md).
- **Weather** is a mean over a lat/lon box, not over the country's shape, and is not weighted by population, load or capacity.

### 9.4 Volume definitions

- Imbalance volumes are in MWh per settlement period; ENTSO-E declares MWH for every country. RS 2025+ is published in MW and divided by 4 (§4 step 1).
- The sign is positive = system long (surplus) everywhere after the AT and HU corrections (§4 step 1). The data alone (the volume–price relation, which matched APG's own sign in every AT month) confirms it for BE, CH, CZ, DE, FI, FR, GR, HR, LV, PL, PT, RO, SI and SK, and for EE with the Baltic dashboard volume (§3.2; volume–price corr negative in every month 2019–2026). For ES the relation is too weak before late 2024 to confirm the sign; RS was confirmed against EMS's IGCC data.
- The hourly countries' values are the sum over the hour (§4 step 4).
- **TODO: hourly countries in 15-min settlement.** FR, LV and RS settle every 15 min from 2025-01, and PT from 2025-04 (§1). Their 2025 targets are hourly means (prices) or sums (volumes) of quarter-hour values, while training used native hourly (FR: 30-min) settlement. The price target is therefore smoother in the test year than in training. Options: model them at 15 min once enough 15-min history exists, or keep the hourly aggregate and state it as the target definition.

### 9.5 Removed publication errors

They are removed in the loader and documented in the constants `BALANCING_PRICE_LIMIT` and `BAD_DATA` in `preprocess_all.py`.

- **HU imbalance volume, Jan–Mar 2019:** every value is positive (absolute values without a sign); from 1 April 2019 the series is signed. Blanked. One-signed months also occur in PL 2019–20, RO 2020-04 and RS 2019–21, all before those countries' training windows.
- **RS imbalance volume, 30–31 Dec 2024:** see §4 step 1. Blanked.

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

- **The network is untuned.** Huber loss aims at the median while R² and RMSE reward the mean. There is no regularisation, and training stops after a few epochs (100 seeds train in about 1 minute per country).
- **SMAPE** is reported but is not meaningful for series that cross zero, which includes both prices and volumes.

### 9.7 Why BA is excluded

Checked against NOSBiH's own publications (nosbih.ba: imbalance prices, hourly system deviation) and the SERC annual reports (derk.ba) on 2026-11-17:
- **The ENTSO-E volume is not the system imbalance.** 2019–2021 is never negative and 52–66% exactly zero. From 2022 the yearly total |volume| is 18–63× SERC's total BRP imbalance (2023: 5,758 GWh against 92 GWh; BiH transmits about 18,700 GWh a year), and it does not correlate with NOSBiH's hourly deviation at any time shift (|r| < 0.16; NOSBiH's averages 10–20 MW).
- **Prices are published in BAM, not EUR.** The XML says BAM, and the values equal NOSBiH's KM/MWh prices in 91–98% of hours (Oct 2024, Jun 2025). Converted at the fixed 1.95583, they are close to SERC's annual average imbalance prices.
- **Prices are administered.** They are pay-as-bid extremes of activated aFRR/mFRR energy times SERC-set coefficients, falling back to aFRR bid prices when nothing is activated, and to 0 for the long price when there are no bids (Energy Community, *State of balancing markets 2023*). That explains the long price of 0 in about 58% of 2020–21 hours and the few distinct, round prices.
- **No test year.** ENTSO-E has no BA data after 2025-12-04 (prices) and 2025-12-08 (volumes). NOSBiH itself still publishes 15-min prices and the hourly deviation, so BA could be rebuilt from those sources if needed.

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

Run of 2026-11-17/18 (all fixes of §9.1, including the AT, HU and RS volume corrections; BA excluded; CZ, SI and EE added on 2026-11-18). Models trained on data before 2025, tuned on 2025 (dev), scored on 2026 to early November (test). Full tables: `results/{dev,test}/summary.txt`; charts: `figures/performance/{dev,test}/`. The results of earlier runs (2025 as the test year, scored with the early-stopping data) are in `archive/2026-11-17/` and are not comparable.

### 11.1 Which model to use

MAE skill = 1 − MAE / MAE of the naive forecast, averaged over horizons, per country and target group (19 countries; 44 groups on dev, 43 on test; single-priced short prices left out).

| | ridge | LightGBM | NN (one seed) | NN ensemble (100 seeds) |
|---|---|---|---|---|
| Median MAE skill, dev 2025 | 0.138 | **0.162** | 0.088 | 0.126 |
| Median MAE skill, test 2026 | 0.066 | **0.092** | 0.020 | 0.055 |
| Groups better than naive, dev | 91% | **100%** | 86% | 93% |
| Groups better than naive, test | 70% | **86%** | 58% | 67% |
| Lowest MAE of all models, test (of 43) | 9 | **26** | – | 2 |

- **LightGBM is the best model** on dev and on test, for prices and volumes, and it degrades least from 2025 to 2026. Naive is best in 6 test groups (CH, CZ, GR and HU long price, GR volume, PT short price), ridge in 9 (DE, ES, FI, FR, PT, RS and SK volume, HR and RS long price), the NN ensemble in 2 (EE long price and volume).
- Adding CZ, SI and EE (2026-11-18) left these conclusions unchanged. Against the 16-country run, median skills moved by at most 0.014 and the shares better than naive by at most 3 points.
- **EE on the Baltic dashboard data (§3.2):** the dev year grew from 2,293 to 8,582 hours, and the test targets are now Elering's final values. The test volume naive MAE fell from 53.9 to 45.4 MWh, because ENTSO-E's 2026 volumes were noisier than Elering's. MAE skill against naive on test (ENTSO-E → dashboard): long price, LightGBM 0.10 → 0.09 and NN ensemble 0.08 → 0.10; volume, LightGBM 0.04 → 0.06 and NN ensemble 0.07 → 0.08.
- **Ridge beats the network** on median skill. Volumes in particular are close to linear.
- **The network is the weakest learned model.** One network beats naive on only 59% of the test groups; averaging 100 seeds helps (median skill 0.020 → 0.052) but stays below ridge. The std of R² over seeds is small (median 0.013 on test), so this is not training noise.
- **By horizon**, every model gains on naive as the horizon grows. At the first horizon the network is far worse than naive (median price skill −0.23 on test), while LightGBM is about even (+0.02). The network does not reproduce the persistence that dominates the first horizon; modelling the change from the value at t instead of the level would likely help all models.
- **Effect of the volume corrections** (test, MAE skill before → after): AT long price LightGBM 0.12 → 0.16, ridge −0.07 → 0.11, NN ensemble −0.13 → 0.08; AT volume LightGBM 0.12 → 0.17; RS long price ridge 0.08 → 0.17, NN ensemble 0.07 → 0.13; the RS volume target fell from 112 to 28 MWh naive MAE (it was 4× too large). HU is unchanged (±0.01), as a constant sign flip is learned either way.

### 11.2 Per country (test 2026; the CH short price on dev)

MAE in EUR/MWh (prices) and MWh (volumes), averaged over horizons; the lowest MAE is in bold.

| Country | Target | MAE naive | MAE ridge | MAE LightGBM | MAE NN ens. | R² naive | R² LightGBM | R² NN ens. | lowest MAE |
|---|---|---|---|---|---|---|---|---|---|
| AT | long price | 59.9 | 53.6 | **50.3** | 55.0 | -0.00 | 0.29 | 0.13 | LightGBM |
| AT | volume | 29.9 | 24.8 | **24.8** | 25.2 | -0.11 | 0.27 | 0.14 | LightGBM (24.78 vs ridge 24.83) |
| BE | long price | 41.7 | 44.5 | **38.6** | 55.8 | -0.00 | 0.15 | 0.19 | LightGBM |
| BE | volume | 30.3 | 27.1 | **26.7** | 28.7 | 0.11 | 0.33 | 0.23 | LightGBM |
| CH | long price | **91.9** | 145.3 | 93.8 | 147.2 | -0.05 | 0.21 | -0.38 | naive |
| CH | short price (dev) | 62.9 | 56.9 | **50.2** | 53.6 | -0.12 | 0.24 | 0.19 | LightGBM |
| CH | volume | 25.5 | 23.5 | **22.3** | 23.3 | -0.11 | 0.14 | 0.08 | LightGBM |
| CZ | long price | **54.1** | 63.8 | 54.9 | 63.6 | -0.53 | 0.15 | 0.10 | naive |
| CZ | volume | 13.7 | 13.1 | **12.4** | 13.7 | 0.13 | 0.29 | 0.15 | LightGBM |
| DE | long price | 52.9 | 46.1 | **44.9** | 52.0 | -0.03 | 0.32 | 0.29 | LightGBM |
| DE | volume | 100.1 | **79.0** | 79.1 | 80.0 | -0.40 | 0.14 | 0.13 | ridge |
| EE | long price | 50.6 | 47.3 | 46.1 | **45.7** | 0.41 | 0.53 | 0.56 | NN ens. |
| EE | volume | 45.4 | 42.5 | 42.5 | **41.7** | 0.28 | 0.41 | 0.43 | NN ens. |
| ES | long price | 31.0 | 27.0 | **26.4** | 27.2 | 0.37 | 0.61 | 0.62 | LightGBM |
| ES | short price | 28.5 | 24.9 | **23.8** | 25.7 | 0.35 | 0.61 | 0.60 | LightGBM |
| ES | volume | 65.8 | **56.4** | 56.8 | 57.0 | -0.35 | 0.08 | 0.08 | ridge |
| FI | long price | 30.2 | 31.8 | **27.7** | 29.4 | 0.28 | 0.51 | 0.48 | LightGBM |
| FI | volume | 131.4 | **124.5** | 125.8 | 138.3 | 0.58 | 0.63 | 0.56 | ridge |
| FR | long price | 55.6 | 47.3 | **44.4** | 46.8 | 0.24 | 0.51 | 0.50 | LightGBM |
| FR | short price | 59.4 | 48.9 | **48.1** | 49.0 | 0.23 | 0.51 | 0.51 | LightGBM |
| FR | volume | 507.5 | **457.0** | 457.8 | 460.1 | 0.18 | 0.35 | 0.35 | ridge |
| GR | long price | **65.4** | 74.7 | 73.4 | 73.7 | 0.11 | 0.38 | 0.38 | naive |
| GR | volume | **47.8** | 53.3 | 56.3 | 58.2 | 0.56 | 0.43 | 0.40 | naive |
| HR | long price | 113.6 | **106.9** | 107.2 | 109.5 | 0.01 | 0.30 | 0.30 | ridge |
| HR | volume | 13.0 | 12.3 | **12.1** | 13.8 | 0.26 | 0.38 | 0.22 | LightGBM |
| HU | long price | **112.8** | 132.0 | 116.9 | 117.8 | -0.25 | 0.20 | 0.20 | naive |
| HU | volume | 17.7 | 16.9 | **16.3** | 17.5 | 0.27 | 0.41 | 0.36 | LightGBM |
| LV | long price | 56.5 | 57.4 | **51.1** | 69.1 | 0.30 | 0.51 | 0.24 | LightGBM |
| LV | volume | 30.7 | 28.6 | **27.9** | 29.1 | 0.25 | 0.40 | 0.35 | LightGBM |
| PL | long price | 45.7 | 43.3 | **43.0** | 46.0 | 0.37 | 0.23 | 0.44 | LightGBM |
| PL | volume | 78.4 | 84.3 | **78.4** | 102.3 | 0.60 | 0.61 | 0.28 | LightGBM (tie with naive) |
| PT | long price | 26.2 | 28.0 | **24.6** | 25.3 | 0.68 | 0.76 | 0.77 | LightGBM |
| PT | short price | **31.4** | 41.7 | 33.2 | 34.9 | 0.50 | 0.63 | 0.61 | naive |
| PT | volume | 168.9 | **142.4** | 152.7 | 142.4 | 0.63 | 0.69 | 0.74 | ridge (tie with NN ens.) |
| RO | long price | 144.9 | 132.6 | **118.8** | 134.6 | -0.15 | 0.25 | 0.24 | LightGBM |
| RO | short price | 143.2 | 128.4 | **114.1** | 125.7 | -0.18 | 0.25 | 0.25 | LightGBM |
| RO | volume | 24.7 | 25.6 | **23.1** | 26.1 | 0.48 | 0.55 | 0.45 | LightGBM |
| RS | long price | 65.2 | **54.0** | 54.3 | 57.0 | 0.15 | 0.51 | 0.47 | ridge |
| RS | volume | 28.0 | **20.3** | 20.4 | 21.3 | -0.54 | 0.13 | 0.13 | ridge |
| SI | long price | 85.3 | 77.7 | **74.7** | 77.5 | -0.10 | 0.22 | 0.27 | LightGBM |
| SI | volume | 10.5 | 10.7 | **9.8** | 10.0 | 0.52 | 0.59 | 0.59 | LightGBM |
| SK | long price | 44.3 | 38.4 | **37.9** | 41.1 | -0.66 | 0.12 | 0.07 | LightGBM |
| SK | short price | 44.8 | 38.6 | **37.6** | 41.2 | -0.67 | 0.11 | 0.07 | LightGBM |
| SK | volume | 8.4 | **7.8** | 8.0 | 7.9 | 0.38 | 0.35 | 0.44 | ridge |

### 11.3 Where no model beats naive in 2026

- **CH long price:** CH moved to single pricing on 1 January 2026, so the 2026 "long" price is the single price (median 150 €/MWh against −2.5 in 2025). Every model learned the dual-price long price; LightGBM comes closest (MAE 93.8 against 91.9).
- **CZ long price:** naive is best at t+15 to t+45 (MAE 39.9 against LightGBM 45.1 at t+15); LightGBM is best from t+60 (58.3 against 60.5) and has a far better R² overall (0.15 against −0.53). CZ trains on only six months (July–December 2024), and its price became calmer each year: the median change between quarter-hours was 23 €/MWh in 2024, 12 in 2025 and 7 in 2026. This favours persistence at short horizons. On dev, LightGBM's skill is 0.18.
- **GR:** trained on 2024 only (35k rows), and its generation series are missing in 2026 and therefore dropped (§4 step 12). Naive wins both targets.
- **HU long price:** HU aFRR prices end on 2026-09-30 and are therefore dropped; MAVIR's PICASSO accession was reported for around October 2026.
- **PT short price:** the short price equals the long price in 28% of 2026 rows, against 0–8% in earlier years.
- **PL volume:** LightGBM only ties naive; PL has 19k training rows and joined PICASSO in July 2025.

### 11.4 SHAP

SHAP explains the network (§7), which is not the best model; the attributions describe that network and should be compared with a TreeSHAP explanation of LightGBM before drawing conclusions about drivers.

Share of total mean |SHAP| per category, summed over all targets (%; 1,000 test rows; network seeds 0–4). Additivity holds to a relative error below 1e-6 everywhere.

| Country | target history | generation | ID wind/solar fc. | DA wind/solar fc. | weather | day-ahead price | calendar | import | balancing prices | load + load fc. | seed agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AT | 19 | 15 | 14 | 8 | 14 | 9 | 5 | 5 | 4 | 4 | 0.91 |
| BE | 15 | 23 | 16 | 12 | 9 | 6 | 4 | 4 | 0 | 5 | 0.90 |
| CH | 26 | 20 | 0 | 7 | 16 | 9 | 8 | 3 | 5 | 2 | 0.79 |
| CZ | 16 | 31 | 4 | 3 | 15 | 3 | 4 | 5 | 2 | 8 | 0.83 |
| DE | 11 | 23 | 17 | 4 | 14 | 3 | 4 | 9 | 3 | 6 | 0.88 |
| EE | 32 | 14 | 0 | 2 | 25 | 5 | 8 | 4 | 3 | 5 | 0.90 |
| ES | 32 | 15 | 11 | 6 | 9 | 10 | 3 | 2 | 0 | 5 | 0.88 |
| FI | 18 | 15 | 13 | 9 | 11 | 4 | 3 | 4 | 11 | 6 | 0.89 |
| FR | 19 | 20 | 0 | 7 | 14 | 9 | 5 | 3 | 7 | 6 | 0.84 |
| GR | 28 | 0 | 17 | 12 | 20 | 7 | 4 | 3 | 4 | 5 | 0.92 |
| HR | 20 | 16 | 9 | 6 | 10 | 15 | 2 | 4 | 6 | 8 | 0.92 |
| HU | 17 | 20 | 18 | 10 | 10 | 3 | 5 | 4 | 0 | 6 | 0.96 |
| LV | 24 | 22 | 3 | 2 | 17 | 4 | 7 | 3 | 0 | 14 | 0.90 |
| PL | 16 | 18 | 9 | 6 | 10 | 7 | 4 | 5 | 11 | 3 | 0.93 |
| PT | 22 | 13 | 10 | 6 | 11 | 18 | 4 | 2 | 5 | 3 | 0.87 |
| RO | 14 | 19 | 8 | 10 | 12 | 2 | 5 | 5 | 4 | 9 | 0.90 |
| RS | 14 | 26 | 0 | 0 | 14 | 11 | 6 | 8 | 12 | 6 | 0.76 |
| SI | 31 | 23 | 0 | 0 | 16 | 11 | 8 | 3 | 0 | 3 | 0.91 |
| SK | 26 | 16 | 2 | 2 | 17 | 5 | 7 | 4 | 8 | 8 | 0.86 |

The remaining share is the generation forecast and the net position (0–8% each). "Seed agreement" is the mean Spearman correlation of the feature rankings between the 5 seeds.

- **Target history** (price and volume at t and before) is the largest or second-largest category almost everywhere (11–32%), highest in EE, ES, SI, GR, CH and SK. The current imbalance volume is a top-3 driver of the price in AT, BE, DE, FR, GR and RO, and the top driver of the volume nearly everywhere.
- **Generation and the wind/solar forecasts** together take 44–51% in BE, DE and HU (actual generation covers every production type); in GR, whose generation is dropped, the wind/solar forecasts alone take 29%. The intraday solar forecast is the top price driver in BE, DE and HU; actual solar generation in RO and LV.
- **The day-ahead price** is the top price driver in AT, FR, GR, HR, PL, PT and RS.
- **Run-of-river hydro** is the top CH price driver, as before.
- **Balancing-energy prices** matter only in FI, PL and RS (11–12%); they are dropped or missing in BE, ES, HU and LV.
- **RS:** after the volume correction, "generation - Biomass" is the network's top driver of the RS volume and second of the price. RS biomass generation is small, so this is more likely a proxy (e.g. for the time of year) than a cause; with seed agreement 0.76, the RS ranking is the least stable.
- **CZ, SI, EE** (added 2026-11-18):
  - **CZ:** the current imbalance volume is the top driver of both price and volume. The load forecast and the day-ahead net position follow; the net position takes 6.5%, the most of any country.
  - **SI:** target history (31%) and the day-ahead price lead. Actual solar generation is the third price driver and the second volume driver.
  - **EE:** target history (32%) and weather (25%, the most of any country) dominate. EE has no intraday and only a partial day-ahead wind/solar forecast, so ERA5 wind and radiation stand in for them.
- **Weather** takes 9–25%. It is ERA5 reanalysis used as a perfect forecast (§8), so its share is an upper bound.
- Seed agreement is 0.76–0.96; lowest in RS, CH and CZ, whose rankings should be read with care.

import altair as alt
import polars as pl


def lag_chart(df, width=800, height=400):

    # Base encodings
    base = (
        alt.Chart(df)
        .encode(
            x=alt.X("country_code:N"),
            y=alt.Y("error:Q", axis=alt.Axis(title=df["error measure"][0], labelFontSize=14, titleFontSize=14)),
            color=alt.Color("country_code:N")
            .scale(scheme="category20")
            .legend(orient="bottom", direction="horizontal", labelFontSize=13, titleFontSize=13, labelLimit=600),
            column=alt.Column("lag:T", header=alt.Header(format="%H:%M")),
        )
        .properties(title=df["Target"][0])
    )

    bar = base.mark_bar()

    # Combine
    return bar  # .properties(width=800, height=300)


def unpivot_errors(df):
    return df.with_columns(
        # Safely remove the suffix (e.g., "_t+15min" or "_t+2h") at the end of the string
        pl.col("Target").str.replace(r"_t\+\d+(?:min|h)$", ""),
        # Extract the number and multiply by the correct milliseconds
        (
            pl.col("Target").str.extract(r"_t\+(\d+)(?:min|h)").cast(pl.Int32)
            * pl.when(pl.col("Target").str.contains(r"_t\+\d+h"))
            .then(60 * 60 * 1000)  # Multiplier for hours
            .otherwise(60 * 1000)  # Multiplier for minutes
        )
        .cast(pl.Datetime("ms"))
        .alias("lag"),
    ).unpivot(
        set(df.columns) - {"Target", "lag", "csv_path"},
        index=["Target", "lag", "csv_path"],
        variable_name="error measure",
        value_name="error",
    )


def compare_results(df):

    targets = df["Target"].unique()
    error_measures = tuple(filter(lambda x: not x.endswith("_STD"), df["error measure"].unique()))
    print(targets.to_list())
    print(error_measures)
    print()
    for meas in error_measures:
        for target in targets:
            chart = lag_chart(df.filter((pl.col("Target") == target) & (pl.col("error measure") == meas)))
            chart.save(f"figures/performance/{meas}_{target.replace(" -","_")}.png", ppi=200)


results = (
    unpivot_errors(pl.read_csv("results/NN_100_seeds_Huber_*_data.csv", include_file_paths="csv_path"))
    .with_columns(pl.col("csv_path").str.extract(r"results/NN_100_seeds_Huber_(\w\w)_data.csv").alias("country_code"))
    .drop("csv_path")
)
print(results)
HUN_df = results.filter(
    (pl.col("country_code") == "HU") & pl.col("error measure").is_in(("RMSE", "MAE", "SMAPE")) & pl.col("Target").str.contains("prices")
).with_columns(pl.col("error") / 360)
results = results.update(HUN_df, on=["Target", "lag", "country_code"])
compare_results(results)

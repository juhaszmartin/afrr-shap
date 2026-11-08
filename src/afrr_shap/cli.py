"""Command-line entry point for the aFRR forecasting pipeline."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(script, countries=None, env=None):
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script)]
    if countries:
        command.append(",".join(countries))
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _run_report(env):
    _run("plot_shap_results.py", env=env)
    _run("plot_sample_days.py", env=env)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "plot_results.py")],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def _parser():
    parser = argparse.ArgumentParser(
        prog="afrr",
        description="Run the aFRR forecasting and SHAP pipeline.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Configuration file, overriding AFRR_CONFIG.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preprocess", "train", "explain", "report", "pipeline"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument(
            "--countries",
            help="Comma-separated country codes, for example DE,FR.",
        )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    if args.config:
        env["AFRR_CONFIG"] = args.config

    countries = args.countries.split(",") if args.countries else None
    if args.command in ("preprocess", "pipeline"):
        _run("preprocess_all.py", countries, env)
        _run("get_entsoe_info.py", env=env)
    if args.command in ("train", "pipeline"):
        _run("train_all_100_seeds.py", countries, env)
    if args.command in ("explain", "pipeline"):
        _run("run_shap_all.py", countries, env)
    if args.command in ("report", "pipeline"):
        _run_report(env)


if __name__ == "__main__":
    main()

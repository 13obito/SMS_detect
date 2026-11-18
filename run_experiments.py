"""Command-line entry point. Run `python run_experiments.py --help`."""
from __future__ import annotations

import argparse
from pathlib import Path

from sms_spam.common import read_json, validate_config
from sms_spam.data import download_data, prepare
from sms_spam.experiments import final_evaluation, run_midsemester, tune


def main():
    code_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="SMS spam experiments; default milestone evaluates validation only.")
    parser.add_argument("--config", type=Path, default=code_root / "config.json")
    parser.add_argument("--workspace", type=Path, default=code_root, help="Directory for data/ and results/.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download the original UCI ZIP and preserve its readme.")
    for command in ["prepare", "midsemester", "compare", "tune"]:
        sub = commands.add_parser(command, help={
            "prepare": "Audit and create all fixed group-disjoint splits.",
            "midsemester": "Fixed word baseline; clean and substitution validation only.",
            "compare": "Controlled word vs character comparison; no augmentation or tuning.",
            "tune": "Four configurations, five seeds; tune on seen validation edits and freeze models."
        }[command])
        sub.add_argument("--download", action="store_true", help="Download data if the raw file is missing.")
        if command != "prepare":
            sub.add_argument("--run-name", default=None)
        if command == "tune":
            sub.add_argument("--low-fpr", action="store_true", help="Also freeze a cutoff targeting <=1%% clean validation ham FPR.")
            sub.add_argument("--rates", type=float, nargs="+", default=None,
                             help="Predeclare final test rates, e.g. 0.1 0.2 0.3. Training uses config edit_probability.")
            sub.add_argument("--bootstrap-unit", choices=["message", "group"], default="message")
    final = commands.add_parser("final", help="Evaluate the reserved test set using frozen settings. Run only at project end.")
    final.add_argument("--frozen-run", default="frozen_development")
    args = parser.parse_args()
    root = args.workspace.resolve()
    if args.command == "download":
        print(download_data(root))
        return
    if args.command == "final":
        final_evaluation(root, args.frozen_run)
        return
    config = read_json(args.config)
    validate_config(config)
    if args.command == "prepare":
        prepare(root, config, args.download)
    elif args.command in ["midsemester", "compare"]:
        run_midsemester(root, config, args.download, args.command == "compare", args.run_name)
    elif args.command == "tune":
        tune(root, config, args.download, args.run_name or "frozen_development",
             args.low_fpr, args.rates, args.bootstrap_unit)


if __name__ == "__main__":
    main()

from src.data.config import load_config_from_yaml, TrMode
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        required=True,
        help="path to config",
    )
    parser.add_argument("--resume", type=str, default="", help="path to resume checkpoint")
    args = parser.parse_args()

    if args.config == "":
        raise ValueError("config file is required")

    config = load_config_from_yaml(args.config)

    if args.resume:
        config.resume = args.resume

    config.mode = TrMode.fit
    return config

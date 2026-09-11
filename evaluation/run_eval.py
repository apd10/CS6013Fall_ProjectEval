"""Run math evaluation against a vLLM OpenAI server."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_eval_examples,
    run_math_eval,
    save_results,
    summarize_predictions,
)


def load_config(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen3.5-4B on a math dataset."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config.",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset path or Hugging Face id. "
            "Defaults to dataset_path in the config."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=(
            "vLLM server port. "
            "Defaults to the port in vllm_base_url."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the number of examples. "
            "Defaults to limit in the config."
        ),
    )

    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        dest="max_concurrency",
        help=(
            "In-flight chat requests against vLLM. "
            "Defaults to max_concurrency in the config."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        dest="max_new_tokens",
        help=(
            "Max new tokens to generate per example. "
            "Defaults to max_new_tokens in the config."
        ),
    )

    return parser.parse_args()


def apply_base_url_port(base_url: str, port: int | None) -> str:
    if port is None:
        return base_url

    parsed = urlparse(base_url)
    hostname = parsed.hostname or "localhost"
    netloc = f"{hostname}:{port}"

    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo = f"{userinfo}:{parsed.password}"
        netloc = f"{userinfo}@{netloc}"

    return urlunparse(parsed._replace(netloc=netloc))


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.dataset is not None:
        cfg["dataset_path"] = args.dataset

    cfg["vllm_base_url"] = apply_base_url_port(
        cfg["vllm_base_url"],
        args.port,
    )

    if args.limit is not None:
        cfg["limit"] = args.limit

    if args.max_concurrency is not None:
        cfg["max_concurrency"] = args.max_concurrency

    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens

    return cfg


def main():

    args = parse_args()

    cfg = apply_cli_overrides(
        load_config(args.config),
        args,
    )

    print("=" * 80, flush=True)
    print("MATH vLLM API EVALUATION", flush=True)
    print("=" * 80, flush=True)

    print(f"Experiment: {cfg['exp_name']}", flush=True)
    print(f"Model:      {cfg['model_name']}", flush=True)
    print(f"Server:     {cfg['vllm_base_url']}", flush=True)
    print(f"Dataset:    {cfg['dataset_path']}", flush=True)
    print(f"Limit:      {cfg.get('limit')}", flush=True)
    print(f"Concurrency: {cfg.get('max_concurrency')}", flush=True)
    print(f"Max tokens: {cfg.get('max_new_tokens')}", flush=True)
    print(f"Config:     {args.config}", flush=True)
    print(flush=True)

    # --------------------------------------------------------
    # Check vLLM API server
    # --------------------------------------------------------

    print("=" * 80, flush=True)
    print("Connecting to vLLM API server", flush=True)
    print("=" * 80, flush=True)

    from openai import OpenAI

    client = OpenAI(
        base_url=cfg["vllm_base_url"],
        api_key=cfg.get("api_key", "EMPTY"),
        timeout=cfg["request_timeout"],
    )

    models = client.models.list()

    print("vLLM API server is available.", flush=True)

    print("Available models:", flush=True)
    for model in models.data:
        print(f"  - {model.id}", flush=True)

    print(flush=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    examples = load_eval_examples(
        dataset_name=cfg["dataset_path"],
        math_instruction=cfg["math_instruction"],
        limit=cfg["limit"],
    )

    print(
        f"Loaded {len(examples)} examples.",
        flush=True,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = run_math_eval(
        examples,
        client,
        model_name=cfg["model_name"],
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        min_p=cfg.get("min_p", 0.0),
        presence_penalty=cfg.get("presence_penalty", 1.5),
        repetition_penalty=cfg.get("repetition_penalty", 1.0),
        enable_thinking=cfg["enable_thinking"],
        max_concurrency=cfg.get("max_concurrency"),
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    summary = summarize_predictions(
        predictions
    )

    print(flush=True)
    print("=" * 80, flush=True)
    print("RESULTS", flush=True)
    print("=" * 80, flush=True)

    print(
        f"Accuracy:   {summary['accuracy']:.4f} "
        f"({summary['num_correct']}/"
        f"{summary['num_examples']})",
        flush=True,
    )

    print(
        f"Parse rate: {summary['parse_rate']:.4f} "
        f"({summary['num_parsed']}/"
        f"{summary['num_examples']})",
        flush=True,
    )

    print("=" * 80, flush=True)

    save_results(
        cfg["output"],
        benchmark="math_eval",
        model_name=cfg["model_name"],
        summary=summary,
        predictions=predictions,
        extra=cfg,
    )


if __name__ == "__main__":
    main()
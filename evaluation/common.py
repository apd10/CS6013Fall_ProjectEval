"""Evaluate a Hugging Face instruction model on a math dataset."""

from __future__ import annotations

import json
import logging
import re
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from math_verify import parse, verify


from openai import OpenAI
from datasets import load_dataset

from math_verify import parse, verify
from math_verify.parser import (
    LatexExtractionConfig,
    ExprExtractionConfig,
)


latex_config = LatexExtractionConfig(
    boxed_match_priority=0,
    try_extract_without_anchor=True,
)

expr_config = ExprExtractionConfig()


@dataclass
class EvalExample:
    example_id: str
    prompt: str
    gold: str
    metadata: dict[str, Any]


@dataclass
class EvalPrediction:
    example_id: str
    gold: str
    prediction: str | None
    prediction_normalized: str | None
    prediction_parsed: list[str]
    gold_normalized: str
    gold_parsed: list[str]
    correct: bool
    response: str
    metadata: dict[str, Any]


def parse_answer(
    answer: str | None,
) -> tuple[str | None, list[str]]:
    """
    Normalize and parse an answer.

    Returns:
        normalized answer
        parsed answer(s), converted to strings
    """

    if answer is None:
        return None, []

    normalized = normalize_answer(answer)

    if normalized is None:
        return None, []

    try:
        wrapped = f"\\({normalized}\\)"

        parsed = parse(
            wrapped,
            extraction_config=[
                latex_config,
                expr_config,
            ],
        )

        parsed_strings = [
            str(x)
            for x in parsed
        ]

        return normalized, parsed_strings

    except Exception as exc:
        logging.warning(
            "Failed to parse answer=%r: %s",
            answer,
            exc,
        )

        return normalized, []


def build_math_prompt(
    problem: str,
    math_instruction: str,
) -> str:

    return (
        f"{math_instruction}\n\n"
        f"Problem:\n"
        f"{problem.strip()}"
    )


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last balanced \\boxed{...} answer."""

    i = text.rfind(r"\boxed")

    if i < 0:
        return None

    start = text.find("{", i)

    if start < 0:
        return None

    depth = 0

    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1

            if depth == 0:
                return text[start + 1:j].strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize common formatting differences."""

    answer = answer.strip()

    # Remove surrounding $ delimiters.
    answer = answer.strip("$").strip()

    # Remove LaTeX spacing commands.
    answer = re.sub(r"\\[,;:!]", "", answer)

    # Normalize common LaTeX commands.
    answer = answer.replace(r"\left", "")
    answer = answer.replace(r"\right", "")

    # Normalize whitespace.
    answer = re.sub(r"\s+", "", answer)

    # Normalize \dfrac and \tfrac to \frac.
    answer = answer.replace(r"\dfrac", r"\frac")
    answer = answer.replace(r"\tfrac", r"\frac")

    # Remove thousands separators.
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)

    # Normalize braces around simple values.
    if (
        len(answer) >= 2
        and answer.startswith("{")
        and answer.endswith("}")
    ):
        answer = answer[1:-1]

    return answer


def answers_match(
    prediction: str | None,
    gold: str,
) -> tuple[
    bool,
    str | None,
    list[str],
    str,
    list[str],
]:

    if prediction is None:
        return (
            False,
            None,
            [],
            normalize_answer(gold),
            [],
        )

    prediction_normalized = normalize_answer(
        prediction
    )

    gold_normalized = normalize_answer(
        gold
    )

    if prediction_normalized is None:
        return (
            False,
            None,
            [],
            gold_normalized,
            [],
        )

    # Parse both answers.
    try:
        prediction_wrapped = (
            f"\\({prediction_normalized}\\)"
        )

        gold_wrapped = (
            f"\\({gold_normalized}\\)"
        )

        predicted_parsed_raw = parse(
            prediction_wrapped,
            extraction_config=[
                latex_config,
                expr_config,
            ],
        )

        gold_parsed_raw = parse(
            gold_wrapped,
            extraction_config=[
                latex_config,
                expr_config,
            ],
        )

        predicted_parsed = [
            str(x)
            for x in predicted_parsed_raw
        ]

        gold_parsed = [
            str(x)
            for x in gold_parsed_raw
        ]

    except Exception as exc:
        logging.warning(
            "math-verify failed: "
            "prediction=%r gold=%r error=%s",
            prediction,
            gold,
            exc,
        )

        return (
            False,
            prediction_normalized,
            [],
            gold_normalized,
            [],
        )

    # Exact match after normalization.
    if prediction_normalized == gold_normalized:
        return (
            True,
            prediction_normalized,
            predicted_parsed,
            gold_normalized,
            gold_parsed,
        )

    # Mathematical equivalence.
    if not predicted_parsed_raw or not gold_parsed_raw:
        return (
            False,
            prediction_normalized,
            predicted_parsed,
            gold_normalized,
            gold_parsed,
        )

    try:
        correct = verify(
            gold_parsed_raw,
            predicted_parsed_raw,
        )

    except Exception as exc:
        logging.warning(
            "verify failed: "
            "prediction=%r gold=%r error=%s",
            prediction,
            gold,
            exc,
        )
        correct = False

    return (
        correct,
        prediction_normalized,
        predicted_parsed,
        gold_normalized,
        gold_parsed,
    )


def get_torch_dtype(dtype: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    return mapping[dtype]


def build_chat_messages(
    prompt: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": prompt,
        }
    ]


def generate_responses(
    client: OpenAI,
    model_name: str,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None,
    presence_penalty: float | None,
    repetition_penalty: float | None,
    enable_thinking: bool,
    max_concurrency: int | None = None,
) -> list[str]:

    n = len(prompts)
    workers = n if not max_concurrency else min(max_concurrency, n)

    extra_body = {
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
        },
    }

    def _one(index: int, prompt: str) -> tuple[int, str]:
        # NOTE: flush=True is required. When stdout is redirected to a file
        # (e.g. under SLURM), Python switches from line-buffering to full
        # block-buffering, so unflushed prints can sit invisible for a long time.
        print(
            f"Submitted request {index}/{n}",
            flush=True,
        )

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                seed=42,
                extra_body=extra_body,
            )
        except Exception:
            print(
                f"FAILED to generate response {index}/{n}",
                flush=True,
            )
            traceback.print_exc()
            raise

        text = response.choices[0].message.content

        if text is None:
            text = ""

        return index, text.strip()

    print(flush=True)
    print("=" * 80, flush=True)
    print(
        f"Sending {n} requests to vLLM "
        f"(concurrency={workers})",
        flush=True,
    )
    print("=" * 80, flush=True)

    responses: list[str] = [""] * n

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
        futures = [
            executor.submit(_one, i, prompt)
            for i, prompt in enumerate(prompts, start=1)
        ]

        for future in as_completed(futures):
            index, text = future.result()
            responses[index - 1] = text
            print(
                f"Received response {index}/{n}",
                flush=True,
            )

    for i, text in enumerate(responses, start=1):
        print(flush=True)
        print("=" * 80, flush=True)
        print(f"MODEL RESPONSE {i}/{n}:", flush=True)
        print(text, flush=True)
        print(flush=True)

    return responses


def load_eval_examples(
    dataset_name: str,
    math_instruction: str,
    limit: int | None = None,
) -> list[EvalExample]:

    print(f"Loading dataset: {dataset_name}", flush=True)

    dataset = load_dataset(
        dataset_name,
        split="train",
    )

    print(
        f"Dataset size: {len(dataset)}",
        flush=True,
    )

    if limit is not None:
        dataset = dataset.select(
            range(min(limit, len(dataset)))
        )

        print(
            f"Using first {len(dataset)} examples.",
            flush=True,
        )

    examples = []

    for row_idx, row in enumerate(dataset):
        print("DEBUG ROW:", flush=True)
        print(repr(row), flush=True)

        # NOTE: wrapped in try/except so that a missing/renamed column
        # (e.g. "problem" / "answer" / "problem_idx" not matching the
        # actual dataset schema) raises a clear, immediately-flushed
        # error instead of silently killing the process right after
        # the last DEBUG ROW print with the traceback only visible in
        # a separate .err log file.
        try:
            problem = row["problem"]
            gold = str(row["answer"])

            examples.append(
                EvalExample(
                    example_id=str(row["problem_idx"]),
                    prompt=build_math_prompt(
                        problem,
                        math_instruction,
                    ),
                    gold=gold,
                    metadata={
                        "problem_type": row.get(
                            "problem_type",
                            None,
                        ),
                    },
                )
            )

        except Exception:
            print(
                f"FAILED to parse row {row_idx}: {row!r}",
                flush=True,
            )
            traceback.print_exc()
            raise

    print(
        f"Created {len(examples)} EvalExample objects.",
        flush=True,
    )

    return examples


def run_math_eval(
    examples: Sequence[EvalExample],
    client: OpenAI,
    *,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    enable_thinking: bool,
    max_concurrency: int | None = None,
) -> list[EvalPrediction]:

    # --------------------------------------------------------
    # Generate model responses
    # --------------------------------------------------------

    prompts = [
        example.prompt
        for example in examples
    ]

    responses = generate_responses(
        client,
        model_name,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        enable_thinking=enable_thinking,
        max_concurrency=max_concurrency,
    )

    # --------------------------------------------------------
    # Evaluate responses
    # --------------------------------------------------------

    predictions: list[EvalPrediction] = []

    for example, response in zip(
        examples,
        responses,
    ):

        # Extract the final answer from \boxed{...}
        prediction = extract_boxed_answer(
            response
        )

        # Normalize, parse, and compare prediction vs gold.
        (
            correct,
            prediction_normalized,
            prediction_parsed,
            gold_normalized,
            gold_parsed,
        ) = answers_match(
            prediction=prediction,
            gold=example.gold,
        )

        # ----------------------------------------------------
        # Print detailed evaluation information
        # ----------------------------------------------------

        print(flush=True)
        print("=" * 80, flush=True)
        print(
            f"Example {example.example_id}",
            flush=True,
        )

        print(
            f"Prediction: "
            f"{prediction!r}",
            flush=True,
        )

        print(
            f"Gold: "
            f"{example.gold!r}",
            flush=True,
        )

        print(
            f"Prediction normalized: "
            f"{prediction_normalized!r}",
            flush=True,
        )

        print(
            f"Gold normalized: "
            f"{gold_normalized!r}",
            flush=True,
        )

        print(
            f"Prediction parsed: "
            f"{prediction_parsed!r}",
            flush=True,
        )

        print(
            f"Gold parsed: "
            f"{gold_parsed!r}",
            flush=True,
        )

        print(
            f"Correct: "
            f"{correct}",
            flush=True,
        )

        print("=" * 80, flush=True)

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        predictions.append(
            EvalPrediction(
                example_id=example.example_id,
                gold=example.gold,
                prediction=prediction,
                prediction_normalized=prediction_normalized,
                prediction_parsed=prediction_parsed,
                gold_normalized=gold_normalized,
                gold_parsed=gold_parsed,
                correct=correct,
                response=response,
                metadata=example.metadata,
            )
        )

    return predictions


def summarize_predictions(
    predictions: Sequence[EvalPrediction],
) -> dict[str, Any]:
    """Calculate evaluation statistics."""

    n = len(predictions)

    n_correct = sum(
        1
        for prediction in predictions
        if prediction.correct
    )

    n_parsed = sum(
        1
        for prediction in predictions
        if prediction.prediction is not None
    )

    return {
        "num_examples": n,
        "num_correct": n_correct,
        "accuracy": (
            n_correct / n
            if n
            else 0.0
        ),
        "num_parsed": n_parsed,
        "parse_rate": (
            n_parsed / n
            if n
            else 0.0
        ),
    }


def save_results(
    output_path: str | Path,
    *,
    benchmark: str,
    model_name: str,
    summary: dict[str, Any],
    predictions: Sequence[EvalPrediction],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save evaluation results as JSON."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "benchmark": benchmark,
        "model_name": model_name,
        "summary": summary,
        "extra": extra or {},
        "predictions": [
            asdict(prediction)
            for prediction in predictions
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote results to {path}", flush=True)
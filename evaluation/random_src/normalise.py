import re
import logging

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


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None

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
) -> bool:

    if prediction is None:
        return False

    prediction = normalize_answer(prediction)
    gold = normalize_answer(gold)

    if prediction is None or gold is None:
        return False

    # Exact match after normalization.
    if prediction == gold:
        return True

    try:
        # math_verify requires LaTeX to be inside
        # a math environment.
        prediction_wrapped = f"\\({prediction}\\)"
        gold_wrapped = f"\\({gold}\\)"

        predicted_parsed = parse(
            prediction_wrapped,
            extraction_config=[
                latex_config,
                expr_config,
            ],
        )

        gold_parsed = parse(
            gold_wrapped,
            extraction_config=[
                latex_config,
                expr_config,
            ],
        )

        if not predicted_parsed:
            return False

        if not gold_parsed:
            return False

        return verify(
            gold_parsed,
            predicted_parsed,
        )

    except Exception as exc:
        logging.warning(
            "math-verify failed: "
            "prediction=%r gold=%r error=%s",
            prediction,
            gold,
            exc,
        )

        return False


# ============================================================
# Test examples
# ============================================================

examples = [
    (r"103", r"103"),
    (r"3375", r"3375"),
    (r"1/576", r"\frac{1}{576}"),
    (r"-984", r"-984"),
    (r"890", r"890"),
    (r"1311/2017", r"\frac{1311}{2017}"),
    (r"\frac{9\sqrt{23}}{23}", r"\frac{9 \sqrt{23}}{23}"),
    (r"1 - \frac{2}{\pi}", r"1-\frac{2}{\pi}"),
    (r"1037", r"1037"),
    (
        r"\text{roots of } 2S^3 - S^2 - 11S + 12 = 0",
        r"\frac{-1+\sqrt{17}}{2}, \frac{-1-\sqrt{17}}{2}",
    ),
    (r"56", r"56"),
    (r"29", r"29"),
    (r"105", r"105"),
    (r"2304", r"2304"),
    (r"200", r"200"),
    (r"6300", r"6300"),
    (r"2 \cdot 26!", r"2^{25} \cdot 26!"),
    (None, r"\frac{2025}{101}"),
    (r"2/3", r"\frac{4}{9}"),
    (None, r"\frac{448}{3}"),
    (r"26", r"26"),
    (r"63", r"63"),
    (r"8\sqrt{10}", r"8\sqrt{10}"),
    (r"100", r"20"),
    (r"\sqrt{23} - 2\sqrt{3}", r"\sqrt{23}-2 \sqrt{3}"),
    (r"9\sqrt{15}", r"9\sqrt{15}"),
    (r"7/18", r"\frac{7}{18}"),
    (r"\sqrt{6}", r"\sqrt{6}"),
    (r"14+4\sqrt{37}", r"14+4\sqrt{37}"),
    (r"35", r"\sqrt{\frac{95}{24}}"),
]


# ============================================================
# Run evaluation
# ============================================================

num_correct = 0

for i, (prediction, gold) in enumerate(
    examples,
    start=1,
):
    correct = answers_match(
        prediction,
        gold,
    )

    if correct:
        num_correct += 1

    print(
        f"Example {i}: "
        f"prediction={prediction!r}, "
        f"gold={gold!r}, "
        f"correct={correct}"
    )


# ============================================================
# Accuracy
# ============================================================

num_examples = len(examples)
accuracy = num_correct / num_examples

print()
print("=" * 80)
print(f"Correct:  {num_correct}/{num_examples}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("=" * 80)
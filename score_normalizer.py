"""Score Normalizer utility node for IQA scores."""
import numpy as np
from .comfy_compat import io
from .iqa_core import get_hash


class IQA_ScoreNormalizer(io.ComfyNode):
    """
    Normalizes IQA scores to a consistent range.
    Useful for combining scores from different metrics that have different scales.
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_ScoreNormalizer",
            display_name="IQA: Score Normalizer",
            category="IQA/Logic",
            inputs=[
                io.Float.Input("score"),
                io.Float.Input("input_min", default=0.0),
                io.Float.Input("input_max", default=1.0),
                io.Float.Input("output_min", default=0.0),
                io.Float.Input("output_max", default=100.0),
                io.Boolean.Input("invert", default=False),
                io.Boolean.Input("clamp", default=True),
            ],
            outputs=[
                io.Float.Output("normalized_score"),
                io.String.Output("score_text"),
                io.Float.Output("raw_scores")
            ]
        )

    @classmethod
    def IS_CHANGED(cls, input_min, input_max, output_min, output_max, invert, clamp, **kwargs):
        return get_hash([input_min, input_max, output_min, output_max, invert, clamp])

    @classmethod
    def VALIDATE_INPUTS(cls, score, input_min, input_max, output_min, output_max, **kwargs):
        if input_min >= input_max:
            return "Input min must be less than input max."
        if output_min >= output_max:
            return "Output min must be less than output max."
        return True

    @classmethod
    def execute(cls, score, input_min, input_max, output_min, output_max, invert, clamp):
        # Handle both single values and lists
        if isinstance(score, (float, int)):
            scores = [float(score)]
        else:
            scores = list(score)

        normalized_scores = []

        for s in scores:
            # Normalize to 0-1 range based on input range
            normalized = (s - input_min) / (input_max - input_min + 1e-10)

            # Invert if requested (lower is better → higher is better)
            if invert:
                normalized = 1.0 - normalized

            # Scale to output range
            output_val = output_min + normalized * (output_max - output_min)

            # Clamp if requested
            if clamp:
                output_val = max(output_min, min(output_max, output_val))

            normalized_scores.append(output_val)

        # Return single value or list depending on input
        if len(normalized_scores) == 1:
            result = normalized_scores[0]
        else:
            result = normalized_scores

        score_text = f"Normalized: {result}"

        return {
            "ui": {"text": [score_text]},
            "result": io.NodeOutput(result, score_text, normalized_scores)
        }

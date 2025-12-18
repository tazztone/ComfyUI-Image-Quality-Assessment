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
                io.Float.Input("score", tooltip="Input score(s) to normalize.\n\nCan be a single value or list of values from batch processing.\nWill be mapped from input range to output range."),
                io.Float.Input("input_min", default=0.0, tooltip="Expected minimum value of the input score.\n\n• Set based on your metric's range\n• LPIPS: 0.0, BRISQUE: 0, SSIM: 0.0\n• Values below this are clipped if 'clamp' is enabled"),
                io.Float.Input("input_max", default=1.0, tooltip="Expected maximum value of the input score.\n\n• Set based on your metric's range\n• LPIPS: 1.0, BRISQUE: 100, SSIM: 1.0\n• Values above this are clipped if 'clamp' is enabled"),
                io.Float.Input("output_min", default=0.0, tooltip="Minimum value of the output range.\n\n• Common: 0.0 for normalized scores\n• Use 0 for percentage outputs"),
                io.Float.Input("output_max", default=100.0, tooltip="Maximum value of the output range.\n\n• Common: 100.0 for percentage scale\n• Use 1.0 for normalized 0-1 range"),
                io.Boolean.Input("invert", default=False, tooltip="Invert the score direction.\n\n• True: Lower input → higher output\n  - Use for metrics where lower = better (LPIPS, noise)\n  - Makes all metrics 'higher = better'\n\n• False: Preserve original direction"),
                io.Boolean.Input("clamp", default=True, tooltip="Clamp output to the specified range.\n\n• True: Values outside output range are clipped (recommended)\n• False: Allow extrapolation beyond range"),
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

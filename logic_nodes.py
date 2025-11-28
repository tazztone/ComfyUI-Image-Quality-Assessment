import torch
from .comfy_compat import io
from .iqa_core import get_hash

class PyIQA_EnsembleNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PyIQA_EnsembleNode",
            display_name="IQA: Ensemble Scorer",
            category="IQA/Logic",
            inputs=[
                io.Float.Input("score_1", default=0.0),
                io.Float.Input("weight_1", default=1.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("score_2", default=0.0),
                io.Float.Input("weight_2", default=1.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("score_3", default=0.0),
                io.Float.Input("weight_3", default=1.0, min=0.0, max=10.0, step=0.1),
                io.Float.Input("score_4", default=0.0),
                io.Float.Input("weight_4", default=1.0, min=0.0, max=10.0, step=0.1),
            ],
            outputs=[
                io.Float.Output("weighted_score"),
                io.String.Output("score_text")
            ]
        )

    @classmethod
    def execute(cls, score_1, weight_1, score_2, weight_2, score_3, weight_3, score_4, weight_4):
        # Handle list inputs (batches)
        # Inputs might be floats or lists of floats.

        # Helper to ensure list
        def to_list(val, ref_len=None):
            if isinstance(val, (float, int)):
                if ref_len:
                    return [val] * ref_len
                return [val]
            return val

        # Determine batch size from scores
        scores = [score_1, score_2, score_3, score_4]
        batch_len = 1
        for s in scores:
            if isinstance(s, list):
                batch_len = max(batch_len, len(s))

        # Normalize inputs
        s1 = to_list(score_1, batch_len)
        s2 = to_list(score_2, batch_len)
        s3 = to_list(score_3, batch_len)
        s4 = to_list(score_4, batch_len)

        # Weights are usually scalars from widgets, but if they were inputs...
        # Assuming weights are scalar for now as they are simple inputs.

        final_scores = []
        total_weight = weight_1 + weight_2 + weight_3 + weight_4

        if total_weight == 0:
            return io.NodeOutput(0.0, "Weights sum to zero")

        for i in range(batch_len):
            val = (s1[i] * weight_1) + (s2[i] * weight_2) + (s3[i] * weight_3) + (s4[i] * weight_4)
            final_scores.append(val / total_weight)

        # If batch size is 1, return scalar, else return list?
        # ComfyUI nodes usually return scalar if not LIST output.
        # But if we want to pass this to another ensemble, we need list.
        # Let's return the list if batch > 1, or scalar if 1.
        # But for V3 type safety, we should be consistent.
        # The prompt/review noted broken logic for lists.
        # If we return a list, and downstream expects float, Comfy handles it if mapped?
        # Actually, let's return the list if we have a list input.

        if batch_len == 1:
            res = final_scores[0]
        else:
            res = final_scores

        return io.NodeOutput(res, f"Ensemble Score: {res}")


class IQA_ThresholdFilter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_ThresholdFilter",
            display_name="IQA: Threshold Filter",
            category="IQA/Logic",
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("scores", forceInput=True),
                io.Float.Input("threshold", default=0.5, step=0.01),
                io.Enum.Input("operation", ["greater", "less"], default="greater")
            ],
            outputs=[
                io.Image.Output("passed_images"),
                io.Float.Output("passed_scores"),
                io.Image.Output("failed_images"),
                io.Float.Output("failed_scores")
            ]
        )

    @classmethod
    def execute(cls, images, scores, threshold, operation):
        if isinstance(scores, (float, int)):
            scores = [scores] * images.shape[0]

        passed_indices = []
        failed_indices = []

        # Handle mismatch
        count = min(len(scores), images.shape[0])

        for i in range(count):
            score = scores[i]
            pass_check = False
            if operation == "greater":
                if score > threshold: pass_check = True
            else:
                if score < threshold: pass_check = True

            if pass_check:
                passed_indices.append(i)
            else:
                failed_indices.append(i)

        # Helper to construct batch
        def build_batch(indices):
            if not indices:
                # Empty batch - return 1 black pixel
                # [1, H, W, C]
                return torch.zeros((1, images.shape[1], images.shape[2], images.shape[3])), [0.0]

            imgs = images[indices]
            sc = [scores[i] for i in indices]
            return imgs, sc

        p_imgs, p_sc = build_batch(passed_indices)
        f_imgs, f_sc = build_batch(failed_indices)

        return io.NodeOutput(p_imgs, p_sc, f_imgs, f_sc)


class IQA_BatchRanker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IQA_BatchRanker",
            display_name="IQA: Batch Ranker",
            category="IQA/Logic",
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("scores", forceInput=True),
                io.Enum.Input("order", ["descending", "ascending"], default="descending"),
                io.Int.Input("take_top_n", default=0, min=0)
            ],
            outputs=[
                io.Image.Output("sorted_images"),
                io.Float.Output("sorted_scores")
            ]
        )

    @classmethod
    def execute(cls, images, scores, order, take_top_n):
        if isinstance(scores, (float, int)):
            scores = [scores] * images.shape[0]

        if len(scores) != images.shape[0]:
            # Mismatch, return as is
            return io.NodeOutput(images, scores)

        data = list(zip(range(len(scores)), scores))
        reverse = (order == "descending")
        data.sort(key=lambda x: x[1], reverse=reverse)

        sorted_indices = [x[0] for x in data]
        sorted_scores = [x[1] for x in data]

        if take_top_n > 0 and take_top_n < len(sorted_indices):
            sorted_indices = sorted_indices[:take_top_n]
            sorted_scores = sorted_scores[:take_top_n]

        sorted_images = images[sorted_indices]

        return io.NodeOutput(sorted_images, sorted_scores)

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io as py_io
from PIL import Image
from sklearn.cluster import KMeans
import skimage.restoration

from comfy_compat import io
from iqa_core import aggregate_scores, tensor_to_numpy, get_hash, InferenceError

# Set matplotlib backend to Agg to avoid UI issues
plt.switch_backend('Agg')

class Analysis_BlurDetection(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Blur Detection",
            display_name="Analysis: Blur Detection",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_blur_map", default=True),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("blur_score"),
                io.Image.Output("blur_map"),
                io.String.Output("interpretation"),
                io.Float.Output("raw_scores")
            ]
        )

    @staticmethod
    def interpret_blur(score):
        if score < 50:
            return f"Very blurry ({score:.1f})"
        elif score < 150:
            return f"Slightly blurry ({score:.1f})"
        elif score < 300:
            return f"Acceptably sharp ({score:.1f})"
        else:
            return f"Very sharp ({score:.1f})"

    @classmethod
    def execute(cls, image, block_size, visualize_blur_map, aggregation):
        scores = []
        maps = []
        interpretations = []

        try:
            img_list = tensor_to_numpy(image)
            for img_np in img_list:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                h, w = gray.shape
                h_blocks = h // block_size
                w_blocks = w // block_size

                blur_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
                block_scores = []

                for i in range(h_blocks):
                    for j in range(w_blocks):
                        block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                        lap = cv2.Laplacian(block, cv2.CV_64F)
                        var = np.var(lap)
                        blur_map[i, j] = var
                        block_scores.append(var)

                score = float(np.mean(block_scores)) if block_scores else 0.0
                scores.append(score)
                interpretations.append(cls.interpret_blur(score))

                if visualize_blur_map:
                    vis_up = cv2.resize(blur_map, (w, h), interpolation=cv2.INTER_NEAREST)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(vis_up, cmap="viridis", aspect="equal")
                    ax.axis("off")
                    cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                    cbar = plt.colorbar(im, cax=cbar_ax)
                    cbar.set_label("Blur Strength (Laplacian Variance)", fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                    cbar.ax.yaxis.set_label_position("left")
                    cbar.ax.yaxis.set_ticks_position("left")

                    buf = py_io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    blur_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    blur_rgb = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
                    maps.append(torch.from_numpy(blur_rgb.astype(np.float32) / 255.0))
                else:
                    maps.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Blur detection failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        final_interp = cls.interpret_blur(final_score)
        maps_out = torch.stack(maps) if maps else torch.zeros((1, 64, 64, 3))

        return {
            "ui": {"text": [f"Blur Score: {final_score:.2f} ({final_interp})"]},
            "result": io.NodeOutput(final_score, maps_out, final_interp, scores)
        }

class Analysis_ColorHarmony(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Color Harmony Analyzer",
            display_name="Analysis: Color Harmony",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("num_clusters", default=3, min=2, max=8),
                io.Boolean.Input("visualize_harmony", default=True),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("harmony_score"),
                io.String.Output("harmony_type"),
                io.Image.Output("hue_wheel_visual"),
                io.Float.Output("raw_scores")
            ]
        )

    @staticmethod
    def hue_distance(h1, h2):
        return min(abs(h1 - h2), 180 - abs(h1 - h2))

    @staticmethod
    def match_harmony(hues):
        if not hues or len(hues) < 2:
            return "Insufficient hues", 0.0

        scores = {}
        diffs = [Analysis_ColorHarmony.hue_distance(hues[i], hues[j]) for i in range(len(hues)) for j in range(i+1, len(hues))]

        if any(170 <= d <= 190 for d in diffs):
            scores["Complementary"] = 1.0
        if all(d < 30 for d in diffs):
            scores["Analogous"] = 1.0
        if any(110 <= d <= 130 for d in diffs) and len(hues) >= 3:
            scores["Triadic"] = 1.0

        if len(hues) >= 3:
            sorted_hues = np.sort(hues)
            for i in range(len(sorted_hues)):
                base = sorted_hues[i]
                others = sorted_hues[:i].tolist() + sorted_hues[i+1:].tolist()
                split1 = (base + 150) % 180
                split2 = (base + 210) % 180
                split_hits = sum(min(abs(o - s), 180 - abs(o - s)) < 20 for o in others for s in [split1, split2])
                if split_hits >= 2:
                    scores["Split-Complementary"] = 1.0
                    break

        if len(hues) >= 4:
            extended_hues = sorted(hues + [(hues[0] + 180) % 180])
            distances = np.diff(extended_hues)
            if len(distances) >= 4:
                std_dev = np.std(distances)
                if std_dev < 20:
                    scores["Square"] = 1.0
                elif all(40 <= d <= 70 for d in distances):
                    scores["Tetradic"] = 1.0

        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            return best[0], best[1]
        return "No clear harmony", 0.0

    @classmethod
    def execute(cls, image, num_clusters, visualize_harmony, aggregation):
        scores = []
        types = []
        visuals = []

        try:
            img_list = tensor_to_numpy(image)
            for img_np in img_list:
                hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                h = hsv_img[:, :, 0].reshape(-1, 1)
                kmeans = KMeans(n_clusters=num_clusters, n_init="auto").fit(h)

                dominant_hues = []
                if len(kmeans.cluster_centers_) > 0:
                     dominant_hues = sorted([int(center[0]) for center in kmeans.cluster_centers_])

                if not dominant_hues:
                    scores.append(0.0)
                    types.append("No dominant hues")
                    visuals.append(torch.zeros((64, 64, 3), dtype=torch.float32))
                    continue

                harmony_type, score = cls.match_harmony(dominant_hues)
                scores.append(float(score))
                types.append(harmony_type)

                if visualize_harmony:
                    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
                    hue_angles = [2 * np.pi * h / 180 for h in dominant_hues]
                    ax.set_theta_direction(-1)
                    ax.set_theta_zero_location('N')
                    ax.set_yticklabels([])
                    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
                    ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])
                    for hue in hue_angles:
                        ax.plot([hue], [1], marker='o', markersize=12, color=plt.cm.hsv(hue / (2 * np.pi)))
                    ax.set_title(harmony_type, fontsize=10)

                    buf = py_io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    vis = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    visuals.append(torch.from_numpy(vis_rgb.astype(np.float32) / 255.0))
                else:
                    visuals.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Color harmony analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        vis_out = torch.stack(visuals) if visuals else torch.zeros((1, 64, 64, 3))

        # For types, just join unique ones or pick most common?
        # Let's return the type corresponding to the first image for now, or list them.
        # User requirement was "update return values ... so your frontend extension displays the results".
        # We can put details in UI text.
        unique_types = sorted(list(set(types)))
        type_str = ", ".join(unique_types) if len(unique_types) < 4 else f"{len(unique_types)} types"

        return {
            "ui": {"text": [f"Harmony Score: {final_score:.2f}", f"Types: {type_str}"]},
            "result": io.NodeOutput(final_score, type_str, vis_out, scores)
        }

class Analysis_Clipping(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Clipping Analysis",
            display_name="Analysis: Clipping",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("mode", ["Highlight/Shadow Clipping", "Saturation Clipping"]),
                io.Int.Input("threshold", default=5, min=1, max=50),
                io.Boolean.Input("visualize_clipping_map", default=True),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("clipping_score"),
                io.Image.Output("clipping_map"),
                io.String.Output("interpretation"),
                io.Float.Output("raw_scores")
            ]
        )

    @classmethod
    def execute(cls, image, mode, threshold, visualize_clipping_map, aggregation):
        scores = []
        maps = []
        interpretations = []

        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                h, w, _ = uint8_img.shape

                if mode == "Highlight/Shadow Clipping":
                    gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
                    shadows = gray <= threshold
                    highlights = gray >= 255 - threshold
                    mask = np.zeros_like(uint8_img)
                    mask[shadows] = [0, 0, 255]      # blue for shadows
                    mask[highlights] = [255, 0, 0]   # red for highlights
                    total_clipped = np.count_nonzero(shadows | highlights)
                    interp = f"Clipped: {100 * total_clipped / (h * w):.2f}%"
                else:  # Saturation Clipping
                    hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)
                    s_channel = hsv[:, :, 1]
                    v_channel = hsv[:, :, 2]
                    saturation_mask = (s_channel >= 255 - threshold) & (v_channel >= 255 - threshold)
                    mask = np.zeros_like(uint8_img)
                    mask[saturation_mask] = [255, 0, 255]  # magenta
                    total_clipped = np.count_nonzero(saturation_mask)
                    interp = f"Sat Clipped: {100 * total_clipped / (h * w):.2f}%"

                score = total_clipped / (h * w)
                scores.append(score)
                interpretations.append(interp)

                if visualize_clipping_map:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(mask)
                    ax.axis("off")
                    buf = py_io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    map_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
                    maps.append(torch.from_numpy(map_rgb.astype(np.float32) / 255.0))
                else:
                    maps.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Clipping analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        maps_out = torch.stack(maps)
        # Use first interpretation or a summary
        final_interp = interpretations[0] if interpretations else ""

        return {
            "ui": {"text": [f"Clipping Score: {final_score:.4f}"]},
            "result": io.NodeOutput(final_score, maps_out, final_interp, scores)
        }

class Analysis_ColorCast(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Color Cast Detector",
            display_name="Analysis: Color Cast",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("tolerance", default=0.05, min=0.01, max=0.5),
                io.Boolean.Input("visualize_color_bias", default=True),
                io.Enum.Input("visualization_mode", ["Channel Difference", "Neutrality Deviation"]),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("cast_score"),
                io.Image.Output("color_bias_map"),
                io.String.Output("interpretation"),
                io.Float.Output("raw_scores")
            ]
        )

    @classmethod
    def execute(cls, image, tolerance, visualize_color_bias, visualization_mode, aggregation):
        scores = []
        maps = []
        interpretations = []

        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                mean_rgb = np.mean(uint8_img.reshape(-1, 3), axis=0)
                mean_norm = mean_rgb / (np.sum(mean_rgb) + 1e-6)

                ref = 1.0 / 3
                delta = mean_norm - ref
                cast_score = float(np.max(np.abs(delta)))

                dominant = np.argmax(delta)
                weakest = np.argmin(delta)
                channels = ['Red', 'Green', 'Blue']
                dominant_name = channels[dominant]
                weakest_name = channels[weakest]

                if cast_score < tolerance:
                    interp = "No significant color cast"
                else:
                    interp = f"Cast: {dominant_name} tint"

                scores.append(cast_score)
                interpretations.append(interp)

                if visualize_color_bias:
                    if visualization_mode == "Channel Difference":
                        diff_rg = uint8_img[:, :, 0].astype(np.int16) - uint8_img[:, :, 1].astype(np.int16)
                        diff_gb = uint8_img[:, :, 1].astype(np.int16) - uint8_img[:, :, 2].astype(np.int16)
                        diff_rb = uint8_img[:, :, 0].astype(np.int16) - uint8_img[:, :, 2].astype(np.int16)
                        diff_map = np.stack([
                            np.clip(diff_rg + 128, 0, 255),
                            np.clip(diff_gb + 128, 0, 255),
                            np.clip(diff_rb + 128, 0, 255)
                        ], axis=-1).astype(np.uint8)
                    else:
                        r, g, b = uint8_img[:, :, 0], uint8_img[:, :, 1], uint8_img[:, :, 2]
                        avg = ((r + g + b) / 3).astype(np.uint8)
                        deviation = np.abs(uint8_img - avg[:, :, np.newaxis])
                        diff_map = np.clip(deviation * 2, 0, 255).astype(np.uint8)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(diff_map)
                    ax.axis("off")
                    buf = py_io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    vis_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    maps.append(torch.from_numpy(vis_rgb.astype(np.float32) / 255.0))
                else:
                    maps.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Color cast detection failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        maps_out = torch.stack(maps)
        final_interp = interpretations[0] if interpretations else ""

        return {
            "ui": {"text": [f"Cast Score: {final_score:.4f}"]},
            "result": io.NodeOutput(final_score, maps_out, final_interp, scores)
        }

class Analysis_ColorTemperature(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Color Temperature Estimator",
            display_name="Analysis: Color Temperature",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Int.Output("kelvin"),
                io.String.Output("temperature_label"),
                io.Image.Output("color_swatch"),
                io.Float.Output("raw_scores")
            ]
        )

    @staticmethod
    def _estimate_color_temperature(img_uint8):
        img_f = img_uint8.astype(np.float32) / 255.0
        avg = img_f.mean(axis=(0,1)).flatten()[:3]
        r, g, b = avg

        X = 0.412453*r + 0.357580*g + 0.180423*b
        Y = 0.212671*r + 0.715160*g + 0.072169*b
        Z = 0.019334*r + 0.119193*g + 0.950227*b
        denom = X + Y + Z + 1e-6
        x = X/denom; y = Y/denom
        n = (x - 0.3320)/(0.1858 - y + 1e-6)
        CCT = 449*n**3 + 3525*n**2 + 6823.3*n + 5520.33

        kelvin = int(round(CCT))
        if kelvin < 3000: lab = "Warm"
        elif kelvin < 4500: lab = "Neutral"
        elif kelvin < 6500: lab = "Cool Daylight"
        else: lab = "Blueish / Overcast"

        return kelvin, lab, avg

    @classmethod
    def execute(cls, image, aggregation):
        kelvins = []
        labels = []
        swatches = []

        try:
            img_list = tensor_to_numpy(image)
            for img_uint8 in img_list:
                kelvin, label, avg_rgb = cls._estimate_color_temperature(img_uint8)
                kelvins.append(kelvin)
                labels.append(label)

                fig, ax = plt.subplots(figsize=(1.28, 0.64), dpi=100)
                ax.axis("off")
                swatch_arr = np.ones((64, 128, 3), dtype=np.float32) * avg_rgb.reshape(1,1,3)
                ax.imshow(swatch_arr)
                text_color = "black" if avg_rgb.sum() > 1.5 else "white"
                ax.text(0.02, 0.6, f"{kelvin}K", color=text_color, fontsize=12, transform=ax.transAxes)

                buf = py_io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                buf.seek(0)
                img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                swatches.append(torch.from_numpy(img_rgb.astype(np.float32) / 255.0))

        except Exception as e:
            raise InferenceError(f"Color temperature estimation failed: {str(e)}")

        final_kelvin_float = aggregate_scores(kelvins, aggregation)
        final_kelvin = int(final_kelvin_float)

        # Re-estimate label for final kelvin
        if final_kelvin < 3000: final_lab = "Warm"
        elif final_kelvin < 4500: final_lab = "Neutral"
        elif final_kelvin < 6500: final_lab = "Cool Daylight"
        else: final_lab = "Blueish / Overcast"

        swatches_out = torch.stack(swatches)

        return {
            "ui": {"text": [f"Temp: {final_kelvin}K ({final_lab})"]},
            "result": io.NodeOutput(final_kelvin, final_lab, swatches_out, kelvins)
        }

class Analysis_Contrast(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Contrast Analysis",
            display_name="Analysis: Contrast",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("method", ["Global", "Local", "Hybrid"], default="Hybrid"),
                io.Enum.Input("comparison_method", ["Michelson", "RMS", "Weber"], default="RMS"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_contrast_map", default=True),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("contrast_score"),
                io.Image.Output("contrast_map"),
                io.Float.Output("raw_scores")
            ]
        )

    @classmethod
    def execute(cls, image, method, comparison_method, block_size, visualize_contrast_map, aggregation):
        scores = []
        maps = []

        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
                h, w = gray.shape
                blocks = []

                if method in ["Local", "Hybrid"]:
                    for y in range(0, h, block_size):
                        for x in range(0, w, block_size):
                            block = gray[y:y + block_size, x:x + block_size]
                            if block.size == 0: continue
                            if comparison_method == "Michelson":
                                c = (block.max() - block.min()) / (block.max() + block.min() + 1e-6)
                            elif comparison_method == "RMS":
                                c = block.std()
                            elif comparison_method == "Weber":
                                c = (block.max() - block.mean()) / (block.mean() + 1e-6)
                            blocks.append(c)
                    local_contrast = np.mean(blocks) if blocks else 0
                else:
                    local_contrast = 0

                if method in ["Global", "Hybrid"]:
                    if comparison_method == "Michelson":
                        global_contrast = (gray.max() - gray.min()) / (gray.max() + gray.min() + 1e-6)
                    elif comparison_method == "RMS":
                        global_contrast = gray.std()
                    elif comparison_method == "Weber":
                        global_contrast = (gray.max() - gray.mean()) / (gray.mean() + 1e-6)
                else:
                    global_contrast = 0

                if method == "Global": score = global_contrast
                elif method == "Local": score = local_contrast
                else: score = (global_contrast + local_contrast) / 2

                scores.append(float(score))

                if visualize_contrast_map and method != "Global":
                    map_h = (h + block_size - 1) // block_size
                    map_w = (w + block_size - 1) // block_size
                    contrast_map = np.zeros((map_h, map_w), dtype=np.float32)
                    idx = 0
                    for y in range(map_h):
                        for x in range(map_w):
                            if idx < len(blocks):
                                contrast_map[y, x] = blocks[idx]
                                idx += 1

                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(contrast_map, cmap="inferno", aspect="equal")
                    ax.axis("off")
                    cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                    cbar = plt.colorbar(im, cax=cbar_ax)
                    cbar.set_label("Local Contrast", fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                    cbar.ax.yaxis.set_label_position("left")
                    cbar.ax.yaxis.set_ticks_position("left")

                    buf = py_io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    maps.append(torch.from_numpy(img.astype(np.float32) / 255.0))
                else:
                    maps.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Contrast analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        maps_out = torch.stack(maps)
        return {
            "ui": {"text": [f"Contrast: {final_score:.4f}"]},
            "result": io.NodeOutput(final_score, maps_out, scores)
        }

class Analysis_Defocus(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Defocus Analysis",
            display_name="Analysis: Defocus",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("method", ["FFT Ratio (Sum)", "FFT Ratio (Mean)", "Hybrid (Mean+Sum)", "Edge Width"]),
                io.Boolean.Input("normalize", default=True),
                io.Enum.Input("edge_detector", ["Sobel", "Canny"], default="Sobel", optional=True),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("defocus_score"),
                io.String.Output("interpretation"),
                io.Image.Output("fft_heatmap"),
                io.Image.Output("high_freq_mask"),
                io.Float.Output("raw_scores")
            ]
        )

    @staticmethod
    def fft_analysis(gray, method):
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        log_mag = np.log1p(magnitude)
        norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

        cx, cy = magnitude.shape[1] // 2, magnitude.shape[0] // 2
        y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
        radius = min(cx, cy) // 4
        mask = (x - cx)**2 + (y - cy)**2 > radius**2

        hf_mag = magnitude * mask
        masked_norm = np.log1p(hf_mag)
        masked_vis = cv2.normalize(masked_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_img = cv2.applyColorMap(masked_vis, cv2.COLORMAP_TURBO)

        sum_val = np.sum(magnitude[mask])
        mean_val = np.mean(magnitude[mask])
        total_val = np.sum(magnitude)

        fft_score_sum = 1.0 - (sum_val / (total_val + 1e-9))
        fft_score_mean = 1.0 - (mean_val / (magnitude.mean() + 1e-9))

        if method == "FFT Ratio (Sum)": score = fft_score_sum
        elif method == "FFT Ratio (Mean)": score = fft_score_mean
        else: score = 0.5 * fft_score_sum + 0.5 * fft_score_mean

        return score, heatmap, mask_img

    @staticmethod
    def edge_width_analysis(gray, detector):
        if detector not in ["Sobel", "Canny"]: detector = "Sobel"

        if detector == "Sobel":
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0) + cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        else:
            edges = cv2.Canny(gray, 100, 200)

        abs_edges = np.abs(edges)
        edge_vis = cv2.normalize(abs_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_vis_color = cv2.applyColorMap(edge_vis, cv2.COLORMAP_TURBO)

        thresholded = (abs_edges > np.mean(abs_edges)).astype(np.uint8)
        dilated = cv2.dilate(thresholded, np.ones((3, 3)))
        eroded = cv2.erode(thresholded, np.ones((3, 3)))
        thickness_map = dilated - eroded

        mask_vis = cv2.merge([(thickness_map * 255).astype(np.uint8)]*3)
        score = np.mean(thickness_map)
        return score, edge_vis_color, mask_vis

    @staticmethod
    def interpret(score):
        if score < 0.2: return f"Very sharp ({score:.2f})"
        elif score < 0.4: return f"Slight defocus ({score:.2f})"
        elif score < 0.6: return f"Moderate defocus ({score:.2f})"
        elif score < 0.8: return f"Significant blur ({score:.2f})"
        else: return f"Severe defocus ({score:.2f})"

    @classmethod
    def execute(cls, image, method, normalize, edge_detector, aggregation):
        scores = []
        heatmaps = []
        masks = []

        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

                if "FFT" in method or method.startswith("Hybrid"):
                    score, fft_vis, mask_vis = cls.fft_analysis(gray, method)
                elif method == "Edge Width":
                    score, fft_vis, mask_vis = cls.edge_width_analysis(gray, edge_detector)
                else:
                    score, fft_vis, mask_vis = 0.0, np.zeros_like(uint8_img), np.zeros_like(uint8_img)

                if normalize:
                    score = max(0.0, min(score, 1.0))

                scores.append(float(score))

                # Convert vis to RGB tensors
                if len(fft_vis.shape) == 2: fft_vis = cv2.cvtColor(fft_vis, cv2.COLOR_GRAY2RGB)
                elif fft_vis.shape[2] == 3: fft_vis = cv2.cvtColor(fft_vis, cv2.COLOR_BGR2RGB) # applyColorMap gives BGR

                if len(mask_vis.shape) == 2: mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)
                elif mask_vis.shape[2] == 3 and not (method=="Edge Width" and "mask_vis" in locals()): # Helper returns specific format
                     # FFT helper mask_vis is applied colormap -> BGR
                     # Edge helper mask_vis is merged -> RGB (manual merge)
                     # Let's check helper:
                     # fft: applyColorMap -> BGR
                     # edge: merge -> RGB (since input was 0-1 mask * 255)
                     if "FFT" in method or method.startswith("Hybrid"):
                         mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
                     # else (Edge Width), mask_vis is already RGB from merge

                heatmaps.append(torch.from_numpy(fft_vis.astype(np.float32) / 255.0))
                masks.append(torch.from_numpy(mask_vis.astype(np.float32) / 255.0))

        except Exception as e:
            raise InferenceError(f"Defocus analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        interp = cls.interpret(final_score)

        heatmaps_out = torch.stack(heatmaps)
        masks_out = torch.stack(masks)

        return {
            "ui": {"text": [f"Defocus: {final_score:.2f} ({interp})"]},
            "result": io.NodeOutput(final_score, interp, heatmaps_out, masks_out, scores)
        }

class Analysis_EdgeDensity(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Edge Density Analysis",
            display_name="Analysis: Edge Density",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("method", ["Canny", "Sobel"], default="Canny"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_edge_map", default=True),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("edge_density_score"),
                io.Image.Output("edge_density_map"),
                io.String.Output("interpretation"),
                io.Image.Output("edge_preview"),
                io.Float.Output("raw_scores")
            ]
        )

    @classmethod
    def execute(cls, image, method, block_size, visualize_edge_map, aggregation):
        scores = []
        maps = []
        previews = []

        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

                if method == "Canny":
                    edges = cv2.Canny(gray, 100, 200)
                else:
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    edges = cv2.magnitude(sobelx, sobely)
                    edges = np.uint8(np.clip(edges / (np.max(edges)+1e-6) * 255, 0, 255))
                    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

                h, w = edges.shape
                h_blocks = h // block_size
                w_blocks = w // block_size
                density_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
                densities = []

                for i in range(h_blocks):
                    for j in range(w_blocks):
                        block = edges[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                        edge_pixels = np.count_nonzero(block)
                        density = edge_pixels / (block_size * block_size)
                        density_map[i, j] = density
                        densities.append(density)

                score = float(np.mean(densities)) if densities else 0.0
                scores.append(score)

                # Preview
                edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                edge_overlay = np.clip(uint8_img * 0.6 + edge_color * 0.4, 0, 255).astype(np.uint8)
                previews.append(torch.from_numpy(edge_overlay.astype(np.float32) / 255.0))

                if visualize_edge_map:
                    vis_up = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_NEAREST)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(vis_up, cmap="magma", vmin=0, vmax=1, aspect="equal")
                    ax.axis("off")
                    cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                    cbar = plt.colorbar(im, cax=cbar_ax)
                    cbar.set_label("Edge Density", fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                    cbar.ax.yaxis.set_label_position("left")
                    cbar.ax.yaxis.set_ticks_position("left")
                    buf = py_io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    map_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
                    maps.append(torch.from_numpy(map_rgb.astype(np.float32) / 255.0))
                else:
                    maps.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Edge density analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        if final_score < 0.05: interp = f"Very smooth ({final_score:.2f})"
        elif final_score < 0.15: interp = f"Soft detail ({final_score:.2f})"
        elif final_score < 0.3: interp = f"Moderate detail ({final_score:.2f})"
        else: interp = f"Dense detail ({final_score:.2f})"

        maps_out = torch.stack(maps)
        previews_out = torch.stack(previews)

        return {
            "ui": {"text": [f"Density: {final_score:.2f} ({interp})"]},
            "result": io.NodeOutput(final_score, maps_out, interp, previews_out, scores)
        }

class Analysis_Entropy(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Entropy Analysis",
            display_name="Analysis: Entropy",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_entropy_map", default=True),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("entropy_score"),
                io.Image.Output("entropy_map"),
                io.String.Output("interpretation"),
                io.Float.Output("raw_scores")
            ]
        )

    @staticmethod
    def compute_entropy(block):
        hist = cv2.calcHist([block], [0], None, [256], [0, 256])
        hist = hist.ravel()
        prob = hist / (np.sum(hist) + 1e-6)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    @staticmethod
    def interpret_entropy(score):
        if score < 2: return f"Very low entropy ({score:.2f})"
        elif score < 4: return f"Low entropy ({score:.2f})"
        elif score < 6: return f"Moderate entropy ({score:.2f})"
        elif score < 7.5: return f"High entropy ({score:.2f})"
        else: return f"Very high entropy ({score:.2f})"

    @classmethod
    def execute(cls, image, block_size, visualize_entropy_map, aggregation):
        scores = []
        maps = []

        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
                h, w = gray.shape
                h_blocks = h // block_size
                w_blocks = w // block_size

                entropy_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
                entropies = []

                for i in range(h_blocks):
                    for j in range(w_blocks):
                        block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                        e = cls.compute_entropy(block)
                        entropy_map[i, j] = e
                        entropies.append(e)

                score = float(np.mean(entropies)) if entropies else 0.0
                scores.append(score)

                if visualize_entropy_map:
                    vis_up = cv2.resize(entropy_map, (w, h), interpolation=cv2.INTER_NEAREST)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(vis_up, cmap="inferno", vmin=0, vmax=8, aspect="equal")
                    ax.axis("off")
                    cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                    cbar = plt.colorbar(im, cax=cbar_ax)
                    cbar.set_label("Entropy (bits)", fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                    cbar.ax.yaxis.set_label_position("left")
                    cbar.ax.yaxis.set_ticks_position("left")
                    buf = py_io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    legend_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    legend_rgb = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
                    maps.append(torch.from_numpy(legend_rgb.astype(np.float32) / 255.0))
                else:
                    maps.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Entropy analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        interp = cls.interpret_entropy(final_score)
        maps_out = torch.stack(maps)

        return {
            "ui": {"text": [f"Entropy: {final_score:.2f}"]},
            "result": io.NodeOutput(final_score, maps_out, interp, scores)
        }

class Analysis_NoiseEstimation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Noise Estimation",
            display_name="Analysis: Noise Estimation",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_noise_map", default=True),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("noise_score"),
                io.Image.Output("noise_map"),
                io.Float.Output("raw_scores")
            ]
        )

    @classmethod
    def execute(cls, image, block_size, visualize_noise_map, aggregation):
        scores = []
        maps = []

        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
                h, w = gray.shape
                h_blocks = h // block_size
                w_blocks = w // block_size

                heatmap = np.zeros((h_blocks, w_blocks), dtype=np.float32)
                block_scores = []

                for i in range(h_blocks):
                    for j in range(w_blocks):
                        block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                        var = np.var(block)
                        heatmap[i, j] = var
                        block_scores.append(var)

                score = float(np.mean(block_scores)) if block_scores else 0.0
                scores.append(score)

                if visualize_noise_map:
                    vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    vis_up = cv2.resize(vis, (w, h), interpolation=cv2.INTER_NEAREST)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(vis_up, cmap='jet', aspect='equal')
                    ax.axis('off')
                    cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                    cbar = plt.colorbar(im, cax=cbar_ax)
                    cbar.set_label('Noise Strength (Variance)', fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                    cbar.ax.yaxis.set_label_position('left')
                    cbar.ax.yaxis.set_ticks_position('left')
                    buf = py_io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    buf.seek(0)
                    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                    legend_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    legend_rgb = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
                    maps.append(torch.from_numpy(legend_rgb.astype(np.float32) / 255.0))
                else:
                    maps.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Noise estimation failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        maps_out = torch.stack(maps)

        return {
            "ui": {"text": [f"Noise: {final_score:.4f}"]},
            "result": io.NodeOutput(final_score, maps_out, scores)
        }

class Analysis_RGBHistogram(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RGB Histogram Renderer",
            display_name="Analysis: RGB Histogram",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output("histogram_image"),
            ]
        )

    @classmethod
    def execute(cls, image):
        histograms = []
        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                red = uint8_img[:, :, 0]
                green = uint8_img[:, :, 1]
                blue = uint8_img[:, :, 2]

                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                ax.hist(red.ravel(), bins=256, color='red', alpha=0.5, label='Red')
                ax.hist(green.ravel(), bins=256, color='green', alpha=0.5, label='Green')
                ax.hist(blue.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
                ax.set_title("RGB Histogram")
                ax.legend()
                fig.tight_layout()

                buf = py_io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, transparent=False, facecolor='white')
                plt.close(fig)
                buf.seek(0)
                pil_image = Image.open(buf).convert("RGB")
                img_np = np.array(pil_image).astype(np.float32) / 255.0
                histograms.append(torch.from_numpy(img_np))

        except Exception as e:
            raise InferenceError(f"RGB Histogram failed: {str(e)}")

        maps_out = torch.stack(histograms)
        return io.NodeOutput(maps_out)

class Analysis_SharpnessFocusScore(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sharpness / Focus Score",
            display_name="Analysis: Sharpness/Focus",
            category="IQA/Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("method", ["Laplacian", "Tenengrad", "Hybrid"], default="Hybrid"),
                io.Boolean.Input("visualize_edges", default=False),
                io.Enum.Input("aggregation", ["mean", "min", "max", "median", "first"], default="mean"),
            ],
            outputs=[
                io.Float.Output("sharpness_score"),
                io.Image.Output("edge_visualization"),
                io.String.Output("interpretation"),
                io.Float.Output("raw_scores")
            ]
        )

    @staticmethod
    def interpret_score(score, method):
        if method == "Laplacian":
            if score < 100: desc = "Very blurry"
            elif score < 300: desc = "Soft focus"
            elif score < 700: desc = "Moderately sharp"
            else: desc = "Very sharp"
        elif method == "Tenengrad":
            if score < 10000: desc = "Very blurry"
            elif score < 25000: desc = "Soft focus"
            elif score < 50000: desc = "Moderately sharp"
            else: desc = "Very sharp"
        elif method == "Hybrid":
            if score < 0.2: desc = "Very blurry"
            elif score < 0.4: desc = "Soft focus"
            elif score < 0.7: desc = "Moderately sharp"
            else: desc = "Very sharp"
        else: desc = "Unknown"
        return desc

    @classmethod
    def execute(cls, image, method, visualize_edges, aggregation):
        scores = []
        maps = []

        try:
            img_list = tensor_to_numpy(image)
            for uint8_img in img_list:
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

                lap = cv2.Laplacian(gray, cv2.CV_64F)
                lap_score = lap.var()

                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
                mag = np.sqrt(gx ** 2 + gy ** 2)
                ten_score = np.mean(mag ** 2)

                if method == "Laplacian":
                    score = lap_score
                    edges = np.abs(lap)
                elif method == "Tenengrad":
                    score = ten_score
                    edges = mag
                elif method == "Hybrid":
                    lap_norm = np.clip(lap_score / 1500, 0, 1)
                    ten_norm = np.clip(ten_score / 50000, 0, 1)
                    score = (lap_norm + ten_norm) / 2
                    edges = np.abs(lap) + mag
                else:
                    score = 0.0
                    edges = np.zeros_like(gray)

                scores.append(float(score))

                if visualize_edges:
                    vis = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
                    maps.append(torch.from_numpy(vis_rgb.astype(np.float32) / 255.0))
                else:
                    maps.append(torch.zeros((64, 64, 3), dtype=torch.float32))

        except Exception as e:
            raise InferenceError(f"Sharpness analysis failed: {str(e)}")

        final_score = aggregate_scores(scores, aggregation)
        interp = cls.interpret_score(final_score, method)
        maps_out = torch.stack(maps)

        return {
            "ui": {"text": [f"Sharpness: {final_score:.2f} ({interp})"]},
            "result": io.NodeOutput(final_score, maps_out, interp, scores)
        }

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.IQA.ScoreDisplay",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.category && nodeData.category.startsWith("IQA") && !nodeData.category.startsWith("IQA/Dataset")) {

            // Add a widget to display the score
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                // Create a text widget
                // In LiteGraph/ComfyUI, text widgets are drawn on canvas.
                // There is no HTML input element unless it's a DOM widget.
                // We just add it and will update its value.
                const w = this.addWidget("text", "last_score", "0.00", function (v) { }, { serialize: false });
            };

            // Update the widget when execution finishes
            const onExecutedOriginal = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecutedOriginal?.apply(this, arguments);

                // The backend returns {"ui": {"text": [score_str]}} which populates 'message.text' here.
                if (message && message.text && message.text[0]) {
                    const score_text = message.text[0];
                    const w = this.widgets.find(w => w.name === "last_score");
                    if (w) {
                        w.value = score_text;
                        // Force redraw to show update immediately
                        if (this.onResize) { this.onResize(this.size); }
                        app.graph.setDirtyCanvas(true, true);
                    }
                }
            }
        }
    },
});

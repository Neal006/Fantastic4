"""
SHAP explainability module for the XGBoost inference server.
Generates per-prediction SHAP values and optional waterfall/bar chart as base64.
"""

import io
import base64
import logging
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("mlinference")


class ShapExplainer:
    """Wraps shap.TreeExplainer for on-demand per-prediction explanations."""

    def __init__(self):
        self._explainer = None
        self._ready = False

    def initialize(self, model):
        """Create the TreeExplainer. Called once after model is loaded."""
        import shap
        logger.info("Initializing SHAP TreeExplainer ...")
        self._explainer = shap.TreeExplainer(model)
        self._ready = True
        logger.info("SHAP explainer ready.")

    @property
    def ready(self) -> bool:
        return self._ready

    def explain(
        self,
        feature_vector: np.ndarray,
        feature_cols: list[str],
        class_names: list[str],
        predicted_class_idx: int,
        generate_plot: bool = True,
        top_n: int = 5,
    ) -> dict:
        """
        Compute SHAP values for a single sample and return structured output.

        Args:
            feature_vector: 1-D scaled feature array (n_features,)
            feature_cols: list of feature column names
            class_names: list of class names
            predicted_class_idx: index of the predicted class
            generate_plot: whether to render a bar chart as base64
            top_n: number of top features to include

        Returns:
            dict with keys:
              - top_features: list of {feature, shap_value, rank}
              - all_values: dict of feature -> shap_value (for the predicted class)
              - class_shap: dict of class_name -> {feature: shap_value} (all classes)
              - plot_base64: base64-encoded PNG of the bar chart (or None)
        """
        if not self._ready:
            return {"error": "SHAP explainer not initialized"}

        X = feature_vector.reshape(1, -1)
        shap_values = self._explainer.shap_values(X)

        # Handle different SHAP output formats:
        # - List of (1, n_features) arrays (older SHAP versions)
        # - Single (1, n_features, n_classes) array (newer SHAP versions for multi-class)
        # - Single (1, n_features) array (binary classification)
        
        if isinstance(shap_values, list):
            # List format: one array per class
            sv_predicted = shap_values[predicted_class_idx][0]  # (n_features,)
            all_class_sv = {
                class_names[i]: {
                    feature_cols[j]: round(float(shap_values[i][0][j]), 6)
                    for j in range(len(feature_cols))
                }
                for i in range(len(shap_values))
            }
        elif len(shap_values.shape) == 3:
            # 3D array format: (1, n_features, n_classes)
            sv_predicted = shap_values[0, :, predicted_class_idx]  # (n_features,)
            all_class_sv = {
                class_names[i]: {
                    feature_cols[j]: round(float(shap_values[0, j, i]), 6)
                    for j in range(len(feature_cols))
                }
                for i in range(len(class_names))
            }
        else:
            # 2D array format: (1, n_features) - binary classification
            sv_predicted = shap_values[0]
            all_class_sv = {
                class_names[0]: {
                    feature_cols[j]: round(float(shap_values[0][j]), 6)
                    for j in range(len(feature_cols))
                }
            }

        # Top N by absolute SHAP value
        abs_sv = np.abs(sv_predicted)
        top_indices = np.argsort(abs_sv)[::-1][:top_n]

        top_features = [
            {
                "feature": feature_cols[idx],
                "shap_value": round(float(sv_predicted[idx]), 6),
                "abs_shap": round(float(abs_sv[idx]), 6),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(top_indices)
        ]

        # All values for the predicted class (sparse — only non-zero)
        all_values = {
            feature_cols[i]: round(float(sv_predicted[i]), 6)
            for i in range(len(feature_cols))
            if abs(sv_predicted[i]) > 1e-6
        }

        # Generate bar chart
        plot_b64 = None
        if generate_plot:
            plot_b64 = self._render_bar_chart(
                top_features, class_names[predicted_class_idx], top_n
            )

        return {
            "top_features": top_features,
            "all_values": all_values,
            "class_shap": all_class_sv,
            "plot_base64": plot_b64,
        }

    def _render_bar_chart(
        self, top_features: list[dict], predicted_class: str, top_n: int
    ) -> Optional[str]:
        """Render a horizontal bar chart of top SHAP features and return base64 PNG."""
        try:
            features = [f["feature"] for f in top_features][::-1]
            values = [f["shap_value"] for f in top_features][::-1]

            colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

            fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.5)))
            ax.barh(features, values, color=colors, height=0.6, edgecolor="none")
            ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
            ax.set_title(
                f"Top {top_n} Feature Contributions — Predicted: {predicted_class}",
                fontsize=13,
                fontweight="bold",
            )
            ax.axvline(x=0, color="#333", linewidth=0.8, alpha=0.6)
            ax.grid(axis="x", alpha=0.2)
            ax.tick_params(axis="y", labelsize=9)
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            b64 = base64.b64encode(buf.read()).decode("utf-8")
            return f"data:image/png;base64,{b64}"

        except Exception as e:
            logger.error("SHAP plot generation failed: %s", e)
            return None


# ── Module-level singleton ───────────────────────────────────────
shap_explainer = ShapExplainer()

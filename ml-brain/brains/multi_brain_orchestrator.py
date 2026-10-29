"""
Multi-Brain Orchestrator (`multi_brain_orchestrator.py`).

Coordinates all 5 Specialized Goal-Horizon AI Brains (`BRAIN-01` through `BRAIN-05`),
linking:
1. The shared Rust `RawFingerprintEngine` (`rust_bridge.py`)
2. The `Mamba-SSM` + `FlexAttention` World Model Backbone (`sequence_backbone.py`)
3. The `LightGBM` + `TreeSHAP` 4-Stream Cause-and-Effect Explainer (`explainer_and_onnx.py`)
4. The Autonomous Options Structure & Greeks Matcher (`options_matcher.py`)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..world_model.explainer_and_onnx import (
    OnnxBrainArtifactExporter,
    OnnxBrainRuntime,
    TreeShapAttributionReport,
    TreeShapRawExplainer,
)
from ..world_model.rust_bridge import (
    RawBarInput,
    RawMarketInternalsInput,
    RawOptionsInput,
    RawShortFloatInput,
    RustAutonomousSignalOutput,
    RustFingerprintBridge,
)
from ..world_model.sequence_backbone import (
    MambaFlexWorldModelBackbone,
    WorldModelPrediction,
)
from .brain_configs import ALL_BRAIN_CONFIGS, BrainHorizonConfig, get_brain_config
from .options_matcher import (
    MatchedOptionsBlueprint,
    OptionCandidateContract,
    OptionsStructureMatcher,
)


@dataclass(frozen=True)
class BrainSignalPackage:
    brain_config: BrainHorizonConfig
    rust_signal: RustAutonomousSignalOutput
    world_model_prediction: WorldModelPrediction
    shap_attribution: TreeShapAttributionReport
    matched_options: MatchedOptionsBlueprint
    latent_embedding_128: list[float]


class MultiBrainOrchestrator:
    """
    Manages the 5 isolated Goal-Horizon AI Brains (`BRAIN-01` through `BRAIN-05`),
    executing inference, structural stop solving, TreeSHAP attribution, Options matching,
    and `.onnx` artifact export.
    """

    def __init__(self, bridge: RustFingerprintBridge | None = None) -> None:
        self.bridge = bridge or RustFingerprintBridge()
        self.backbones: dict[str, MambaFlexWorldModelBackbone] = {
            brain_id: MambaFlexWorldModelBackbone(brain_id=brain_id)
            for brain_id in ALL_BRAIN_CONFIGS
        }
        self.explainers: dict[str, TreeShapRawExplainer] = {
            brain_id: TreeShapRawExplainer(backbone)
            for brain_id, backbone in self.backbones.items()
        }
        self.options_matcher = OptionsStructureMatcher()

    def evaluate_ticker_for_brain(
        self,
        brain_id: str,
        symbol: str,
        bars: Sequence[RawBarInput],
        options: RawOptionsInput | None = None,
        short_float: RawShortFloatInput | None = None,
        internals: RawMarketInternalsInput | None = None,
        is_optionable: bool = True,
        available_contracts: Sequence[OptionCandidateContract] | None = None,
    ) -> BrainSignalPackage:
        cfg = get_brain_config(brain_id)
        opts_in = options or RawOptionsInput(
            dominant_gamma_wall_strike=bars[-1].close_price * 1.02
        )
        sf_in = short_float or RawShortFloatInput()
        int_in = internals or RawMarketInternalsInput()

        # 1. Build scale-independent 4-stream tensor and structural levels via shared Rust engine
        tensor_out = self.bridge.build_sequence_tensor(bars, opts_in, sf_in, int_in)
        rust_sig = self.bridge.evaluate_autonomous_signal(
            brain_id=brain_id,
            symbol=symbol,
            bars=bars,
            options=opts_in,
            short_float=sf_in,
            internals=int_in,
        )

        # 2. Run Mamba-SSM + FlexAttention sequence backbone
        backbone = self.backbones[brain_id]
        wm_pred = backbone.selective_ssm_and_flex_attention_forward(tensor_out.matrix_2d)

        # 3. Compute exact TreeSHAP 4-Stream attribution
        shap_report = self.explainers[brain_id].explain_sequence(tensor_out.matrix_2d)

        # 4. Match optimal Options contract & structure alongside stock shares
        matched_opt = self.options_matcher.match_optimal_structure(
            brain_id=brain_id,
            symbol=symbol,
            direction=rust_sig.direction,
            underlying_entry_price=rust_sig.entry_price,
            underlying_stop_price=rust_sig.structural_stop_price,
            underlying_target_1_price=rust_sig.target_1_price,
            underlying_target_2_price=rust_sig.target_2_price,
            is_optionable=is_optionable,
            front_iv=opts_in.front_iv,
            back_iv=opts_in.back_iv,
            available_contracts=available_contracts,
        )

        return BrainSignalPackage(
            brain_config=cfg,
            rust_signal=rust_sig,
            world_model_prediction=wm_pred,
            shap_attribution=shap_report,
            matched_options=matched_opt,
            latent_embedding_128=tensor_out.latent_embedding_128,
        )

    def export_all_brains_to_onnx(
        self,
        output_dir: Path,
        model_version: str = "v1.0.0-champion",
    ) -> dict[str, Path]:
        exported: dict[str, Path] = {}
        for brain_id, backbone in self.backbones.items():
            path = OnnxBrainArtifactExporter.export_brain_to_onnx(
                backbone=backbone,
                output_dir=output_dir,
                model_version=model_version,
            )
            # Immediately verify the exported ONNX artifact loads cleanly via OnnxBrainRuntime
            _ = OnnxBrainRuntime(path)
            exported[brain_id] = path
        return exported


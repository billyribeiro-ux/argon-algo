"""
The 5 Selectable Goal-Horizon AI Brains & Options Contract Matcher (`ml-brain/brains`).
"""

from .brain_configs import (
    ALL_BRAIN_CONFIGS,
    BrainHorizonConfig,
    compute_brain_reward_objective,
    get_brain_config,
)
from .multi_brain_orchestrator import BrainSignalPackage, MultiBrainOrchestrator
from .options_matcher import (
    MatchedOptionsBlueprint,
    OptionCandidateContract,
    OptionsStructureMatcher,
)

__all__ = [
    "ALL_BRAIN_CONFIGS",
    "BrainHorizonConfig",
    "get_brain_config",
    "compute_brain_reward_objective",
    "OptionCandidateContract",
    "MatchedOptionsBlueprint",
    "OptionsStructureMatcher",
    "BrainSignalPackage",
    "MultiBrainOrchestrator",
]


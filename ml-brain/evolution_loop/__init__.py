"""
The 3-Speed Continuous Self-Learning & Evolution Engine (`ml-brain/evolution_loop`).

- `speed1_intraday`: Intra-Day Tick-by-Tick Dynamic Exit & Confidence Threshold Adapter.
- `speed2_nightly_autopsy`: Daily 4:15 PM EST Post-Market Autopsy, Hard-Negative Mining & Missed-Move Discovery.
- `speed3_weekend_tournament`: Weekend Champion vs. Challenger Tournament with Purged & Embargoed Walk-Forward CV.
"""

from .speed1_intraday import (
    ActivePositionState,
    IntradayAdaptationResult,
    Speed1IntradayAdapter,
)
from .speed2_nightly_autopsy import (
    ClosedSignalRecord,
    MissedMarketMoverRecord,
    NightlyAutopsyReport,
    Speed2NightlyAutopsyEngine,
)
from .speed3_weekend_tournament import (
    Speed3WeekendTournamentEngine,
    TournamentEvaluationMetrics,
    WeekendTournamentResult,
)

__all__ = [
    "ActivePositionState",
    "IntradayAdaptationResult",
    "Speed1IntradayAdapter",
    "ClosedSignalRecord",
    "MissedMarketMoverRecord",
    "NightlyAutopsyReport",
    "Speed2NightlyAutopsyEngine",
    "TournamentEvaluationMetrics",
    "WeekendTournamentResult",
    "Speed3WeekendTournamentEngine",
]


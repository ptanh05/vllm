"""ERS scoring engine — official Phase 1 formula."""
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class ERSConfig:
    """Phase-1 scoring parameters."""
    f_ttft_s: float = 0.100    # TTFT Floor (100ms)
    c_ttft_s: float = 1.500    # TTFT Ceiling (1500ms)
    f_tpot_s: float = 0.020    # TPOT Floor (20ms)
    c_tpot_s: float = 0.045    # TPOT Ceiling (45ms)
    gamma: float = 2.0
    w: float = 0.5

    def to_dict(self):
        return asdict(self)


ACCURACY_DROP_THRESHOLD_OK = 0.10
ACCURACY_DROP_THRESHOLD_FAIL = 0.16


@dataclass
class RequestTiming:
    """Per-request timing data."""
    request_id: str = ""
    success: bool = False
    ttft_s: float = 0.0        # seconds
    tpot_list_s: list[float] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0
    total_latency_s: float = 0.0
    error: str = ""

    @property
    def tpot_mean_s(self) -> float:
        if not self.tpot_list_s:
            return 0.0
        return statistics.mean(self.tpot_list_s)


@dataclass
class ERSRequestScore:
    request_id: str
    s_ttft: float
    s_tpot: float
    s_request: float
    ttft_s: float
    tpot_mean_s: float
    num_output_tokens: int
    success: bool


class ERSScorer:
    """Computes ERS according to the official Phase-1 formula."""

    def __init__(self, config: ERSConfig | None = None):
        self.config = config or ERSConfig()

    def score_one(self, timing: RequestTiming) -> ERSRequestScore:
        if not timing.success or timing.num_output_tokens == 0:
            return ERSRequestScore(
                request_id=timing.request_id,
                s_ttft=0.0, s_tpot=0.0, s_request=0.0,
                ttft_s=timing.ttft_s,
                tpot_mean_s=timing.tpot_mean_s,
                num_output_tokens=timing.num_output_tokens,
                success=False,
            )

        def s_score(value, floor, ceiling):
            if value <= floor:
                return 1.0
            if value >= ceiling:
                return 0.0
            raw = (ceiling - value) / (ceiling - floor)
            return raw ** self.config.gamma

        s_ttft = s_score(timing.ttft_s, self.config.f_ttft_s, self.config.c_ttft_s)
        s_tpot = s_score(timing.tpot_mean_s, self.config.f_tpot_s, self.config.c_tpot_s)
        s_request = self.config.w * s_ttft + (1 - self.config.w) * s_tpot

        return ERSRequestScore(
            request_id=timing.request_id,
            s_ttft=s_ttft, s_tpot=s_tpot, s_request=s_request,
            ttft_s=timing.ttft_s,
            tpot_mean_s=timing.tpot_mean_s,
            num_output_tokens=timing.num_output_tokens,
            success=True,
        )

    def compute_ers(self, scores: list[ERSRequestScore]) -> float:
        if not scores:
            return 0.0
        return statistics.mean([s.s_request for s in scores])


# ---- Accuracy penalty functions ----

def accuracy_penalty(delta: float) -> float:
    """f(Δ) piecewise linear accuracy decay."""
    if delta <= ACCURACY_DROP_THRESHOLD_OK:
        return 1.0
    if delta >= ACCURACY_DROP_THRESHOLD_FAIL:
        return 0.0
    return 1.0 - (delta - ACCURACY_DROP_THRESHOLD_OK) / (
        ACCURACY_DROP_THRESHOLD_FAIL - ACCURACY_DROP_THRESHOLD_OK
    )


def final_score(ers: float, accuracy_drop: float) -> float:
    """Final score = 100 × ERS × f(Δ)."""
    return 100.0 * ers * accuracy_penalty(accuracy_drop)

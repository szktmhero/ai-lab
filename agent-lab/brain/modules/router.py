"""Inspectable routing policy; it cannot solve the task itself."""

from __future__ import annotations

from ..core.types import BrainState, RouterDecision


class SalienceRouter:
    def __init__(self, conflict_threshold: float = 0.2):
        self.conflict_threshold = conflict_threshold

    def decide(
        self,
        state: BrainState,
        *,
        enable_memory: bool,
        enable_error_monitor: bool,
        adaptive: bool,
        state_coupling: bool,
    ) -> RouterDecision:
        if not adaptive:
            retrieve_memory = enable_memory
            robust = enable_error_monitor
            reason = "fixed route: fast + deliberate + robust"
        elif not state_coupling:
            # The ablation retains a fixed memory route while disconnecting all
            # dynamic state from routing decisions.
            retrieve_memory = enable_memory
            robust = False
            reason = "dynamic state observed but causally disconnected"
        else:
            retrieve_memory = enable_memory and (
                state.cognitive_load >= 0.5
                or state.uncertainty >= 0.4
                or state.novelty >= 0.35
            )
            conflict_or_change = (
                state.prediction_error >= self.conflict_threshold
                or state.novelty >= 0.55
            )
            resource_constrained = state.resource_pressure >= 0.9
            robust = (
                enable_error_monitor
                and conflict_or_change
                and not resource_constrained
            )
            if robust:
                reason = "conflict or novelty activated robust cortex"
            elif resource_constrained and conflict_or_change:
                reason = "resource pressure suppressed optional robust processing"
            else:
                reason = "low conflict and novelty kept the economical route"

        modules = ["perception", "bounded_workspace", "fast_controller"]
        if retrieve_memory:
            modules.append("episodic_memory")
        if enable_error_monitor:
            modules.append("error_monitor")
        modules.append("deliberative_cortex")
        if robust:
            modules.append("robust_cortex")
        modules.append("action_interface")
        return RouterDecision(
            activated_modules=tuple(modules),
            retrieve_memory=retrieve_memory,
            use_robust_cortex=robust,
            reason=reason,
        )

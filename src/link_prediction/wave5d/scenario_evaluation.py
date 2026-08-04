"""Scenario evaluation helpers (Wave 5D pilot + Wave 5E full matrix)."""

from __future__ import annotations

from dataclasses import dataclass

SCENARIOS = (
    "clean",
    "hidden_confounder",
    "selection_bias",
    "no_overlap",
    "noisy_documentation",
    "multihospital",
)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    train_scenario: str
    test_scenario: str
    mode: str  # in_scenario | cross_scenario


def in_scenario_specs() -> list[ScenarioSpec]:
    return [
        ScenarioSpec(s, s, s, "in_scenario") for s in SCENARIOS
    ]


def cross_scenario_specs() -> list[ScenarioSpec]:
    return [
        ScenarioSpec(
            name=f"clean_to_{s}",
            train_scenario="clean",
            test_scenario=s,
            mode="cross_scenario",
        )
        for s in SCENARIOS
        if s != "clean"
    ]

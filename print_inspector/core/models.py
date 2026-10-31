"""Shared data types for inspection results."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InspectionStatus(str, Enum):
    """Overall verdict for one inspection.

    NOT_INSPECTED is the state before any comparison has run. REVIEW is used
    whenever the result cannot be trusted (poor capture, unreliable alignment),
    so an uncertain inspection is never reported as PASS.
    """

    NOT_INSPECTED = "not_inspected"
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"

    @property
    def label(self) -> str:
        return "NOT INSPECTED" if self is InspectionStatus.NOT_INSPECTED else self.name


class Severity(str, Enum):
    """WARNING: usable, but results may be degraded. CRITICAL: cannot be trusted."""

    WARNING = "warning"
    CRITICAL = "critical"


class QualityLevel(str, Enum):
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"


@dataclass(frozen=True)
class QualityIssue:
    """One finding of a check stage (capture quality, alignment, ...)."""

    code: str
    severity: Severity
    message: str


def level_from_issues(issues: tuple[QualityIssue, ...] | list[QualityIssue]) -> QualityLevel:
    if any(i.severity is Severity.CRITICAL for i in issues):
        return QualityLevel.POOR
    if issues:
        return QualityLevel.MARGINAL
    return QualityLevel.GOOD


def sort_issues(issues: list[QualityIssue]) -> tuple[QualityIssue, ...]:
    """Critical issues first, otherwise in the order they were found."""
    return tuple(sorted(issues, key=lambda i: i.severity is not Severity.CRITICAL))

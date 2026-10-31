from print_inspector.core.alignment import AlignmentMetrics, AlignmentResult
from print_inspector.core.capture_quality import QualityMetrics, QualityReport
from print_inspector.core.decision import preliminary_status
from print_inspector.core.models import InspectionStatus, QualityIssue, Severity

import numpy as np


def _quality(*severities):
    issues = tuple(QualityIssue(f"q{i}", s, "msg") for i, s in enumerate(severities))
    return QualityReport(metrics=QualityMetrics(image_width=1, image_height=1, megapixels=0.0), issues=issues)


def _alignment(*severities, rectified=True):
    issues = tuple(QualityIssue(f"a{i}", s, "msg") for i, s in enumerate(severities))
    image = np.zeros((2, 2, 3), np.uint8) if rectified else None
    return AlignmentResult(metrics=AlignmentMetrics(), issues=issues, rectified=image)


def test_nothing_run_yet():
    assert preliminary_status(None, None) == (InspectionStatus.NOT_INSPECTED, [])


def test_reliable_inputs_never_pass():
    status, reasons = preliminary_status(_quality(Severity.WARNING), _alignment())
    assert status is InspectionStatus.NOT_INSPECTED and reasons == []


def test_poor_quality_gives_review():
    status, reasons = preliminary_status(_quality(Severity.CRITICAL), _alignment())
    assert status is InspectionStatus.REVIEW and "quality" in reasons[0]


def test_poor_alignment_gives_review():
    status, reasons = preliminary_status(_quality(), _alignment(Severity.CRITICAL))
    assert status is InspectionStatus.REVIEW and "aligned" in reasons[0]


def test_failed_alignment_without_image_gives_review():
    status, _ = preliminary_status(None, _alignment(rectified=False))
    assert status is InspectionStatus.REVIEW


def test_both_reasons_listed():
    _, reasons = preliminary_status(_quality(Severity.CRITICAL), _alignment(Severity.CRITICAL))
    assert len(reasons) == 2


# ---------------------------------------------------------------- registration precedence

from print_inspector.core.registration import RegistrationMetrics, RegistrationResult  # noqa: E402


def _registration(*severities, image=True):
    issues = tuple(QualityIssue(f"r{i}", s, "msg") for i, s in enumerate(severities))
    return RegistrationResult(
        metrics=RegistrationMetrics(), issues=issues, registered=np.zeros((2, 2, 3), np.uint8) if image else None
    )


def _alignment_with(code):
    return AlignmentResult(
        metrics=AlignmentMetrics(), issues=(QualityIssue(code, Severity.CRITICAL, "msg"),),
        rectified=np.zeros((2, 2, 3), np.uint8),
    )


def test_poor_registration_gives_review():
    status, reasons = preliminary_status(_quality(), _alignment(), _registration(Severity.CRITICAL))
    assert status is InspectionStatus.REVIEW and "registered" in reasons[0]


def test_verified_registration_supersedes_indirect_alignment_estimates():
    status, _ = preliminary_status(_quality(), _alignment_with("residual_distortion"), _registration())
    assert status is InspectionStatus.NOT_INSPECTED


def test_alignment_blockers_still_apply():
    for code in ("label_outside_photo", "outline_mismatch", "invalid_corners"):
        status, _ = preliminary_status(_quality(), _alignment_with(code), _registration())
        assert status is InspectionStatus.REVIEW, code


def test_unreliable_registration_does_not_rescue_alignment():
    status, reasons = preliminary_status(
        _quality(), _alignment_with("residual_distortion"), _registration(Severity.CRITICAL)
    )
    assert status is InspectionStatus.REVIEW and len(reasons) == 2


# ---------------------------------------------------------------- comparison

from print_inspector.core.comparison import ComparisonResult  # noqa: E402


def _comparison(*severities):
    issues = tuple(QualityIssue(f"c{i}", s, "msg") for i, s in enumerate(severities))
    empty = np.zeros((2, 2), np.uint8)
    return ComparisonResult(
        metrics=None, issues=issues, candidates=(), expected=np.zeros((2, 2, 3), np.uint8),
        gray_difference=empty, structural_difference=empty, edge_difference=empty,
        combined_score=np.zeros((2, 2), np.float32), candidate_mask=empty,
    )


def test_unreliable_comparison_gives_review():
    status, reasons = preliminary_status(_quality(), _alignment(), _registration(), _comparison(Severity.CRITICAL))
    assert status is InspectionStatus.REVIEW and "comparison" in reasons[0]


def test_candidates_alone_are_not_a_verdict():
    status, _ = preliminary_status(_quality(), _alignment(), _registration(), _comparison(Severity.WARNING))
    assert status is InspectionStatus.NOT_INSPECTED


# ---------------------------------------------------------------- PASS / FAIL / REVIEW

import pytest  # noqa: E402

from print_inspector.config import DecisionConfig  # noqa: E402
from print_inspector.core.decision import OperatorMark, decide  # noqa: E402
from print_inspector.core.defects import (  # noqa: E402
    Defect,
    DefectCategory,
    DefectEvidence,
    DefectReport,
    DefectSeverity,
    DetectionReliability,
)

CFG = DecisionConfig(fail_min_severity="major", max_minor_defects=2, allow_pass_with_warnings=True)


def _defect(defect_id, severity="major", reliable=True):
    blank = np.zeros((4, 4, 3), np.uint8)
    evidence = DefectEvidence((0, 0), blank, blank, blank, blank[:, :, 0], blank[:, :, 0], blank[:, :, 0])
    return Defect(
        id=defect_id, category=DefectCategory.EXTRA_PRINT, category_certain=True, category_note="",
        reliability=DetectionReliability.RELIABLE if reliable else DetectionReliability.UNCERTAIN,
        reliability_notes=(), severity=DefectSeverity(severity), severity_notes=(),
        x=0, y=0, width=4, height=4, centroid=(2.0, 2.0), area_px=10, length_px=4.0, thickness_px=3.0,
        area_mm2=0.1, length_mm=0.3, thickness_mm=0.2, score=3.0, peak_sigma=10.0, max_difference=100.0,
        mean_difference=50.0, signed_difference=-50.0, delta_e=30.0, delta_l=-30.0, delta_c=1.0,
        polarity="darker", ink_fraction=0.0, edge_fraction=0.0, text_overlap=0.0, min_ssim=0.5,
        missing_edge_px=0, extra_edge_px=3, scale=1, candidate_ids=(1,), evidence=evidence,
    )


def _report(*defects):
    return DefectReport(defects=tuple(defects), filtered=(), px_per_mm=10.0, scale_estimated=False,
                        text_mask=np.zeros((2, 2), np.uint8))


def _verdict(report, marks=None, quality=None, cfg=CFG):
    return decide(quality or _quality(), _alignment(), _registration(), _comparison(), report, marks or {}, cfg)


def test_no_defects_pass():
    decision = _verdict(_report())
    assert decision.status is InspectionStatus.PASS


def test_minor_defects_within_limit_pass():
    decision = _verdict(_report(_defect(1, "minor"), _defect(2, "minor")))
    assert decision.status is InspectionStatus.PASS and decision.minor_count == 2
    assert "minor" in decision.reasons[0]


def test_reliable_major_defect_fails():
    decision = _verdict(_report(_defect(1, "major")))
    assert decision.status is InspectionStatus.FAIL and decision.failing == (1,)


def test_uncertain_major_defect_needs_review():
    decision = _verdict(_report(_defect(1, "critical", reliable=False)))
    assert decision.status is InspectionStatus.REVIEW and decision.review == (1,)


def test_operator_confirmation_turns_review_into_fail():
    report = _report(_defect(1, "major", reliable=False))
    decision = _verdict(report, {1: OperatorMark.CONFIRMED})
    assert decision.status is InspectionStatus.FAIL and "Confirmed" in decision.reasons[0]


def test_operator_dismissal_allows_pass():
    report = _report(_defect(1, "critical"), _defect(2, "minor"))
    decision = _verdict(report, {1: OperatorMark.DISMISSED})
    assert decision.status is InspectionStatus.PASS


def test_marked_for_review():
    decision = _verdict(_report(_defect(1, "minor")), {1: OperatorMark.NEEDS_REVIEW})
    assert decision.status is InspectionStatus.REVIEW and decision.review == (1,)


def test_too_many_reliable_minor_defects_fail():
    report = _report(*(_defect(i, "minor") for i in range(1, 4)))
    decision = _verdict(report)
    assert decision.status is InspectionStatus.FAIL and "minor defects" in decision.reasons[0]


def test_too_many_minor_findings_some_uncertain_need_review():
    report = _report(_defect(1, "minor"), _defect(2, "minor", reliable=False), _defect(3, "minor", reliable=False))
    assert _verdict(report).status is InspectionStatus.REVIEW


def test_unreliable_inputs_give_review_even_with_reliable_defect():
    decision = _verdict(_report(_defect(1, "major")), quality=_quality(Severity.CRITICAL))
    assert decision.status is InspectionStatus.REVIEW
    assert any("confirm or dismiss" in r for r in decision.reasons)


def test_confirmed_defect_fails_even_with_unreliable_inputs():
    decision = _verdict(_report(_defect(1, "major")), {1: OperatorMark.CONFIRMED}, quality=_quality(Severity.CRITICAL))
    assert decision.status is InspectionStatus.FAIL


def test_unreliable_inputs_never_pass():
    assert _verdict(_report(), quality=_quality(Severity.CRITICAL)).status is InspectionStatus.REVIEW


def test_warnings_can_be_required_clean():
    strict = DecisionConfig(fail_min_severity="major", max_minor_defects=2, allow_pass_with_warnings=False)
    assert _verdict(_report(), quality=_quality(Severity.WARNING), cfg=strict).status is InspectionStatus.REVIEW
    assert _verdict(_report(), quality=_quality(Severity.WARNING)).status is InspectionStatus.PASS


@pytest.mark.parametrize("threshold, expected", [("major", InspectionStatus.FAIL), ("critical", InspectionStatus.PASS)])
def test_rejection_severity_is_configurable(threshold, expected):
    cfg = DecisionConfig(fail_min_severity=threshold, max_minor_defects=5, allow_pass_with_warnings=True)
    assert _verdict(_report(_defect(1, "major")), cfg=cfg).status is expected


def test_no_comparison_yet_is_not_a_verdict():
    decision = decide(_quality(), _alignment(), _registration(), None, None, {}, CFG)
    assert decision.status is InspectionStatus.NOT_INSPECTED

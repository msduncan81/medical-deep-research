"""Data model for the GRADE pipeline (Core GRADE, per outcome, per body of evidence).

Certainty is rated per outcome across a body of evidence (GRADE 1; Core GRADE 2025). These models carry
what the graph nodes need: the confirmed outcomes with their certainty target (null or MID), the map of
systematic reviews used to avoid double counting, and the data extracted per study x outcome.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ..research.models import ScoredStudy


class OutcomeImportance(StrEnum):
    CRITICAL = "critical"  # GRADE importance 7-9
    IMPORTANT = "important"  # GRADE importance 4-6


class OutcomeKind(StrEnum):
    BINARY = "binary"
    CONTINUOUS = "continuous"
    TIME_TO_EVENT = "time_to_event"


class CertaintyTarget(StrEnum):
    NULL = "null"  # "is there any effect?"
    MID = "mid"  # "is the effect important?" (minimal important difference)


class StudyDesign(StrEnum):
    RCT = "rct"
    NON_RANDOMIZED = "non_randomized"
    SYSTEMATIC_REVIEW = "systematic_review"
    OTHER = "other"


class RiskOfBias(StrEnum):
    LOW = "low"
    HIGH = "high"
    UNCLEAR = "unclear"


class OutcomeSpec(BaseModel):
    id: str = Field(description="Short stable identifier, e.g. 'o1'.")
    name: str = Field(description="Outcome as it will appear in the Summary of Findings table.")
    importance: OutcomeImportance = OutcomeImportance.IMPORTANT
    importance_score: int = Field(default=6, ge=1, le=9, description="GRADE 1-9 scale (7-9 critical).")
    kind: OutcomeKind = OutcomeKind.CONTINUOUS
    desirable_direction: str = Field(default="decrease", description="'decrease' or 'increase' is better.")
    target: CertaintyTarget = CertaintyTarget.NULL
    mid_value: float | None = Field(default=None, description="Minimal important difference, absolute.")
    mid_unit: str | None = None
    mid_source: str | None = Field(default=None, description="Citation or rationale for the MID.")
    rationale: str = Field(default="", description="Why this outcome matters for the question.")


class SubgroupHypothesis(BaseModel):
    """A priori subgroup hypothesis with its expected direction (Core GRADE 3: at most ~3, pre-specified)."""

    id: str = Field(description="Short id, e.g. 's1'.")
    variable: str = Field(description="What distinguishes the subgroups, e.g. 'co-interventions'.")
    levels: list[str] = Field(default_factory=list, description="Two or three levels, e.g. ['alone', 'with titration'].")
    expected_larger_benefit: str = Field(default="", description="The level expected to benefit more.")
    rationale: str = ""


class ComparisonSpec(BaseModel):
    short_label: str = Field(default="", description="2-6 word name of the intervention for sentences.")
    intervention: str = Field(default="", description="What counts as the intervention for this question.")
    comparator: str = Field(default="", description="What counts as the comparator.")
    subgroups: list[SubgroupHypothesis] = Field(default_factory=list)


class OutcomeProposal(BaseModel):
    comparison: ComparisonSpec = Field(default_factory=ComparisonSpec)
    outcomes: list[OutcomeSpec] = Field(default_factory=list)
    notes: str = ""


class ReviewCoverage(BaseModel):
    """One systematic review and the primary studies of the pool that it includes."""

    review_ref: int = Field(description="Reference number of the review in the ranked list.")
    title: str
    year: int | None = None
    covered_refs: list[int] = Field(default_factory=list, description="Ranked primary studies it includes.")
    reference_count: int = 0
    network: bool = Field(default=False, description="Network meta-analysis (outside Core GRADE).")
    chosen: bool = False
    reason: str = ""


class ReviewMap(BaseModel):
    reviews: list[ReviewCoverage] = Field(default_factory=list)
    groups: list[list[int]] = Field(default_factory=list, description="Reviews that share primary studies.")
    covered_by_chosen: dict[int, int] = Field(
        default_factory=dict, description="primary study ref -> chosen review ref that already includes it"
    )
    corrected_covered_area: float | None = None
    notes: list[str] = Field(default_factory=list)


class StudyOutcomeData(BaseModel):
    """What one study (or review) reports for one outcome."""

    study_ref: int = 0
    outcome_id: str
    reported: bool = False
    design: StudyDesign = StudyDesign.OTHER
    n_intervention: int | None = None
    n_control: int | None = None
    events_intervention: int | None = None
    events_control: int | None = None
    effect_measure: str | None = Field(default=None, description="RR, OR, HR, MD, SMD, RD ...")
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    studies_in_estimate: int | None = Field(default=None, description="For reviews: trials pooled.")
    participants_in_estimate: int | None = None
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome_definition: str = ""
    follow_up: str = ""
    risk_of_bias: RiskOfBias = RiskOfBias.UNCLEAR
    risk_of_bias_rationale: str = ""
    review_grade: str | None = Field(default=None, description="Certainty reported by a review, if any.")
    favours: str = Field(default="unclear", description="'intervention', 'control', 'neither' or 'unclear'.")
    heterogeneity_i2: float | None = Field(default=None, description="I² (%) reported by a review.")
    is_network_meta_analysis: bool = False
    source_text: str = Field(default="abstract", description="'abstract' or 'fulltext'.")
    quote: str = Field(default="", description="Short verbatim passage supporting the numbers.")


class StudyExtraction(BaseModel):
    study_ref: int = 0
    comparison_match: str = Field(
        default="unclear",
        description="'yes' when the study tests the confirmed intervention vs comparator, 'partial' when close "
                    "(e.g. a variant of the intervention), 'no' when it tests a different comparison",
    )
    comparison_note: str = ""
    subgroup_levels: dict[str, str] = Field(
        default_factory=dict, description="subgroup hypothesis id -> level of this study (or 'unclear')"
    )
    items: list[StudyOutcomeData] = Field(default_factory=list)


class GradeStage1Result(BaseModel):
    comparison: ComparisonSpec = Field(default_factory=ComparisonSpec)
    seeded_studies: list[ScoredStudy] = Field(
        default_factory=list, description="Trials added from the reference lists of included reviews."
    )
    outcomes: list[OutcomeSpec] = Field(default_factory=list)
    outcomes_confirmed_by_user: bool = False
    review_map: ReviewMap = Field(default_factory=ReviewMap)
    extractions: list[StudyExtraction] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GradeResult(GradeStage1Result):
    """Stage 1 plus the per-outcome assessment (bodies of evidence, domains, certainty in modes A and B)."""

    profiles: list[dict] = Field(default_factory=list)

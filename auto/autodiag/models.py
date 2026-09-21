"""Shared data models.

Two families live here:

* Vehicle-side snapshot types (what we read from the car and the media inputs).
* Diagnosis output types (what the reasoning layer must return). These double
  as the JSON schema for Claude's structured output, so they forbid extra keys.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Vehicle snapshot
# --------------------------------------------------------------------------- #


class DTC(StrictModel):
    code: str = Field(description="Diagnostic trouble code, e.g. P0301")
    description: str = ""
    status: Literal["stored", "pending", "permanent"] = "stored"


class PIDValue(StrictModel):
    name: str
    value: float | str | None
    unit: str = ""


class VehicleInfo(StrictModel):
    vin: str | None = None
    year: int | None = None
    make: str | None = None
    model: str | None = None
    engine: str | None = None
    mileage: int | None = None


class VehicleSnapshot(StrictModel):
    """Everything we know from the OBD port at one point in time."""

    vehicle: VehicleInfo = Field(default_factory=VehicleInfo)
    dtcs: list[DTC] = Field(default_factory=list)
    freeze_frame: list[PIDValue] = Field(default_factory=list)
    live: list[PIDValue] = Field(default_factory=list)
    readiness: dict[str, str] = Field(default_factory=dict)
    source: str = "unknown"

    def pid(self, name: str, section: str = "live") -> PIDValue | None:
        for p in getattr(self, section):
            if p.name == name:
                return p
        return None

    def redacted(self) -> "VehicleSnapshot":
        """Copy with the VIN removed."""
        copy = self.model_copy(deep=True)
        copy.vehicle.vin = None
        return copy


class Flag(StrictModel):
    """A derived observation computed in code from PID values."""

    name: str
    severity: Literal["info", "warn"]
    detail: str
    pids: list[str] = Field(default_factory=list)
    search_terms: str = ""


class AudioFeatures(StrictModel):
    path: str
    duration_s: float
    rms_db: float
    dominant_hz: list[float]
    pulse_rate_hz: float | None = Field(
        default=None,
        description="Detected repeating impulse rate (ticks/knocks per second), if any",
    )
    notes: list[str] = Field(default_factory=list)


class DiagnosticInput(StrictModel):
    snapshot: VehicleSnapshot
    symptoms: str = ""
    audio: AudioFeatures | None = None
    image_paths: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


class Chunk(StrictModel):
    chunk_id: str
    doc_id: str
    title: str
    page: int | None = None
    section: str | None = None
    text: str
    score: float = 0.0
    handle: str = ""  # c1, c2, ... assigned after fusion

    def location(self) -> str:
        parts = []
        if self.section:
            parts.append(f'section "{self.section}"')
        if self.page is not None:
            parts.append(f"p.{self.page}")
        return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Diagnosis output (structured output schema)
# --------------------------------------------------------------------------- #

EvidenceKind = Literal["obd", "manual", "symptom", "audio", "image", "general_knowledge"]


class Evidence(StrictModel):
    kind: EvidenceKind
    detail: str = Field(description="What was observed or quoted, one or two sentences")
    source: str = Field(
        description=(
            "Chunk handle (e.g. c3) for manual evidence, PID or DTC name for OBD evidence, "
            "empty string otherwise"
        )
    )


class Cause(StrictModel):
    rank: int
    title: str = Field(description="Short name of the fault, e.g. 'Cylinder 1 ignition coil failure'")
    probability: float = Field(description="Estimated probability between 0 and 1; all causes sum to about 1")
    reasoning: str = Field(description="Why this cause fits the evidence, 2-4 sentences")
    evidence: list[Evidence]
    contradicting_evidence: list[str] = Field(description="Observations that argue against this cause; may be empty")


class NextTest(StrictModel):
    order: int
    name: str
    procedure: str = Field(description="Concrete steps a technician can follow")
    discriminates: list[str] = Field(description="Cause titles this test confirms or rules out")
    expected_if_faulty: str
    expected_if_ok: str
    tools_needed: list[str]
    est_minutes: int
    source: str = Field(description="Chunk handle if the procedure came from a manual, else empty string")


class Diagnosis(StrictModel):
    summary: str = Field(description="Two or three sentence plain-language summary for the driver")
    severity: Literal["drive_normally", "drive_with_caution", "limit_driving", "do_not_drive"]
    causes: list[Cause]
    next_tests: list[NextTest]
    missing_information: list[str] = Field(description="Data that would most improve confidence")
    safety_notes: list[str]


class ValidationReport(StrictModel):
    invalid_citations: int = 0
    downgraded: list[str] = Field(default_factory=list)
    unknown_cause_refs: list[str] = Field(default_factory=list)
    probability_sum_before: float = 0.0


class DiagnosisResult(StrictModel):
    diagnosis: Diagnosis
    chunks: list[Chunk]
    flags: list[Flag]
    validation: ValidationReport
    model: str
    backend: str = "claude"
    usage: dict[str, int] = Field(default_factory=dict)
    fallback_model: str | None = None

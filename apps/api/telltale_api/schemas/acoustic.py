from typing import Any

from pydantic import BaseModel

from .common import Recommendation, SoundStatus


class Series(BaseModel):
    times: list[float]
    values: list[float]


class Spectrogram(BaseModel):
    times: list[float]
    band_hz: list[float]
    values: list[list[float]]  # [band][time] in dB


class BandDeviation(BaseModel):
    band_hz: list[float]
    z: list[float]  # (clip mean - normal reference mean) / normal std, per band


class Waveform(BaseModel):
    duration_s: float
    min: list[float]
    max: list[float]


class AcousticAnalysis(BaseModel):
    machine_type: str
    machine_id: str | None = None
    filename: str | None = None
    method: str = "autoencoder"  # "ensemble" | "autoencoder"
    score: float
    threshold: float
    severity: float  # 1.0 = exactly at the alert level; higher = more unusual
    p_value: float | None = None  # share of normal recordings at least this unusual (conformal)
    unusual_percentile: float | None = None  # "more unusual than X% of normal recordings"
    calibration: str | None = None  # "global" | "enrolled:<name>" - which normal clips the p-value is relative to
    status: SoundStatus
    is_anomalous: bool
    duration_s: float
    sample_rate: int
    peak_time_s: float
    frame_errors: Series
    spectrogram: Spectrogram
    band_deviation: BandDeviation
    waveform: Waveform
    recommendations: list[Recommendation] = []
    model: dict[str, Any] = {}


class FaultProbability(BaseModel):
    class_name: str
    label: str
    probability: float


class FaultDiagnosis(BaseModel):
    filename: str | None = None
    predicted_class: str
    label: str
    confidence: float
    prediction_set: list[str] = []  # conformal set: classes that can't be ruled out
    prediction_set_labels: list[str] = []
    coverage: float | None = None  # target probability that the set contains the true class
    uncertain: bool = False
    rpm_used: float | None = None  # shaft speed used for order analysis
    rpm_assumed: bool = False  # True if the speed was unknown and a default was assumed
    segment_agreement: float
    segments: int
    duration_s: float
    probabilities: list[FaultProbability]
    recommendations: list[Recommendation] = []
    model: dict[str, Any] = {}


class EnrollmentStatus(BaseModel):
    machine_type: str
    name: str
    clips: int
    calibrated: bool  # enough clips (>= 19) for this condition's own false-alarm calibration
    clips_needed_for_calibration: int


class SampleClip(BaseModel):
    name: str
    machine_type: str | None
    label: str | None
    size_bytes: int

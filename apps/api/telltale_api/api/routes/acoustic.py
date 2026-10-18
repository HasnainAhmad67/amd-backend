import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from ...schemas.acoustic import AcousticAnalysis, EnrollmentStatus, FaultDiagnosis, SampleClip
from ..deps import AppState, get_state

router = APIRouter()
AUDIO_EXT = {".wav", ".flac", ".ogg"}  # demo clips served to the browser


async def _read_upload(file: UploadFile, state: AppState) -> io.BytesIO:
    limit = int(state.settings.max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File larger than {state.settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(422, "Empty file")
    return io.BytesIO(data)


@router.get("/models")
def acoustic_models(state: AppState = Depends(get_state)) -> dict:
    """Machine types with a trained sound model, plus the bearing fault classifier if present."""
    return {
        "anomaly": [
            {
                "machine_type": mt,
                "dataset": d.metadata.get("dataset"),
                "threshold": d.threshold,
                "test_auc": d.metadata.get("test_metrics", {}).get("overall", {}).get("roc_auc"),
            }
            for mt, d in sorted(state.registry.acoustic.items())
        ],
        "bearing": (
            {
                "classes": state.registry.bearing.classes,
                "class_labels": state.registry.bearing.class_labels,
                "test_macro_f1": state.registry.bearing.metadata["test_metrics"]["macro_f1"],
            }
            if state.registry.bearing
            else None
        ),
    }


@router.post("/analyze", response_model=AcousticAnalysis)
async def analyze(
    file: UploadFile = File(..., description="WAV / FLAC / OGG recording"),
    machine_type: str | None = Form(None, description="fan | pump | slider | valve | gearbox"),
    machine_id: str | None = Form(None, description="Attach the result to a fleet machine"),
    state: AppState = Depends(get_state),
) -> dict:
    """Score a recording against the normal-sound model for its machine type."""
    if machine_id and machine_id not in state.fleet.machines:
        raise HTTPException(404, f"Unknown machine '{machine_id}'")
    if machine_type is None:
        if not machine_id:
            raise HTTPException(422, "Provide machine_type or machine_id")
        machine_type = state.fleet.machines[machine_id].spec.acoustic_profile
        if not machine_type:
            raise HTTPException(422, f"Machine '{machine_id}' has no acoustic_profile")
    detector = state.registry.require_acoustic(machine_type)
    buf = await _read_upload(file, state)
    try:
        result = await run_in_threadpool(detector.analyze, buf)
    except (ValueError, RuntimeError) as exc:  # soundfile raises RuntimeError subclasses for bad audio
        raise HTTPException(422, f"Could not analyse audio: {exc}") from exc

    result.update(machine_id=machine_id, filename=file.filename)
    result["recommendations"] = state.playbook.for_acoustic(machine_type, result)
    result["model"] = {
        "dataset": detector.metadata.get("dataset"),
        "created_at": detector.metadata.get("created_at"),
        "test_metrics": detector.metadata.get("test_metrics", {}).get("overall"),
        "reference_band_hz": detector.metadata["reference_spectrum"]["band_hz"][:: max(1, detector.cfg.n_mels // 32)],
    }
    if machine_id:
        state.fleet.record_acoustic(machine_id, result, file.filename)
    return result


def _resolve_type(state: AppState, machine_type: str | None, machine_id: str | None) -> str:
    if machine_id and machine_id not in state.fleet.machines:
        raise HTTPException(404, f"Unknown machine '{machine_id}'")
    if machine_type:
        return machine_type
    if machine_id and state.fleet.machines[machine_id].spec.acoustic_profile:
        return state.fleet.machines[machine_id].spec.acoustic_profile
    raise HTTPException(422, "Provide machine_type or a machine_id with an acoustic_profile")


@router.post("/enroll", response_model=EnrollmentStatus)
async def enroll(
    files: list[UploadFile] = File(..., description="NORMAL recordings of this machine (10 s each works well)"),
    machine_type: str | None = Form(None),
    machine_id: str | None = Form(None),
    name: str | None = Form(None, description="Condition name (defaults to machine_id)"),
    state: AppState = Depends(get_state),
) -> dict:
    """Teach the sound model what THIS machine sounds like when healthy - no retraining.

    With 19+ clips the machine also gets its own false-alarm calibration (Mondrian conformal).
    """
    mt = _resolve_type(state, machine_type, machine_id)
    detector = state.registry.require_acoustic(mt)
    cond = name or machine_id or "custom"
    if len(files) > 100:
        raise HTTPException(413, "At most 100 clips per request")
    bufs = [await _read_upload(f, state) for f in files]
    try:
        status = await run_in_threadpool(detector.enroll, cond, bufs)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, f"Could not enroll: {exc}") from exc
    return {"machine_type": mt, **status}


@router.get("/enrolled", response_model=list[EnrollmentStatus])
def list_enrolled(machine_type: str | None = None, state: AppState = Depends(get_state)) -> list[dict]:
    out = []
    for mt, det in sorted(state.registry.acoustic.items()):
        if machine_type and mt != machine_type:
            continue
        for name in sorted(getattr(det, "enrolled", {})):
            out.append({"machine_type": mt, **det.enrollment_status(name)})
    return out


@router.delete("/enrolled/{machine_type}/{name}")
def delete_enrolled(machine_type: str, name: str, state: AppState = Depends(get_state)) -> dict:
    detector = state.registry.require_acoustic(machine_type)
    if not getattr(detector, "enrolled", None) or not detector.clear_enrollment(name):
        raise HTTPException(404, f"No enrolled clips named '{name}' for {machine_type}")
    return {"deleted": name}


@router.post("/diagnose", response_model=FaultDiagnosis)
async def diagnose(
    file: UploadFile = File(...),
    rpm: float | None = Form(None, gt=0, le=20000, description="Shaft speed; improves accuracy (order analysis)"),
    machine_id: str | None = Form(None, description="Use this fleet machine's live speed from telemetry"),
    state: AppState = Depends(get_state),
) -> dict:
    """Classify a bearing recording: healthy / inner race / outer race / ball / looseness / combined."""
    classifier = state.registry.require_bearing()
    if rpm is None and machine_id:
        st = state.fleet.machines.get(machine_id)
        if st is None:
            raise HTTPException(404, f"Unknown machine '{machine_id}'")
        if st.reading:
            rpm = float(st.reading["rotational_speed_rpm"])
    buf = await _read_upload(file, state)
    try:
        result = await run_in_threadpool(classifier.diagnose, buf, rpm)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(422, f"Could not analyse audio: {exc}") from exc
    result["filename"] = file.filename
    result["recommendations"] = state.playbook.for_bearing(result)
    result["model"] = {
        "dataset": classifier.metadata.get("dataset"),
        "split_strategy": classifier.metadata.get("split_strategy"),
        "test_macro_f1": classifier.metadata["test_metrics"]["macro_f1"],
    }
    return result


@router.get("/samples", response_model=list[SampleClip])
def list_samples(state: AppState = Depends(get_state)) -> list[dict]:
    """Demo clips in data/samples. Name: <type>_<label>_<n>.wav (type "bearing" = bearing fault demo clips)."""
    root = state.settings.samples_dir
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        parts = p.stem.split("_")
        out.append(
            {
                "name": p.name,
                "machine_type": parts[0] if len(parts) >= 2 else None,
                "label": "_".join(parts[1:-1]) if len(parts) >= 3 else None,
                "size_bytes": p.stat().st_size,
            }
        )
    return out


@router.get("/samples/{name}")
def get_sample(name: str, state: AppState = Depends(get_state)) -> FileResponse:
    root = state.settings.samples_dir.resolve()
    path = (root / name).resolve()
    if path.parent != root or path.suffix.lower() not in AUDIO_EXT or not path.is_file():
        raise HTTPException(404, "Sample not found")
    return FileResponse(path, media_type="audio/wav" if path.suffix.lower() == ".wav" else None, filename=path.name)

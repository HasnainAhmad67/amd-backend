"""Dataset downloader.

    telltale-download ai4i                                  #   ~0.5 MB  sensor data (UCI #601)
    telltale-download mimii-due --machine fan               #   ~1.1 GB  per machine type (Zenodo 4740355)
    telltale-download mimii --machine fan --snr 6           #  ~7-11 GB  per type+SNR    (Zenodo 3384388)
    telltale-download bearing                               #    ~75 MB  bearing sounds (Mendeley n9y9c7xrz3)
    telltale-download samples --machine fan                 #   copy a few test clips into data/samples for the UI
    telltale-download list                                  #   show everything with sizes + licences

Everything lands in data/raw/<dataset>/ (override with TELLTALE_DATA_DIR). Archives are kept
in data/raw/_archives/ so re-running skips the download.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from ..paths import raw_dir

AI4I_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
MIMII_DUE_URL = "https://zenodo.org/records/4740355/files/{name}?download=1"
MIMII_URL = "https://zenodo.org/records/3384388/files/{name}?download=1"
# Mendeley's public "download all" endpoint redirects to a short-lived signed S3 link (GET only).
BEARING_URL = "https://data.mendeley.com/public-api/zip/n9y9c7xrz3/download/1"
MIMII_DUE_TYPES = ("fan", "gearbox", "pump", "slider", "valve")
MIMII_TYPES = ("fan", "pump", "slider", "valve")
MIMII_SNRS = ("-6", "0", "6")

CATALOG = """
dataset     source                          size              licence          used for
---------   -----------------------------   ---------------   --------------   ------------------------------
ai4i        UCI #601 (AI4I 2020)            0.5 MB            CC BY 4.0        sensor failure-risk model
mimii-due   Zenodo 4740355 (DCASE21 T2)     ~1.1 GB / type    CC BY-NC-SA 4.0  sound anomaly detector (default)
mimii       Zenodo 3384388                  7-11 GB / type    CC BY-SA 4.0     sound anomaly detector (more data)
bearing     Mendeley n9y9c7xrz3             75 MB             CC BY 4.0        bearing fault classifier
""".strip()


def _download(url: str, dest: Path, timeout: float = 60, retries: int = 10) -> Path:
    """Stream ``url`` to ``dest`` with resume + retries (stdlib only, so the API image can use it too).

    Slow or flaky Wi-Fi often drops long downloads without an error, so: the size is checked against
    Content-Length, an interrupted download continues from where it stopped (HTTP Range), and only a complete
    file is renamed from ``<name>.part`` to ``<name>``. Re-running the command also resumes.
    """
    import time

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is only in the [train] extra
        tqdm = None

    if dest.exists():
        print(f"[ok] already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get] {url}")
    total: int | None = None
    stalled = 0  # consecutive attempts that added no bytes; any progress resets it
    attempt = 0
    while stalled < retries:
        attempt += 1
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            break
        headers = {"User-Agent": "telltale-downloader"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https URLs
                if have and r.status != 206:  # server ignored the Range header: start over
                    have = 0
                length = int(r.headers.get("content-length") or 0)
                total = (have + length) if length else total
                bar = tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=dest.name) if tqdm else None
                with open(tmp, "ab" if have else "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                        if bar:
                            bar.update(len(chunk))
                if bar:
                    bar.close()
        except (OSError, urllib.error.URLError) as exc:
            print(f"[retry {attempt}] connection problem: {exc}")
        size = tmp.stat().st_size if tmp.exists() else 0
        if total is None or size >= total:
            break
        stalled = 0 if size > have else stalled + 1
        print(f"[retry {attempt}] got {size / 1e6:.0f} of {total / 1e6:.0f} MB - resuming in a few seconds")
        time.sleep(min(30, 2 * (stalled + 1)))
    size = tmp.stat().st_size if tmp.exists() else 0
    if total is not None and size < total:
        raise RuntimeError(
            f"Download incomplete ({size / 1e6:.0f} of {total / 1e6:.0f} MB). Run the same command again - it resumes."
        )
    tmp.replace(dest)
    return dest


def _extract(archive: Path, target: Path) -> None:
    """Unzip once. A marker file makes re-runs skip it (re-extracting 1 GB is slow and races with readers)."""
    target.mkdir(parents=True, exist_ok=True)
    marker = target / f".extracted-{archive.name}"
    if marker.exists() and marker.stat().st_mtime >= archive.stat().st_mtime:
        print(f"[ok] already extracted: {archive.name}")
        return
    if not zipfile.is_zipfile(archive):
        archive.unlink()  # corrupt or truncated: delete so the next run downloads it again
        raise RuntimeError(f"{archive.name} was damaged (incomplete download) and has been deleted. Run the command again.")
    print(f"[unzip] extracting {archive.name} -> {target}")
    with zipfile.ZipFile(archive) as zf:
        root = target.resolve()
        for member in zf.infolist():  # guard against path traversal
            if not (target / member.filename).resolve().is_relative_to(root):
                raise RuntimeError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(target)
    marker.touch()


def _archives() -> Path:
    return raw_dir() / "_archives"


def get_ai4i(timeout: float = 60) -> Path:
    """Download + extract AI4I 2020 if needed. Returns the CSV path."""
    csv = raw_dir() / "ai4i2020" / "ai4i2020.csv"
    if not csv.is_file():
        archive = _download(AI4I_URL, _archives() / "ai4i2020.zip", timeout=timeout)
        _extract(archive, raw_dir() / "ai4i2020")
    print(f"[ok] AI4I ready at {csv}")
    return csv


def get_mimii_due(machines: list[str], include_eval_train: bool) -> None:
    for m in machines:
        names = [f"dev_data_{m}.zip"] + ([f"eval_data_{m}_train.zip"] if include_eval_train else [])
        for name in names:
            archive = _download(MIMII_DUE_URL.format(name=name), _archives() / "mimii_due" / name)
            _extract(archive, raw_dir() / "mimii_due")
    print(f"[ok] MIMII DUE ready under {raw_dir() / 'mimii_due'}")


def get_mimii(machines: list[str], snr: str) -> None:
    for m in machines:
        name = f"{snr}_dB_{m}.zip"
        archive = _download(MIMII_URL.format(name=name), _archives() / "mimii" / name)
        _extract(archive, raw_dir() / "mimii" / f"{snr}dB")
    print(f"[ok] MIMII ready under {raw_dir() / 'mimii' / f'{snr}dB'}")


def get_bearing() -> Path:
    """Download + extract the Mendeley bearing-sound dataset (MATLAB .mat recordings) if needed."""
    target = raw_dir() / "bearing_sound"
    already_there = target.is_dir() and any(target.rglob("*.mat"))
    if not already_there:
        try:
            archive = _download(BEARING_URL, _archives() / "bearing_sound.zip", timeout=120)
        except Exception as exc:
            target.mkdir(parents=True, exist_ok=True)
            raise SystemExit(
                f"Automatic download failed ({exc}). Download it by hand instead:\n"
                "  1. Open https://data.mendeley.com/datasets/n9y9c7xrz3/1 and click 'Download All'.\n"
                f"  2. Unzip it into {target}\n"
                "  3. Run telltale-train-bearing again."
            ) from exc
        _extract(archive, target)
    print(f"[ok] Bearing dataset ready under {target}")
    return target


def _spread_over_sections(clips: list, n: int) -> list:
    """Pick n clips round-robin over sections (source domain first), not the first n of one section.

    Selection is by position only, never by model score, so demo clips stay representative.
    """
    pool = [c for c in clips if c.domain == "source"] or clips
    by_group: dict[str, list] = {}
    for c in pool:
        by_group.setdefault(c.group, []).append(c)
    queues = [by_group[g] for g in sorted(by_group)]
    picked: list = []
    depth = 0
    while len(picked) < n and any(depth < len(q) for q in queues):
        picked += [q[depth] for q in queues if depth < len(q)][: n - len(picked)]
        depth += 1
    return picked


def make_samples(machines: list[str], per_label: int = 3) -> None:
    """Copy a few held-out normal/anomalous clips into data/samples/<type>_<label>_<n>.wav for demos."""
    from ..acoustic.data import discover

    out = raw_dir().parent / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for m in machines:
        for dataset in ("mimii_due", "mimii"):
            try:
                clips = discover(dataset, raw_dir() / dataset, m)
            except FileNotFoundError:
                continue
            for label in ("normal", "anomaly"):
                picked = _spread_over_sections([c for c in clips if c.split == "test" and c.label == label], per_label)
                for i, c in enumerate(picked):
                    shutil.copyfile(c.path, out / f"{m}_{label}_{i}.wav")
            print(f"[ok] {m}: samples from {dataset} -> {out}")
            break
        else:
            print(f"[skip] {m}: no MIMII / MIMII DUE data found; download it first")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["ai4i", "mimii-due", "mimii", "bearing", "samples", "list"])
    parser.add_argument("--machine", action="append", help="Machine type (repeatable). Default: fan")
    parser.add_argument("--snr", choices=MIMII_SNRS, default="6", help="MIMII only: noise level in dB")
    parser.add_argument("--include-eval-train", action="store_true", help="MIMII DUE: also fetch extra training data")
    args = parser.parse_args(argv)

    machines = args.machine or ["fan"]
    if args.dataset == "list":
        print(CATALOG)
    elif args.dataset == "ai4i":
        get_ai4i()
    elif args.dataset == "mimii-due":
        bad = set(machines) - set(MIMII_DUE_TYPES)
        if bad:
            sys.exit(f"Unknown MIMII DUE machine type(s) {bad}; choose from {MIMII_DUE_TYPES}")
        get_mimii_due(machines, args.include_eval_train)
    elif args.dataset == "mimii":
        bad = set(machines) - set(MIMII_TYPES)
        if bad:
            sys.exit(f"Unknown MIMII machine type(s) {bad}; choose from {MIMII_TYPES}")
        free_gb = shutil.disk_usage(raw_dir().parent if raw_dir().parent.exists() else ".").free / 1e9
        print(f"Note: each MIMII archive is 7-11 GB and needs ~2x that while extracting ({free_gb:.0f} GB free).")
        get_mimii(machines, args.snr)
    elif args.dataset == "samples":
        make_samples(machines)
    else:
        get_bearing()


if __name__ == "__main__":
    main()

"""Registry of public mammography datasets and a small downloader."""

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class Source:
    name: str
    homepage: str
    access: str          # physionet | tcia | portal | on-request | direct
    url: str = None
    sha256: str = None
    notes: str = ""
    aliases: tuple = field(default_factory=tuple)


SOURCES = {
    "vindr": Source(
        name="VinDr-Mammo",
        homepage="https://physionet.org/content/vindr-mammo/",
        access="physionet",
        notes="Credentialed PhysioNet access. After approval, mirror the files with wget.",
        aliases=("vindr-mammo", "vindr_mammo"),
    ),
    "cbis": Source(
        name="CBIS-DDSM",
        homepage="https://www.cancerimagingarchive.net/collection/cbis-ddsm/",
        access="tcia",
        notes="Download the .tcia manifest and fetch with the NBIA Data Retriever.",
        aliases=("cbis-ddsm", "cbis_ddsm", "ddsm"),
    ),
    "cmmd": Source(
        name="CMMD",
        homepage="https://www.cancerimagingarchive.net/collection/cmmd/",
        access="tcia",
        notes="TCIA collection with biopsy-confirmed labels.",
        aliases=("chinese-mammography",),
    ),
    "inbreast": Source(
        name="INBreast",
        homepage="https://www.kaggle.com/datasets/tommyngx/inbreast2012",
        access="on-request",
        notes="Obtain from the authors or a mirror; extract under data/raw/INBreast.",
        aliases=("inbreast2012",),
    ),
    "mosmed": Source(
        name="MosMed-MMG",
        homepage="https://mosmed.ai/datasets/",
        access="portal",
        notes="Registration required on the mosmed.ai portal.",
        aliases=("mosmeddata", "mosmed-mmg"),
    ),
}


def find(name):
    key = name.strip().lower()
    if key in SOURCES:
        return SOURCES[key]
    for s in SOURCES.values():
        if key == s.name.lower() or key in s.aliases:
            return s
    raise KeyError(f"unknown dataset '{name}'; known: {', '.join(sorted(SOURCES))}")


def _checksum(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, out_dir, overwrite=False):
    """Download a dataset that exposes a direct URL, else raise with instructions."""
    src = find(name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src.url is None:
        raise RuntimeError(
            f"{src.name} needs manual/credentialed access ({src.access}).\n"
            f"Homepage: {src.homepage}\n{src.notes}"
        )

    target = out_dir / Path(src.url).name
    if target.exists() and not overwrite:
        print(f"{target} already present, skipping")
        return target

    print(f"downloading {src.name} -> {target}")
    with urlopen(src.url) as resp, open(target, "wb") as fh:
        shutil.copyfileobj(resp, fh)

    if src.sha256 and _checksum(target) != src.sha256:
        target.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {src.name}")
    return target


def show_access(names=None):
    for key in names or list(SOURCES):
        s = find(key)
        print(f"\n{s.name} ({s.access})")
        print(f"  homepage: {s.homepage}")
        if s.notes:
            print(f"  {s.notes}")

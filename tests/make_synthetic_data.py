"""Build a tiny synthetic DICOM dataset for testing."""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
except ImportError as exc:
    raise ImportError("pydicom is required to build synthetic DICOMs") from exc


def _write_dicom(path, label, photometric, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 1000, size=(64, 64), dtype=np.uint16)
    if label == 1:
        img[24:40, 24:40] = rng.integers(3000, 4000, size=(16, 16), dtype=np.uint16)
    if photometric == "MONOCHROME1":
        img = img.max() - img

    meta = Dataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = img.shape
    ds.PixelData = img.astype(np.uint16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))


def make_synthetic(root, per_class=12):
    """Write DICOMs + annotations.csv under root; return the CSV path."""
    root = Path(root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0
    for label in (0, 1):
        for _ in range(per_class):
            photometric = "MONOCHROME1" if idx % 2 else "MONOCHROME2"
            fname = f"img_{idx:03d}.dcm"
            _write_dicom(data_dir / fname, label, photometric, seed=idx)
            rows.append({
                "image": f"data/{fname}",
                "label": label,
                "dataset": "synthetic",
                "birads": 5 if label == 1 else 1,
                "density": 2,
            })
            idx += 1

    csv_path = root / "annotations.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
    print("wrote", make_synthetic(out))

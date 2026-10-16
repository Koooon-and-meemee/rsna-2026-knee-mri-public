"""Turn a DICOM series into a fixed-size uint8 volume.

Pure functions over already-loaded pydicom datasets, with no I/O and no
project imports, because this module is embedded verbatim into the Kaggle
kernel that does the preprocessing. There is one copy of this logic and it
runs identically here and there.

Four choices carry most of the risk:

**Slices are ordered by ImagePositionPatient projected onto the slice normal,
not by InstanceNumber.** InstanceNumber is a scanner-assigned label - often
reversed, sometimes duplicated within a stack, and wrong without saying so.

**Laterality is normalised.** Medial and Lateral are defined relative to the
body's midline, so which side of the image they fall on depends on which knee
was scanned - and 5 of the 12 targets (both menisci, both femorotibial
compartments, MCL) are laterality-specific. Left unnormalised, those five
labels are trained against an axis the model cannot observe: the same
pathology sits on opposite sides of the frame depending on which knee walked
in. This was previously unhandled entirely - only slice order within a study
was considered, never which knee the study even was - and it shows in the
per-finding scores: Lateral Meniscus and Lateral OA, the two labels most
exposed to this, were the two weakest of the twelve in every run measured
(gold AUC 0.83-0.87 against 0.90-0.98 for their medial counterparts).

The correction differs by plane. Coronally and axially the medial-lateral axis
lies in the image plane, so mirroring the array axis aligned with patient-x
maps one knee onto the other. Sagittally, medial-lateral is the *through-plane*
axis - no in-plane pixel is on the "wrong side," what differs is which end of
the stack is medial, so the slice order is reversed instead.

Which knee was scanned is read from geometry rather than the `Laterality` DICOM
tag, because that tag is Type 2C (may legitimately be absent) and is missing on
roughly half of real acquisitions, generally by whole vendors rather than
scattered series - silently treating those as one fixed side would mirror every
study from the other side. Position and orientation are recorded on every
image regardless, and give the same answer: in patient coordinates (LPS),
+x points to the patient's left, so the sign of the image centre's x-coordinate
says which knee this is. The centre is used rather than ImagePositionPatient
itself, which is a corner half a field of view away - enough to flip the sign
for a knee scanned near the midline. A study centred within a few mm of the
midline is left unresolved rather than guessed, since inside that band the
sign carries no information.

**Cropped to a fixed physical extent before resizing, not resized from the raw
frame.** Acquisition matrices vary across this corpus (multiple Rows/Columns
values have been observed at multiple pixel spacings), and a fixed-pixel
resize from a varying field of view hands the encoder images whose physical
scale differs between studies - a 256px frame can be anywhere from roughly
0.4 to 0.6 mm/pixel depending on which scanner and protocol produced it. Two
studies at the same encoder input size are then not comparable, and a model
trained on the mix cannot separate "this pixel span is a lesion" from "this
pixel span is what a 0.15 mm/px scanner happens to produce." Cropping to a
fixed millimetre extent before the resize fixes the physical scale of every
input to the same value, which is also the condition under which increasing
the pixel count can buy anything: without it, more pixels can encode more of a
field of view that varies, rather than more detail of a fixed one.
CROP_MM=130 is chosen to sit below the smallest field of view seen in this
corpus while still containing the joint on every plane; if a series' field of
view is smaller than CROP_MM the crop is skipped rather than padded; the
percentile normalisation below then also runs on the cropped extent.

**Normalisation is per volume, by percentile.** MR intensity has no absolute
units; the same tissue reads differently across the 16 sites and across
sequences at one site, so any fixed window is meaningless. Clipping at the
0.5/99.5 percentiles stops one bright artefact from crushing everything else
into a handful of levels.
"""

from __future__ import annotations

import numpy as np

CROP_MM = 130.0
MIDLINE_DEAD_ZONE_MM = 8.0


def slice_order(datasets: list) -> list[int]:
    """Indices that sort the slices along the through-plane axis."""
    try:
        o = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
        normal = np.cross(o[:3], o[3:])
        pos = [
            float(np.dot(np.asarray(d.ImagePositionPatient, dtype=float), normal))
            for d in datasets
        ]
        if len(set(pos)) < len(pos) // 2:  # degenerate geometry, not trustworthy
            raise ValueError("positions are not distinct")
        return list(np.argsort(pos))
    except Exception:
        nums = [float(getattr(d, "InstanceNumber", i)) for i, d in enumerate(datasets)]
        return list(np.argsort(nums))


def _geometry(ds) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Row/column direction cosines and the image-plane normal, or None."""
    try:
        o = np.asarray(ds.ImageOrientationPatient, dtype=float)
        row_dir, col_dir = o[:3], o[3:]
        normal = np.cross(row_dir, col_dir)
        return row_dir, col_dir, normal
    except Exception:
        return None


def plane_of(ds) -> str:
    """Sagittal / coronal / axial from the axis the slice normal is closest to."""
    geo = _geometry(ds)
    if geo is None:
        return "unknown"
    _, _, normal = geo
    ax = int(np.argmax(np.abs(normal)))
    return ("sagittal", "coronal", "axial")[ax]


def knee_side(ds) -> str | None:
    """'left' / 'right' from patient-coordinate geometry, or None if unresolved.

    LPS convention: +x points to the patient's left. The frame centre in
    patient coordinates is p + row_dir * col_spacing * cols/2
                              + col_dir * row_spacing * rows/2
    (rows step along col_dir, columns step along row_dir - PixelSpacing is
    [row spacing, column spacing], i.e. the physical step between adjacent
    ROWS is the first value and applies when moving along the column
    direction vector, and vice versa).
    """
    try:
        geo = _geometry(ds)
        if geo is None:
            return None
        row_dir, col_dir, _ = geo
        pos = np.asarray(ds.ImagePositionPatient, dtype=float)
        sp = [float(x) for x in ds.PixelSpacing]
        rows, cols = int(ds.Rows), int(ds.Columns)
        centre = pos + col_dir * sp[0] * (rows / 2.0) + row_dir * sp[1] * (cols / 2.0)
        x = centre[0]
        if abs(x) < MIDLINE_DEAD_ZONE_MM:
            return None
        return "left" if x > 0 else "right"
    except Exception:
        return None


def _resize2d(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear resize without torch, so the kernel needs no extra deps."""
    if a.shape == (h, w):
        return a
    y = np.linspace(0, a.shape[0] - 1, h)
    x = np.linspace(0, a.shape[1] - 1, w)
    y0 = np.clip(np.floor(y).astype(int), 0, a.shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, a.shape[0] - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, a.shape[1] - 1)
    x1 = np.clip(x0 + 1, 0, a.shape[1] - 1)
    wy = (y - y0)[:, None]
    wx = (x - x0)[None, :]
    top = a[np.ix_(y0, x0)] * (1 - wx) + a[np.ix_(y0, x1)] * wx
    bot = a[np.ix_(y1, x0)] * (1 - wx) + a[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bot * wy


def _center_crop(a: np.ndarray, ds, crop_mm: float) -> np.ndarray:
    """Centre crop to a fixed physical extent, or return `a` unchanged if the
    field of view is already smaller than crop_mm (nothing to crop)."""
    try:
        sp = [float(x) for x in ds.PixelSpacing]  # [row spacing, col spacing]
        n_rows = int(round(crop_mm / sp[0]))
        n_cols = int(round(crop_mm / sp[1]))
    except Exception:
        return a
    if n_rows >= a.shape[0] or n_cols >= a.shape[1]:
        return a
    r0 = (a.shape[0] - n_rows) // 2
    c0 = (a.shape[1] - n_cols) // 2
    return a[r0 : r0 + n_rows, c0 : c0 + n_cols]


def _find_x_axis(row_dir: np.ndarray, col_dir: np.ndarray) -> str | None:
    """Which array axis (row=0 / col=1) tracks patient-x, for in-plane mirroring.

    Rows step along col_dir, columns step along row_dir (see knee_side). The
    axis to flip is whichever direction vector has the larger x-component.
    """
    if abs(col_dir[0]) > abs(row_dir[0]):
        return "rows"
    if abs(row_dir[0]) > abs(col_dir[0]):
        return "cols"
    return None


def to_volume(
    datasets: list,
    depth: int = 32,
    size: int = 256,
    crop_mm: float | None = CROP_MM,
    normalize_laterality: bool = True,
) -> np.ndarray:
    """(depth, size, size) uint8 volume from a list of pydicom datasets."""
    order = slice_order(datasets)
    ordered = [datasets[i] for i in order]

    plane = plane_of(ordered[0]) if ordered else "unknown"
    side = knee_side(ordered[len(ordered) // 2]) if ordered else None
    # canonical = left: a right-sided study is mirrored onto the left frame
    flip = normalize_laterality and side == "right"

    frames = []
    for ds in ordered:
        a = ds.pixel_array.astype(np.float32)
        if crop_mm:
            a = _center_crop(a, ds, crop_mm)
        frames.append(_resize2d(a, size, size))
    vol = np.stack(frames)  # (D, size, size)

    if flip:
        geo = _geometry(ordered[0])
        if plane == "sagittal":
            vol = vol[::-1]
        elif geo is not None:
            axis = _find_x_axis(geo[0], geo[1])
            if axis == "rows":
                vol = vol[:, ::-1, :]
            elif axis == "cols":
                vol = vol[:, :, ::-1]

    # depth: pick `depth` evenly spaced positions and linearly blend neighbours,
    # which both downsamples a thick stack and upsamples a thin one
    d = vol.shape[0]
    if d != depth:
        z = np.linspace(0, d - 1, depth)
        z0 = np.clip(np.floor(z).astype(int), 0, d - 1)
        z1 = np.clip(z0 + 1, 0, d - 1)
        wz = (z - z0)[:, None, None]
        vol = vol[z0] * (1 - wz) + vol[z1] * wz

    lo, hi = np.percentile(vol, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)
    return np.rint(vol * 255).astype(np.uint8)

"""Extract the DICOM header fields that identify a scanner, one row per series.

Our validation splits studies by report text, so a model can be trained and
validated on the same MRI scanner. A competitor measured what that is worth: a
gradient-boosted model on DICOM metadata alone reaches 0.6516 macro AUC under
random folds and 0.5981 when folds are grouped by scanner, so 0.053 of it is
site memorisation that does not transfer. The gap is largest on the
osteoarthritis columns (0.074-0.081) and on fracture (0.086).

That is a probe of metadata predicting labels, not of our image model - but the
same leak is available to an image model through scanner-specific appearance,
and our splits do nothing to prevent it. If the hidden test contains scanners
absent from training, our held-out number is optimistic by an unknown amount.

This kernel produces the field needed to find out: the five-part fingerprint
Manufacturer + ManufacturerModelName + SoftwareVersions + ImagingFrequency +
ReceiveCoilName, which partitions the corpus into roughly 265 groups with the
top 20 covering 45.5% of studies.

Reading one header per series rather than per slice: these fields are constant
within a series, and 24,371 reads is minutes where 730,000 would be hours.
stop_before_pixels is what makes it cheap - the pixel data is the file.
"""

import glob
import json
import os
import time

import pandas as pd
import pydicom

COMP = "rsna-knee-abnormality-detection"
FIELDS = [
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
    "ImagingFrequency", "ReceiveCoilName",
    # not part of the fingerprint, but free to collect while the header is open
    # and plausibly useful: field strength separates 1.5T from 3T sites, and
    # PatientSex was documented in train.csv then removed from it by the hosts.
    "MagneticFieldStrength", "PatientSex", "PatientAge",
    "SeriesDescription", "ScanningSequence", "SequenceVariant",
    "SliceThickness", "SpacingBetweenSlices", "RepetitionTime", "EchoTime",
    "FlipAngle", "EchoTrainLength", "PixelBandwidth", "InstitutionName",
]

t0 = time.time()
root = os.path.dirname(glob.glob(f"/kaggle/input/**/{COMP}/train.csv", recursive=True)[0]) \
    if glob.glob(f"/kaggle/input/**/{COMP}/train.csv", recursive=True) \
    else f"/kaggle/input/competitions/{COMP}"
print("ROOT =", root, flush=True)

series_dirs = sorted(glob.glob(f"{root}/train_series/*/*"))
print(f"{len(series_dirs)} series", flush=True)

rows, failed = [], 0
for i, d in enumerate(series_dirs, 1):
    files = glob.glob(f"{d}/*.dcm")
    if not files:
        failed += 1
        continue
    try:
        # one header, no pixels; the first file in the series is representative
        ds = pydicom.dcmread(sorted(files)[0], stop_before_pixels=True)
        r = {"StudyInstanceUID": os.path.basename(os.path.dirname(d)),
             "SeriesInstanceUID": os.path.basename(d),
             "n_slices": len(files)}
        for f in FIELDS:
            v = getattr(ds, f, None)
            r[f] = str(v) if v is not None else ""
        rows.append(r)
    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"  {os.path.basename(d)[:16]} failed: {type(e).__name__}", flush=True)
    if i % 2000 == 0:
        print(f"  {i}/{len(series_dirs)}  {(time.time()-t0)/60:.1f}m", flush=True)

df = pd.DataFrame(rows)
df.to_parquet("/kaggle/working/scanner_meta.parquet", index=False)

fp = (df.Manufacturer + "|" + df.ManufacturerModelName + "|" + df.SoftwareVersions
      + "|" + df.ImagingFrequency + "|" + df.ReceiveCoilName)
df["fingerprint"] = fp
per_study = df.groupby("StudyInstanceUID").fingerprint.agg(lambda s: s.mode().iloc[0])
vc = per_study.value_counts()
print(f"\n{len(df)} series, {failed} unreadable, {(time.time()-t0)/60:.1f}m")
print(f"{per_study.nunique()} distinct fingerprints over {len(per_study)} studies")
print(f"top 20 cover {vc.head(20).sum()/len(per_study):.1%}")
print(vc.head(10).to_string())

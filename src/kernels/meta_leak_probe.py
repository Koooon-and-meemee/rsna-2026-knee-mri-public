"""Does DICOM metadata alone predict the findings? (leak check)

If protocol metadata carries most of the signal, a leaderboard score of 0.932
three days into an image competition needs no image model to explain it - and
if it is that predictive, ignoring it is a unilateral handicap.

Trained on the 4,407 student pseudo-labels; scored on the 58 expert-labelled
studies, which the pseudo-labels never saw as ground truth.
"""
import os, glob, json, warnings
import numpy as np, pandas as pd, pydicom
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
# locate the label dataset rather than assuming its mount path - the first run
# died on a hardcoded path while the dataset was still being attached
_hits = glob.glob("/kaggle/input/**/pseudo_student_all.parquet", recursive=True)
if not _hits:
    raise SystemExit("label dataset not attached: " + str(os.listdir("/kaggle/input")))
LAB = os.path.dirname(_hits[0])
print("LAB =", LAB)
LABELS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA",
          "PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

train = pd.read_csv(f"{ROOT}/train.csv")
series = pd.read_csv(f"{ROOT}/train_series.csv")
pseudo = pd.read_parquet(f"{LAB}/pseudo_student_all.parquet")
is_gold = train[LABELS].notna().all(axis=1)
print(f"{len(train)} studies, {int(is_gold.sum())} gold")

TAGS = ["SeriesDescription","ProtocolName","ScanningSequence","SequenceVariant",
        "ScanOptions","MRAcquisitionType","SliceThickness","RepetitionTime",
        "EchoTime","FlipAngle","MagneticFieldStrength","Manufacturer",
        "ManufacturerModelName","PixelSpacing","Rows","Columns","SpacingBetweenSlices",
        "InstitutionName","BodyPartExamined","PatientPosition","EchoTrainLength"]

rows = []
studies = sorted(series.StudyInstanceUID.unique())
for i, st in enumerate(studies):
    d = f"{ROOT}/train_series/{st}"
    try:
        rec = {"StudyInstanceUID": st, "n_series": 0}
        descs = []
        for ser in os.listdir(d):
            fs = glob.glob(f"{d}/{ser}/*.dcm")
            if not fs:
                continue
            h = pydicom.dcmread(fs[0], stop_before_pixels=True)
            rec["n_series"] += 1
            rec.setdefault("n_slices", 0)
            rec["n_slices"] += len(fs)
            for t in TAGS:
                v = getattr(h, t, None)
                if v is None:
                    continue
                if t in ("PixelSpacing",):
                    v = float(v[0])
                if isinstance(v, (int, float)) or (hasattr(v, "real") and not isinstance(v, str)):
                    try: rec[f"{t}_max"] = max(rec.get(f"{t}_max", -1e9), float(v))
                    except Exception: pass
                else:
                    descs.append(f"{t}={v}")
        rec["desc_blob"] = " | ".join(sorted(set(descs)))
        rows.append(rec)
    except Exception as e:
        pass
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(studies)}", flush=True)

meta = pd.DataFrame(rows).fillna(np.nan)
print("metadata frame:", meta.shape)

# text fields -> hashed bag of tokens (keeps it dependency-free and fast)
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=512, alternate_sign=False, token_pattern=r"[^ |]+")
X_txt = hv.transform(meta.desc_blob.fillna("")).toarray()
num_cols = [c for c in meta.columns if c not in ("StudyInstanceUID","desc_blob")]
X = np.hstack([meta[num_cols].to_numpy(dtype=float), X_txt])
print("feature matrix:", X.shape)

meta = meta.reset_index(drop=True)
gold_uid = set(train.loc[is_gold, "StudyInstanceUID"])
is_g = meta.StudyInstanceUID.isin(gold_uid).to_numpy()
y_soft = pseudo.set_index("StudyInstanceUID").reindex(meta.StudyInstanceUID)
y_true = train.set_index("StudyInstanceUID").reindex(meta.StudyInstanceUID)

aucs = {}
for c in LABELS:
    tr_y = (y_soft.loc[~is_g, c].to_numpy() > 0.5).astype(int)
    if len(np.unique(tr_y)) < 2: continue
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_depth=6)
    m.fit(X[~is_g], tr_y)
    p = m.predict_proba(X[is_g])[:, 1]
    t = y_true.loc[is_g, c].to_numpy().astype(int)
    if len(np.unique(t)) > 1:
        aucs[c] = roc_auc_score(t, p)
    print(f"  {c:18} gold AUC {aucs.get(c, float('nan')):.3f}", flush=True)

print(f"\nMETADATA-ONLY macro AUC on gold 58: {np.mean(list(aucs.values())):.4f}")
print("reference: keyword 0.597 | base 9B 0.861 | student 0.899 | leaderboard #1 0.932")
json.dump({k: float(v) for k, v in aucs.items()}, open("/kaggle/working/meta_auc.json","w"), indent=1)

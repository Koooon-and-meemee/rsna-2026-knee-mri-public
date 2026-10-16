"""Build and push the model submission kernel.

The checkpoint reaches the kernel one of two ways. --train-tag attaches a
finished Kaggle training kernel as a kernel_source, so the checkpoint is read
where it was written. --dataset attaches a Kaggle Dataset instead, which is
what local training needs: the weekly GPU quota is spent, so models are now
trained on the 5090 and the resulting 58MB checkpoint is uploaded.

The pushed script is volume.py + net.py + net2d.py + the driver, concatenated.
These are the same files the preprocessing and training kernels use, which is
the point: a test volume must be built exactly like a training volume, and the
checkpoint must load into the module it was saved from. Which architecture to
build is read from the checkpoint's own metadata, not passed in here.

--dataset takes a comma-separated list, one dataset per ensemble member. The
kernel loads every image_model.pt it finds and averages their probabilities, so
adding a member is an argument here rather than a code change - and because the
DICOM decode is shared, a second member costs a fraction of a second run.

    uv run python src/push_submit_model.py --train-tag v1
    uv run python src/push_submit_model.py --dataset qwrfsdr/rsna-knee-image-model-m2 --tag m2
    uv run python src/push_submit_model.py --tag ens3 \
      --dataset qwrfsdr/rsna-knee-image-model-m31newshards,qwrfsdr/rsna-knee-image-model-m33up336new
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER = "qwrfsdr"
COMP = "rsna-knee-abnormality-detection"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tag", default=None, help="which rsna-knee-train-* to load")
    ap.add_argument("--dataset", default=None,
                    help="comma-separated datasets holding image_model.pt, one per "
                         "ensemble member, e.g. qwrfsdr/rsna-knee-image-model-m2")
    ap.add_argument("--tag", default=None, help="submission kernel suffix")
    ap.add_argument("--max-series", type=int, default=0,
                    help="use at most N series per study (efficiency lever)")
    ap.add_argument("--by-plane", action="store_true",
                    help="spend --max-series across imaging planes, not UID order")
    ap.add_argument("--slice-stride", type=int, default=1,
                    help="keep every Nth slice at inference (efficiency lever)")
    ap.add_argument("--blend-coat", action="store_true",
                    help="append the Global96 arm and write rank-mean blends. "
                         "Needs the package dataset in --dataset; it carries no "
                         "image_model.pt so it adds no member to our arm.")
    ap.add_argument("--blend-weight", type=float, default=None,
                    help="which blend weight submission.csv carries. The tail\n                         writes every weight to its own file; this only picks\n                         the default one, for when Kaggle will not accept a -f\n                         other than submission.csv.")
    ap.add_argument("--blend-public", type=Path, default=None,
                    help="a pulled public notebook dir (kaggle kernels pull -m): its "
                         "code cells run as a second arm and are rank-blended; its "
                         "dataset/kernel/model sources are attached")
    ap.add_argument("--public-weight", type=float, default=None,
                    help="which public blend weight submission.csv carries")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("-m", "--message", default="3d cnn on student pseudo-labels")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if bool(a.train_tag) == bool(a.dataset):
        raise SystemExit("give exactly one of --train-tag or --dataset")

    datasets = a.dataset.split(",") if a.dataset else []

    # A dataset created seconds ago is not attachable yet: `kaggle datasets
    # create` returns as soon as the upload lands and Kaggle finishes indexing
    # afterwards. Push before that and the kernel mounts only the competition
    # data, spends 276 seconds in the recursive fallback looking for weights
    # that were never mounted, and dies with "no checkpoint attached". That
    # cost a kernel run at 09:57 on 2026-08-09. Every member is checked: one
    # unindexed dataset out of three still loses the whole run.
    if datasets and not a.dry_run:
        for dset in datasets:
            # The Global96 package is attached as a second arm, not a member: it
            # ships checkpoints under its own names and no image_model.pt, which
            # is exactly why find_all() below does not pick it up as one of ours.
            if "global96" in dset:
                continue
            r = subprocess.run(["uv", "run", "kaggle", "datasets", "files", dset],
                               capture_output=True, text=True, cwd=ROOT)
            if "image_model.pt" not in r.stdout:
                raise SystemExit(
                    f"{dset} is not ready yet (no image_model.pt listed).\n"
                    f"Kaggle is still indexing it; wait and re-run.\n"
                    f"  {r.stdout.strip() or r.stderr.strip()}"
                )

    # With several datasets attached there is no single name to fall back on, so
    # an ensemble push has to be given --tag rather than inheriting one.
    if len(datasets) > 1 and not a.tag:
        raise SystemExit("--tag is required when attaching more than one dataset")
    slug = f"rsna-knee-sub-{a.tag or a.train_tag or datasets[0].split('/')[-1]}"
    body = "\n".join([
        (ROOT / "src" / "kernels" / "volume.py").read_text()
        if (ROOT / "src" / "kernels" / "volume.py").exists()
        else (ROOT / "src" / "volume.py").read_text(),
        (ROOT / "src" / "kernels" / "net.py").read_text(),
        (ROOT / "src" / "kernels" / "net2d.py").read_text(),
        (ROOT / "src" / "kernels" / "submit_kernel.py").read_text(),
    ] + ([(ROOT / "src" / "kernels" / "blend_coat_tail.py").read_text()]
         if a.blend_coat else [])
      + ([(ROOT / "src" / "kernels" / "blend_public_tail.py").read_text()]
         if a.blend_public else []))

    # The public notebook's code cells are embedded verbatim and run as a second
    # arm; see blend_public_tail.py. Its mounts come along, or it finds nothing.
    extra = {"dataset_sources": [], "kernel_sources": [], "model_sources": []}
    if a.blend_public:
        if a.blend_coat:
            raise SystemExit("--blend-public and --blend-coat each rewrite submission.csv; pick one")
        import base64
        import zlib
        nb_path = next(a.blend_public.glob("*.ipynb"))
        nb = json.loads(nb_path.read_text())
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for c in cells:
            for line in c.splitlines():
                if line.startswith(("!", "%")):
                    raise SystemExit(f"{nb_path.name} has a notebook magic, not script-safe: {line!r}")
        # A notebook may put `from __future__ import` at the top of any cell;
        # a script allows it only at the top of the file. Joined as-is, the
        # first pub13 run died on line 475 before doing anything, and the
        # outer py_compile could not see it inside the embedded blob. Hoist
        # them, then compile the joined script here rather than on Kaggle.
        future = []
        for i, c in enumerate(cells):
            keep = []
            for line in c.splitlines():
                if line.startswith("from __future__ import"):
                    if line not in future:
                        future.append(line)
                else:
                    keep.append(line)
            cells[i] = "\n".join(keep)
        text = "\n".join(future + [""]) + "\n\n".join(cells)
        compile(text, nb_path.name, "exec")
        script = text.encode()
        body = body.replace("__PUBLIC_SCRIPT_B64__",
                            base64.b64encode(zlib.compress(script, 9)).decode())
        if a.public_weight is not None:
            marker = "PRIMARY_WEIGHT = 0.50"
            if marker not in body:
                raise SystemExit("blend_public_tail.py has no PRIMARY_WEIGHT to replace")
            if a.public_weight not in (0.35, 0.50, 0.65):
                raise SystemExit(f"--public-weight {a.public_weight} is not in BLEND_WEIGHTS")
            body = body.replace(marker, f"PRIMARY_WEIGHT = {a.public_weight}")
        meta = json.loads((a.blend_public / "kernel-metadata.json").read_text())
        for k in extra:
            extra[k] = [x for x in meta.get(k) or [] if x]
        print(f"  blend-public: {nb_path.name}, {len(cells)} cells, {len(script)//1024} KiB, "
              f"w={a.public_weight or 0.5}; +{sum(map(len, extra.values()))} sources")

    # predict_study runs the model, so it must keep its no_grad decorator. An
    # edit that inserted a helper just above `def predict_study` once landed
    # between the decorator and the function: the kernel then ran inference with
    # grad enabled, every series died on .numpy(), and all three dev studies fell
    # back to priors while the kernel still exited 0 and wrote a submission.
    if "@torch.no_grad()\ndef predict_study" not in body:
        raise SystemExit("submit_kernel.py: predict_study lost its @torch.no_grad() "
                         "- inference would run with grad and fall back to priors")

    if a.blend_coat:
        # The tail moves submission.csv aside and rewrites it. If the arm above
        # ever stopped writing that file the tail would fail on the move and the
        # run would end with our arm's predictions unreachable.
        if 'sub.to_csv("/kaggle/working/submission.csv"' not in body:
            raise SystemExit("submit_kernel.py no longer writes "
                             "/kaggle/working/submission.csv - the blend tail "
                             "has nothing to move aside")
        if not any("global96" in d for d in datasets):
            raise SystemExit("--blend-coat needs the Global96 package in --dataset")
        print("  blend-coat: Global96 runs as a second arm, rank-mean blends written")

    if a.blend_weight is not None:
        if not a.blend_coat:
            raise SystemExit("--blend-weight does nothing without --blend-coat")
        marker = "PRIMARY_WEIGHT = 0.30"
        if marker not in body:
            raise SystemExit("blend_coat_tail.py has no PRIMARY_WEIGHT to replace")
        body = body.replace(marker, f"PRIMARY_WEIGHT = {a.blend_weight}")
        if f"BLEND_WEIGHTS = (0.20, 0.30, 0.40, 0.50)" in body and \
                a.blend_weight not in (0.20, 0.30, 0.40, 0.50):
            raise SystemExit(f"--blend-weight {a.blend_weight} is not in BLEND_WEIGHTS, "
                             "so no file would be written at it")
        print(f"  blend weight: submission.csv carries w={a.blend_weight}")

    if a.max_series:
        marker = 'MAX_SERIES = int(os.environ.get("MAX_SERIES", 0))'
        if marker not in body:
            raise SystemExit("submit_kernel.py has no MAX_SERIES default to replace")
        body = body.replace(
            marker,
            f'MAX_SERIES = int(os.environ.get("MAX_SERIES", {a.max_series}))')
        print(f"  max series {a.max_series}: at most {a.max_series} series per study")

    if a.by_plane:
        if not a.max_series:
            raise SystemExit("--by-plane does nothing without --max-series")
        marker = 'BY_PLANE = int(os.environ.get("BY_PLANE", 0))'
        if marker not in body:
            raise SystemExit("submit_kernel.py has no BY_PLANE default to replace")
        body = body.replace(marker, 'BY_PLANE = int(os.environ.get("BY_PLANE", 1))')
        print("  by plane: the series budget is spread across sagittal/coronal/axial")

    if a.slice_stride > 1:
        # The kernel reads SLICE_STRIDE from the environment, and Kaggle gives
        # us no way to set one, so bake it into the source we push. Replacing
        # the default rather than appending keeps the definition in one place.
        marker = 'SLICE_STRIDE = int(os.environ.get("SLICE_STRIDE", 1))'
        if marker not in body:
            raise SystemExit("submit_kernel.py has no SLICE_STRIDE default to replace")
        body = body.replace(
            marker,
            f'SLICE_STRIDE = int(os.environ.get("SLICE_STRIDE", {a.slice_stride}))')
        print(f"  slice stride {a.slice_stride}: every {a.slice_stride}th slice")

    d = Path(tempfile.mkdtemp())
    (d / "sub.py").write_text(body)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}",
        "title": slug,
        "code_file": "sub.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        # the default P100 cannot run torch's conv3d; see push_train_image.py
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": "false",
        "competition_sources": [COMP],
        "dataset_sources": datasets + [x for x in extra["dataset_sources"] if x not in datasets],
        "kernel_sources": ([f"{USER}/rsna-knee-train-{a.train_tag}"] if a.train_tag else [])
                          + extra["kernel_sources"],
        "model_sources": extra["model_sources"],
    }, indent=1))

    src = ", ".join(datasets) or f"rsna-knee-train-{a.train_tag}"
    print(f"{USER}/{slug} <- checkpoint from {src}")
    if a.dry_run:
        print(d / "sub.py")
        return

    r = subprocess.run(["uv", "run", "kaggle", "kernels", "push", "-p", str(d)],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip() or r.stderr.strip())
    print(
        "\nonce it completes, submit with (BOTH -v and -f are required):\n"
        f"  uv run kaggle competitions submit {COMP} \\\n"
        f"    -k {USER}/{slug} -v <version> -f submission.csv -m '{a.message}'"
    )


if __name__ == "__main__":
    main()

# Kernel scripts

Scripts pushed to Kaggle and run there, kept here so the experiments are
reproducible and reviewable. They are not importable modules: each one runs
standalone inside a kernel, where `/kaggle/input` is a mounted filesystem and
this repository does not exist.

- `meta_leak_probe.py` — can DICOM *metadata* alone predict the 12 findings?
  Trained on the 4,407 student pseudo-labels, scored on the 58 expert studies,
  reading no pixels. Motivated by a leaderboard score of 0.932 three days into
  the competition, which is fast for an image-only task with 58 labels. If
  protocol metadata carries the signal, the rank ordering says more about which
  sequences a radiologist ordered - itself a record of what they suspected -
  than about what the images show.

Note on paths: kernels must locate their inputs by globbing rather than
hardcoding. `/kaggle/input/<dataset-slug>/...` is not reliable immediately
after a dataset is created, and the failure is a bare FileNotFoundError.

## Kernel platform constraints, learned the hard way

- **Accelerator must be requested explicitly.** `enable_gpu: true` alone
  assigns a Tesla P100, whose sm_60 is absent from the preinstalled torch's
  arch list, so every conv3d dies with "no kernel image is available".
  Kaggle's own docs say "NvidiaTeslaP100 is not usable with the default Kaggle
  image". Set `machine_shape: "NvidiaTeslaT4"` - and note the API accepts a
  wrong or misspelled value *silently*, falling back to the same P100, so
  `gpu_probe.py` prints the device actually received rather than the one asked
  for.
- **Two concurrent GPU sessions, maximum.** A third push fails with "Maximum
  batch GPU session count of 2 reached", which bounds how wide a sweep runs.
- **Logs are only served for completed runs.** A running kernel returns an
  empty log, so mid-run progress is not observable from outside.
- **Datasets are not immediately mountable after creation.** A kernel pushed
  seconds later sees nothing, and the failure is a bare FileNotFoundError.
- **Output files come back through the same rate-limited API**, so kernels
  write a few large shards rather than one file per item.

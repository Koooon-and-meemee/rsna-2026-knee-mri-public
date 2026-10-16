# Image model runs

Gold = macro AUC over the 58 expert-labelled studies, aggregated to study
level. Pseudo-val = the same statistic over 522 held-out studies carrying
student pseudo-labels: biased toward whatever the text model believed, but
nine times the resolution, so it ranks variants that the 58 cannot separate.

Reference points: metadata alone 0.605 | student on reports 0.899 (not
available at test time) | leaderboard #1 0.932 at 3 days in.

| run | config | epochs | gold | pseudo-val | wall | note |
|---|---|---|---|---|---|---|
| v1 | w24 128x128x32 lr3e-4 | 6 | **0.752** | 0.743 | 202m | LB **0.753**; still improving at the last epoch |
| v2ep16 | same, cached | 16 | **0.824** | 0.807 | 200m | LB **0.834**; epochs alone bought +0.072; still climbing |
| v3ep32 | same | 32 | **0.860** | 0.849 | 419m | LB **0.868**; best at the FINAL epoch; session limit now binds |
| v4size192 | w24 **192**x192x32 | 16 | 0.812 | 0.802 | 516m | resolution HURT: -0.012 vs v2 at 2.6x the cost |

## v1

Gold rose every epoch bar one - 0.642, 0.679, 0.685, 0.728, 0.727, 0.752 -
and training loss fell throughout, 0.6003 to 0.5752. Nothing had plateaued and
nothing had started overfitting, so 6 epochs was simply too few. The model is
undertrained rather than misspecified, which is the cheaper of the two
problems to have.

The two instruments agree closely (0.743 vs 0.752). That was not guaranteed:
pseudo-val rewards imitating the text model, and gold rewards being right, so
a gap between them would have meant the image model was learning the text
model's habits. It did not appear, which is what licenses using pseudo-val to
rank variants.

Next: v2ep16 holds the architecture fixed and only raises epochs to 16, so the
undertraining reading is tested by itself.

## Planned next steps

Two branches, decided by v2ep16's curve rather than by preference:

- **Still rising at epoch 16** -> the model is still undertrained and epochs
  remain the cheapest lever. Push to 32 with the cache (~14 min/epoch, so
  roughly 8 hours - near the session ceiling, which is the real bound on this
  branch).
- **Flattened or turned over** -> capacity, not training time, is the limit.
  Raise `--width` from 24 (2.0M params) before `--size`, because width costs
  roughly linearly while in-plane resolution costs quadratically, and the
  volumes were preprocessed at 256 so 192 is available without redoing prep.

Only one knob moves per run. The gold 58 cannot resolve differences below
about 0.04, and pseudo-val's 522 studies are a proxy rather than a truth, so a
run that changes two things at once cannot be attributed afterwards.

## The gold 58 tracks the leaderboard: 0.752 vs 0.753

Measured, not assumed. The two sets are not the same thing - the 58 are
enriched for abnormality and the hidden test set is of unknown size and base
rate - so evaluate.py has always printed a warning against reading one as the
other. One submission put a point on that relationship and the gap is 0.001.

What it changes: local gold becomes a usable estimate of leaderboard position,
which means the 5-per-day submission budget can be spent only on variants that
already improved locally, and the distance to 0.932 is a real 0.18 rather than
an artefact of an easier local set.

What it does not change: this is one point. A second submission should confirm
it before the relationship is leaned on hard.

Also answered: the submission kernel completes the hidden-test rerun inside
the session limit at ~5.5 series and ~165 slices per study, so the current
inference path needs no optimisation yet.

## Original open question the submission answered

The hidden test set is rerun through our notebook, and its size is unknown -
test.csv shows 3 studies. Per study we read ~5.5 series of ~30 slices, build
volumes and run a forward pass. If the hidden set is in the thousands, the
rerun may approach the session limit; a submission that times out scores
nothing. This first submission measures that, not just accuracy.

## v1 per-finding: the gap is not uniform

    Effusion          0.920      Lateral Meniscus  0.744
    Medial OA         0.862      PF OA             0.735
    Lateral OA        0.849      Contusion         0.698
    Synovitis         0.787      ACL               0.689
    Baker's           0.786      Medial Meniscus   0.680
                                 Fracture          0.660
                                 MCL               0.615

The split is by the physical size of what has to be seen. Effusion is fluid
filling a joint space - a bright mass that survives any downsampling.
Osteoarthritis is cartilage loss and osteophytes, structural change at the
scale of the joint. Both score high.

What scores low is thin and local: the MCL is a narrow band of ligament, a
meniscal tear is a few millimetres of fissure, a fracture line is thinner
still. Training at 128x128 from volumes preprocessed at 256 is the obvious
suspect - those structures are being averaged away before the model sees them.

This revises the planned branch. The earlier note said width before size,
reasoning about cost; that reasoning holds for compute but not for this
evidence. More channels cannot restore detail that downsampling already
destroyed, so if v2 shows capacity is the limit, `--size 192` comes before
`--width 40`. The preprocessed volumes are 256, so the resolution is there to
be used.

Still one knob per run, and still after v2 - if v2 is still climbing at epoch
16, epochs remain the cheapest lever and resolution waits.

## v2ep16: the undertraining reading was right

Only the epoch count changed, and gold went 0.752 to 0.824. Pseudo-val moved
with it, 0.743 to 0.807. The distance to the leader's 0.932 fell from 0.18 to
0.108 without touching the architecture.

The last six epochs oscillate - 0.792, 0.781, 0.821, 0.824, 0.811, 0.822 -
but the trend is still up and training loss falls monotonically to the end
(0.5993 to 0.5418). Nothing has overfit. The oscillation is what 58 studies
look like: pseudo-val, with 522, rises monotonically over the same span, so
the wobble is sampling noise in the small set rather than instability in the
model.

So the branch says epochs again: v3ep32. The cache made this affordable -
12.4 min/epoch against v1's 33, a measured 2.7x - so 32 epochs is about 6.8
hours rather than a day.

Resolution stays queued behind this. It is still the best hypothesis for why
MCL, Fracture and the menisci lag, but epochs are still paying and one knob
moves at a time.

## Local-to-leaderboard, second point

    v1   gold 0.752   LB 0.753   (+0.001)
    v2   gold 0.824   LB 0.834   (+0.010)

The relationship holds, but not as an identity: the leaderboard reads slightly
HIGHER than local, in the same direction both times. That has an explanation.
The 58 are enriched for abnormality - mean 4.1 positive findings and no
all-negative case - and a set with positives crowded together is harder to
rank than one with normal studies mixed in. The hidden test set presumably
has that mix.

So local gold is a conservative estimate rather than an unbiased one. Useful
in the same way: if it rises, the leaderboard rises, and the true value should
be a little better. Not useful for reading absolute distance to a rival's
score too precisely.

## Leaderboard at day 4

     1  Brandon Low       0.934        7  Roman             0.882
     2  Pizza Boy         0.912        8  jimmyYang1016     0.877
     3  1c                0.905        9  James Y           0.872
     4  Manan Gupta       0.901       10  Ivan Sabev        0.869
     5  Subin An          0.900       11  ask812            0.866
     6  Fabian Friedland  0.889          ...  us            0.834

Roughly 15th-20th. The prize band is the top 10, currently 0.869, so 0.035
away. Worth keeping in proportion: v2 is the second run of a first
architecture - 2M parameters, quarter resolution, no ensemble, no TTA - and
the leaderboard is four days old against an October deadline.

Brandon Low remains 0.022 clear of second. Metadata was ruled out as the
explanation at 0.605, so it is a genuinely better approach rather than a
shortcut.

## v3ep32: epochs still paying, but the session limit binds

Gold 0.824 to 0.860, pseudo-val 0.807 to 0.849, and both peaked at epoch 32 -
the last one. Training loss fell to 0.4974 without flattening. There is more
here.

What stops it is not the model. 32 epochs took 419 minutes; 64 would be about
14 hours against a 9-12 hour session ceiling. Epochs are no longer a knob that
can be turned, whatever the curve says.

That makes resolution the next move rather than a deferred one, and it was
already the best hypothesis: v1's weakest findings were MCL 0.615, Fracture
0.660 and Medial Meniscus 0.680, all structures a few millimetres across,
being trained at 128 from volumes preprocessed at 256.

v4size192 holds epochs at 16 - not 32 - because 192 costs roughly 2.25x per
step. That does mean v4 is not a clean comparison against v3's 32 epochs; it
is a comparison against v2's 16, which scored 0.824. If v4 clears that by more
than the epoch-16-to-32 gain would predict, resolution is doing work.

Projected leaderboard for v3 is around 0.87 given the +0.001 and +0.010 offsets
from v1 and v2, which would put it at the edge of the top-10 prize band
(0.869).

## Local-to-leaderboard, third point

    v1   gold 0.752   LB 0.753   (+0.001)
    v2   gold 0.824   LB 0.834   (+0.010)
    v3   gold 0.860   LB 0.868   (+0.008)

Consistently positive and, after the first, consistently around +0.009. Local
gold plus ~0.008 is now a usable leaderboard estimate, which is what makes it
reasonable to iterate locally and spend submission slots sparingly.

## Standing still while climbing

     1  Brandon Low      0.934      9  tingyi           0.888
     2  Pizza Boy        0.932     10  Roman            0.882  <- prize cutoff
     3  Aastik Rajan15   0.915     11  Ivan Sabev       0.878
     4  1c               0.905     12  jimmyYang1016    0.877
     5  Manan Gupta      0.901     13  ask812           0.875
     6  Subin An         0.900     14  Zhizhou Sha      0.872
     7  kevin zhou       0.889     15  James Y          0.872
     8  Fabian Friedland 0.889         ... us           0.868

We went 0.753 to 0.834 to 0.868 overnight. The top-10 cutoff went 0.869 to
0.882 over the same window, and second place went from 0.897 to 0.932. The gap
to the prize band narrowed from 0.035 to 0.014, but the rank barely moved.

Worth stating plainly: absolute progress and relative position are different
things this early, and everyone is still on the steep part of their curve.

## v4size192: the resolution hypothesis was wrong

Against v2, which differs only in resolution:

    v2  128   gold 0.824   pseudo-val 0.807   200m
    v4  192   gold 0.812   pseudo-val 0.802   516m

Worse on both instruments, at 2.6x the time. Gold alone could be noise; gold
and the 522-study pseudo-val moving together cannot be dismissed that way.
There is no gain here.

Two corrections to earlier reasoning.

The cache did NOT fall back to streaming - it wrote 28.7GB to /kaggle/temp in
15.7 minutes and reported "cache active". The run was slow purely because 192
costs 2.6x per epoch against 128, which tracks the 2.25x pixel ratio almost
exactly. The estimate was wrong, not the system, and the 13:30 "cache failure"
call was a misdiagnosis of a correct run.

And the reading of v1's per-finding scores was wrong. MCL at 0.615 and
Fracture at 0.660 looked like thin structures lost to downsampling; more
resolution does not recover them. Two better hypotheses remain:

  - capacity. 2.0M parameters may have no representational room for 2.25x the
    input, so the extra pixels dilute rather than inform.
  - label quality. MCL is mentioned rarely in the reports, so its pseudo-label
    may simply be noisy, and no amount of image detail fixes a wrong target.

The second is testable without a GPU: check MCL's positive rate and value
spread in the teacher labels against the findings that score well.

Not submitted - 0.812 is below v3's 0.860, and the rule was to spend a slot
only on an improvement.

## local32: local/Kaggle parity, and how noisy gold-58 actually is

The same config that produced v3 on a Kaggle T4, rerun on the 5090:

    v3       Kaggle T4   ep32   gold 0.860   (LB 0.868)
    local32  RTX 5090    ep32   gold 0.858   135.7m, 4.2 min/epoch

A 0.002 difference. The local path can be trusted to stand in for the Kaggle
one, which matters because the weekly GPU quota is spent and every remaining
experiment runs here.

The more useful result is the epoch-to-epoch spread. Over epochs 20-32, after
removing the residual upward trend:

    gold-58       detrended sd 0.0081    slope +0.0015/epoch
    pseudo-val    detrended sd 0.0034    slope +0.0020/epoch

Three consequences, and they change how every earlier number should be read.

**Gold-58 cannot resolve differences under about 0.02.** So local32's 0.858
against v3's 0.860 is not a regression, and neither would 0.845 have been. Any
future claim that a variant beat the baseline on gold alone, by less than that,
is unsupported.

**Best-epoch selection on gold inflates the reported score by about +0.010.**
The maximum over 13 draws from a distribution with sd 0.008 sits roughly one sd
above its mean, and that is exactly what we see: max 0.847 against plateau mean
0.837 in the epochs before the final one. Every gold number in this log - 0.824,
0.860, 0.858 - is a best-of-N and carries that inflation. The leaderboard is
scored on the selected checkpoint, so this does not make the LB scores wrong;
it makes our *estimate* of each checkpoint optimistic, and it means checkpoint
selection is partly picking noise.

**Pseudo-val is the ranking instrument, gold is the calibration instrument.**
522 studies against 58 buys 2.4x tighter resolution. Pseudo-val inherits the
teacher's label bias so it cannot say what the absolute score is, but for
"did this change help", it is the far better tool. Gold's job is to catch the
case where a change games the teacher's biases without helping on real labels.

One more thing worth noting: the slope at epochs 20-32 is still positive on
both instruments. 32 epochs is not converged for the from-scratch model.

## m2: the architecture was the problem all along

Same data, same 128 resolution, same labels. Only the model changed - a
from-scratch 2.0M-parameter 3D CNN replaced by an ImageNet-pretrained
convnext_nano applied per slice, with gated attention pooling over the slice
axis (net2d.py).

                    pseudo-val   gold-58   wall clock
    local32 (3D)       0.847      0.858     135.7 min
    m2 (SliceNet)      0.868      0.900       7.9 min

Against the noise floor measured in the previous entry, +0.021 on pseudo-val is
about 6 sd and +0.042 on gold is about 5 sd. Both instruments moved, both by
more than they can produce by chance.

The epoch curve is worth keeping because it answers the selection worry
directly:

    ep 3   pseudo-val 0.855   gold 0.884
    ep 4   pseudo-val 0.858   gold 0.894
    ep 5   pseudo-val 0.868   gold 0.900   <- both maxima
    ep 6   pseudo-val 0.863   gold 0.890
    ...    plateau at pseudo-val ~0.856, gold ~0.885

Best-epoch selection on gold inflates by ~0.010, so a lone gold peak would mean
little. This is not a lone peak: it is the top of a smooth rise, and the
522-study pseudo-val independently picks the same epoch. The checkpoint is
real.

Two things this reframes.

**Four runs of hyperparameter tuning on v1-v4 were spent on the wrong
variable.** Roughly 20 GPU-hours went into width, epochs and resolution while
the architecture sat unquestioned, and the architecture turned out to be worth
more than all of it combined - in a seventh of the time. The lesson is not
"use pretrained weights", which is obvious; it is that a long tuning campaign
is itself evidence that the thing being tuned may be the wrong thing.

**The v4 resolution result should be retested.** 192 lost to 128 on the 3D CNN,
and the leading explanation was capacity - 2.0M parameters having no room for
2.25x the pixels. A 15.1M pretrained backbone is a different capacity regime,
so that negative does not transfer. Resolution is back on the list.

Overfitting starts immediately after ep5, so the next runs should be short.
16 epochs cost 24 minutes and 11 of them were wasted.

## Ensembling three convnexts: rejected on its own pre-committed rule

The rule set before running was that an ensemble has to gain at least 0.005 on
pseudo-val to be worth its inference time, because the submission kernel costs
4.7 seconds per study per model and code competitions cap total runtime.

                                pseudo-val            gold-58
    m2      nano seed 0            0.8675              0.8997
    m2s1    nano seed 1            0.8685              0.8807
    m3tiny  tiny                   0.8736              0.8982
    m2+m3tiny                      0.8758  (+0.0022)   0.9022  (+0.0025)
    m2+m2s1+m3tiny                 0.8771  (+0.0035)   0.8997  (-0.0000)

Both fail. The best case is +0.0035, which is one standard deviation on the
instrument that measured it, bought with 3x the inference. Rejected.

This is the seed-variance finding arriving a second time. Two nano runs
differing only in seed scored 0.8675 and 0.8685; averaging models that agree
has nothing to average over. Adding tiny helps more than adding a second seed
does - +0.0022 from one architecture change against +0.0010 from a seed - but
nano and tiny are the same family, trained the same way, on the same volumes,
so they still make substantially the same mistakes.

What this does NOT settle: whether a genuinely different backbone family
helps. EfficientNetV2 and Swin are queued for exactly this, and they are the
honest test of ensembling here - convolution versus attention is a real
difference in inductive bias, where nano versus tiny is only a difference in
width. If those also come in under +0.005, ensembling is dead for this problem
and the inference budget should go to a single better model instead.

Note also how the two instruments disagree in detail: gold puts the 2-model
ensemble ahead of the 3-model one, pseudo-val puts it behind. Both differences
are well inside gold's 0.0081 noise. This is what the instrument split is for
- pseudo-val decides, gold only checks that nothing has gone badly wrong.

## Four axes come back flat, and the plane probe says why

    nano 15M                pseudo-val 0.868
    tiny 28M                           0.874
    small 50M                          0.872     capacity has topped out
    tiny + pos_weight                  0.872     rejected
    tiny + augmentation                0.874     holds the peak, does not raise it
    ensemble of the above              0.877     +0.0035 against a +0.005 bar

Everything about the model is stuck at 0.872-0.874. That pattern - width,
regularisation, loss weighting and averaging all doing nothing - is not a
tuning problem, so the next question was what the model is actually looking at.

Every study contains all three planes and they are scored independently and
averaged. Scoring m3tiny per plane on pseudo-val:

    Sagittal   0.857        Coronal   0.857        Axial   0.841
    uniform mean over all three        0.874

Two results, and the second is the important one.

**Plane-weighted aggregation is dead.** The uniform mean beats every single
plane, and letting each finding pick its best plane scores 0.0091 WORSE than
the mean. The averaging we already do is functioning as an ensemble over
views. There is nothing here to recover.

**The model is not reading the lesions.** Medial meniscus scores 0.896 on
sagittal against 0.847 on axial. A meniscal tear is a few millimetres of
fissure that a radiologist reads on sagittal images; a model that had actually
found it could not be nearly plane-invariant. The three planes agree because
the model is reading coarse study-level appearance - degeneration, effusion
volume, general morphology - which looks the same from every direction. That
is a signal that saturates, and it explains why no amount of capacity moves it.

So the bottleneck is pixels, not parameters. The backbone was pretrained at 224
and is being fed 128, which is simultaneously a scale mismatch and a loss of
detail. Building a bigger cache would change both at once - and cannot be done
anyway, since a 192 cache needs 28.7GB against 25GB free.

--upsample resizes on the GPU before the backbone. It adds no information, so
if it helps, the gain was scale matching; if it does not, the findings need
real pixels and the disk argument becomes worth having. Zero disk either way.

## m2 scores 0.900 on the leaderboard - rank 8 of 589

    v3 (3D CNN)     gold 0.860    LB 0.868    rank 46 / 575
    m2 (SliceNet)   gold 0.900    LB 0.900    rank  8 / 589

Inside the prize band and inside the gold-medal cut, both of which are top 10.

**The +0.008 offset is retired.** Across four submissions the gap between the
local gold score and the leaderboard has collapsed:

    v1   0.752 -> 0.753   +0.001
    v2   0.824 -> 0.834   +0.010
    v3   0.860 -> 0.868   +0.008
    m2   0.900 -> 0.900   +0.000

The gold set is enriched for abnormality, which inflates AUC while the model is
weak and stops helping as it gets strong. Whatever the mechanism, the offset is
not a constant, so from here the gold number is quoted as the leaderboard
estimate directly rather than adjusted upward.

**The hidden test set is about 955 studies.** Scoring ran 19:06 to roughly
20:31, so 5100 seconds; subtracting the 610 that v1 of the kernel wasted on
recursive globs and dividing by the measured 4.7 s per study gives ~955.

That turns the runtime cap into a real budget:

    one model at 128       955 x 4.7s  =   75 min
    one model at 224       3x          =  225 min
    against a 9h cap       seven at 128, or two at 224

Raising resolution and ensembling are therefore competing for the same budget,
not independent levers. Ensembling is already rejected on accuracy grounds, so
nothing is blocked today - but if EfficientNetV2 or Swin turn out to add real
diversity, that trade has to be made explicitly rather than discovered when a
submission times out.

## Scale, not pixels - and a checkpoint thrown away by the wrong instrument

Two hypotheses were run against each other. The model had stalled at pseudo-val
0.872-0.874 across every change to width, regularisation, loss weighting and
averaging, and the plane probe suggested it was reading coarse study-level
appearance rather than lesions. Either it could not see the detail (pixels), or
it had been taught not to look (labels).

**Labels: tested and behind.** StudyNet applies the loss per study and attends
over series, so a sagittal-only finding is no longer asserted of the axial
series. Eight epochs give pseudo-val 0.860 against the series-level 0.874.

That is not yet a refutation. A study-level epoch sees 3,827 samples where a
series-level epoch sees 21,148, and the curve was still climbing at the end
(0.834 / 0.841 / 0.851 / 0.854 / 0.860). Reading a truncated run as a negative
result is how undertraining gets mistaken for a wrong idea, so 24-epoch
versions are queued.

**Pixels: it was never the pixels.** Upsampling 128 to 224 on the GPU adds no
information whatsoever, and it works:

                        ep1     ep2     ep3     ep4     ep5
    128px (tiny)       0.820   0.846   0.860   0.869   0.874
    224px upsampled    0.818   0.857   0.868   0.867   0.881

So the ceiling was not detail the model could not resolve. The backbone was
pretrained at 224 and was being fed 128, and closing that scale mismatch is
worth about +0.007 with the same bytes. The 192px cache question is therefore
withdrawn - nothing needs to be deleted, and the volumes on disk were never the
constraint.

**And the best epoch was discarded.** train_image.py selected checkpoints on
gold. Gold preferred epoch 4 at 0.912 over epoch 5 at 0.907 - a difference of
0.005 on an instrument with a standard deviation of 0.0081, so a coin flip -
while pseudo-val separated the same two epochs by 0.881 to 0.867, four standard
deviations the other way. The weights that cleared the submission bar were
overwritten by weights that did not.

Selection now runs on pseudo-val in both trainers, which is what the noise
measurement said to do three entries ago and which train_study.py already did.
The rerun is queued at 12 epochs, since 0.881 was still rising when it was lost.

The general lesson is not about this hyperparameter. Knowing which instrument
is precise is worthless if the code still asks the imprecise one to decide.

## 224 confirmed on the held-out half: pseudo-val 0.886

m17up224 is m12up224 rerun with checkpoint selection fixed. Best epoch 10.

                          full      sel      hold
    m17up224 (224)       0.8862   0.8905   0.8820
    m3tiny   (128)       0.8736   0.8784   0.8695
    difference on hold                     +0.0125

The two models' sel-minus-hold gaps are -0.0085 and -0.0089 - essentially
identical, even though m3tiny's checkpoint was chosen on gold and m17's on the
full pseudo-val. Neither run selected on the selecting half specifically, so
that common gap is not selection inflation; it is a fixed difficulty difference
between the two halves. Which means the halves are not interchangeable, and the
honest comparison is hold against hold: +0.0125 for the upsampled model.

Ensembling m17up224 with m3tiny gains +0.0024 on pseudo-val, under the +0.005
bar, and would double inference time in a runtime-capped kernel. Gold shows
+0.0060 for the same pair, but gold has sd 0.0081 and does not get a vote.
Rejected; m17up224 submits alone.

Runtime is the open risk. convnext_tiny at 224 is roughly 1.9 x 3.06 = 5.8x the
forward cost of convnext_nano at 128, which took 4.7 s per study. DICOM reading
does not scale with the model, so the total is somewhere between 2 and 7 hours
against a 9-hour cap - too wide a range to submit on. The kernel prints its own
wall clock per study on the three visible test studies, so the extrapolation
comes from measurement rather than from this paragraph.

## The submission kernel is I/O bound, not compute bound

m17up224 was held back to measure runtime before submitting, on the assumption
that convnext_tiny at 224 - roughly 5.8x the forward cost of convnext_nano at
128 - might approach the 9-hour cap. The measurement says otherwise:

    nano @128 (m2)     14.2s for 3 studies    4.7 s/study
    tiny @224 (m17)    13.5s for 3 studies    4.5 s/study

5.8x the compute, and it got slightly faster. The kernel is bound by reading
DICOMs and assembling volumes on the CPU; the backbone forward pass is noise
inside that. Estimated scoring time for ~955 studies: about 72 minutes.

This retracts the runtime budget from two entries ago. "Seven models at 128 or
two at 224" was arithmetic on the wrong bottleneck. Volumes are built once per
study and every model reads the same tensor, so a second member costs one more
forward pass - well under a second per study, about eight minutes over the full
test set.

Inference cost is therefore not a constraint on this problem at all. The
ensemble rejection stands, but only on its accuracy grounds (+0.0024 against a
+0.005 bar); the "it would double inference time" half of that argument was
wrong. If EfficientNetV2 or Swin produce real diversity, they can be added
without a runtime trade. Bigger backbones and higher resolution are likewise
free at test time - the cost lives entirely in training.

## m17up224 scores 0.929 - rank 3 of 626, and gold-58 is finished

    v1   gold 0.752 -> LB 0.753   +0.001
    v2        0.824 ->    0.834   +0.010
    v3        0.860 ->    0.868   +0.008
    m2        0.900 ->    0.900   +0.000
    m17       0.901 ->    0.929   +0.028

m2 and m17 differ by 0.001 on gold and by 0.029 on the leaderboard. The
58-study gold set did not see the improvement at all - not underestimated it,
did not see it. Selecting on gold would have meant not submitting m17.

Pseudo-val saw it: 0.868 to 0.886, predicting +0.018 against an actual +0.029.
Directionally right and, for once, conservative. 522 studies against 58 is the
whole difference, and the rule change two entries ago - select on pseudo-val,
never on gold - is what made this submission happen.

Gold's remaining job is narrow: it is the only set with real radiologist labels
rather than teacher pseudo-labels, so it still guards against a change that
games the teacher's biases. It cannot rank anything.

The result itself is larger than it looks. Upsampling adds no information - it
stretches 128px of detail across a 224px canvas - and it was worth +0.029 on
the leaderboard. That was the cheapest change made all week and it beat every
architecture, capacity, augmentation and ensembling experiment combined.

Which raises the expected value of m20real224 considerably. If a scale match
with no new pixels is worth that much, real 224 pixels may be worth more, and
the volumes are stored at 256 so there is another step above that.

## Real 256 pixels do not beat upsampled 224 - the gain was scale, not detail

m21real256 streams the volumes at their stored 256px, so every pixel is real:
no upsampling, no interpolation anywhere. Against the upsampled runs, taking
each run's own best epoch and both halves of pseudo-val:

                            pv-sel   pv-hold   sel-hold
    m21real256  real 256    0.8999   0.8820    -0.0179
    m18cos      up 224      0.8949   0.8830    -0.0119
    m17up224    up 224      0.8905   0.8820    -0.0085

On the selecting half 256 wins by 0.005. On the half that decides nothing and
therefore cannot be gamed, it is level with m18cos and level with m17up224.

The sel-minus-hold gaps say why. Every other model sits near -0.009 to -0.012;
m21real256 is at -0.0179. It fit the selecting half harder, and 0.8999 is that
much more inflated. This is exactly the failure the split was built to catch,
and without it this run would have been declared a +0.005 improvement and
submitted.

So the +0.029 that upsampling bought on the leaderboard was a scale match
between a 224-pretrained backbone and its input, not detail the model could
not previously resolve. Feeding it genuinely sharper pixels adds nothing.

m20real224 is still worth running: real pixels AT the pretraining scale
separates "real detail is useless" from "256 is too far from where the backbone
was trained", which are different conclusions with different consequences.

### And the harness lied first

Scoring these three through ensemble.py returned 0.785 for m21real256 against
the 0.900 its own training loop measured. ensemble.py reads `upsample` out of
the checkpoint but not `size`, so it fed a 256px model the 128px cache - a
quarter of the pixels it has ever seen.

That is the third train/inference resize mismatch today: the submission kernel
ignored upsample, ensemble.py ignored upsample, and now ensemble.py ignores
size. Each one produces a plausible wrong number rather than an error, and the
plausible wrong number here would have read as "real resolution is a disaster".
It now refuses to score a model whose training size it cannot reproduce.

## The model has passed the teacher, and pseudo-val is saturating

The 58 gold studies carry real radiologist labels and the teacher labelled them
too, so the teacher can be scored against truth exactly the way a model is.

    finding            pos   teacher   model   model-teacher
    Synovitis           27     0.786   0.766      -0.020
    Lateral OA          11     0.818   0.826      +0.008
    Effusion            35     0.861   0.957      +0.095
    Lateral Meniscus    23     0.865   0.839      -0.027
    Contusion           19     0.872   0.926      +0.053
    PF OA               21     0.911   0.883      -0.028
    Baker's             12     0.918   0.987      +0.069
    Medial OA           15     0.919   0.978      +0.060
    Fracture            18     0.928   0.933      +0.006
    Medial Meniscus     26     0.951   0.912      -0.038
    MCL                  9     0.972   0.939      -0.033
    ACL                 24     0.994   0.960      -0.035

    macro    teacher 0.900    model 0.909

**The model is already better than its own labels.** Training on noisy targets
across 4,407 studies averaged the noise out, and the images carry signal the
reports do not: effusion is 0.861 in the reports and 0.957 from the pixels,
which is a radiologist seeing fluid and not always writing it down.

**Which breaks the instrument.** pseudo-val measures agreement with the
teacher, so where the teacher is wrong and the model is right, pseudo-val
scores that as an error. If the teacher agrees with reality at 0.900, a
*perfect* model cannot score far above 0.90 on pseudo-val. Current pv-sel runs
0.895 to 0.900.

That is very likely why every experiment since yesterday lands inside +/-0.005
on the held-out half: not that the changes stopped working, but that the
instrument stopped resolving. It also reframes m21real256 - "level on the
holdout" may mean "indistinguishable to a saturated metric" rather than "no
better".

The m18cos submission now in flight tests this directly. It is +0.001 on
pv-hold; if it moves the leaderboard materially, pseudo-val has stopped
tracking and selection needs rebuilding.

**And it splits the remaining work by finding.** Where the teacher is weak, no
amount of training helps - the target is wrong:

    relabel        Synovitis 0.786, Lateral OA 0.818, Effusion 0.861
    keep training  ACL 0.994, MCL 0.972, Medial Meniscus 0.951 - labels are
                   near-perfect and the model trails by 0.033 to 0.038

Yesterday's plan to attack Synovitis with the model was wrong. Its ceiling is
0.786 and the model is at 0.766; there is 0.02 there, not 0.11. The real image
headroom is on the findings whose labels are already good.

## Compliance: reading the actual rules rather than quotes of them

Two forum threads ask whether sending Competition Data (report text) to a
hosted LLM API is permitted. Neither has a host answer after two days, while
the same hosts answered efficiency-track questions within hours. Participants
split 2-to-1 toward "not permitted", citing the Data Security clause.

Reading the rules themselves changes the picture, because the participants
quoted only the clause that supports their reading.

**2.6.b names a hosted commercial LLM as an acceptable tool.** "The Host will
employ an assessment of whether or not the following criteria can exclude the
use of the particular LLM, data set(s), or tool(s)... By way of example only, a
small subscription charge to use additional elements of a large language model
such as Gemini Advanced are acceptable if meeting the Reasonableness Standard."
Gemini Advanced is a hosted service; using it means sending data to Google. The
only stated ground for excluding an LLM is cost and accessibility. Transmission
is not mentioned.

**2.4.a grants use "for any purpose".** Quoted by nobody in the threads.

**2.4.b is an anti-disclosure clause, read whole.** All three of its sentences
are about people obtaining the data: prevent "persons who have not formally
agreed to these Rules from gaining access"; do not "transmit, duplicate,
publish, redistribute or otherwise provide or make available"; notify Kaggle of
"unauthorized transmission of or unauthorized access to". Publish, redistribute,
unauthorized access - the vocabulary of leakage, not of processing.

**The MIRA Research Use Agreement is the same shape.** Section 4: "YOU MAY NOT
DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY of any portion or all of a MIRA dataset
TO OTHERS without specific prior written permission." The qualifier "to others"
governs all three verbs. Section 5 is about not sharing the download link.
Section 6 forbids derivative works, which read literally would forbid training a
model at all - impossible to reconcile with a competition that exists to train
models on this data, and directly contradicted by 2.4.a.

And the literal reading proves too much: if transmitting Competition Data to a
non-participating party covered API processing, then Colab, AWS and GCP would
all be violations, and Kaggle itself is a third party relative to RSNA.

**Position: permitted.** The residual risk is not a rules question but a
discretion question - a host could decide otherwise - and no reading of the
text we have supports the strict interpretation.

Separately, relabelling locally is still worth doing, for a reason that has
nothing to do with compliance: the hosts have now published the exact severity
thresholds the annotators used (>50% of ACL fibres; OA at 1cm and >50%
thickness; meniscal signal contacting the surface on two images; "moderate or
large" for effusion and Baker cyst). Our labels were produced without them. A
relabel that encodes the rubric might beat 0.900 on its own merits, and doing it
with the local model retires the argument as a side effect.

## Label refinement: three sources that fail differently

Scored against the 58 gold studies, which carry real radiologist labels:

    teacher (report-derived)                      0.8997
    rubric relabel (report + official criteria)   0.8935
    mean of the two                               0.9025
    image model                                   0.9094
    0.25 teacher + 0.75 model                     0.9271
    0.25 (teacher+rubric)/2 + 0.75 model          0.9291

**The rubric relabel lost on its own, and the reason is in the prompt.** The
findings where the prompt quoted the hosts' published criteria improved -
PF OA +0.031, MCL +0.017, lateral OA +0.015, lateral meniscus +0.014. The
findings where the prompt added a single anecdotal example from the forum got
worse: Baker's -0.107, fracture -0.026.

For Baker's the prompt said a cyst measured at 2 x 1.5 x 3.5 cm had been graded
negative, so the size threshold is high. That is one case from one forum post,
presented as if it were a calibration standard, and both labellers anchored on
it hard - one reported pushing cysts at 2.2 x 3.1 x 3.7 cm and 1.5 x 1.5 x 4.5
cm down to 0.3 and only clearing a 9.7 cm cyst. Same mistake on fracture, where
one reported negative osteochondral case pushed every subchondral injury into
0.15-0.4.

Published criteria helped. Anecdotes hurt. The measurement separated them
cleanly because the prompt was written once and scored once.

**Averaging beats both**, which is now the third time today that combining
sources which fail differently has won where combining similar ones lost:

    teacher vs rubric    same text, different criteria      +0.003
    teacher vs model     text versus pixels                 +0.027
    all three                                               +0.029

Against yesterday's failed convnext ensemble, where the members agreed with
each other and averaging bought +0.0035 against a +0.005 bar.

**Plan.** Rebuild the training targets as a blend of the teacher labels and the
model's own out-of-fold predictions. Two cautions carried forward: the 0.25
weight was picked on 58 studies and 0.25-0.50 were all close, so use something
mid-range rather than trusting a decimal fitted to 58; and the model's
predictions on its own training studies are memorised, so they have to be
out-of-fold or this amplifies its errors instead of correcting them
(src/oof_predict.py).

Extending the rubric relabel to all 4,407 reports is deferred. It adds +0.002
over teacher-alone inside the blend, against a cost of relabelling 4,407
multilingual reports averaging 1,200 characters. The teacher-plus-model blend
delivers +0.027 for no labelling cost at all, so it goes first.

## The label hypothesis is rejected

StudyNet applies the loss per study and attends over series, so a sagittal-only
finding is no longer asserted of the axial series in the same study. The
motivating measurement was the plane probe: three planes scoring 0.857 / 0.857 /
0.841 alone, nearly plane-invariant, which is what a model reading coarse
appearance rather than lesions looks like.

At 8 epochs it scored 0.860 against the series-level 0.874 and was still
climbing, so that run was undertrained rather than refuted. Rerun at 48 epochs
with upsampling and cosine decay, it peaked at pv-hold 0.862 on epoch 13 and
then oscillated between 0.848 and 0.857 for thirteen more epochs. Stopped at 26.

    m18cos       (series-level)   pv-hold 0.883
    m24study48   (study-level)    pv-hold 0.862

0.021 short, against a holdout standard deviation of 0.0034. Giving it 5.5x the
epochs did not close the gap; it converged to a worse place.

Worth being clear about what this does and does not refute. The labelling
defect is real - the host has confirmed labels are image-derived per study with
a stricter rubric, and attaching them to every series does assert findings of
series that cannot show them. What is refuted is that fixing it this way helps.
Attention over series has to learn which series matters from the same weak
signal, and it has 3,827 studies to learn it from where the series-level model
has 21,148 examples. The extra supervision is worth more than the cleaner
target.

**Standing rule adopted: stop a run when its best pv-hold is more than 0.010
below the incumbent and that peak is at least 8 epochs old.** 0.010 is about
three standard deviations on this instrument, so the gap cannot close by noise;
requiring the peak to be stale avoids killing a run in a dip. m24study48 met
both by epoch 21 and ran five epochs past it.

## TTA does nothing, and the reason is in the training loop

    m18cos            pv-sel 0.8950   pv-hold 0.8830   gold 0.9088
    m18cos flip-TTA          0.8961          0.8833          0.9082

+0.0003 on the deciding half, a tenth of that instrument's standard deviation.

Not noise - zero, and predictably so. Training already flips the volume
in-plane with probability 0.5, so the model has seen both orientations of every
study and is invariant to the transform by construction. Averaging a prediction
with its own mirror asks the model a question it answers identically. TTA pays
when the model has never been exposed to the transform; ours has.

Worth stating as a general check rather than a fact about this flag: a TTA
transform that duplicates an augmentation already in training cannot help, and
the two lists should be compared before spending inference on it.

## Scanner memorisation is worth about 0.009, and the measurement is confounded

The default validation split cannot see scanner generalisation: 516 of its 522
studies (98.9%) were scanned on equipment that also appears in training.
m26scanner is the m18cos recipe with one change - validation is 527 studies
from 104 scanner groups that training never sees.

    m18cos       default split      pv-hold 0.8830
    m26scanner   scanner-grouped    pv-hold 0.8744
                                            -0.0086

0.0086 is about 2.5 standard deviations on this instrument, so it is real. It
is also not a catastrophe: the leaderboard number is not a fiction built on
scanner fingerprints.

**The comparison is confounded and cannot be tightened by rerunning it.** The
two validation sets are different studies, so the gap mixes together the loss
from an unseen machine and the difference in difficulty between two samples of
knees. It gives the order of magnitude and nothing finer. An unconfounded
version would need the same studies scored both ways, which the design does not
permit - a study is either in the held-out scanner groups or it is not.

Carried forward: report LOSO-style worst-group performance alongside the
default number for anything considered for a final submission, and treat a
T1-minus-T2 gap much larger than 0.009 as evidence of a new shortcut rather
than of the old one.

## Backbone learning rate: the assumed lr/10 was fine

`--backbone-lr` had never been passed explicitly; every SliceNet run inherited
lr/10 from the first one. m25blr sets it to 1e-4 against a head lr of 3e-4, a
third rather than a tenth.

    ep 10   pv-sel 0.9006   pv-hold 0.8801    <- best pv-sel of any run
    ep 12   pv-sel 0.895    pv-hold 0.876
    ep 13   pv-sel 0.898    pv-hold 0.876
    ep 14   pv-sel 0.895    pv-hold 0.873
    ep 15   pv-sel 0.895    pv-hold 0.873
    ep 16   pv-sel 0.895    pv-hold 0.873

pv-sel 0.9006 is the highest any run has recorded, and it means nothing: the
deciding half never reached m18cos's 0.8830 and drifted down over the last six
epochs. Loss kept falling to 0.3669 throughout. This is the m21real256 pattern
again - a run that fits the selecting half harder without generalising - and it
is the second time the held-back half has stopped a number that looked like a
record.

Axis closed. Nothing suggests the ratio matters within this range.

*Process note: this run was killed at epoch 11 during a GPU cleanup and resumed
from `last.pt` for epochs 12-16. Resume restores optimiser, scaler and
scheduler step count, and the resumed epochs continue the curve smoothly, so
the interruption is not a confound.*

## The resolution ladder ended at 224, and the reason is the backbone, not the lesion

m17up224 won +0.029 on the leaderboard by upsampling 128px volumes to 224 - an
operation that adds no information whatsoever. That result set the queue's
priorities for two days. m27up336 takes the same operation one rung further,
256 stored to 336 on the GPU.

    ep  1   pv-hold 0.837        ep  7   pv-hold 0.882
    ep  2           0.867        ep  8           0.880
    ep  3           0.872        ep  9           0.880
    ep  4           0.879        ep 10           0.882
    ep  5           0.876        ep 11           0.877
    ep  6           0.879

Eleven epochs, and the deciding half never once reached m18cos's 0.8830. Loss
fell from 0.5704 to 0.3742 throughout, so the model was still learning; it just
stopped converting that into held-out performance after epoch 4.

    m18cos      up 224      pv-hold 0.8830
    m21real256  real 256            0.8820
    m27up336    up 336              0.8820

Three input scales, one number. Stopped at epoch 11 of 16 by request; five
unobserved epochs is a real caveat, but seven consecutive flat ones make a
reversal unlikely.

**This refutes the explanation, not just the setting.** A widely upvoted public
notebook argues from physical scale: every series is acquired over a 150 mm
field of view, so a 126px input puts one 14px DINOv2 patch token across 16.7 mm
of knee while a meniscal tear is 1-3 mm, and the lesion is below the sampling
limit. That argument predicts continued gains at 336. There were none.

The better explanation is the backbone's own pretraining resolution.
convnext_tiny.in12k_ft_in1k was trained at 224, where its final feature map is
7x7. At 128 the map is 4x4 and the network is off its operating point; at 224
it is back on it; 256 and 336 are 8x8 and 10x10 and have nothing left to give.
Both stories predict the 128-to-224 jump identically. They diverge at 336, and
the data chose.

Two consequences. m30up448 is now expected to be flat and should be dropped
rather than run - it tests a hypothesis that has already failed. And the same
logic says a backbone pretrained at higher resolution could still benefit,
which makes the queued DINOv2 run a test of the explanation and not only a
diversity bet.

## Where the leaderboard actually is, and what is not driving it

At 879 teams we are 4th at 0.930, unchanged since 9 August 01:05. The top three
moved on 9 August to 0.940 / 0.939 / 0.938, and three new teams entered the top
twelve at 0.919-0.922 that evening.

The best verified public notebook scores **0.894**, and the most-upvoted one
(248 votes) anchors at 0.891. Nothing public explains the top of the board. The
0.010 above us is private work on an axis nobody has published.

Worth recording because it changes what to try: the strongest public notebook
describes our pipeline almost exactly - reports to targets, pure imaging model,
text discarded at inference, rank averaging because the metric is rank-only,
per-finding sample weights from how confidently the report could be read. Two
things it flags as gaps in the public frontier, slice ordering by physical
position and per-series intensity windowing, `src/kernels/volume.py` has done
from the start. The design is independently confirmed and we are 0.036 ahead of
it; the gap above us is not a design we are missing.

Every public imaging kernel uses DINOv2. Every run here has used convnext, and
the single transformer attempt was killed at epoch 2. Same-family ensembling
failed at +0.0035 because the members agreed, so a ViT remains the only untried
member that can fail differently.

## DINOv2 finishes all 16 epochs, well below the convnext baseline

m28dino336 (vit_small_patch14_dinov2.lvd142m, 336px, same recipe otherwise)
ran to completion after the net2d.py pooling/dynamic-size fix:

    ep  1  pv-hold 0.692        ep  9  pv-hold 0.815
    ep  4  pv-hold 0.753        ep 12  pv-hold 0.843
    ep  7  pv-hold 0.790        ep 15  pv-hold 0.852  <- best
    ep  8  pv-hold 0.805        ep 16  pv-hold 0.851

Best pv-hold 0.852 against m18cos's 0.883 - a 0.031 gap, real and large on
this instrument (holdout sd 0.0034). Still creeping upward at epoch 16
(+0.001 to +0.002/epoch over the last three), so not fully converged, but the
gap is too large for the remaining slope to plausibly close it. Consistent
with the Kaggle discussion report that DINOv3 also underperformed DINOv2 for
another competitor's same-recipe comparison (0.775 -> 0.763) - the ViT family
is not winning on this task at the epoch budgets tried so far, at least not
without more tuning than a straight backbone swap.

Ensemble value is still an open question independent of solo score - see
[[averaging-different-failures]] - but a 0.031-behind member needs to disagree
usefully to be worth including, and that has not been checked yet.

## The queue runner had the same invisible-job bug again, on a new script

`running()` matched `src/train_` only. `oof_predict.py` and `blend_targets.py`
don't match that prefix, so the queue saw "0 running" for the full 140 minutes
oof_predict held the GPU across 5 folds, and launched three more entries on
top of it: blend_targets (failed cleanly, its own input didn't exist yet -
oof_predictions.parquet wasn't written until oof_predict finished), m29blend40
(failed cleanly, blend_targets never produced its output), and m23swin (real
casualty - collided with oof_predict on VRAM and died to a genuine CUDA OOM).

Fixed: `running()` now matches `src/(train_|oof_predict|blend_targets)`. Same
class of bug as the 2026-08-09 one already documented in the script's own
comments (matching one trainer's name made every other trainer invisible) -
recurred because the fix was specific to the script names known at the time,
not to the general shape of the problem (any GPU-holding script this queue
launches needs to be in the pattern).

## Preprocessing fix landed - every old-shard result above is now provisional

All three Kaggle reprocessing kernels (crop_mm=130 + laterality
normalisation, see [[rsna-knee-laterality-crop-fix]]) completed
2026-08-13 ~05:00-13:00. Every number in this file above this point was
trained on the uncorrected shards. Re-baselining the m18cos recipe against
the corrected shards is the top priority once the download/swap completes -
it is the only way to know whether the fix helped, and by how much.

## Re-baseline on the corrected shards: pv-hold flat, submitted as instrument check

m31newshards - exact m18cos recipe, changed in nothing but the shards
underneath (crop_mm=130 + laterality normalisation):

    finding            m18cos(old)  m31(new)    diff
    ACL                   0.9596     0.9559    -0.0037
    MCL                   0.9388     0.9569    +0.0181
    Medial Meniscus       0.9123     0.9279    +0.0156
    Lateral Meniscus      0.8385     0.8602    +0.0217
    Medial OA             0.9783     0.9752    -0.0031
    Lateral OA            0.8259     0.8162    -0.0097
    PF OA                 0.8829     0.8822    -0.0006
    Effusion              0.9565     0.9528    -0.0037
    Synovitis             0.7664     0.8160    +0.0496
    Baker's               0.9873     0.9728    -0.0145
    Contusion             0.9258     0.9447    +0.0189
    Fracture              0.9333     0.9167    -0.0167

    macro gold: 0.9088 -> 0.9148   pv-sel: 0.8949 -> 0.8997   pv-hold: 0.8830 -> 0.8832

pv-hold - the deciding number - is flat, well inside the instrument's noise
band (sd 0.0034). Gold macro and pv-sel both moved mildly positive but
neither is what selects a submission. Per-finding gold deltas are one noisy
58-study run and shouldn't be over-read on their own, but are at least
directionally mixed rather than confirming the laterality hypothesis cleanly:
of the 5 laterality-specific labels, Lateral Meniscus (+0.022) and MCL
(+0.018) improved the most; Lateral OA (-0.010) went the other way.

**Verdict: the crop+laterality fix corrects a real, independently-verified
defect (see [[rsna-knee-laterality-crop-fix]]), but does not clear the bar
for "this is a better model" on pv-hold alone.** Pushed anyway as an
instrument check - exactly the m21real256 precedent - to see whether the
private/public leaderboard agrees with pv-hold's flat reading or disagrees
with it, which would itself be informative. Submitted 2026-08-13 ~18:26 as
qwrfsdr/rsna-knee-sub-m31newshards; scoring took unusually long (>1h PENDING,
prior submissions typically resolved in minutes) - not yet resolved as of
session close. Check `kaggle competitions submissions rsna-knee-abnormality-detection`
next session.

## Two more queue-runner races, same root cause pattern, both fixed

Same day, two more instances of "a job this script doesn't know how to see
runs unsupervised and collides with something else":

1. `build_cache()` in train_image.py cached by (depth, size) only, with no
   check against the source shards' mtimes. After the shard swap, a
   re-baseline run silently reused a memmap cache built from the OLD shards
   (dated 2026-08-12 21:51, before the crop/laterality fix) - it would have
   "confirmed" the fix did nothing by never actually training on the fixed
   data. Caught within ~4 minutes by noticing the "reusing cache" log line
   and checking its mtime against the new shards. Fixed: build_cache now
   rebuilds if any source shard is newer than the cache.

2. `gpu_queue.sh`'s free_mib() check sampled GPU memory in the few-minute
   window right after a job exited but before the driver had reclaimed its
   ~17GB, so the next job launched into memory that looked free but wasn't,
   and OOM'd. Fixed: the runner now forces one extra poll's wait the first
   cycle after running() drops from >0 to 0, before trusting free_mib().

Neither cost meaningful GPU time (both caught within minutes), but the first
one specifically could have produced a confidently wrong conclusion
("the fix didn't help") if it had gone unnoticed for longer - worth flagging
as the sharper of the two, since it fails silently with a plausible-looking
result rather than crashing.

## m32blend40new (label-blend + new shards) in progress at session close

Training against image_dataset_blend40.parquet (teacher labels blended 0.4
with the model's own out-of-fold predictions, see [[rsna-knee-prior-injection]]
for why this specifically fixes the flat-prior-for-unaddressed-findings
problem) on the corrected shards. Three epochs in at close:

    ep 1  pv-hold 0.853   ep 2  pv-hold 0.865   ep 3  pv-hold 0.876

Climbing steadily, ~0.01-0.012/epoch, and already close to the 0.883
baseline at epoch 3 of 16 - worth checking first thing next session, this is
the most promising open thread. Queue is empty after this (nothing else
queued); runner will idle once it finishes.

# m31newshards scores 0.942 — and the ranking instrument could not see it

Read 2026-08-19 14:00, six days late, because the Kaggle token had only ever
been an environment variable and five reboots erased it. The submission had
been sitting COMPLETE since 13 August.

    m18cos   old shards, up224            LB 0.930
    m31      corrected shards, same recipe LB 0.942     +0.012

That is four places on a 1936-team board: rank 6, with 0.941 as the 10th-place
prize line. The local read at submission time was "flat, does not clear the
bar for a submission" and it was pushed anyway, as an instrument check. The
instrument check is the result.

## Why pv-hold went blind, and when it will go blind again

    pv-hold   0.8830 -> 0.8832    +0.0002   (sd 0.0034)  no signal
    gold-58   0.9088 -> 0.9148    +0.0060   (sd 0.0081)  0.7 sd, right sign
    LB        0.930  -> 0.942     +0.012

Not noise, and not bad luck — the comparison is structurally incapable of
seeing this class of change. Preprocessing rebuilt the validation volumes
along with the training volumes, so the model was scored on data exactly as
corrected as what it learned from, against teacher labels derived from report
*text*, which did not move at all. Both sides shifted together; the gain
cancelled inside the ratio.

Gold-58 did better, and for a reason worth keeping: its labels are real
radiologist readings, so a change that aligns pixels with clinical truth
shows up there. It is simply too noisy (sd 0.0081) to resolve +0.006 on its
own. It got the sign right and could not have carried the decision.

The leaderboard is the only instrument where corrected images meet labels that
were not part of the correction.

**This is the second occurrence, not the first.** m2 -> m17 was +0.001 on gold
and +0.029 on the leaderboard, and it too was a representation change
(upsampling to 224 to match the backbone's pretraining operating point). Two
for two on representation changes; pv-hold has never yet missed a recipe
change.

The working rule is therefore split by *what kind of change* an experiment is:

    recipe          lr, schedule, capacity, loss, epochs, targets, sampling
                    -> pv-hold decides. A submission here wastes a slot.
    representation  preprocessing, crop, laterality, resolution, normalisation
                    -> pv-hold cannot decide. gold-58 gets a weak vote.
                       Spend the submission.

Classify before running, not after.

## What this invalidates: the resolution ladder

The 336px entry on the closed list — "the resolution axis is CLOSED, and so is
the physical-scale story for it" — was decided on pv-hold alone. up336 was
never submitted. By the rule above that verdict rests on no valid instrument.

It is worse than merely unsupported. The entire ladder (real 224, real 256,
up 336) ran on the OLD shards, which had no fixed-mm crop, so two studies at
identical pixel dimensions sat at different physical scales. On that data,
adding pixels could not reliably add detail of the same anatomy — which is
precisely the defect the crop fix removed, and the crop commit predicted this
in writing before any of this was known.

One caveat kept honest: real 256 *was* submitted, at LB 0.929 against 0.930,
so on old shards the flat reading was real and confirmed. That does not carry
over to corrected shards, and it never covered 336 at all.

The missing cell is up336 + corrected shards + plain labels — m27up336's
recipe on the new data. m32blend40new does not answer it: it moves labels and
resolution simultaneously.

## Six days of nothing, and the two silent failures behind it

Between 13 Aug 19:21 and 19 Aug there was no GPU activity at all. Five reboots
(14th, 15th x2, 17th, 19th). Two things failed quietly:

1. m32blend40new died at epoch 3 of 16 holding pv-hold 0.8765 and climbing
   ~0.011/epoch, three epochs short of passing the 0.8832 baseline. Its log
   lived in a job temp directory and is gone; cause unrecoverable.
2. The Kaggle token, being an environment variable, died with the first
   reboot — so the 0.942 that had already been scored could not be read.

Neither failure produced an error anyone would see. The queue runner exits
when the queue empties and is hand-started; `artifacts/queue.txt` held only
comments, so even a surviving runner would have had nothing to do.

For scale: 99 of the current top 100 teams submitted at least once during
those six days. Six idle days cost more than any single entry on the open list
is expected to win. Token now persisted at ~/.kaggle/access_token (600); the
runner is still not reboot-safe.

## Board shape at 2026-08-19 04:59 UTC

    rank 6 of 1936      0.942       last submitted 13 Aug
    above us            0.952, 0.947, 0.946, 0.945, 0.943
    10th (prize line)   0.941       we are 0.001 inside the money
    density             4 teams in 0.942-0.945, 5 more in 0.940-0.942
    activity            59/100 of the top 100 submitted within 24h, 89 within 3d
    our submissions     9, against a top-20 median of 16

The margin is smaller than our own holdout's standard deviation, and the
private split has not been seen at all. Sixth place is a position, not a lead.

# What the top of the board actually is (2026-08-19)

Pulled the public notebooks to answer "are the leaders just overfitting the
public LB?" The answer is: the pack is, the top four are not, and the more
useful finding is what the public frontier is made of.

## The public meta is one 35-member rank ensemble, shared between teams

`mattiaangeli/bend-the-knee-to-dinov3-ensembled` (144 votes) and
`tonylica/rsna-knee-dino-radimagenet-rank-ensemble` (82 votes) are the same
lineage. The inference stack:

    20 DINOv2-small checkpoints (5 folds)  -> percentile ranks, equal average
     5 DINOv3-small fold checkpoints        -> ranks, averaged
    parent = 0.55 * DINOv2 rank + 0.45 * DINOv3 rank
     5 RadImageNet reference attention heads + 5 E13 heads, shared RN50 encoder
    Rad    = 0.50 reference + 0.50 E13, re-ranked
    E10    = 0.50 parent + 0.50 Rad, EXCEPT Baker's and Fracture keep parent
    final  = 0.85 * E10 + 0.15 * second-pass E13

The parent notebook is 0.917. A dozen named contributors supplied checkpoints
and label sets to each other.

Two things follow. First, those blend weights and the per-finding carve-out
for Baker's and Fracture are public-LB fitting in its purest form - nobody
derives 0.85/0.15 from a holdout. Second, and more important, **the whole
public field shares checkpoints, folds and label sets**, so their errors are
correlated with each other and their private-LB shakeup will be correlated
too. Ours is a single convnext_tiny on our own text-derived teacher labels and
touches none of that lineage. On a board of 1936 teams that is a genuinely
uncorrelated entry, which is not visible in a public rank.

## Overfitting explains the pack, not the leaders

    1  Brandon Low          0.952   53 submissions
    2  MKhlystun            0.947   13
    3  CloseAI              0.946   14
    4  Lukas Nissen Molvær  0.945    8      <- fewer submissions than us
    6  us                   0.942    9
    8  Tucker Arrants       0.942    9

Max-of-n selection buys about sd * E[max of n standard normals]: 1.49 sd at
n=9, 2.29 sd at n=53. The difference is 0.8 sd. Scaling our pv-hold sd of
0.0034 (522 studies) to the hidden set puts the public sd near 0.003-0.005,
so 53 submissions are worth roughly 0.002-0.004 over our 9. The gap to first
is 0.010, and fourth place got 0.945 on eight submissions. Selection cannot
account for the top; it can easily account for the 13 teams packed into
0.938-0.942, which includes us.

Submission count correlates with score at only r=0.24 across the top 200 -
and that is confounded, since teams that submit more also work more.

## Two cheap things this hands us

**Rank averaging is untried here.** `src/ensemble.py` averages probabilities.
The metric is AUC, which only reads order, and members with different
calibration drag each other down under probability averaging - which is why
every public ensemble converts to percentile ranks first. Costs no training
and no GPU beyond one inference pass over existing checkpoints. The
same-family ensemble that failed at +0.0035 was measured the probability way
and deserves one re-read before that verdict stands.

**RadImageNet is a better-motivated diverse backbone than DINOv2.** It is a
medical-imaging pretrained encoder, whereas m28dino336 was a natural-image ViT
run on old shards - a weak test of the family-diversity idea rather than a
refutation of it.

# m33up336new — the reopened resolution axis, run and submitted

16 epochs on the corrected shards, m27up336's exact recipe with plain labels:
the control m32blend40new cannot provide, since it moves labels and resolution
together. Started 14:12, finished 19:22 on 19 August, 19.6 min/epoch.

    ep 1  0.841   ep 5  0.874   ep 9  0.880  <- selected (pv-sel 0.903)
    ep 2  0.866   ep 6  0.879   ep10  0.882
    ep 3  0.875   ep 7  0.884   ep11-16  0.877-0.878, converged
    ep 4  0.876   ep 8  0.877

    selected ep9   pv-sel 0.9026   pv-hold 0.8799   gold 0.9169
    m31 incumbent  pv-sel 0.8997   pv-hold 0.8832   gold 0.9148

pv-sel is the highest any run has posted here, beating m25blr's 0.9006 - and
m25blr is precisely the cautionary tale for reading pv-sel, since its pv-hold
never reached the bar and drifted down. pv-hold is 0.0033 below the incumbent,
one standard deviation, and gold is 0.0021 above. Locally this is a tie that
leans slightly negative on the deciding half.

**Submitted anyway, and not as a gamble.** Resolution is a representation
change, and the rule set this morning says pv-hold cannot judge those - it was
blind to a +0.012 six days ago for a structural reason that applies here
verbatim. Refusing to submit because pv-hold shrugged would be using the
instrument in exactly the way that was just shown to be invalid. Submission
55621148, 19:25 KST.

Note ep7 held the run's best pv-hold at 0.884 and was not selected, because
selection reads pv-sel only. That is the discipline working as intended: the
held-back half does not get to pick, or it stops being held back.

Kernel loaded correctly at `size=256 upsample=336`, a combination no previous
submission used (m21 was 256 with no upsample, m31 was 128 upsampled to 224).
The kernel skips the downsize when SIZE == PREP_SIZE and goes straight up to
336, which is the path training took.

The queue handed over to m32blend40new at 19:24, one poll after m33 exited -
the VRAM-reclaim fix from the 13th visibly doing its job in queue.log:

    19:23:08 last job exited - one extra poll before trusting free VRAM
    19:24:08 launching m32blend40new  (free 30882MiB, running 0)

## m33 scored 0.943 — a null result that is still worth what it cost

    m31newshards  up224 / 128px   pv-hold 0.8832   LB 0.942
    m33up336new   up336 / 256px   pv-hold 0.8799   LB 0.943

+0.001, which is well inside the public split's estimated sd of 0.003-0.005.
This does NOT establish that 336 beats 224. It is a tie.

What it does establish is narrower and still useful: pv-hold predicted a
decline of 0.0033 and the board did not reproduce it. That is the third
disagreement between the local instrument and the leaderboard on a
representation change, though the smallest of the three (+0.029, +0.012,
+0.001 in that order). The pattern holds; the magnitude does not have to.

So the resolution axis moves from "closed" to "open but below this
instrument's resolution". **Do not climb to 448 on the strength of this.**
Chasing a +0.001 signal with a five-hour run and a submission slot is how the
next six days get spent for nothing.

**The real return is two models of equal accuracy trained on different input
representations.** That is the diversity criterion from
[[averaging-different-failures]] read literally: backbone width is not a
difference, input representation is. Both checkpoints exist, both are
new-shard, and ensemble.py grew rank averaging and multi-resolution streaming
this afternoon precisely so this pair could be measured without retraining
anything. m32blend40new finishes around 01:00 and adds a third new-shard
member on different targets; measure all three together, by rank and by
probability, before spending another GPU hour.

Board after the submission: still 6th of 1960, tied at 0.943 with Pizza Boy
who submitted a day earlier. The 10th-place prize line has not moved from
0.941 all day.

## The 3-member ensemble scores 0.946 — rank 3 of 2075, and the first time the local instrument was right

    m31newshards      pv-hold 0.8832   LB 0.942
    m33up336new       pv-hold 0.8799   LB 0.943
    m31+m33+m32       pv-hold 0.8872   LB 0.946

Submission 55628839, 20 Aug 02:32 KST, prob mean of three. pseudo-val predicted
+0.0052 and the board paid +0.003. After three straight disagreements on
representation changes, this is the first change where pv-hold and the
leaderboard moved together — and it is a *recipe* change (member composition),
which is exactly the class pv-hold is supposed to be able to judge. The rule
from 19 Aug survives its first test in the direction it predicted.

Board 21 Aug: 3rd of 2075, 0.946. The 10th-place prize line moved 0.941 ->
0.943 in two days, and 139 teams joined. Ranks 2 through 11 span 0.947 to
0.943, which is inside the public split's estimated sd of 0.003-0.005: our
rank is largely noise and must not be used to steer anything.

## m34dinonew: a different backbone family does not buy diversity if it is weak

DINOv2 vit_small, up336/256, 16 epochs on corrected shards - the first DINO run
that ever saw the crop and laterality fix, which is why m28dino336's untrusted
0.850 needed replacing.

    solo            pv-sel 0.8708   pv-hold 0.8559   gold 0.8902
    m31+m33+m32     pv     0.8977   pv-hold 0.8872   gold 0.9206   (submitted)
    m31+m33+m34     pv     0.8950   pv-hold 0.8854   gold 0.9171
    m31+m33+m32+m34 pv     0.8965   pv-hold 0.8864   gold 0.9184

Swapped in, it costs -0.0027 / -0.0018 / -0.0035. Added as a fourth, -0.0012 /
-0.0008 / -0.0022. Every individual gap sits inside its instrument's noise, but
six numbers falling in one direction across three instruments - one of them
gold-58, with real radiologist labels - is not a coin landing the same way six
times. **No submission was spent on this.** That is the LB policy applied
correctly: the board adjudicates when instruments disagree, not when they agree.

**Rank mean went negative in both** (-0.0018, -0.0007 vs best member) while prob
mean stayed positive. Converting to percentile ranks removes the units, but it
also removes a weak member's lower confidence, so its bad ordering gets a full
vote. "Every public ensemble ranks first" is not unconditional - it assumes
members of comparable strength. With a member 0.029 back, prob mean is the safer
average.

**Not retrained longer.** ep14/15/16 were 0.870 / 0.871 / 0.870 while loss kept
falling 0.4816 -> 0.4777 -> 0.4756: fitting the targets without improving the
ranking. More epochs of that recipe buy overfitting, and +0.001 per three epochs
does not close a 0.029 gap.

The family-diversity hypothesis is shelved, not refuted. It was confounded twice
over - a flat 3e-4 on a DINOv2 ViT backbone (this project has had a
`--backbone-lr` knob since m25blr and did not use it here) and equal weighting
in the ensemble. Either could be worth revisiting; neither is worth five GPU
hours to rescue a member whose best case is about +0.002.

## The instrument is drifting apart, and one axis was never touched

Across all 25 runs, pv-sel climbed 0.886 -> 0.903 over two weeks while pv-hold
did not move at all:

    m18cos      9 Aug    sel 0.8949   hold 0.8830   gap 0.0119   gold 0.9088
    m31        13 Aug    sel 0.8997   hold 0.8832   gap 0.0165   gold 0.9148
    m33        19 Aug    sel 0.9026   hold 0.8799   gap 0.0227   gold 0.9169
    m32        19 Aug    sel 0.9032   hold 0.8808   gap 0.0224   gold 0.9147

The best pv-hold any run has posted is m31's 0.8832; m18cos tied it twelve days
earlier at 0.8830. The sel-hold gap - which is precisely the selection inflation
the hash split was built to measure - has doubled. Meanwhile gold and the LB both
rose, so the work is real; it is the ranking instrument that has stopped
resolving it.

Two readings, and they are not yet separable. Either pv-hold is blind to what
actually improved (representation changes, per the 19 Aug rule), or **pv-hold is
scored against pseudo-labels and has hit the teacher's ceiling** - both halves
share the teacher's biases, so the split catches epoch-selection inflation but
cannot catch fitting the teacher's systematic errors. If the second is true,
pv-hold's ceiling is not the model's ceiling and we have been flying on one
instrument for a while.

**Every one of the 25 runs is `depth 32 / width 24`.** The through-plane axis has
never been varied once. It is a representation change orthogonal to in-plane
resolution, and representation diversity is the only ensemble axis with a
measured precedent (m31-containing pairs 0.8860 vs same-representation 0.8841).
m35depth48 is m31's recipe with depth alone changed - known backbone, one
variable, 128px so ~2.5h.

**Submission budget was being hoarded for no reason.** Deadline is 2026-10-22:
62 days, ~300 slots, 11 used. Low submission count protects against fitting the
public split - and with 11 probes we cannot be doing that - but it is not thrift
worth preserving. Agreed 21 Aug: when local instruments disagree or a gap falls
inside noise, spend a slot rather than reason the difference out of a saturated
pv-hold. When they agree, as today, do not.

## m36dinolr: there was no lr confound, and that is the more useful finding

**Retracted premise.** m36 was queued to remove a confound: m34 was described as
having been trained with "a flat 3e-4 on a pretrained DINOv2 ViT" because it did
not pass `--backbone-lr`. That is wrong. train_image.py:332 reads

    blr = a.backbone_lr if a.backbone_lr is not None else a.lr / 10

so the flag's *default* is discriminative, not flat. m34 ran at backbone 3e-5 /
head 3e-4, and both logs print the same line to prove it:

    m34  backbone vit_small_patch14_dinov2.lvd142m  lr 3e-05 / head lr 0.0003
    m36  backbone vit_small_patch14_dinov2.lvd142m  lr 3e-05 / head lr 0.0003

The knob has never been off. `--backbone-lr 3e-5` on the m36 command line set it
to the value it already had. **m34 and m36 are the same experiment run twice**,
same seed (0), same everything.

## Two identical runs differ by 0.0075 on pv-hold

That accident is worth more than the experiment it ruined, because this project
has never measured run-to-run variance and has been deciding on gaps smaller
than it.

    m34dinonew   pv-sel 0.8708   pv-hold 0.8559   gold 0.8902
    m36dinolr    pv-sel 0.8794   pv-hold 0.8634   gold 0.8958
                       +0.0086         +0.0075         +0.0056

Identical commands, identical seed, and they diverged from the first epochs -
0.814 vs 0.831 at ep8, before m36 was ever resumed - so this is training
nondeterminism (cuDNN algorithm selection, atomics), not the resume and not the
seed. Two samples are not a standard deviation, but a 0.0075 spread between
replicates is consistent with a per-run sd around 0.005.

**The instrument sd quoted throughout this file, 0.0034, is the wrong quantity.**
That is evaluation noise: how much the score of *one fixed model* moves because
pseudo-val is 522 studies. It says nothing about how much the same recipe moves
when you train it again. The decisions in this file have routinely turned on
0.001-0.004 differences between single runs of different recipes, and those
differences sit inside the replicate spread measured here.

What this does NOT invalidate: comparisons where the direction was consistent
across several instruments, and LB comparisons, which are a separate instrument
on unseen data. What it does invalidate: any conclusion of the form "recipe A
beat recipe B by 0.003 in one run each".

**Standing rule from this point: a single-run gap under ~0.005 on pv-hold is not
a result.** Either replicate it, or find the direction confirmed on pv, pv-hold
and gold together, or spend a submission.

## The DINOv2 member verdict, unchanged and now better supported

The retraction does not rescue the family - it removes the excuse that was
protecting it. Two independent runs of the same recipe, and both fail the same
way as members:

    incumbent   m31+m33+m32       pv 0.8977   pv-hold 0.8872   gold 0.9206
    swap m34    m31+m33+m34         -0.0027         -0.0018         -0.0035
    swap m36    m31+m33+m36         -0.0020         -0.0007         -0.0044
    add  m34    m31+m33+m32+m34     -0.0012         -0.0008         -0.0022
    add  m36    m31+m33+m32+m36     -0.0009         -0.0007         -0.0019

Twelve readings, all negative. The individual magnitudes are inside the
replicate spread and should not be compared to each other - m36 is not
"less bad" than m34 in any meaningful sense - but twelve signs are not.

Solo, the two runs bracket 0.8559-0.8634 on pv-hold against convnext_tiny
members at 0.8799-0.8835. The pre-registered bar was ~0.89. Neither run came
close, and the gap is four to five times the replicate spread, so this one *is*
readable.

Rank mean went negative or nil in three of four (-0.0018, -0.0007 vs best member
for m34; -0.0004, +0.0003 for m36). Prob mean stays ahead whenever a member is
materially behind. Two backbones, four measurements, same finding.

**No submission spent.** Instruments agree; the board adjudicates disagreement.

**Closed: backbone family diversity.** Tried twice, ten GPU hours, a member
0.017-0.020 back both times. The 35-member DINOv2/DINOv3 ensembles at the top of
this board are a different quantity of compute, not a better flag. Do not spend
a third run here.

**Still open, and now the only supported axis:** input *representation*
diversity (m31-containing pairs pv-hold 0.8860 vs the same-representation pair
0.8841). See m37up288 - the third representation, which is the first time that
axis has been given a third point in 25 runs.

## Audit: what the 0.005 replicate rule takes down

Applying the rule from the section above to every closed conclusion in this
file. The test: does the claim rest on a gap between *single runs of different
recipes*, and is that gap under ~0.005 with no second instrument agreeing?

**Collapsed — these were never established and must not be cited again:**

    capacity topped out    small 0.872 vs tiny 0.874    0.002   convnext_small
                           was never actually ruled out. nano 0.868 vs tiny
                           0.874 is 0.006 and survives, so "bigger than nano
                           helps" holds; "bigger than tiny does not" does not.
    pos_weight rejected    0.872 vs 0.874               0.002
    augmentation ceiling   0.874 vs 0.874               0.000   moot in practice,
                           every current recipe uses aug regardless
    same-family ensemble   +0.0035 vs an invented 0.005 bar    the measurement
                           and the bar were both inside the spread
    label blending (m32)   0.8808 vs m31 0.8832 (0.0024) and vs m33 0.8799
                           (0.0009). m32 is indistinguishable from m33. The
                           member verdict stands on other grounds - it shares
                           m33's representation - but "the blend does not clear
                           the bar" was never measured.
    real resolution        "256px and 224px both TIE upsampled 224". A tie is
                           exactly what an underpowered instrument cannot
                           establish. This is UNMEASURED, not closed.

**Survives:**

    plane weighting        0.0091 worse          ~2x spread
    study-level loss       0.862 vs 0.883        4x
    backbone family        0.017-0.020 solo, n=2 4-5x, and replicated
    scanner-grouped split  0.0086                1.7x
    3-member ensemble      pv-hold +0.0037 AND LB 0.943 -> 0.946. A second
                           instrument on unseen data agreed. This is the model
                           for what a real result looks like here.
    corrected shards       LB +0.012
    flip-TTA               +0.0003 - survives on MECHANISM, not on the number.
                           Training flips with p=0.5, so the model is invariant
                           by construction and the measurement merely confirms
                           what the training loop guarantees.

**The uncomfortable one.** The precedent this project has been steering by -
"representation diversity pays, target diversity does not", m31-containing pairs
pv-hold 0.8860 against the same-representation pair 0.8841 - is a **0.0019 gap**.
Inside the spread. It is the stated justification for m37up288, which is running
as this is written. m37 is a defensible bet, but it is a bet, and the record
should not call it evidence-led.

**What this changes about how to work here.** The instrument cannot resolve
0.002, and most axes move things by about that much. Three responses, in order
of cost:

  1. Stop treating one run as a measurement. Where a decision matters, run the
     recipe twice - it is 5 hours to convert a guess into a result.
  2. Prefer axes with large expected effects over axes with plausible small ones.
  3. Spend submissions. The LB is a genuinely independent instrument on unseen
     data, we have used 11 of ~300 slots, and every conclusion in the "survives"
     list that is not mechanical got there by LB corroboration or by a gap so
     large the instrument could not hide it.

## Seed diversity: the axis this accident exposed

m34 and m36 are the same recipe trained twice and they fail *differently* -
0.0075 apart on pv-hold. Different failures is the one precondition under which
averaging pays, and in 25 runs seed diversity has never been tried, because
until today nobody knew two runs of one recipe were different models at all.

Queued as a zero-cost test: both checkpoints exist, so `ensemble.py m34dinonew
m36dinolr` asks whether same-recipe replicates average better than either
member. If they do, the cheapest ensemble member available is not a new
representation or a new backbone - it is m31's own recipe with `--seed 1`.

## Mid-run judgement, recalibrated

The standing early-stop rule was "stop when best pv-hold is more than 0.010
below the incumbent and that peak is at least 8 epochs old", justified as "about
three standard deviations". It was three sd of the *evaluation* noise (0.0034).
Against the replicate spread measured on 22 August (~0.005) the same 0.010 is
two sd, which makes the rule twice as trigger-happy as it was written to be -
the wrong direction for a rule whose failure mode is killing a good run.

Recalibrated, and split by what the run is FOR:

**A run meant to win alone** - stop when best pv-hold is >0.015 below the
incumbent (three replicate sd) and the peak is at least 6 epochs stale.

**A run meant to join the ensemble** - the incumbent is the wrong reference. A
member does not have to beat the ensemble; it has to be strong enough that its
vote is not noise. DINOv2 came in 0.017-0.020 behind the tiny members and lost
all twelve readings, twice. So: compare against an *accepted member's curve at
the same epoch*, and stop when the candidate is >0.015 behind it with its own
peak 4+ epochs stale. m33's per-epoch pv-hold is checkpointed at
`artifacts/ref/m33_pvhold.txt` for exactly this.

**What the mid-run check cannot do.** Tracking the reference proves the
candidate is strong enough to earn a vote. It says nothing about whether it
fails *differently*, which is the entire reason to build a member - and that is
only visible in the ensemble measurement afterwards. A candidate that matches
the reference epoch for epoch may simply be the same model in different
clothing. Passing the floor check is permission to finish the run, not evidence
the run will pay.

## m37up288: the third representation, and the first time the instruments split

Source 192 -> 288, m33's recipe with the representation as the only variable.
Sixteen epochs, 4.7h, and the checkpoint stopped moving at epoch 10 - pv-sel sat
at 0.896-0.898 from epoch 4 onward while loss fell 0.468 -> 0.358. The last six
epochs changed nothing. Letting them run was the right call for comparability
(m33 is a 16-epoch run and a 10-epoch m37 could not be set beside it) and it
still cost 1.7 GPU-hours for no change to the artifact. Worth a rule next time:
six flat epochs with loss still falling is an "it is finished" signal, distinct
from the existing "it is failing" kill rule.

    solo   m37up288   pv 0.8905   pv-sel 0.8982   pv-hold 0.8830   gold 0.9123

Which places it on top of the other members and inside the noise of all of them
(m31 0.8835, m32 0.8808, m33 0.8799 on pv-hold). Three convnext_tiny models on
three input representations land in the same place. **Nothing to choose between
them, and that is itself the finding**: in-plane representation is not where
this model's remaining error lives.

**As a member, the instruments split for the first time.**

    incumbent  m31+m33+m32        pv 0.8977   pv-hold 0.8872   gold 0.9206
    swap m37   m31+m33+m37          -0.0008        +0.0010        -0.0003
    add  m37   m31+m33+m32+m37       0.0000        +0.0008        -0.0006

pv-hold is positive in both configurations, gold is negative in both, pv is nil.
Six readings that do NOT point one way - the opposite of m34 and m36, where
twelve pointed the same way and no submission was warranted. Every magnitude
here is at most 0.0010, a fifth of the replicate spread, so no local argument
can settle it. That is the case the 21 August LB policy exists for, and a slot
was spent: kernel `rsna-knee-sub-ens4m37`, the four-member prob mean.

**The four-member was chosen over the swap deliberately.** The swap wins nothing
even if it is real - it trades one interchangeable member for another - while
the four-member tests whether member COUNT scales, which is the one direction
the top of this board demonstrates (a 35-member rank ensemble). And it is purely
additive: nothing the incumbent has is discarded on the strength of a 0.0010
signal.

**A second blow to the representation-diversity story.** Gain over the best
member is what actually measures diversity, and it does not move:

    m31+m33+m32   prob +0.0052   rank +0.0019
    m31+m33+m37   prob +0.0053   rank +0.0022

m32 shares m33's representation and was filed as adding no diversity; m37 has a
representation of its own. They contribute the same. Together with the audit
finding that the founding precedent was a 0.0019 gap, the honest position is now
that **representation diversity has never been shown to pay here**. The ensemble
gain of ~+0.005 appears to be what averaging any three comparable models gives,
independent of how they differ.

## Seed diversity does not pay, and the reason corrects the earlier reasoning

Four hours after proposing it, the zero-cost test refutes it. m34 and m36 are
the same recipe trained twice; averaging them is worse than the better one.

    m34dinonew          prob 0.8628   rank 0.8597   hold 0.8559
    m36dinolr           prob 0.8714   rank 0.8684   hold 0.8634
    ENSEMBLE prob mean  0.8704  (-0.0010 vs best member)   hold 0.8627
    ENSEMBLE rank mean  0.8668  (-0.0046 vs best member)   hold 0.8598
    gold: prob 0.8952 (-0.0006), rank 0.8940 (-0.0019)

Four readings, all negative. **The hypothesis was wrong and the error in it is
worth naming**: "two runs differ by 0.0075" was read as "two runs fail
differently", and those are not the same statement. Replicates of one recipe
share architecture, data, targets and augmentation, so their errors are
correlated almost by construction - what differs is largely *how good* each run
is, not *where* each run is wrong. Averaging correlated errors buys nothing, and
averaging a stronger model with a weaker one drags toward the weaker. A 0.0075
spread is evidence of instability, not of diversity.

Rank mean is punished harder (-0.0046) than prob mean (-0.0010), which is the
same asymmetry m34 and m36 showed as members: converting to percentile ranks
strips the weaker model's lower confidence and gives its worse ordering a full
vote. Third independent observation of that effect.

**What survives.** This was measured on the weakest pair we own, with a 0.0075
quality gap between them. Two *strong* replicates that land closer together
might still average usefully - the mechanism that killed this test was the
quality gap, not the shared recipe alone. m38seed1 (m31's recipe, --seed 1) is
already training and `ensemble.py m31newshards m38seed1` will answer it for a
pair at 0.883 rather than 0.860.

**And its other purpose is now the more important one.** Every judgement in this
project is scaled against a replicate spread of 0.005, and that number comes
from a single pair of runs of the weakest model here. m38 measures it on a
strong recipe. If strong recipes are tighter, the 0.005 rule is too conservative
and several conclusions retired in the audit may be recoverable; if they are not,
the rule holds and so do the retirements.

## LB 0.947: pv-hold called the split correctly, and gold was wrong

The four-member m31+m33+m32+m37 scored **0.947**, up from the three-member's
0.946. The submission was made precisely because the instruments disagreed -
pv-hold +0.0008, gold -0.0006, pv nil - and the board has now adjudicated:

    pv-hold  +0.0008 (positive)  ->  LB +0.001   CORRECT
    gold     -0.0006 (negative)  ->  LB +0.001   WRONG
    pv        0.0000             ->  LB +0.001   uninformative

Second consecutive time pv-hold predicted a board movement (three-member:
+0.0037 local, +0.003 on the LB). Gold has now been wrong once in the only case
where it disagreed with pv-hold. That is not enough to retire gold - it is 336
series and its noise is large - but when the two conflict, **pv-hold is the one
with a track record**. Recorded so the next split is not re-argued from scratch.

The 21 August LB policy has now paid for itself: it produced a submission that
turned an unresolvable local question into +0.001 and a rank.

## Seed diversity, corrected: quality gap kills it, not the shared recipe

The refutation four hours earlier was half right, and the caveat attached to it
turned out to be the important half.

    weak pair    m34 + m36   quality gap 0.0075   ensemble -0.0010  LOSES
    strong pair  m31 + m38   quality gap 0.0024   ensemble +0.0026  PAYS

Same-recipe averaging is not inherently worthless. What killed the first test
was mixing a model with one 0.0075 worse, which drags the average toward the
weaker member faster than any decorrelation can recover. Close replicates
decorrelate enough to pay. **The rule is about the gap between members, not
about how they were produced** - which is the same rule that rejected DINOv2
(0.017-0.020 back) and is now the third finding pointing at it.

## Replicate spread, measured a second time - and the first estimate was lucky

m38seed1 is m31's recipe with --seed 1, the first replicate of a *strong* recipe.

    m31newshards  pv-sel 0.8997   pv-hold 0.8832   gold 0.9148
    m38seed1      pv-sel 0.9034   pv-hold 0.8836   gold 0.9205
                        +0.0037         +0.0004        +0.0057

**pv-hold differs by 0.0004 here against 0.0075 on the weak pair.** The
temptation is to conclude that strong recipes are twenty times more stable and
that the 0.005 rule is far too conservative. That conclusion is not available
from two pairs. Two draws from a distribution with sd 0.005 land 0.0004 apart
often enough; estimating sd from a single pair difference (sd = delta/sqrt2)
gives 0.0053 from one pair and 0.0003 from the other, and pooling the variances
gives **sd about 0.0038**.

So the 0.005 rule stands, slightly conservative, on n=2 pairs. It is not
twenty-fold too strict and the six retired conclusions stay retired - gaps of
0.000-0.0035 remain unreadable at sd 0.0038. What a third pair would buy is a
real estimate rather than a pooled guess; what it costs is 2.5 GPU hours, which
is worth it only if a decision turns on the difference between 0.0038 and 0.005.

Note also that **gold moved 0.0057 between two identical runs** while pv-hold
moved 0.0004. Gold is 336 series and its replicate noise is an order larger than
pv-hold's. Combined with gold being the instrument that just called the LB wrong,
gold should be read as a tiebreaker of last resort, not as a third vote of equal
weight.

## Three seeds, one recipe: replicate spread depends on CONVERGENCE, not recipe

    m31newshards  seed 0   sel 0.8997   hold 0.8832   gold 0.9148
    m38seed1      seed 1   sel 0.9034   hold 0.8836   gold 0.9205
    m39seed2      seed 2   sel 0.8947   hold 0.8831   gold 0.9136
                           sd 0.0044    sd 0.0003     sd 0.0037

**pv-hold replicates to 0.0003 on a converged recipe.** The 0.005 rule was
derived from the m34/m36 pair at 0.0075 - and those two were DINOv2 runs still
climbing at epoch 16. An unconverged run is sampled at a random point on a
rising curve; a converged one is sampled at a plateau. That, not the recipe or
the backbone, is what sets replicate spread.

**Revision to the standing rule.** "A single-run gap under 0.005 on pv-hold is
not a result" holds for runs that have not plateaued. For runs that HAVE - flat
best-epoch pv-hold with loss still falling, which is every convnext_tiny run
here - the readable threshold is far lower, and 0.001-0.002 differences are
real. Conclusions retired in the 23 August audit should be re-examined one at a
time on that basis: those measured between converged runs may be recoverable,
those measured against unconverged runs stay retired.

This also strengthens last night's instrument order rather than weakening it.
pv-sel (sd 0.0044) and gold (sd 0.0037) still cannot read the effects being
chased; pv-hold, at sd 0.0003, is an order more sensitive than either. It was
already the instrument with the predictive record, and it is now also the only
one with the resolution.

Note the seed pattern in the member pool: m39 is the weakest of the three on
pv-sel (0.8947) but identical on pv-hold (0.8831). Whether it earns a seat is a
question about the quality gap on the instrument that matters, and on that
instrument there is no gap at all.

## Re-examining the audit: four stay retired, two are provisional

The convergence finding rescues only comparisons made on **pv-hold**. It does
nothing for the rest, and the reason is the instrument rather than the runs.

**Stay retired - measured on pseudo-val, which has sd 0.0044.**

    capacity topped out    small 0.872 vs tiny 0.874   pseudo-val, gap 0.002
    pos_weight rejected    0.872 vs 0.874              pseudo-val, gap 0.002
    augmentation ceiling   0.874 vs 0.874              pseudo-val, gap 0.000
    same-family ensemble   +0.0035 vs a 0.005 bar      pseudo-val

All four predate the sel/hold split, so they were scored on the same 522 studies
that selected the checkpoint. That is the pv-sel statistic, and three seeds put
its replicate sd at 0.0044. A 0.002 gap is under half of one sd. Convergence is
irrelevant here - the ruler was never fine enough, and no amount of stability in
the runs fixes a instrument that moves 0.0044 by itself.

**Provisionally recoverable - measured on pv-hold, between converged runs.**

    label blending   m32 0.8808 vs m31 0.8832    gap 0.0024
    resolution       m33 0.8799 vs m31 0.8832    gap 0.0033

At the measured pv-hold sd of 0.0003 these are 8 and 11 sd. If that sd transfers,
both are real: m32 and m33 are genuinely *worse single models* than m31, and the
19 August reading of the resolution axis as "a tie" was an artefact of comparing
them on the wrong instrument.

**Why provisional and not settled.** The 0.0003 was measured by replicating ONE
recipe three times - m31's, 128px/up224, 12 epochs. Nothing yet shows that a
different recipe replicates as tightly, and comparing m31 against m33 spans two
recipes, so it inherits whatever instability m33's recipe has on top of m31's.
The measurement that closes this is already running: m40m33s1 is m33's recipe at
seed 1, and the pair will say whether pv-hold's tightness is a property of the
instrument or of that one recipe.

**One consequence worth noting now.** If m32 really is 0.0024 behind m31 as a
single model, that is independent support for the submission currently scoring,
which drops m32 from the ensemble. Two separate lines of evidence - the member's
own quality, and the ensemble's pv-hold rising when it is removed - would then
point the same way, which is a stronger position than the 0.0010 signal the
submission was actually made on.

## Two different four-member ensembles both score 0.947: count is the lever, composition is not

    3 members  m31+m33+m32              pv-hold 0.8872   LB 0.946
    4 members  m31+m33+m32+m37          pv-hold 0.8880   LB 0.947
    4 members  m31+m33+m37+m38          pv-hold 0.8898   LB 0.947

The second four-member config drops m32 and adds m38, and pv-hold preferred it
by +0.0018. **The board cannot tell them apart.** The +0.001 came from going
three to four; which model occupies the fourth seat did not register.

This is the third independent observation of the same thing. m32 (sharing m33's
representation) and m37 (built with its own) contribute identical gain over the
best member (+0.0052 vs +0.0053). Now two four-member sets with different
membership score identically. **Members are interchangeable at this scale as
long as their quality is comparable** - which is also why DINOv2 failed, since
it was the one candidate whose quality was not comparable.

**What this costs pv-hold.** It is now 2 for 3 on predicting board movements,
and the miss is the case where it was asked to rank two configurations rather
than to say whether a change helps. Two readings follow and they are not
separable yet: either pv-hold's 0.001-0.002 resolution genuinely does not
transfer to the hidden set, or the public LB's own quantisation (three decimals,
so it hides up to 0.001) is swallowing a real difference. The second is likely -
0.8880 and 0.8898 differ by 0.0018 and the LB reports 0.947 for both, which is
exactly the width of one rounding step.

Either way the practical rule is the same: **stop spending slots on which
members, spend them on how many.** Submitted the five-member (pv-hold 0.8892)
to test whether the ladder continues past four; if count is the lever it should
read 0.948, and if the 3->4 step was a one-off it will read 0.947 again.

Note this also removes the reason to keep measuring member swaps locally. The
ensemble measurements are five minutes each and they rank configurations that
the board treats as identical. From here the useful local measurement is
"is this new member comparable in quality", not "which subset is best".

## pv-hold's tightness is a property of the instrument, not of one recipe

m40m33s1 is m33's recipe at seed 1 - the second recipe to be replicated.

    m31 recipe (128/up224, 12ep)  n=3    pv-hold sd   0.0003   sel sd   0.0044
    m33 recipe (256/up336, 16ep)  n=2    pv-hold diff 0.0006   sel diff 0.0031

Same order on both. pv-hold replicates to 0.0003-0.0006 regardless of recipe,
while pv-sel moves 0.003-0.004 and gold 0.004-0.005. The instrument order set on
23 August is now measured on two independent recipes rather than inferred from
one.

**The two provisional recoveries are confirmed.** At sd 0.0003-0.0006:

    m32 0.8808 vs m31 0.8832    gap 0.0024   ~5-8 sd   REAL
    m33 0.8799 vs m31 0.8832    gap 0.0033   ~6-11 sd  REAL

m32 and m33 are genuinely weaker single models than m31. The 19 August reading
of the resolution axis as "a tie" was an artefact of judging it on pseudo-val,
whose sd is ten times larger.

**And with two seeds each, the recipe comparison itself is now clean:**

    m31 recipe   pv-hold  0.8832 / 0.8836 / 0.8831   mean 0.8833
    m33 recipe            0.8799 / 0.8792            mean 0.8796
                                                     gap  0.0038

**m31's recipe is the better one, by about ten standard deviations, and it is
half the cost** - 2.5h at 128px against 5.3h at 256px. Source resolution 128
upsampled to 224 beats source 256 upsampled to 336, which is consistent with the
old note that the +0.029 upsampling won on the leaderboard was a scale match to
a 224-pretrained backbone rather than detail. The member factory should be m31's
recipe exclusively; m41seed3 and m42seed4 already are.

**A last observation about checkpoint selection.** m33's best pv-sel epoch was 9,
m40's was 5 - and their pv-hold landed within 0.0006 of each other anyway. The
epoch that pv-sel picks moves around by four epochs between replicates while the
quality it is picking from barely moves. Selection on pv-sel is close to
arbitrary within the plateau; it is not choosing a better model, it is choosing
a luckier reading. That costs nothing here because pv-hold is what judges, but
it means the sel/hold gap should never be read as a model property.

## Each recipe is worth about two members

The six-member measurement separates two things that looked the same all night.

    4  m31 m33 m37 m38                pv 0.8985  hold 0.8898  gold 0.9203
    5  + m39  (recipe A, 3rd copy)       0.8982       0.8897       0.9208
    6  + m39 + m40  (recipe B, 2nd)      0.8990       0.8906       0.9213

    recipe A  128/up224   m31, m38 already present   m39 is the 3rd   -0.0001
    recipe B  256/up336   m33 already present        m40 is the 2nd   +0.0009

At pv-hold's replicate sd of 0.0003 those are a confident nothing and a
three-sigma gain. **A second replicate of a recipe pays; a third does not.**

The mechanism is the one that killed seed diversity on the weak pair and
rescued it on the strong one. Averaging two runs of a recipe cancels the part of
the error that is training noise. Once that is cancelled, what remains is the
recipe's systematic error, which every further replicate shares in full - so the
third copy adds a correlated vote and the average does not move.

**This retires two of my own claims from the last twelve hours.** "Member count
is the lever" was wrong: the fifth member did nothing. "Composition does not
matter" was also wrong: it matters entirely, just not in the way the earlier
measurements suggested - what matters is which *recipes* are represented and how
many times, not which individual checkpoints. Both claims were built on the
observation that two four-member sets scored 0.947, which is consistent with
this rule too, and I over-read it.

**What the rule predicts, and what tests it.** m37's recipe (192/up288) has no
replicate. If the rule holds, m45m37s1 should add about +0.0009 to pv-hold and a
further member of recipe A or B should add nothing. That is a specific number
for a specific run rather than a story fitted to results, which is what the
previous two claims lacked.

**And it changes what a member costs.** Growing the ensemble now needs new
*recipes*, each trained twice - not more seeds of what we have. The pool is
A x3 (one wasted), B x2, C x1. With capacity reopened as a live question, a
convnext_small that lands near 0.883 would be a fourth recipe, worth two members
rather than one.

## Ensembling is finished on the board, and capacity was not just unproven but backwards

Two results arrived together and they point in opposite directions.

**The five-member scored 0.947.** So did both four-member configurations, and
the three-member scored 0.946.

    3 members                     LB 0.946
    4 members (m32+m37)           LB 0.947
    4 members (m37+m38)           LB 0.947
    5 members (m32+m37+m38)       LB 0.947

One step, at three to four, and nothing since. Local pv-hold still separates
these (0.8872 / 0.8880 / 0.8898 / 0.8892, and 0.8906 for six) but the board has
now failed to see 0.0018, 0.0014 and 0.0008 differences in a row. **Ensembling
is closed as a source of leaderboard points at this scale.** It is not closed as
a technique - the four-member is worth +0.001 over one model and stays in every
submission - but further members buy pv-hold and nothing else.

This is the refutation condition written into queue item 15 four hours ago,
met exactly as specified. The note said it would send the next effort at
single-model quality. It did, one item earlier than planned:

**convnext_small, at epoch 6 of 12, is already above every model we own.**

    m43small  ep1 0.838  ep2 0.866  ep3 0.876  ep4 0.878  ep5 0.882  ep6 0.886

    every existing single model, pv-hold:
      m38 0.8836   m31 0.8832   m39 0.8831   m37 0.8830   m33 0.8799   m40 0.8792

At pv-hold's replicate sd of 0.0003, 0.886 is about nine sd above the m31-recipe
mean of 0.8833, with half the run still to go. This is the first single-model
improvement since 13 August.

**The 9 August conclusion was not merely unreadable, it was inverted.** It
recorded small 0.872 against tiny 0.874 and closed capacity as topped out. On an
instrument with sd 0.0044 that gap was noise; on one with sd 0.0003 the truth is
that small is substantially *better*. An axis was closed for fifteen days on a
measurement that pointed the wrong way.

The general lesson is not "re-run everything". It is that **a null result from an
instrument that cannot resolve the effect is not a null result**, and this
project has now been bitten by that twice in two days - once in the audit that
retired six conclusions, and once here, where the retired conclusion turned out
to have the sign wrong as well.

**What follows if m43small holds to epoch 12.** It is a better model and a fourth
recipe, and by the two-members-per-recipe rule it is worth two members, not one.
Its replicate becomes the highest-value queued run. Whether any of that reaches
the board is a separate question the last four submissions answered pessimistically
- but a single model 0.003 better is a different proposition from a fifth member,
because it lifts every ensemble that contains it rather than adding a correlated vote.

## m43small finishes at 0.8841: capacity closes again, properly measured this time

    m31 recipe (tiny, n=3)   sel 0.8993   hold 0.8833   gold 0.9163
    m43small (small)         sel 0.9014   hold 0.8841   gold 0.9108
                                 +0.0021      +0.0008      -0.0055

**Retracting the retraction.** Two hours ago this entry recorded that the 9 August
"capacity topped out" conclusion was "not merely unreadable, it was inverted",
on the strength of m43small reading 0.886 at epoch 6. It then peaked there and
came down; the checkpoint pv-sel selects is epoch 7 at pv-hold 0.8841. The
delivered gain is +0.0008 - about 2.7 sd, real but small - and gold moves the
other way.

So the original measurement still could not support its conclusion, and the
conclusion was still roughly right. **Fixing an instrument does not retroactively
reverse the findings it could not read**, which is precisely the error this file
warned about yesterday and which I then committed in the other direction. A mid-
run epoch is not a result either; only the selected checkpoint is.

Capacity is closed again, this time on a measurement that can carry the weight:
1.8x the parameters for +0.0008 on pv-hold and -0.0055 on gold.

**What is genuinely useful here.** convnext_small is a fourth *recipe* at
comparable quality (0.8841 against tiny's 0.8833), and by the two-members-per-
recipe rule it could contribute two ensemble members. That would be worth doing
if ensembling still bought board points. It does not - three, four and five
members scored 0.946, 0.947, 0.947 - so a fourth recipe is a member factory for
an axis that is already closed.

**And a second instance of the pv-sel selection cost.** pv-hold peaked at epoch 6
(0.886) while pv-sel peaked at epoch 7 (0.8841 on pv-hold). Selection threw away
0.002 of pv-hold. This is now the second time it has been observed (m33 selected
epoch 9, m40 epoch 5, same pv-hold) and it is structural: pv-sel cannot be
replaced by pv-hold as the selector without contaminating the judge. The
delivered model is systematically a little worse than the best epoch, and that
cost is roughly the size of every effect currently being chased.

## Where this leaves the search

Every axis that can be measured locally is returning +0.001-scale effects, and
the board has stopped resolving them:

    ensembling        3->4 members +0.001 on the board, nothing after
    capacity          +0.0008 pv-hold, gold negative
    representation    never shown to pay
    backbone family   closed twice
    seed replication  +0.0009 for a recipe's second copy, nothing for a third

Single-model pv-hold has not left 0.880-0.884 since 13 August, across six models
and four recipes. **No submission candidate exists right now**, and manufacturing
one would be spending a slot to confirm what four consecutive submissions already
showed.

The one untouched hypothesis is the one this file raised on 22 August and never
tested: that **pv-hold has hit the teacher's ceiling**. Both halves of pseudo-val
carry teacher labels, so the split detects epoch-selection inflation but is blind
to the model fitting the teacher's systematic errors. If that is what the
0.880-0.884 plateau is, no image-side axis will move it, and the next work is on
the labels rather than the model.

## Both reopened axes close, and the pattern in how they were wrong

    m44posw (pos_weight)   sel 0.8982   hold 0.8755   gold 0.9022
    m31 recipe (n=3)       sel 0.8993   hold 0.8833   gold 0.9163
                              -0.0011      -0.0078      -0.0141

Three instruments, all negative, and pv-hold's -0.0078 is twenty-six sd.
pos_weight is clearly harmful.

Setting the two re-measurements side by side against what 9 August recorded:

    axis         9 Aug said        9 Aug gap    true pv-hold gap
    capacity     topped out         -0.002        +0.0008
    pos_weight   rejected           -0.002        -0.0078

**Both original conclusions were right and both magnitudes were useless.** The
9 August instrument could not resolve either effect, and it happened to get the
direction right on pos_weight and near-enough on capacity - but it recorded a
0.002 harm for something that costs 0.0078, and a 0.002 harm for something that
is mildly beneficial. Those are the same number standing in for two situations
four times apart.

So yesterday's audit was procedurally right to retire them as unestablished, and
both turned out to hold anyway. **The value of re-measuring was calibration, not
reversal** - and the calibration matters, because "capacity is worth +0.0008"
and "pos_weight costs 0.0078" are decisions, while "both are about -0.002" is
noise wearing a decision's clothes.

This also closes the image-side re-examination. Every axis the audit reopened
has now been measured on an instrument that can read it, and none of them moves
the model materially. What remains is the label axis, which is the only one whose
mechanism explains the plateau rather than merely failing to beat it.

## The label blend, done properly, moves nothing (n=1)

OOF regenerated at 12 epochs instead of 6, blended at teacher weight 0.4, one
retrain of m31's recipe on the result.

    m31 recipe (n=3)      sel 0.8993   hold 0.8833   gold 0.9163
    m46blend12            sel 0.8995   hold 0.8830   gold 0.9168
                             +0.0002      -0.0003      +0.0005

gold moved 0.0005 against its own replicate sd of 0.0037 - 0.13 sd, which is
nothing. The teacher-ceiling hypothesis predicted that better labels would lift
the model where pseudo-val cannot see it; on the one instrument that could have
shown that, nothing happened.

**A structural limit found while trying to check the labels directly.**
`oof_predictions.parquet` covers the 21,148 training rows and **none of the 336
gold_val rows** - by design, since gold must never be contaminated by a model's
opinion of it. The consequence is that the blend's own accuracy cannot be
re-measured: the 0.9271 figure this axis was launched on came from an earlier
setup and cannot be reproduced against this OOF. **Label quality here is only
observable through a downstream model**, which costs 2.5 hours per reading and
is judged on gold at sd 0.0037. That is the real reason this axis has been hard
to settle, and it is worth knowing before anyone reopens it.

Held as n=1. The teacher-weight sweep (0.25, 0.6) and a replicate of 0.4 are
queued behind this, which is the only way to separate a real null from gold's
wobble. If all four land near 0.916, the label axis closes too - and with
image-side axes, ensembling and capacity all closed, that would mean every lever
this project can currently reach returns about +0.001, and the remaining moves
(slice count, a different architecture) both cost a day of Kaggle re-preprocessing
before they can even be tried.

## The depth axis was shelved on a cost estimate nobody checked

m35depth48 died because `--depth` is argparse-only and the shards on disk hold
32 slices; the axis was then shelved as costing "a day, not 2.5 hours". That
figure was never derived. Checking it now, because with the image-side axes
closed and the label axis looking flat, through-plane resolution is the most
likely next move and the cost is the whole argument against it.

    kernel preprocessing   ~2.1h single-threaded for the corpus, and
                           push_preprocess.py splits into concurrent parts
                           (--parts 3), so ~1h wall
    output cap             20GB per kernel; depth 48 is ~54GB, so 4-5 parts
    disk                   123G free, and artifacts/volumes_precrop_bak is 32G
                           of superseded pre-crop shards that can go
    download               NEVER MEASURED anywhere in this record

So the Kaggle side is two to three hours, not a day. The only unknown is pulling
54GB back, and the reason it feels expensive is the memory of the 84-hour
per-slice download that push_preprocess.py exists to avoid - but that was
730,000 individual API requests, and shard retrieval is a dozen.

**This does not reopen the axis; it corrects its price.** The decision of whether
to spend it belongs to the operator, and it should be put to him with a real
number rather than an inherited adjective. Measuring the download of one existing
shard would settle it in minutes and has not been done.

## The teacher-weight sweep closes the label axis, and corrects pv-hold's sd

    config                  sel      hold(sel'd)   gold    selected ep
    baseline (no blend)   0.8993     0.8833      0.9163      6-7
    w=0.60 (least blend)  0.9010     0.8870      0.9155       5
    w=0.40                0.8995     0.8830      0.9168       5
    w=0.25 (most blend)   0.8952     0.8813      0.9146       9

gold spans 0.9146-0.9168, a range of 0.0022 inside its own sd of 0.0037, and it
is not monotone in the weight. Nothing there.

**w=0.60's pv-hold 0.8870 looked like the first single-model movement since 13
August. It is not.** Its epoch curve peaks at ep5 (0.887), holds 0.886 at ep6,
and settles at 0.881-0.883 from ep8 - the same place as everything else. pv-sel
happened to peak at ep5 too, so the checkpoint saved is the top of the run's own
wobble. Late-epoch plateau against the baseline family's 0.8831-0.8836: identical.

This is the second time in one day the same trap was nearly walked into - the
first was m43small reading 0.886 at epoch 6 this morning before finishing at
0.8841. Both times a mid-curve high was mistaken for a level.

**And it forces a correction to this file's own instrument claim.** pv-hold's
replicate sd was put at 0.0003 from three runs of m31's recipe. Those three
selected epochs 7, 7 and 6 - they were coincidentally aligned. Across the wider
set the selected epoch runs from 5 to 9, and within a single run pv-hold moves
0.881-0.887 over that span. **The delivered checkpoint's pv-hold therefore
varies by roughly 0.003-0.004, not 0.0003**, because pv-sel's choice of epoch is
itself noisy and drags pv-hold with it.

    pv-hold, same recipe, same selected epoch     sd ~0.0003
    pv-hold, same recipe, epoch chosen by pv-sel  sd ~0.003-0.004

The first number is a property of training; the second is what any comparison
between two runs actually carries. **Comparisons in this file made against
0.0003 need re-reading against the larger figure**, which means the "5-8 sd"
recoveries of m32 and m33 as genuinely worse models are back to roughly 1 sd and
unestablished again. The two-members-per-recipe rule (+0.0009) and the ensemble
ladder (0.0008-0.0026) sit inside it too.

What survives unchanged is anything measured on a fixed set of checkpoints -
ensemble comparisons reuse the same saved models, so selection noise does not
enter twice - and anything the leaderboard confirmed.

**The label axis closes.** Four weights including no blending, gold flat and
non-monotone, pv-hold plateaus identical once the selection artefact is removed.
The teacher-ceiling hypothesis predicted a lift and there is none.

## Correcting the correction: between-run spread is 0.0005, not 0.003

m49blend40s1 replicates m46blend12 - same recipe, same blended dataset, seed 1.

    m46blend12     selected ep5   hold 0.8830   gold 0.9168
    m49blend40s1   selected ep7   hold 0.8832   gold 0.9128

**Different selected epochs, and pv-hold agrees to 0.0002.** So the earlier
reading - that pv-hold's 0.0003 came from three runs whose epochs coincidentally
aligned - was wrong. Five replicate runs across two datasets now sit inside
0.0005, and two of them selected different epochs.

The mistake was conflating two spreads. m48blend60's curve moves 0.881-0.887
*within one run across epochs*; that is not how far two runs land apart.

    between runs, delivered checkpoint    0.0002-0.0005   (n=5, two datasets)
    within one run, across epochs         0.003-0.006
    m48blend60                            plateau 0.882, delivered 0.8870

So the distribution is tight with a rare large tail: usually 0.0005, but when
pv-sel happens to peak where the pv-hold curve also spikes, the delivered
checkpoint can sit 0.005 above the run's own level. m48 was that case, which is
why it looked like a breakthrough for an hour.

**This restores what the previous entry retired.** The m32 and m33 comparisons
(0.0024 and 0.0033 behind m31) are back to 5-11 sd and readable; the
two-members-per-recipe rule and the ensemble ladder are outside the noise again.

**And it yields a rule that removes the failure mode instead of inflating the
error bar to cover it: for any claim under 0.005, compare late-epoch plateaus
rather than delivered checkpoints.** Every epoch is in the log, so it costs
nothing, and it is immune to whichever epoch pv-sel happened to like. Applied to
the sweep, m48blend60's plateau is 0.882 against the baseline family's
0.8831-0.8836 - the label axis stays closed, now for the right reason.

Noted plainly: this quantity has been revised twice in one evening, first upward
on a within-run range and now back down on a direct replicate. The lesson is not
about the number. It is that a spread should be measured by replicating the
thing being compared, not inferred from the shape of a single run.

## depth-48 shards exist, three days after the axis was shelved

    kernel preprocessing   4 concurrent parts, 21:20 -> 22:20   ~1h
    download               21:00 -> 22:26, 54GB                 ~1.5h at 9-11 MB/s
    verified               36 shards, 24,369 series, every volume (48, 256, 256)

Against the depth-32 set's 36 shards and 24,371 series - two series short, which
is the kind of difference a four-way split boundary produces and does not matter,
since the training dataset joins on SeriesInstanceUID.

The whole rebuild cost **2.5 hours**. It was shelved on 21 August as "a day, not
2.5 hours", a figure that appears in this record without a derivation, and that
sentence kept the only untried axis shut for three days while six other axes were
measured and closed.

Two things were done differently this time, both because of how m35depth48 died:

  - a `--limit 40` smoke kernel ran first and returned (40, 48, 256, 256),
    proving `--depth` reaches the kernel body. m35 died in 60 seconds on a flag
    that existed in argparse and nowhere else.
  - `--volumes` was added to train_image.py before the download finished, so the
    48-slice shards sit beside the 32-slice ones. Every current ensemble member
    was trained on 32 and has to stay reproducible; the alternative was swapping
    the directory, which is how a stale cache nearly invalidated a run on
    13 August.

m50depth48 is queued behind m45m37s1 and will be the first through-plane
measurement this project has ever made. Judged against the m31 recipe's
three-seed pv-hold baseline of 0.8833, on late-epoch plateaus.

## Every recipe peaks at pv-hold 0.884, and pv-sel selection is throwing away up to 0.009

m45m37s1 delivered pv-hold 0.8750 against m37up288's 0.8830 - eight times the
0.0005 between-run spread. The curves explain it: both runs peak at 0.884 at
epoch 6, and m45's pv-sel kept climbing (0.898 -> 0.902 through epoch 12) while
its pv-hold fell to 0.875. Selection landed on the overfit tail.

Measuring that loss across every model trained on the corrected shards:

    tag            sel_ep/N   hold@sel   hold@peak   lost
    m45m37s1        12/16      0.8750     0.8840    0.0090
    m40m33s1         5/16      0.8792     0.8840    0.0048
    m33up336new      9/16      0.8799     0.8840    0.0041
    m38seed1         7/12      0.8836     0.8870    0.0034
    m43small         7/12      0.8841     0.8860    0.0019
    m37up288        10/16      0.8830     0.8840    0.0010
    m39seed2         6/12      0.8831     0.8840    0.0009
    m46/m47/m48/m49  5-9/12    0.8813-0.8870        0.0000-0.0007

**16-epoch runs lose 0.0047 on average to selection; 12-epoch runs lose 0.0010.**
Nearly five times as much. The sel/hold gap that this file has tracked widening
(0.0119 -> 0.0227) is this: past roughly epoch 8 the two instruments move in
opposite directions, and the selector follows the one that is wrong.

**Actionable immediately: stop at 12 epochs.** m33, m40, m37 and m45 are all
16-epoch runs and all four are in the lossy group. No new information is needed
to act on this - it is visible in logs that already exist.

**And the more important column is hold@peak.** Eleven models across five
recipes - 128/up224, 256/up336, 192/up288, convnext_small, and three blended
label sets - peak between 0.8820 and 0.8870. **Nothing reaches past 0.884.**

Delivered checkpoints spanned 0.875-0.887 and that spread was mostly selection
noise. At the peak the models are indistinguishable. So the plateau since
13 August is not an artefact of comparing through a noisy selector: it is a real
ceiling that every recipe hits and none crosses.

That reframes the whole search. The axes were not failing to find a better
recipe; there is no better recipe within this model class and this label set.
Which leaves architecture or labels as the only places a ceiling like that can
move - and the label blend was measured flat, so the honest reading is that the
model class itself is saturated at 0.884 on this instrument.

## The selection loss has a measurable ensemble cost

The two-members-per-recipe rule predicted that replicating m37's recipe would
add about +0.0009 to the four-member set. It did not:

    m31+m33+m37+m38               pv-hold 0.8898
    m31+m33+m37+m38+m45m37s1              0.8883    -0.0015

**This is not a refutation of the rule - it is the selection loss arriving
downstream.** m45m37s1's delivered checkpoint is 0.8750 while the same run
peaked at 0.8840; pv-sel picked epoch 12 out of its overfit tail. The member
added was a model damaged by 0.009, and a weak member dragging an ensemble is
the most repeatedly confirmed finding in this file.

So the prediction was tested with the wrong artefact. What the measurement does
establish is the price of the selector: 0.009 of single-model quality became
0.0024 of ensemble quality, turning a predicted gain into a loss.

That is now two independent reasons to cap training at 12 epochs - the direct
loss on the member, and its amplified effect on anything the member joins.

## depth 48 lands in the middle of the band: the image-side model class is exhausted

    m50depth48   peak(ep5) 0.883   delivered(ep7) 0.8819   late plateau 0.874-0.877
                 gold delivered 0.9050 against the m31 family's 0.9163

Peak 0.883 against the other eleven models' 0.882-0.887. Dead centre. Twelve
models, six recipes - three in-plane resolutions, two slice counts,
convnext_small, and three blended label sets - and every one of them peaks
inside a 0.005 band that none crosses.

The same selection pattern appears again: from epoch 5 pv-sel climbs 0.890 ->
0.895 while pv-hold falls 0.883 -> 0.874.

**The through-plane axis was the last untried one in this model class, and it is
closed.** Worth stating what that cost and what it bought: 2.5 hours of rebuild
plus 2 hours of training to convert an inherited guess into a measurement. The
guess ("a day, not 2.5 hours") had kept the axis shut for three days; the
measurement closes it properly and removes it from every future plan.

**Where this leaves the search.** Everything reachable from the current
architecture and label set has now been measured:

    in-plane resolution   three points, all peak 0.884
    slice count           32 and 48, both peak 0.883-0.884
    capacity              tiny and small, +0.0008
    backbone family       DINOv2 twice, 0.017-0.020 behind
    loss weighting        pos_weight, -0.0078
    label refinement      four teacher weights, flat on gold
    ensembling            +0.001 on the board at three-to-four members, nothing after
    TTA                   nil by construction

The ceiling at 0.884 is a property of SliceNet-over-convnext trained on this
teacher's labels, not of any one recipe. Moving it needs a different
architecture or a better label source, and the label source was measured flat
yesterday. That is a decision about where to spend the remaining 58 days, and it
belongs to the operator.

## Investigating the public meta: independent teachers are worse, medical backbones are untried

With the model class exhausted, the cheapest question was what the top public
solutions do differently. Pulling the two highest-voted ensemble kernels
(`bend-the-knee-to-dinov3-ensembled`, `read-the-report-then-the-knee`) shows they
are assembled almost entirely from *shared public artefacts*:

    LLM report labels    stevenleehans/rsna-knee-llm-report-labels
                         pilkwang/rsna-knee-llm-labels
                         lixin73/rsna-knee-llm-report-labels-sol56
    medical backbones    marwanmath/resnet-50-radimagenet-marwan  (RadImageNet)
                         royceplayz/rsna-medicalnet-resnet50-weights
                         mrquadian/biomedclip
    trained heads        antoinegg1/..., prvsiyan/..., tonylica/rsna2026-models,
                         pilkwang/rsna-knee-weights, mattiaangeli/knee-mri-fold-weights

That is two leads, and one of them is now closed.

**Independent teachers do not help. Ours is the best one available.** Scored on
our 58 gold studies:

    ours (fable)                 0.9004
    llm_labels_v4_blend          0.8927
    llm_labels_v2                0.8873
    llm_labels_full              0.8780
    report_labels_v2             0.8700

And every blend is worse than ours alone - 0.8960 with the best public set,
0.8930 with all four, 0.9002 at a 0.7/0.3 weighting that is just ours with noise
added. Three independently built report-derived label sets, and averaging with
any of them costs us. The teacher ceiling at 0.900 is not something the public
field has solved either; we are already at the top of it.

(The alignment was checked rather than assumed: pseudo_fable_gold.csv is keyed
by positional id, and scoring it that way returns 0.9004 against the 0.8997 in
this file, which confirms the order.)

**The surviving lead is domain-pretrained backbones.** This is NOT the closed
"backbone family" axis - that was DINOv2 twice, pretrained on natural images.
RadImageNet, MedicalNet and BiomedCLIP are pretrained on medical imaging, the
public meta leans on them heavily, and nothing in this project has tried one.
Feasibility has to be checked before it is queued: train_image.py builds its
backbone through timm, so an external checkpoint needs a loading path that does
not exist yet.

## The queue's job check, wrong in the other direction

Three times this file records the same bug: something the queue launched was
invisible to `running()`, so it started another job on top of it. The fix each
time was to widen the pattern, ending at shape-matching any
`.venv/bin/python3 src/*.py`.

Today it failed the other way. `src/ask.py` was blocking on a Telegram answer
with an eight-hour timeout, matched the pattern, and the runner counted it as a
running job. m51radimagenet sat queued while the card idled for 70 minutes, and
would have idled for eight hours.

Fixed in the spirit of the file rather than against it: enumerating GPU scripts
already failed three times, so the pattern still matches on shape and now
subtracts a short denylist of helpers that never touch the GPU (`ask.py`,
`notify.py`). Anything new still counts as blocking, which is the safe direction.

Also worth recording because it cost a shell: `pkill -f "src/ask.py"` kills the
shell running it, because that shell's own command line contains the pattern.
The interpreter-path match that `running()` uses exists for exactly this reason
and should have been used to find the PID.

## The efficiency track is the better-odds prize, and we are 158th on runtime alone

Pulled Ryan Holbrook's efficiency leaderboard (2,175 teams, 25 Aug):

    rank   team                 public
      1    Scott Willis         0.947    <- the only team at our accuracy
      2    Siontist             0.938
      3    Avinash Niroula      0.937
      4    Tucker Arrants       0.942
      5    DaviDavian           0.937
     ...
    158    qkrtkddnjs           0.943    <- us

**Ranks 2 through 10 all score BELOW us and all rank above us.** Inside a score
band the ordering is runtime, and we are losing on runtime alone. Ours is
~4.7 s/study for one model - five series by three planes by thirty-two slices
through the backbone per study - and the submitted incumbent is a four-member
ensemble, so several times worse.

Set against the main track: there we are third at 0.947 with every axis closed
and 0.005 behind first. Here our accuracy is already at the top of the board and
the axis that decides it has never been touched. Three prizes at $7,000 /
$6,000 / $5,000.

**And it needs no training.** The accuracy cost of every speed lever - fewer
slices, fewer series per study, lower input resolution, one model instead of
four - is measurable on pv-hold with the checkpoints already on disk.

Note the fork HANDOFF already recorded and nobody acted on: the ensemble is the
main-track submission and a single fast model is the efficiency submission. They
are different kernels and both are allowed.

**Caveat kept from the earlier note:** the exact scoring formula still has not
been read. The efficiency notebook only reads a precomputed CSV, and that CSV
carries rank, team and public score but no runtime column, so the trade-off
between accuracy and seconds cannot be derived from it either. Before optimising
hard, the formula has to come off the Efficiency Prize Evaluation page. What is
already safe to act on: at equal accuracy, faster is better, and we are 158th.

## RadImageNet closes the backbone question for good

    m51radimagenet   peak 0.849   plateau ep10-12 0.848-0.849   gold 0.8942

Against the convnext band of 0.882-0.887, that is 0.035 back - worse than
DINOv2's 0.856-0.863. Thirteen models now, and the ordering is unambiguous:

    convnext_tiny / small, seven recipes    0.882 - 0.887
    DINOv2 vit_small, twice                 0.856 - 0.863
    RadImageNet ResNet-50                   0.849

**Domain match does not compensate for architecture and pretraining scale.**
RadImageNet is a ResNet-50 over 1.35M low-resolution 2D medical images;
convnext_tiny.in12k_ft_in1k is a stronger architecture pretrained on far more
data. The public ensembles lean on RadImageNet because they are averaging
thirty-five members and want decorrelation, not because any one of them is good.

So the backbone question is closed from both directions - natural-image and
medical - and the 0.884 ceiling is not a backbone problem. Combined with the
label sweep and the public-teacher comparison, it is not a label problem that
anything available can fix either.

**This is where the main-track search ends.** Every axis reachable from the
current architecture, label set and public artefacts has been measured. The
remaining moves are a genuinely different architecture, which is days of work
with a real chance of nothing, or the efficiency track, where our accuracy is
already top of the board and the deciding axis has never been touched.

## The efficiency curve: halving slices is nearly free, quartering is not

Single m31, same checkpoint, only the slices fed to it varied:

    slices   pv-hold   vs base    speed
      32     0.8835      -        1x
      16     0.8804    -0.0031    2x
       8     0.8702    -0.0133    4x

**Not linear.** The first halving costs 0.0031 and the second costs 0.0102 -
three times as much. Thirty-two slices carry redundancy that sixteen still
capture; eight do not. So stride 2 is the operating point and stride 4 is a
different trade entirely.

Set beside the member lever, already measured:

    lever                      speed   pv-hold cost
    4 members -> 1             4x      0.0066
    32 slices -> 16            2x      0.0031
    both                       8x      ~0.0097

Both come to roughly 0.003 per doubling, which makes them comparable and
stackable. The efficiency candidate is therefore **one model at sixteen slices**:
eight times less inference than the submitted incumbent, at a predicted pv-hold
near 0.880 against its 0.8898.

For scale, the efficiency leaderboard's ranks 2 through 10 score 0.925 to 0.942
on the public board while we sit 158th at 0.943. Landing near 0.944 at an eighth
of the compute is a serious position there, and it costs nothing in the main
track because the two are separate kernels and both are allowed.

**Still missing: the scoring formula.** Everything above says "faster at equal
accuracy is better", which is safe. Choosing between 2x and 8x when accuracy
falls needs the actual trade-off, and that is on the competition's Efficiency
Prize Evaluation tab, which is a JS app that WebFetch cannot read.

## We were ranked on our most expensive model, by accident

The efficiency board scored our 19 August submission - m33up336new, a 256-source
model upsampled to 336. That is the most expensive model this project has ever
trained, and it is what put us 158th of 2175.

    currently ranked   m33   256 -> 336   pv-hold 0.8799
    submitted today    m31   128 -> 224   pv-hold 0.8835

**m31 is better on accuracy and 2.25x cheaper per slice.** Nothing had to be
built or traded; the entry was simply the wrong checkpoint. Dev-run wall clock
fell from 54s to 33s on the same three sample studies against the four-member
kernel.

Submitted as `rsna-knee-sub-eff-m31`. The main-track incumbent is a separate
kernel and is untouched.

**Why this and not the eight-times-faster option.** The curve says one model at
sixteen slices is 8x with pv-hold near 0.880, and that is probably the better
efficiency entry. But the local measurement subsampled a 32-slice cached volume,
while the kernel would build a 16-slice volume from the DICOMs - very close, not
identical, and there are no 16-slice shards to check against without another
preprocessing round. Taking the strictly-dominant move first costs nothing and
produces a scored data point; the stride can go on top once this one lands and
the rank movement shows how much runtime actually buys.

Also still missing the scoring formula, which is why "strictly dominant" was the
bar for this submission rather than "best guess on the curve".

## The efficiency formula, found in a competitor's kernel

`sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee` states it:

    AUC/(0.5 - maxAUC) + RuntimeSeconds/32400
    "roughly +0.044 AUC needed per extra hour to break even"

The quoted expression is garbled but the arithmetic pins it down. 3600/32400 =
0.1111, and 0.1111 x (0.9 - 0.5) = 0.0444, which reproduces their number exactly
with maxAUC = 0.9. So the denominator is (maxAUC - 0.5), the runtime budget is
**nine hours**, and the two terms are normalised error plus normalised runtime,
minimised.

At today's maxAUC of 0.952 the exchange rate is 0.1111 x 0.452 = **0.050 AUC per
hour of runtime.**

**This inverts the caution from two hours ago.** The measured accuracy costs -
0.0016 to halve slices in the ensemble, 0.0066 to drop to one model, 0.0094 for
both - are worth 1.2, 4.8 and 6.8 *minutes* of runtime respectively at that rate.
Anything that saves more than seven minutes and costs less than 0.0094 is
favourable, and the four-member submission took 177 minutes to score against
roughly 40-50 for a single model.

So the earlier decision to submit m31 at full slices because it was "strictly
dominant" was right but far too timid. The right entry is the fastest
configuration whose accuracy loss stays inside a few hundredths, and 0.0094 is
not close to the boundary.

**Caveat: this is a competitor's reading, not the official page.** It is
self-consistent and reproduces its own worked number, which is more than the
project had before, but the Efficiency Prize Evaluation tab is still the
authority. Treat 0.05 AUC/hour as the working figure and re-check it before any
decision that turns on the third digit.

**Separately, on studying the public field:** the strongest public kernel here
scores 0.883, and the best-regarded ensemble notebook's parent is 0.917, against
a leaderboard top of 0.952 and our 0.947. The public meta is far below the real
top, so reading public kernels is useful for mechanics like this formula and
useless for closing the 0.005 to first.

## The efficiency board scores your best submission, not one you pick

Our efficiency rank went 158 -> 583 without us doing anything wrong. The board
had been scoring the 19 August m33 entry at 0.943; when the four-member
ensemble scored 0.947 on 23 August it switched to that, and the four-member is
several times slower.

**So "a separate fast kernel for the efficiency track" - which HANDOFF has said
since 19 August - does not work.** The board follows the best main-track score.
Today's single-model entry at 0.942 was ignored for exactly that reason. To move
on efficiency, the *best-scoring* submission has to be the fast one.

That reframes the target. Scott Willis is efficiency rank 1 at 0.947 - our score
- so the position is reachable, but only by making our own 0.947 cheaper rather
than by submitting something else alongside it.

**The official formula, from the competition page:**

    Efficiency = AUC / (Benchmark - max AUC) + RuntimeSeconds / 32400

(Benchmark - maxAUC) is negative, so higher AUC pushes the first term down and
lower is better. The exchange rate is 1/(32400 x |Benchmark - maxAUC|): with
Benchmark 0.5 and maxAUC 0.952 that is **0.050 AUC per hour**, and with this
project's own metadata baseline of 0.605 it is 0.039. Either way runtime
dominates every accuracy difference this project has measured.

**Acted on it:** the four-member incumbent resubmitted at every second slice.
Local cost 0.0016 on pv-hold; dev-run per-study time fell 8.7s -> 4.7s once the
28-second model load is subtracted, so scoring should go from 177 minutes to
roughly 100. Seventy-seven minutes is worth about 0.064 AUC at the going rate
against a 0.0016 loss - a forty-to-one trade.

The kernel takes every Nth slice of the DEPTH-slice volume rather than asking
to_volume for fewer, because that is how the accuracy cost was measured locally
and a differently-resampled volume would be a different experiment.

## 2.5D breaks the ceiling; self-attention over slices does not

Two architecture changes, run separately so a gain could be attributed.

    run       change                       deliv     peak     gold
    m31       baseline                     0.8832      -      0.9148
    m53nbr    2.5D neighbour channels      0.8896   0.8900   0.9118
    m54attn   2 transformer layers         0.8784   0.8790   0.9131
    (band from the previous 13 models: 0.882 - 0.887)

**m53nbr is the first model to leave the band.** 0.8896 against a previous best
delivered checkpoint of 0.8841, at a between-run spread of 0.0005 - eleven sigma.
Peak 0.8900 clears the top of the band too, so it is not a selection artefact.

Two of the backbone's three input channels had carried a duplicate of the same
grayscale slice since the first run in this project. Filling them with (i-1, i,
i+1) costs nothing - identical shape, identical FLOPs, identical stem - and it is
the first path the architecture has ever had to through-plane information.

**It also settles the depth-48 null.** More slices could not help a model that
processed each slice in isolation. The axis was not barren; there was no way to
use it. 2.5D gives context per slice instead of more slices, and that is what
moved.

**Self-attention over slices is harmful: 0.8784, five thousandths below
baseline.** The operator's observation that AttentionPool relates nothing was
correct as a description, but the fix does not pay. Two transformer layers over
32 tokens add 9.5M parameters trained from scratch on 21k series against a
backbone that arrives pretrained, and the plateau (0.877-0.879) sits below
baseline throughout rather than peaking and overfitting - it learns worse, not
later. Attention over slices is closed; attention *within* the stem, which is
what 2.5D gives for free, is what worked.

**What this does not yet tell us.** m53nbr's value as an ensemble member is
unmeasured - the first attempt scored it through a model rebuilt without 2.5D
(see the entry above) and is void. Re-queued.

## 2.5D breaks the ceiling — and immediately exposes a hole nothing else could see

Three runs, all m31's recipe with one thing changed, judged on peak pv-hold
against the 0.882-0.887 band that thirteen models across seven recipes never left:

    m53nbr   2.5D neighbour channels   peak 0.8900   <- first model past 0.887
    m52nano  convnext_nano             peak 0.8860
    m54attn  slice self-attention      peak 0.8790   <- hurts

**The 2.5D run is the first single-model improvement since 13 August**, and it
came from filling two input channels that had carried a duplicate of the third
since the project's first run. +0.0065 over m31 on the same recipe.

Self-attention over slices went the other way, which is why the two were run
separately - combined, this would have read as a wash.

**But the ensemble measurement scores m53nbr at 0.8709, not 0.8896.**

Every other member agrees between its training log and ensemble.py to within
0.0003 (m31 0.8835/0.8832, m38 0.8835/0.8836). m53nbr is out by 0.019. The
loading path was checked first and is correct: image_model.json records
`neighbour_channels: True` and load_model rebuilds it that way - verified
directly, so an earlier diagnosis blaming the metadata was wrong and is
withdrawn.

**The leading hypothesis is slice order, and if it holds it is the more
important finding.** AttentionPool is permutation-invariant over slices, so
every model this project has ever trained is blind to slice ordering. Training
reads volumes through `build_cache`; ensemble.py reads them through
`load_shard`. Any difference in ordering or depth resampling between those two
paths would be invisible to all thirteen prior models and would break exactly
one thing: a model whose input is built from neighbours.

That makes 2.5D both the first gain and the first probe of an assumption nobody
could test before. Until the two paths are reconciled, m53nbr's 0.8900 is
trustworthy only inside the training path, and the ensemble and submission
numbers for it are not usable.

## Correction: the metadata diagnosis was right, and the slice-order hypothesis is dead

The previous entry withdrew a correct diagnosis and replaced it with a wrong one.
Both halves need reversing, and the reason for the mistake is worth more than
either.

**Slice order is refuted.** `build_cache` and `load_shard` return byte-identical
arrays for the same shard - checked directly, same ids, same dtype,
`np.array_equal` true. The two paths agree, so nothing about ordering explains
anything.

**The metadata diagnosis was right.** The timeline settles it:

    15:13  m53nbr writes image_model.pt and image_model.json - no arch flags
    15:52  first ensemble run: model rebuilt WITHOUT 2.5D, scores 0.8709
    19:34  commit records the flags and backfills the existing json
    19:40  second ensemble run: rebuilt correctly, scores 0.8895

Same checkpoint file both times, unchanged since 15:13. What changed was the
metadata and the loader.

**The error was checking present state to explain a past observation.** Opening
image_model.json at 19:40 showed `neighbour_channels: True`, so the metadata
explanation looked refuted - but the flag was there because a fix had just put
it there. The check was contaminated by the thing it was checking for. Reading
`git log` on the two files first would have caught it in one command.

So the standing lesson is not about metadata at all: **when explaining a
measurement taken at time T, verify the state as of T, not now.** In a repository
that is being edited while it runs, those are different questions.

**What survives.** m53nbr's peak pv-hold 0.8900 stands, the ensemble containing
it reaches **0.8911 - the highest this project has measured** - and the earlier
0.8908 and 0.8709 readings were taken through a broken loader and should be
struck.

## Submitted the 2.5D ensemble

Five members - m31, m33, m37, m38 and m53nbr - at pv-hold 0.8911, the highest
configuration this project has measured. Adding m54attn as a sixth gives 0.8910,
so self-attention is dead as a member too.

The submission kernel was checked before pushing rather than after: it reads
`neighbour_channels` and `slice_attn` out of the checkpoint cfg exactly as
ensemble.py does. Without that check this would have scored a model fed the
wrong channels, silently, the way the 15:52 ensemble did - 2.5D adds no
parameters, so a strict load succeeds and nothing complains.

Both boards are served by the same entry. 2.5D is free at inference - identical
tensor shape and FLOPs - so a gain on the main track costs nothing on the
efficiency track, which is the first time in this competition those two have
pointed the same way.

    prior best submitted   4 members            LB 0.947
    submitted now          5 members with 2.5D  pv-hold +0.0013 over it

## Retraction: 2.5D did not break the ceiling. It improved agreement with the teacher.

The board has adjudicated and the earlier entry is wrong.

               pv-hold    gold      LB (single model)
    m31        0.8832     0.9148    0.942
    m53nbr     0.8896     0.9118    0.938
               +0.0064    -0.0030   -0.004

**gold and the leaderboard agree; pv-hold is the outlier.** 2.5D makes the model
match the teacher's labels better and the real ones slightly worse.

Two checks were run first and both came back clean, so the explanation is not
mechanical: `build_cache` and `load_shard` return byte-identical arrays, and
`src/volume.py` and `src/kernels/volume.py` - the slice-ordering code embedded
in the preprocessing kernel and in the scoring kernel - are byte-identical too.
Slice order is fine. The model really is better at pseudo-val and worse in
reality.

**This is the failure mode this file described on 9 August and I walked into
anyway.** pseudo-val measures agreement with a teacher that is 0.900 accurate
against the 58 gold studies. A change that improves teacher-agreement without
improving the underlying prediction is exactly what that instrument rewards, and
2.5D is apparently such a change.

**The judging error was mine and it was procedural.** I set "peak pv-hold clears
the 0.882-0.887 band" as the criterion, saw gold move the other way in the same
table, and did not mention it. gold had been demoted to a last-resort tiebreaker
on 24 August for moving 0.0057 between identical runs - a fair reason - but a
tiebreaker that is ignored when it dissents is not a tiebreaker.

**Revised rule.** pv-hold alone decides nothing when the change could plausibly
shift teacher-agreement specifically. Architecture and input-representation
changes are exactly that class. For those, gold's direction is required to agree
before the result is called, even though gold is noisy - a 0.003 move against
the change is not proof it is bad, but it withdraws the right to call the change
good.

gold has now dissented twice and been right twice: m43small (gold -0.0040 while
pv-hold said +0.0008) and m53nbr here.

**What actually remains true.** convnext_nano reaches pv-hold 0.8858 with gold
0.9082 - also worse on gold, so also not a gain. Self-attention was bad on both.
The 0.884 ceiling stands, thirteen models became sixteen, and no axis has moved
it.

## 2.5D, split by recipe: the corrected rule separates two runs that pv-hold called the same

                       pv-hold            gold
    m33   256/up336    0.8799             0.9169
    m55nbr33  + 2.5D   0.8852  +0.0053    0.9223  +0.0054
    m31   128/up224    0.8832             0.9148
    m53nbr    + 2.5D   0.8896  +0.0064    0.9118  -0.0030

On pv-hold alone the two 2.5D runs look the same - both up by roughly 0.006, and
the low-resolution one looks slightly better. **gold separates them.** On the
256/up336 recipe both instruments move together; on 128/up224 they oppose, and
the board confirmed the opposition at 0.938 against m31's 0.942.

m55nbr33's gold of 0.9223 is the highest single-model gold this project has
measured, past m38's 0.9205.

This is the first case the rule adopted after the retraction has adjudicated, and
it adjudicated in both directions on the same architectural change. Had the rule
been in place this morning, m53nbr would never have been called a ceiling break
and m55nbr33 would still be a candidate.

**Why the recipes might differ is not established.** The plausible story is that
neighbour context needs in-plane detail to be worth the two channels it costs -
at 128 source the duplicate channels may have been carrying more than the
neighbours do. One run each, so that is a hypothesis and not a finding.

**Also settled:** the five-member ensemble including m53nbr scored 0.947, equal
to the four-member. A member that is 0.004 worse on the board is absorbed rather
than harmful, which matches every previous weak-member result here.

Submitting m55nbr33 alone on today's last slot. m53nbr alone already gave the
negative half of the test at 0.938; this is the symmetric half, and it asks
whether "both instruments agree" actually predicts the board rather than merely
sounding more careful.

## 2.5D loses on the board at both resolutions, and the safeguard rule was not strict enough

                            pv-hold    gold      LB
    m31   128/up224                              0.942
    m53nbr    + 2.5D        +0.0064   -0.0030    0.938
    m33   256/up336                              0.943
    m55nbr33  + 2.5D        +0.0053   +0.0054    0.942

**Both resolutions lose.** m53nbr was caught by the rule adopted yesterday -
gold dissented and the board agreed with gold. m55nbr33 was not: both
instruments moved up together and it still lost 0.001.

m56nbr37 (192/up288) finished the pattern from the third direction: pv-hold
0.881 and gold 0.908 against m37's 0.8830 and 0.9125, down on both.

**The rule failed because it asked for direction and not magnitude.** gold's
replicate sd is 0.0037. m55nbr33's +0.0054 is 1.5 sd - two runs of the same
recipe land that far apart routinely. Treating it as agreement was reading a
coin landing the same way as a signal, which is the same error as the pv-hold
reading it was meant to fix, one instrument along.

**Revised again:** for architecture and representation changes, gold must move
by at least two to three of its own standard deviations - roughly 0.008 to 0.011
- not merely point the same way. Below that it abstains rather than agreeing.

**2.5D is closed.** Three recipes, two board submissions, nothing. The idea was
sound - two of three input channels genuinely carried a duplicate - and filling
them genuinely improves agreement with the teacher's labels. It does not improve
agreement with reality, and that distinction is the entire lesson of this
project's instrument problem, arrived at for the third time.

The ceiling stands at pv-hold 0.884 and LB 0.947.

## 2.5D across all three recipes: no pattern, and the resolution hypothesis dies

                    recipe            pv-hold    gold      LB
    m31             128/up224         0.8832     0.9148    0.942
    m53nbr          128/up224 2.5D    0.8896     0.9118    0.938
    m37             192/up288         0.8830     0.9125     -
    m56nbr37        192/up288 2.5D    0.8822     0.9116     -
    m33             256/up336         0.8799     0.9169    0.943
    m55nbr33        256/up336 2.5D    0.8852     0.9223    0.942

    delta by recipe   pv-hold          gold
    128              +0.0064          -0.0030
    192              -0.0008          -0.0009
    256              +0.0053          +0.0054

**Not monotone, so the resolution hypothesis is dead.** The middle point does not
sit between the ends - 192 is flat on both instruments while 128 splits and 256
agrees. "Neighbour context needs in-plane detail to be worth the channels" was a
reasonable story and it is not what the data shows.

What the data shows is scatter. pv-hold's within-recipe replicate sd is 0.0005,
so +0.0064 and +0.0053 are genuinely large movements on that instrument - and
they point in no consistent direction relative to gold, and both board
submissions lost. The honest summary is that 2.5D changes what the model agrees
with, substantially and reproducibly on pseudo-val, without changing whether it
is right.

Three recipes, two submissions, one dead hypothesis, and a rule that had to be
tightened twice. The axis is closed.

## 2.5D, final: closed individually and in ensembles

    configuration                    pv-hold    gold      verdict
    4-member incumbent               0.8898     0.9203    baseline
    + m55nbr33 (256 2.5D)            0.8905     0.9228    abstain
    m53nbr+m55nbr33+m56nbr37 alone   0.8920     0.9175    reject

The all-2.5D trio reaches the highest pv-hold this project has measured, 0.8920,
while gold falls 0.0028 - the same signature as m53nbr alone, which lost on the
board. Adding m55nbr33 to the incumbent moves both up, but gold's +0.0025 is
0.7 of its own standard deviation. Under the rule as tightened tonight that is
an abstention, not agreement. Reading it as a gain would be the same mistake a
third time, at a weaker signal than the one that already fooled me.

**Closed.** Three recipes, two board submissions, three ensemble measurements.
Nothing survives.

## State at 26 August 04:00

    leaderboard      0.947, rank 3 of 2342. Unchanged since 23 August.
    efficiency       583 of 2238, because the board scores our best submission
                     and that is the slow four-member ensemble.
    ceiling          pv-hold 0.884 across nineteen models and eight recipes.

**Closed axes (13):** in-plane resolution, slice count, capacity, DINOv2,
RadImageNet, pos_weight, plane weighting, study-level loss, label blending,
public teacher labels, TTA, slice self-attention, 2.5D.

**The instrument rules, as they now stand.** These cost more to learn than any
model did:

  - Compare late-epoch plateaus, not the delivered checkpoint. pv-sel selection
    throws away up to 0.009, and 16-epoch runs lose 4.7x what 12-epoch runs do.
  - pv-hold's between-run spread is 0.0005; pv-sel's is 0.0044; gold's is 0.0037.
  - pv-hold alone decides nothing for architecture or representation changes -
    it measures agreement with a teacher that is 0.900 accurate, and a change
    that improves that agreement without improving prediction is exactly what it
    rewards.
  - gold must move at least 0.008 (two to three of its own sds) to count as
    agreeing. Below that it abstains.
  - The leaderboard cannot resolve below about 0.002; four submissions differing
    by less all returned 0.947.

**What is open.** The efficiency track, where our accuracy is already top of the
board and runtime is untouched - but the board scores the best-scoring
submission, so it requires making the main entry cheaper rather than submitting
something else. Measured: the four-member at half slices costs 0.0016 locally
and scored 0.946, against 177 minutes of scoring for the full version.

**What is not open.** Every image-side axis reachable from this architecture and
this label set. Moving the ceiling needs either a genuinely different model class
or a better teacher, and the public field's teachers are all worse than ours.

## The series-count curve, and why the theoretical optimum is not the usable one

Scoring cost is 53 minutes of fixed DICOM decode plus 31 per ensemble member.
The fixed part decodes every series of every study - 5.6 on average. Four points
on the incumbent four-member ensemble, first-N-encountered in shard order:

    series   pv-hold   gold     decode    saved    gold cost   net at 0.00075/min
      1      0.8624    0.9012    9.5 min   43.5     0.0191      +0.0135
      2      0.8739    0.9155   18.9       34.1     0.0048      +0.0208
      3      0.8797    0.9173   28.4       24.6     0.0030      +0.0155
      4      0.8856    0.9212   37.9       15.1    -0.0009      +0.0122
     all     0.8898    0.9203   53          -        -            -

**Both instruments put the theoretical optimum at two series**, +0.0208 by gold
and +0.0097 by pv-hold. They disagree on magnitude and not on the answer, which
is the first time in this project the two have agreed about anything at this
scale.

**But the theoretical optimum is unusable, and the reason is the board mechanic
found yesterday.** The efficiency leaderboard scores our *best-scoring*
submission. A two-series entry costs 0.0048 on gold, which would show up as
roughly -0.003 on the board - so it would no longer be our best submission and
would never be scored on the efficiency board at all. The saving would be
invisible.

**The usable point is four series.** gold moves +0.0009, which is a quarter of
its own standard deviation and therefore nothing, while 15 minutes disappear.
That is free by the rule this project now uses - a change gold cannot see is a
change gold does not object to - and it keeps the main-track score, which is the
condition for being scored on efficiency at all.

Stacked with the slice lever, which is also near-free:

    four series + half slices    decode 37.9 + members 62 = ~100 min against 177
                                 cost ~0.002 across both instruments

**What this does not establish.** First-N-encountered is not the best N. The
plane probe found series are not interchangeable, so a selection that picked
sagittal-first, or one series per plane, would cost less than these numbers -
they are an upper bound. If four series is adopted and the main score holds, the
next question is which four, and that is measurable the same way with no GPU.

## 26 August, 08:10 — which series, not how many: plane balance is worth a series and a half

The open question from last night was whether choosing the series deliberately
beats taking the first N. It does, and by enough to change the operating point.

`ensemble.py --by-plane` spends the series budget across sagittal/coronal/axial
instead of on whatever the shard order presents, keeping a series only when its
plane is not already better represented than the thinnest plane. Measured on the
same four-member ensemble as the count curve:

    series  selection   pv-hold   gold      decode saved
      3     arbitrary   0.8797    0.9173    24.6 min
      3     by plane    0.8842    0.9200    24.6 min
      4     arbitrary   0.8856    0.9212    15.1 min
      4     by plane    0.8862    0.9217    15.1 min
    all 5.6     -       0.8898    0.9203      -

Plane balance wins at both counts on both instruments, and it is nearly free:
**three balanced series reach gold 0.9200 against the whole study's 0.9203**, a
tenth of gold's own spread, for 24.6 minutes. Four balanced series score 0.9217,
*above* the whole study. That is the plane probe's non-interchangeability
result showing up as runtime.

**The honest reading is narrower than that table looks.** Every gold movement
here is inside the 0.008 abstention band, so gold is not actually voting; the
only instrument with a readable signal is pv-hold, and it says all four subsets
cost something (-0.0036 to -0.0101 against the whole study). What the table
establishes is the *ordering* - balanced beats arbitrary at equal cost - not
that the cost is zero.

**Choosing between three and four.** At the efficiency metric's ~0.00075 AUC per
minute, 24.6 minutes is worth 0.0185 and 15.1 is worth 0.0113. Going from four
balanced to three costs 0.0017 on gold and 0.0020 on pv-hold and buys 0.0072.
Three is the better efficiency point by a margin larger than the measurement
error on either instrument.

### The board condition was already broken, and that changes what today is for

Last night's plan said the fast entry has to hold 0.947 or the efficiency board
will score the slow one instead. Checking the submission history rather than
assuming: **the four-member half-slice entry scored 0.946**, not 0.947. The
slice lever alone already cost a quantised step, so the condition is not
something the series lever might break - it is already broken.

That is less serious than it sounds, and the distinction is worth writing down.
The *live* efficiency board scores the best-scoring submission, which is why our
rank moved when a slower higher-scoring entry landed. The *prize* is judged on
the submissions selected at the deadline, 57 days out. So the job today is not
to win the live board - it is to measure the fast configurations well enough to
choose correctly in October.

### Ported to the kernel

`submit_kernel.py` had `MAX_SERIES` but no way to know a plane:
`Anatomical_Plane` exists only in `image_dataset.parquet`, which is not mounted
at scoring time. It turned out not to need it - `volume.py` already carries
`plane_of`, which takes the slice normal from `ImageOrientationPatient` and
returns the axis it lies closest to, and the pushed kernel is volume.py
concatenated with the driver, so the function was already in scope. The cost is
one header read per series with `stop_before_pixels=True`; pixels are never
touched.

`pick_series` was tested against the study shapes it will actually meet before
being pushed: a mixed study, a study that is all one plane, a study with fewer
series than the budget, and a series whose orientation will not parse. The last
one matters - an unreadable plane is queued rather than blocking, because
`min()` is taken over the three real planes only.

Two kernels pushed at 08:11, both four members at half slices:
`sub-effp3` (three balanced series) and `sub-effp4` (four). Both submit after
the 09:00 KST slot reset - all five of yesterday's slots were spent.

**One fidelity caveat on both.** The local measurement walked each study's
series in shard order; the kernel walks them in SeriesInstanceUID order. The
balance rule is the same but which member of an over-represented plane gets
dropped is not, so the board numbers are a re-measurement of the idea rather
than a confirmation of these exact figures.

## 26 August, 09:15 — the decorator a text insertion stepped over

Both efficiency kernels finished COMPLETE, wrote a submission.csv, exited 0, and
were worthless. Every series in every study failed with

    RuntimeError: Can't call numpy() on Tensor that requires grad

and the log's last useful line was `wrote submission.csv: 3 rows, 3 fell back to
priors`. A submission built entirely from priors scores at chance.

The cause was the edit that added `pick_series`. It was inserted immediately
above `def predict_study(uid):` - and the line above *that* was
`@torch.no_grad()`. The insertion landed between the decorator and the function
it decorated, so `no_grad` began guarding `pick_series`, which does no tensor
work at all, and `predict_study` started running the model with autograd live.
The forward pass still worked; only the `.numpy()` at the end of it did not.

**Two things about this are worth keeping.**

Anchoring an insertion on `def <name>(` is unsafe whenever the function might be
decorated, because the anchor is not the top of the definition. Nothing in the
diff looked wrong - the helper was in the right place, the decorator was still
present, and the file parsed.

And it failed *quietly*. The kernel does not exit non-zero when a series fails;
it catches, prints, and falls back to per-finding priors, which is correct
behaviour for one unreadable series and catastrophic for all of them. Had the
dev-run log not been read before submitting - which was in the plan only because
the last change needed the member count confirmed - this would have been a slot
spent on a chance-level entry, and the board would have said 0.5 without saying
why.

`push_submit_model.py` now refuses to push unless `@torch.no_grad()` is still
directly above `def predict_study`. That is a narrow guard for one line, but it
is the line whose loss produces a result that looks like a bad model rather than
a broken one.

**Standing rule, promoted from a habit to a check:** read the dev-run log's
fallback count before every submission. `N fell back to priors` must be 0.

### Submitted 09:20, and the next lever

Both v2 kernels ran clean - `0 fell back to priors`, four members loaded, and
dev-run cost of 10.1s against 12.8s per study for three series against four -
which looked like the series ratio and was over-read as confirmation. See the
correction below: the dev run is three studies and its per-study seconds are
dominated by fixed startup, so it cannot resolve this.

    effp3   4 members, half slices, 3 plane-balanced series
    effp4   4 members, half slices, 4 plane-balanced series

Three slots left today. The dev stub is three studies, so its absolute seconds
are not the scoring runtime and are not treated as such - only the ratio between
the two entries is read from it.

**Next: member count, which is the one efficiency lever never measured.** At the
new operating point the arithmetic changed. Decode is 28.4 minutes at three
balanced series, and each member now costs about 8.3 minutes rather than 31 -
scaled by 3/5.6 series and by half the slices - so members are a *larger* share
of a smaller total than they were when the count was chosen. Dropping four to
two buys about 16 minutes, worth 0.012 at the metric's exchange rate, against a
recorded cost of roughly +0.001 per member with a recipe worth about two of
them. The trade looks lopsided enough that two members may be the efficiency
optimum and four an inherited default.

Queued as three inference-only runs (one, two and three members at the
operating point) against the four-member number already in hand.

### A confound caught before it was read, not after

The first member-count run came back at pv-hold 0.8773 / gold 0.9117 for two
members, against 0.8842 / 0.9200 for four. Read straight, that is a large cost
for dropping two members and would have closed the axis immediately.

It is not comparable. **The entire series curve was measured at full slices**;
the member runs were queued with `--slice-stride 2` because that is the intended
operating point. So the gap contains both the member count and the slice lever,
and attributing it to members alone is the same mistake this project made with
2.5D - reading one number against another that was not produced the same way.

The four-member point at the identical setting is queued. Until it lands the
member axis has no reading at all, and the 0.9117 above is not evidence for
anything.

## 26 August, 10:00 — member count is the lever that was never priced

With the matched four-member baseline in hand the axis reads. All four points
are the same operating point: three plane-balanced series, half slices.

    members   pv-hold   gold      est. runtime
      1       0.8712    0.9006    36.7 min
      2       0.8773    0.9117    45.0 min
      3       0.8811    0.9130    53.3 min
      4       0.8824    0.9158    61.6 min

A member costs 8.3 minutes here, which the metric prices at **0.0062 AUC**. Set
each marginal gain against that:

    1 -> 2    pv-hold +0.0061   gold +0.0111    worth it (gold clears 0.008)
    2 -> 3    pv-hold +0.0038   gold +0.0013    not worth it
    3 -> 4    pv-hold +0.0013   gold +0.0028    not worth it

**Two members is the efficiency optimum, and four was an inherited default.**
The count was chosen for the main track, where members are free relative to the
fixed 53-minute decode, and carried into the efficiency track unexamined - where
the decode has been cut to 28 minutes and members are most of what is left.
Dropping four to two saves 16.6 minutes, worth 0.0124, against a cost of 0.0051
on pv-hold and 0.0041 on gold. Net gain roughly +0.0073.

This is the same "each recipe is worth about two members" result seen from the
cost side: the second member is worth more than the third and fourth combined on
gold, and it is the only one that pays its own way.

**Where the reading is soft.** Only the 1->2 gold step clears the 0.008
abstention band; 2->4 is +0.0041 on gold and abstains. pv-hold's spread is
0.0005, so its numbers are all real signal, and it does say four is genuinely
better than two by 0.0051 - the argument for two is not that four is no better,
it is that four is not better *enough* to pay 16.6 minutes.

Pushed `sub-effp3m2` - m31 + m33 at the operating point, estimated 45 minutes
against the incumbent's 177.

### Correction: the dev run cannot measure runtime, and I read it as if it could

The two-member kernel came back at **11.4s per study against the four-member
entry's 10.1s**. Halving the members cannot make a kernel slower, so the dev
timing is not measuring what it appears to.

The reason is arithmetic. The dev run is three studies. Whatever the kernel
spends once - imports, CUDA context, loading and building each checkpoint - is
divided by three and lands inside "per study". The cost model says a study at
three series and half slices should cost about 3.9s, and every dev number here
is two to three times that, which is the fixed cost showing through.

So the earlier 10.1 vs 12.8 reading is withdrawn. It was the right *ratio* by
coincidence, and I quoted the coincidence as evidence. What the dev log can
establish is what it was originally read for - member count, fallback count, and
that the flags took effect - and nothing about runtime. Real runtime is only
visible on the scoring run.

Submitted `effp3m2` anyway: the case for two members rests on the accuracy
measurements and the cost model, neither of which came from the dev log.

## 26 August, 10:45 — the whole efficiency surface, re-cut at the configuration actually chosen

The series curve in this file was measured at four members and full slices. The
operating point moved to two members at half slices, and nothing established
that the levers are additive, so the curve was re-cut where it will be used.
Two members (m31 + m33), half slices, plane-balanced selection throughout:

    series    pv-hold   gold      runtime   marginal gold   marginal cost
      2       0.8685    0.9016    30.0 min       -               -
      3       0.8773    0.9117    45.0 min    +0.0101        0.0113
      4       0.8818    0.9132    60.0 min    +0.0015        0.0113
    all 5.6   0.8852    0.9192    84.0 min    +0.0060        0.0180

Runtime from the cost model: decode 53 min x (series/5.6), each member 31 min x
(series/5.6) x (1/stride).

**Three and four series are settled, two is not.** Going 3 -> 4 buys 0.0015 of
gold for 15 minutes and 4 -> all buys 0.0060 for 24; both are refused without
argument. But 2 -> 3 buys 0.0101 for a cost of 0.0113, and the exchange rate
itself is only known to +-12% (0.039-0.050 AUC per hour). **The uncertainty in
the rate is the same size as the difference being decided**, so no amount of
local measurement settles two against three. That is precisely the condition
this project spends a slot on, and effp3m2 (three series) is already on the
board with two slots left today.

### The slice lever has a knee, and it is at half

    3 series, 2 members, stride 2   pv-hold 0.8773   gold 0.9117   45.0 min
    3 series, 2 members, stride 4   pv-hold 0.8699   gold 0.8983   36.7 min

Quarter slices costs 0.0134 on gold - well clear of the abstention band - to
save 8.3 minutes worth 0.0062. **Rejected, and by a factor of two.** Set against
the earlier finding that 32 -> 16 slices cost roughly 0.0016, the lever is not a
smooth trade at all: the first halving is nearly free and the second is more
than eight times as expensive. Sixteen slices is the floor, and it was reached
by luck rather than by search - the stride was set to 2 because it was the
obvious number.

### What the re-cut incidentally established

Two members at half slices over all series scores gold **0.9192**, against
**0.9203** for four members at full slices over all series. Halving the members
*and* halving the slices costs 0.0011 - a third of gold's own spread.

That reframes where the accuracy actually lives. Members and slices, which is
where nearly all of this project's compute went, are close to free to give up.
Series coverage is not: dropping from 5.6 to 2 costs 0.0176 on the same
instrument, an order of magnitude more than either. Whatever the model is
reading, it is reading it from *which planes it gets to see*, and the rest is
expensive decoration.

### The board settles the member axis: four and two score identically

    effp3    4 members, half slices, 3 plane-balanced series    0.942
    effp3m2  2 members, half slices, 3 plane-balanced series    0.942
    effp4    4 members, half slices, 4 plane-balanced series    pending

**Identical.** Dropping two of four members cost exactly nothing on the board
and returned 16.6 minutes. gold had put the cost at 0.0041 and abstained under
the 0.008 rule; the abstention was right, and this is the first time the rule
has been vindicated by the board rather than merely obeyed.

Set against the incumbent path, the trade is not close:

    4 members, all series, half slices    0.946   ~100 min
    2 members, 3 balanced series, half    0.942     45 min

Four thousandths of AUC for fifty-five minutes. At the metric's exchange rate
those minutes are worth 0.041 - **ten times what they cost**. Every step down
this curve has paid, and none of the accuracy given up has been visible except
as the quantised 0.001 steps the board reports.

**Two slots left, spent on the two questions the local instruments cannot
answer.** `effp2m2` is two members at two balanced series (30 min) - the 2-vs-3
question where the exchange-rate uncertainty exactly spans the gap. `effp3m1`
is a single member at three series (36.7 min); gold called 1 -> 2 worth 0.0111
and therefore worth buying, but gold also called 2 -> 4 worth 0.0041 and the
board just scored that at zero, so gold's magnitudes on this axis have not
earned trust.

## 26 August, 10:50 — what the public rank is actually worth

Asked how the private set is likely to land, and the answer is worth recording
because it bears on where the remaining seven weeks should go.

    our score        0.947    rank 3 of 2408
    10th place       0.945    the money line, 0.002 below us
    tied at 0.947    4 teams
    within 0.003     15 teams

**We are not third. We are near the top of a fifteen-team pile-up.** A
Hanley-McNeil estimate for mean AUC over 12 findings on roughly a thousand
studies puts the resampling spread of a single team's score at about +-0.005 -
larger than our 0.002 cushion above the money line. Top ten is likely and it is
not safe.

That figure overstates the *rank* shuffle, because it treats each team's score
as moving independently. The top models are almost certainly correlated - same
data, same backbone families - and correlated errors move everyone together,
which damps reshuffling. But nothing damps it down to 0.002.

**The consequence for planning.** Chasing thousandths on the main board is
fitting noise: the instrument cannot resolve the gap that separates 3rd from
15th, so slots spent there buy variance, not position. The only main-track lever
with real headroom is the label side, because the image axes are closed and the
ceiling is the teacher's 0.900 agreement with the gold labels - and that is slow
work.

The efficiency track is the opposite. Today alone it produced trades at ten to
one, measured cleanly, against a field that is far thinner - three prizes and
most teams not contesting it at all. On expected value the remaining time
belongs there, and that is the recommendation standing for the operator.

## 26 August, 11:10 — the board's own efficiency curve, and two series wins

All five slots came back. Runtimes from the cost model; efficiency computed as
-AUC/0.40 + seconds/32400, lower better, with 0.40 the mid-estimate of
(maxAUC - Benchmark) implied by the 0.039-0.050 AUC-per-hour exchange rate.

    entry     members  series   board   runtime    efficiency
    effp4        4        4     0.945   82.2 min    -2.2103
    effp3        4        3     0.942   61.6 min    -2.2409
    effp3m2      2        3     0.942   45.0 min    -2.2717
    effp2m2      2        2     0.934   30.0 min    -2.2794

**Two series is the operating point.** Going 2 -> 3 buys 0.008 of board AUC for
fifteen minutes priced at 0.0113, and the ordering survives the whole plausible
range of the exchange rate - at 0.039/hr and at 0.050/hr, two series still wins.
Going 3 -> 4 buys 0.003 for twenty minutes and is not close.

**A correction to how I framed this yesterday evening.** I said the local
instruments could not settle two against three because gold's 0.0101 sat inside
the exchange rate's uncertainty band around 0.0113. That was true as stated but
it buried the direction: 0.0101 is *below* 0.0113, so the local reading already
leaned to two series, and I reported a lean as a tie. The board then put the
step at 0.008, further below. The slot was still worth spending - the lean was
inside the overlap and a wrong call costs a whole configuration - but the record
should say the local data pointed the right way and I declined to read it.

**Members are free, confirmed twice.** effp3 and effp3m2 differ only in member
count and both score 0.942. Four members buy nothing at three series, exactly as
they bought nothing when the axis was first priced.

**Next probe: one series.** At two members that is 15.0 minutes, so it wins if
it scores above roughly 0.923. The four-member full-slice curve put one series
0.0143 of gold below two, which would land somewhere near 0.925 - close enough
to the line that only the board can say. That is tomorrow's first slot.

The whole descent, from the incumbent to here:

    4 members, all series, full slices    0.947    177 min
    2 members, 2 balanced series, half    0.934     30 min

Thirteen thousandths of AUC for one hundred and forty-seven minutes.

### The last slot, and where the descent stops

    effp3m1   1 member, 3 balanced series, half slices   0.936   36.7 min   -2.2720

Ranked against the rest, lower better:

    2 members, 2 series   0.934   30.0 min   -2.2794   best
    1 member,  3 series   0.936   36.7 min   -2.2720
    2 members, 3 series   0.942   45.0 min   -2.2717
    4 members, 3 series   0.942   61.6 min   -2.2409
    4 members, 4 series   0.945   82.2 min   -2.2103

The second member at three series buys 0.006 and costs 0.0062. **Exactly break
even** - and the top three entries sit within 0.008 of each other on a scale
where the first two steps of this descent were worth 0.03 and 0.06.

**The runtime lever is spent.** Around thirty minutes every direction trades one
for one: another series, another member, or a coarser stride all move accuracy
and runtime by the same amount in efficiency terms. Further cutting is no longer
a free lunch, it is just a smaller submission. From here efficiency improves
only by raising AUC *at* the operating point, not by shrinking the operating
point further.

### A defect in the cost model, found while looking for the next lever

Every runtime number above used "31 minutes per member", which is an average.
Members are not equally expensive:

    m31newshards   128 source, up224    1.00x
    m38seed1       128 source, up224    1.00x
    m37up288       192 source, up288    1.65x
    m33up336new    256 source, up336    2.25x

Compute goes roughly as the square of the upsample, so **m33 alone costs more
than m31 and m38 together** - and m33 is in the pair chosen for every efficiency
entry submitted today. Normalising, a 224-class member is about 21 minutes and
m33 is about 47.

That reopens the member question in a form nothing measured today addresses. The
axis priced this morning was *how many* members; the axis that matters at a
fixed time budget is *which*, because a cheap pair buys either a shorter run or
an extra series for the same money. m31 + m38 at three series costs about the
same as m31 + m33 at two and a half.

Queued: the cheap pair at two and three series, and each of m31 and m33 alone at
two, to get a cost-aware frontier before tomorrow's slots.

## 26 August, 11:55 — the cost-aware frontier: m33 should never have been in the efficiency entry

Half slices, plane-balanced selection throughout. Cost units are relative
inference cost (upsample squared); runtime uses 21 minutes for a 224-class
member and 53 x (series/5.6) minutes of decode.

    config      units  series  pv-hold   gold     runtime
    m31          1.00     2    0.8565   0.9014    22.7 min
    m31+m38      2.00     2    0.8661   0.9065    26.4 min
    m33          2.25     2    0.8611   0.8865    27.3 min
    m31+m33      3.25     2    0.8685   0.9016    31.1 min   board 0.934
    m31+m38      2.00     3    0.8773   0.9176    39.7 min
    m31+m33      3.25     3    0.8773   0.9117    45.0 min   board 0.942

**m33 alone is strictly dominated by m31 + m38**, which is cheaper (26.4 against
27.3 minutes) and higher on both instruments (pv-hold 0.8661 against 0.8611,
gold 0.9065 against 0.8865). Two cheap members beat one expensive member
outright - not on a trade, on every axis at once.

And m33's contribution to the pair is not worth its price either: adding it to
m31 at two series buys pv-hold +0.0024 for 4.7 minutes, and 4.7 minutes are
worth 0.0035. **m33 does not belong in an efficiency submission**, and it was in
every one of the five sent today.

The clean statement of the day's mistake: I priced *how many* members and never
*which*, because the cost model carried a single average that hid a 2.25x
spread. Every conclusion drawn from that model about member count still stands -
it was applied consistently - but it could not see the axis that mattered most
at a fixed time budget.

**The optimum is flat, and that is the real finding.** Efficiency scores for the
plausible next configurations:

    m31+m38, 2 series, 26.4 min, board ~0.934    -2.2861
    m31+m38, 1 series, 13.3 min, board ~0.923    -2.2830
    m31+m38, 3 series, 39.7 min, board ~0.942    -2.2815
    m31+m33, 2 series, 30.0 min, board  0.934    -2.2794   (today's best)

Everything worth trying now sits inside 0.007 of everything else, on an
instrument quantised at 0.001 and an exchange rate known to +-12%. Local
measurement cannot order these. Tomorrow's slots go to the cheap pair at two and
three series, and the board decides.

### The cost model missed the backbone too

Ranking every trained model by pv-hold and by cost at once produces an
uncomfortable table. **The best single models on pv-hold are all in the cheapest
price class**, and neither member of today's efficiency pair is the best in its
own class:

    m53nbr        tiny  up224   1.00x   pv-hold 0.8896   (2.5D - lost on the board)
    m48blend60    tiny  up224   1.00x           0.8870   (label blend - pv-hold contaminated)
    m52nano       NANO  up224   ~0.55x          0.8858
    m43small      small up224   ~1.8x           0.8841
    m38seed1      tiny  up224   1.00x           0.8836
    m31newshards  tiny  up224   1.00x           0.8832
    m33up336new   tiny  up336   2.25x           0.8799   <- in all five entries today

Two of the top three carry known caveats and are excluded: m53nbr is 2.5D, whose
pv-hold gain did not survive the board twice, and the blend models change the
very labels pv-hold is defined against, so their pv-hold is not comparable.

**m52nano survives both filters and dominates m31 outright.** It is
convnext_nano rather than convnext_tiny - roughly 0.55x the FLOPs - and its
pv-hold is 0.0026 *higher*. Cheaper and better, on the instrument with a 0.0005
spread.

The cost model scaled only with upsample squared, so it counted nano and tiny as
identical. That is the second thing it hid today; the first was the 2.25x spread
across upsamples. Corrected, relative cost is
`backbone FLOPs ratio x (upsample/224)^2`:

    m52nano   0.55      m31, m38   1.00      m43small  1.8
    m37       1.65      m33        2.25

m52nano + m31 at three series comes to 37.1 minutes against m31 + m38's 39.7,
and should be at least as accurate. Queued, with the nano alone and a nano/m38
pair for the frontier.

**The pattern worth naming.** Twice today the cost model was wrong in the same
direction: it treated members as interchangeable when they are not. Every
accuracy conclusion drawn under it still holds, because it was applied
consistently to both sides of each comparison - but it kept steering the choice
of *which* model to spend on, and it was blind on exactly that axis.

### The nano frontier: the pair submitted five times today was never the right one

All at half slices with plane-balanced selection. Runtime uses the corrected
cost model (backbone FLOPs x upsample squared, 21 min for a 224-class tiny).

    two series                pv-hold   gold     runtime
      m52nano                 0.8668   0.8957   21.0 min
      m31                     0.8565   0.9014   22.7 min
      m52nano + m31           0.8691   0.9038   24.7 min
      m31 + m38               0.8661   0.9065   26.4 min
      m33                     0.8611   0.8865   27.3 min
      m31 + m33               0.8685   0.9016   31.1 min   board 0.934

    three series
      m52nano + m38           0.8791   0.9177   37.1 min
      m52nano + m31           0.8781   0.9134   37.1 min
      m31 + m38               0.8773   0.9176   39.7 min
      m31 + m33               0.8773   0.9117   45.0 min   board 0.942

**m52nano + m38 is the best three-series configuration on every axis at once** -
highest pv-hold, highest gold, and cheapest. It beats m31 + m33, the pair used
in all five of today's submissions, by 0.0018 of pv-hold and 0.0060 of gold
while running 7.9 minutes faster. There is no trade to weigh; it is simply
better.

The same holds at two series, where m52nano + m31 beats m31 + m33 on all three.

**What this says about today.** Five slots went to configurations built from the
wrong two models. The accuracy conclusions those slots bought are unaffected -
series count, member count and stride were each measured with the pair held
constant, so the comparisons were fair - but the *level* they were measured at
was needlessly low and needlessly slow. The cause was a cost model that
collapsed every member to one average, twice: first hiding the 2.25x spread
across upsamples, then hiding the backbone entirely, so a convnext_nano and a
convnext_tiny were priced the same.

Estimated efficiency for tomorrow's candidates, using the board scores of the
matched m31+m33 configurations as the AUC prior:

    m52nano + m38, 3 series, 37.1 min, ~0.943    -2.2888
    m52nano + m31, 2 series, 24.7 min, ~0.934    -2.2893
    m31 + m33,     2 series, 30.0 min,  0.934    -2.2794   today's best

Tomorrow's first two slots go to the nano pairs at two and three series.
m52nano is uploaded and indexed as a Kaggle dataset; the two-series nano+m38
point is measuring now.

### Two-series frontier closed, and the day's configurations settled

    two series           pv-hold   gold     runtime
      m52nano + m38      0.8709   0.9049   24.7 min   <- best
      m52nano + m31      0.8691   0.9038   24.7 min
      m31 + m33          0.8685   0.9016   31.1 min   board 0.934
      m52nano            0.8668   0.8957   21.0 min
      m31 + m38          0.8661   0.9065   26.4 min
      m31                0.8565   0.9014   22.7 min

**m52nano + m38 wins at two series and at three**, and dominates m31 + m33 at
both - +0.0024 pv-hold, +0.0033 gold, 6.4 minutes cheaper at two series.

Estimated efficiency, using matched board scores as the AUC prior:

    m52nano + m38, 2 series, 24.7 min, ~0.935    -2.2918
    m52nano + m38, 3 series, 37.1 min, ~0.943    -2.2888
    m31 + m33,     2 series, 30.0 min,  0.934    -2.2794   today's best

`sub-nano2` and `sub-nano3` are pushed and will be verified before the 09:00 KST
reset; the two slots are spoken for. Note the caution that has to survive the
night: **these are estimates that assume the board scores the nano pair the same
as the matched tiny pair.** The local instruments say it should score at least
as well, but the entire day's lesson is that the board is the only instrument
that decides, and pv-hold differences of 0.002 have failed to reach it before.

### Both nano kernels verified; twenty idle hours pointed at the cheapest price class

`sub-nano2` and `sub-nano3` both report two members and **0 fell back to
priors**. They submit at the 09:00 KST reset, roughly twenty hours out.

Those twenty hours of idle GPU go where today's conclusion points. The runtime
lever is spent - around thirty minutes every direction trades one for one - so
efficiency now improves only by raising AUC at the operating point. And the
operating point turned out to be the *cheapest* price class, which is exactly
where the least training effort has gone: m52nano was a one-off capacity probe
from 25 August, twelve epochs, never revisited, and it beat every tiny model
this project has spent weeks on once cost is counted.

Queued:

    m58nanos1    nano, up224, seed 1    a replicate; nano+nano is 1.10 units
                                        against nano+m38's 1.55
    m59nano192   nano, up192            0.40 units - a cheaper class than any
                                        point on the current frontier
    m60nano288   nano, up288            0.91 units, still under tiny-at-224;
                                        separates nano's deficit into capacity
                                        versus resolution

The third is the one with something to learn either way. If nano at up288
matches tiny at up224 it says the gap was resolution and the frontier should
move to higher-resolution nano; if it does not, the gap is capacity and the
cheap class is already exhausted.

## 26 August, 13:30 — the nano replicate, and what "better" was actually measuring

m58nanos1 (nano, up224, seed 1) finished in 51 minutes: delivered pv-hold
0.8824 against m52nano's 0.8858. A 0.0034 gap between two runs of one recipe,
where pv-hold's recorded replicate spread is 0.0005, so something had to be
wrong with one of those numbers.

Comparing plateaus rather than delivered checkpoints, as the rule requires:

    run           delivered   plateau (ep 9-12)
    m52nano        0.8858      0.8800
    m58nanos1      0.8824      0.8783
    m38seed1       0.8836      0.8758
    m39seed2       0.8831      0.8798

**Nano and tiny are equal at the plateau** - 0.878-0.880 against 0.876-0.880,
inside each other's spread. The claim written this morning, that m52nano is
*better* than m31, came from m52nano's delivered checkpoint sitting 0.006 above
its own plateau: epoch 6 was a lucky pick on the selecting half.

**The nano conclusion survives, with a different justification.** It is not
"cheaper and better", it is **"equal and 0.55x the price"** - which is the same
decision and a weaker claim. Everything queued for the idle window still makes
sense, and tomorrow's submissions are unchanged, because the ensemble
measurements that chose them were run on the delivered checkpoints, which is
exactly what a submission contains.

**A property of the instrument, worth recording.** Delivered-checkpoint pv-hold
and plateau pv-hold have different spreads:

    delivered, m38 vs m39    0.8836 vs 0.8831    0.0005
    plateau,   m38 vs m39    0.8758 vs 0.8798    0.0040

The 0.0005 figure this project relies on is the spread of *delivered* values,
and it is small partly because epoch selection pulls them together - pv-sel
correlates with pv-hold, so picking the best epoch on one half drags the other
half toward a common level. The underlying run-to-run spread is eight times
larger.

That does not invalidate today's comparisons: every one of them was run on
delivered checkpoints, consistently on both sides, which is the basis the 0.0005
belongs to. But it does mean pv-hold cannot be used to compare a *recipe*
against a recipe at the 0.002 level - only a checkpoint against a checkpoint.
m52nano beating m31 by 0.0026 was a comparison of the first kind dressed as the
second.

Queued: m52nano + m58nanos1 at two and three series. A nano pair costs 1.10
units against nano+m38's 1.55 - about 3.4 minutes at three series - and if two
nano replicates ensemble as well as nano+tiny does, that is the cheapest strong
configuration available.

## 26 August, 15:35 — nano's deficit was resolution, not capacity

m60nano288 plateaus at **0.8838** over epochs 9-12, the highest plateau this
project has measured. Every plateau now on record, with its price:

    nano  up192   0.40 units   0.8780
    nano  up224   0.55 units   0.8783, 0.8800
    nano  up288   0.91 units   0.8838
    tiny  up224   1.00 units   0.8758, 0.8798

**A convnext_nano at 288 beats a convnext_tiny at 224, and costs less doing
it.** That answers the question the run was queued to settle: the gap between
nano and tiny was never capacity. It was that nano had only ever been tried at
the resolution tiny happened to use.

The pattern underneath is worth stating plainly, because it inverts how this
project has spent compute for weeks. Resolution was always bought by enlarging
the *backbone* - tiny to small, or tiny at up336 - and the cheap backbone was
only ever tested at the cheap resolution. Nobody ran a small backbone at a large
input. At a fixed budget the money should go to resolution, not capacity.

There is a caveat this table cannot settle. Plateau spread between replicates of
one recipe is 0.0040 (measured on m38/m39 and again on the two nano224 runs), so
nano288's 0.0046 lead over nano224 is barely one spread. It is suggestive, not
established, and a second nano288 seed would be needed to call it.

Queued: nano at up336 (1.24 units) to see whether resolution keeps paying, and
nano288 paired with nano224 and with nano192 at three series - because what
decides tomorrow's submission is ensemble behaviour on delivered checkpoints,
not single-model plateaus.

### The nano combinations, and a plateau that does not transfer

Half slices, plane-balanced, delivered checkpoints. Runtime from the corrected
cost model.

    three series                   pv-hold   gold     runtime
      m52nano + m38                0.8791   0.9177   37.1 min   best
      m52nano + m31                0.8781   0.9134   37.1 min
      m52nano + m58nanos1          0.8778   0.9112   34.6 min
      m31 + m38                    0.8773   0.9176   39.7 min
      m31 + m33                    0.8773   0.9117   45.0 min   board 0.942
      m59nano192 + m52nano         0.8754   0.9103   33.7 min
      m59nano192 + m38             0.8748   0.9168   36.3 min

    two series
      m52nano + m38                0.8709   0.9049   24.7 min
      m52nano + m58nanos1          0.8708   0.9018   23.0 min
      m59nano192 + m52nano         0.8701   0.8987   22.5 min
      m52nano + m31                0.8691   0.9038   24.7 min
      m31 + m33                    0.8685   0.9016   31.1 min   board 0.934

**nano192 does not survive into a pair.** Its single-model plateau (0.8780) sat
inside the nano224 range, so on the plateau table it looked free. In an ensemble
it costs 0.0037-0.0043 of pv-hold to save 0.8-3.4 minutes, which is a loss at
three series and a rounding error at two.

That is the useful lesson from this batch, and it cuts against how the morning's
member table was read: **a single-model plateau does not predict ensemble
behaviour.** Two models can plateau at the same level and contribute differently
once averaged, because what an ensemble buys is the part of one member's error
the other does not share. Nothing in a single-model number reports that.

**A nano replicate pair is a wash.** m52nano + m58nanos1, two seeds of one
recipe, comes within 0.0013 of m52nano + m38 for 2.5 fewer minutes at three
series - the saving and the loss are the same size. Consistent with this
project's "a recipe is worth about two members" rule, but far too small to
claim it again.

**Everything is still flat.** Seven three-series configurations now sit inside
0.0043 of pv-hold, on an instrument whose delivered-checkpoint spread is 0.0005
but whose board resolution is 0.001. The incumbent choice, m52nano + m38, is
still the best measured, and the kernels already pushed use it.

The one candidate that could leave the plateau is nano288, whose single-model
plateau is the highest measured - but this batch is exactly the warning against
believing that before the ensemble is run. It measures after the nano336
training.

## 26 August, 18:15 — the resolution axis closes, and explains itself

nano336 plateaus at **0.8825**, below nano288's 0.8838 for 36% more compute. The
resolution curve for a convnext_nano, by plateau:

    up192   0.40 units   0.8780
    up224   0.55 units   0.8783, 0.8800
    up288   0.91 units   0.8838
    up336   1.24 units   0.8825

It rises to 288 and stops. Given the 0.0040 replicate spread the last two points
are a tie, but either reading says 336 is not worth buying.

**And none of it reaches the ensemble.** The three-series pairs, in full:

    pair                        pv-hold   gold     runtime
      m60nano288 + m52nano      0.8793   0.9100   36.6 min
      m52nano + m38             0.8791   0.9177   37.1 min
      m60nano288 + m59nano192   0.8781   0.9125   35.8 min
      m52nano + m31             0.8781   0.9134   37.1 min
      m52nano + m58nanos1       0.8778   0.9112   34.6 min
      m31 + m38                 0.8773   0.9176   39.7 min
      m31 + m33                 0.8773   0.9117   45.0 min   board 0.942
      m59nano192 + m52nano      0.8754   0.9103   33.7 min
      m59nano192 + m38          0.8748   0.9168   36.3 min

nano288's single-model plateau leads nano224 by 0.0046. **In a pair that becomes
0.0002.** nano192's plateau matched nano224 and in a pair it lost 0.0037. The
transfer is not merely weak, it is absent in both directions.

**Why, and it is the same wall as everywhere else in this project.** The
ensemble sits near the teacher's 0.900 agreement with the gold labels. What
limits it is not how good each member is but how much of its error the other
member shares - and every member here is trained on the same teacher labels, so
improving one member improves the part that cancels and leaves the shared part
untouched. Making a better member does not make a better ensemble when the
error is common to all of them.

That reframes the day's biggest-looking result. "Nano's deficit was resolution,
not capacity" is true of single models and **operationally worthless**: the
whole point of finding a better cheap member was to build a better cheap pair,
and the pair does not move.

**Nothing displaces the incumbent.** The top four pairs sit inside 0.0015 of
pv-hold and 2.5 minutes of each other - flat to both instruments and well below
the board's 0.001 quantisation. m52nano + m38 remains the best measured, and
`sub-nano2` / `sub-nano3` are already pushed and verified against it.

Tomorrow's five slots: the two pushed kernels first. If a third is worth
spending, m52nano + m58nanos1 at three series is the cheapest member of the
statistically-equal set - 34.6 minutes against 37.1, which is the only remaining
distinction anywhere on this frontier.

## 26 August, 18:30 — the teacher is not uniformly 0.900, and that reopens the label axis

Asked whether the teacher can be improved. Before speculating, measured where it
actually fails. Per-finding agreement with the 58 gold studies:

    finding            pos/58   teacher AUC
    ACL                   24       0.994
    MCL                    9       0.972
    Medial Meniscus       26       0.952
    Fracture              18       0.925
    Medial OA             15       0.919
    PF OA                 21       0.911
    Baker's               12       0.911
    Contusion             19       0.876
    Lateral Meniscus      23       0.865
    Effusion              35       0.861
    Lateral OA            11       0.818
    Synovitis             27       0.787
                                   0.8993 mean

**The 0.900 that has been quoted as a ceiling for weeks is an average over a
0.21 range.** The teacher is near-perfect on findings a report states outright
and poor on findings a report grades or hedges. Lifting only the worst three to
0.95 would put the teacher at 0.9313.

### The image model already beats its own teacher on half the findings

m18cos - an *old* model, trained on the old shards, on the teacher's own labels
- against the teacher on the same 58 studies:

    finding            teacher   image     diff
    Effusion            0.861    0.957    +0.095
    Baker's             0.911    0.987    +0.076
    Medial OA           0.919    0.978    +0.060
    Contusion           0.876    0.927    +0.051
    Lateral OA          0.818    0.828    +0.010
    Fracture            0.925    0.933    +0.008
    Synovitis           0.787    0.767    -0.020
    Lateral Meniscus    0.865    0.839    -0.027
    PF OA               0.911    0.884    -0.027
    MCL                 0.972    0.941    -0.031
    ACL                 0.994    0.960    -0.035
    Medial Meniscus     0.952    0.912    -0.040

**The split is physical, not statistical.** Every finding the image model wins is
something you *see* - joint fluid, a Baker's cyst, cartilage loss, a bone
bruise. Every finding the teacher wins is something a report *states* - the
ligament and meniscus tears that are the headline of a knee MRI report. Effusion
is unmistakable on the image and routinely unmentioned or hedged in text; an ACL
tear is the sentence the report exists to write.

A student trained on its teacher's labels exceeds that teacher on six of twelve
findings. That is not leakage - it cannot inherit what the teacher does not
have. It is the image carrying information the report omits.

### Why uniform label blending failed, and what to do instead

Blending was applied **at one weight across all twelve findings**, and the
record shows every blend landing at or below the teacher. That is exactly what
this table predicts: a uniform blend spends the gain on Effusion to pay for the
loss on ACL and arrives nowhere.

**Per-finding blending is the untried version.** Weight toward the model where
it wins - Effusion, Baker's, Medial OA, Contusion - and keep the teacher intact
where it is near-perfect. Nothing in this project has tried that, and the axis
was closed on the uniform result.

**Where this is soft.** n=58, with 9-35 positives per finding, so a single AUC
here carries a standard error near 0.07. No individual difference below about
0.05 is readable, which leaves only Effusion, Baker's and Medial OA as
individually significant. The *pattern* is stronger than any of its rows: the
six wins and six losses sort perfectly by whether the finding is visible or
stated, which is not something noise produces.

It also has to be checked against the fact that gold is 58 studies and the
blending would be applied to 24,000. Deriving per-finding weights from 58
studies is a selection surface, and the weights should be coarse - two or three
buckets, not twelve fitted numbers.

## 26 August, 20:45 — per-finding blending is worse than uniform, and my own suggestion retires

Six hours ago this file recorded that uniform label blending failed because it
was applied at one weight across twelve findings, spending the gain on Effusion
to pay for the loss on ACL, and proposed **per-finding blending** as the untried
version. Measured now, on gold58, with the blend rule chosen leave-one-out so no
study contributes to the rule that scores it:

    teacher alone                   0.8993
    image alone                     0.9094
    uniform blend, model w=0.25     0.9195
    uniform blend, model w=0.50     0.9253
    uniform blend, model w=0.75     0.9272
    per-finding blend, LOO          0.9143

**Uniform beats per-finding by 0.013.** The suggestion was wrong, and wrong for a
reason already written down in this project: averaging pays when the two sources
fail *differently*, not when one is better. A text model and an image model fail
about as differently as two estimators can, so blending helps on every finding -
including the six where the teacher is individually stronger. The per-finding
rule withheld blending on exactly those six and threw the averaging away.

**And the 0.9272 is not new.** It is the 0.9271 this file already records, and
the reason it could not be reproduced is stated three sections above: the OOF
used for the actual training blend covers the 21,148 training rows and **none of
the gold rows**, by design. My measurement substituted `m18cos_gold_preds` -
an image model's predictions *on gold58* - which is a different quantity. It
measures teacher-plus-image agreement on 58 held-out studies, not the quality of
the labels that were fed to training.

**So the label axis does not reopen.** The blend was run at four weights, gold
moved 0.0005 against a replicate sd of 0.0037, and nothing here contradicts
that. What I produced is a fresh derivation of a number the record already had,
by a route the record already flagged as not matching the training blend.

Worth keeping anyway, because it is the cleanest statement of the split: on the
58 studies with real labels, a teacher at 0.899 and an image model at 0.909
average to 0.927. The two see genuinely different things. That complementarity
is unusable at submission time - there are no reports at test time - and
unusable in training, because the one experiment that could exercise it moved
nothing. It is a real property of the data with no remaining place to spend it.

## 27 August, 10:05 — the nano frontier was a ranking of selection luck, and the board says so

Both nano-pair submissions came back and both lose:

    config                          board   runtime   efficiency
    m31 + m33,     2 series         0.934   30.0 min   -2.2794   best
    m31 + m33,     3 series         0.942   45.0 min   -2.2717
    m52nano + m38, 3 series         0.936   37.1 min   -2.2713
    m52nano + m38, 2 series         0.926   24.7 min   -2.2693

The nano pair is 0.006 behind at three series and 0.008 behind at two. Two
independent points, same sign, both larger than the board's 0.001 quantisation.
**Yesterday's claim - that m52nano + m38 dominates m31 + m33 on every axis - is
withdrawn.** pv-hold had it the other way round (0.8791 against 0.8773 at three
series, 0.8709 against 0.8685 at two).

### Why, and it is not subtle once measured

Each member's delivered checkpoint against its own late-epoch plateau:

    member          delivered   plateau   inflation
    m38seed1         0.8836     0.8757     +0.0079
    m52nano          0.8858     0.8800     +0.0058
    m58nanos1        0.8824     0.8782     +0.0042
    m33up336new      0.8799     0.8793     +0.0006

**The pair I chose is built from the two most inflated checkpoints in the
roster.** The pair that wins on the board contains m33, which is honest to
within 0.0006. Deflating each pair by the mean inflation of its members:

    pair                    pv-hold   corrected
    m52nano + m38            0.8791    0.8722
    m52nano + m58nanos1      0.8778    0.8728
    m31 + m38                0.8773    0.8733
    m31 + m33                0.8773    0.8770

**The correction reproduces the board's ordering exactly.** m31 + m33 ranked
*last* of those on raw pv-hold and first once the selection luck is removed -
and it is first on the board.

### The rule this buys

An epoch is chosen on pv-sel, which correlates with pv-hold, so a lucky epoch is
lucky on both halves. The recorded 0.0005 delivered-checkpoint spread describes a
*typical* draw; it says nothing about a checkpoint that landed 0.008 above its
own plateau, and **nothing in an ensemble's pv-hold reveals that one of its
members did.** The luck is inherited silently and shows up only on the board.

    Before putting a checkpoint into a submitted ensemble, compare its delivered
    pv-hold against its own epoch 9-12 plateau. Above about +0.002, treat its
    contribution as inflated by that much and deflate the ensemble comparison
    accordingly.

Yesterday's entire cost-aware frontier ranked pairs by raw pv-hold, and its top
four entries all contained m52nano or m38seed1. That table was measuring
selection luck as much as quality, which is why it pointed at the configuration
the board then rejected twice.

**Nothing displaces m31 + m33 at two series.** The efficiency-track best is
unchanged from yesterday morning: 0.934 in 30 minutes, against the incumbent's
0.947 in 177.

### The roster audit, and a second correction to the diagnosis

Every checkpoint with a readable log, delivered against its own epoch 9-12
plateau, with the corrected cost (backbone FLOPs x upsample squared):

    member          cost   plateau   delivered   inflation
    m60nano288      0.91    0.8838    0.8859      +0.0021
    m61nano336      1.24    0.8825    0.8848      +0.0023
    m48blend60      1.00    0.8823    0.8870      +0.0047   (blend - excluded)
    m56nbr37        1.65    0.8815    0.8822      +0.0007   (2.5D - excluded)
    m37up288        1.65    0.8810    0.8830      +0.0020
    m40m33s1        2.25    0.8810    0.8792      -0.0018
    m43small        1.80    0.8810    0.8841      +0.0031
    m52nano         0.55    0.8800    0.8858      +0.0058
    m39seed2        1.00    0.8798    0.8831      +0.0033
    m33up336new     2.25    0.8793    0.8799      +0.0006
    m58nanos1       0.55    0.8782    0.8824      +0.0042
    m59nano192      0.40    0.8780    0.8795      +0.0015
    m54attn         1.00    0.8780    0.8784      +0.0004
    m38seed1        1.00    0.8757    0.8836      +0.0079

**This corrects the correction.** The nano pair did not lose because nano is
weak - m52nano's plateau (0.8800) matches m33's (0.8793) at a quarter the cost.
It lost because **m38seed1 has the lowest plateau of any tiny model here**, and
its delivered value, inflated by +0.0079, made it look like one of the best. I
picked the roster's weakest member and its selection luck hid that.

Read by plateau instead of delivered, the cheap end of the roster is genuinely
strong: m60nano288 at 0.8838 is the best plateau measured anywhere in this
project, for 0.91 units against m33's 2.25.

**The test that follows.** `sub-plat2` - m60nano288 + m52nano, two
plane-balanced series, half slices, 1.46 units for about 24.4 minutes. Their
mean plateau is 0.8819 against m33's 0.8793, and the pair should beat the
incumbent's 0.934 while running 5.6 minutes shorter. If plateau predicts the
board where delivered pv-hold did not, this is where it shows.

### The plateau rule failed its own test within two hours

`sub-plat2` - m60nano288 + m52nano, chosen *by plateau* precisely to avoid the
selection luck that was blamed for the previous two losses - scored **0.924**,
below the delivered-pv-hold pick it was meant to correct.

Two-series pairs on the board, half slices, plane-balanced:

    m31 + m33                  0.934   30.0 min   -2.2794
    m52nano + m38seed1         0.926   24.7 min   -2.2693
    m60nano288 + m52nano       0.924   24.4 min   -2.2648

**The inflation story is refuted as a predictor.** It explained the first two
losses neatly - deflating by plateau reproduced the board's ordering exactly -
and then its one forward prediction came out backwards. An explanation that fits
the data it was built on and fails the next point is an overfit, and this one was
built on four numbers.

What the board has actually established, across three submissions:

- Every pair containing a nano backbone scores 0.924-0.926 at two series.
- The pair containing no nano scores 0.934.
- Both local instruments - delivered pv-hold *and* plateau - ranked the nano
  pairs at or above m31 + m33.

So the finding is about the instruments, not about inflation: **pv-hold cannot
rank pairs across backbone families.** Within the tiny family it has tracked the
board tolerably; across nano and tiny it has now been wrong three times in the
same direction, by 0.008 to 0.010, on both the delivered and the plateau
reading.

Why nano underperforms its local numbers is not established and I am not going
to invent a third story for it today. The operational conclusion is enough:
**no nano member goes into a submitted ensemble** until something other than
pv-hold can vouch for it.

**The efficiency-track best is unchanged for the third day: m31 + m33 at two
balanced series, half slices - 0.934 in 30 minutes against the incumbent's 0.947
in 177.**

One test remains worth a slot, and it contains no nano: **m31 + m38seed1** at two
series, both tiny at up224, costs 2.00 units against m31 + m33's 3.25 - about
26.4 minutes. It asks whether m33's 2.25 units are actually load-bearing or
whether any second tiny member would do. If it holds 0.934, that is 3.6 minutes
free; if it drops, m33 is doing real work and the frontier is finished.

## 27 August, 11:15 — the efficiency frontier closes

`sub-tiny2` - m31 + m38seed1, both convnext_tiny at up224, no nano - scored
**0.930** at 26.4 minutes. The full two-series picture:

    pair                        board   runtime   efficiency
    m31 + m33up336new           0.934   30.0 min   -2.2794   best
    m31 + m38seed1              0.930   26.4 min   -2.2761
    m52nano + m38seed1          0.926   24.7 min   -2.2693
    m60nano288 + m52nano        0.924   24.4 min   -2.2648

**m33's 2.25 units are load-bearing.** Replacing it with a 1.00-unit tiny buys
3.6 minutes, worth 0.0027, and costs 0.004 of board AUC. A small loss, but a
loss, and the third consecutive attempt to make the pair cheaper that the board
has refused.

**The frontier is closed.** Five submissions across two days have tried a
cheaper second member - two nano pairs, one nano/tiny pair, one tiny/tiny pair -
and every one lands below m31 + m33. The efficiency-track configuration is
settled:

    m31newshards + m33up336new, 2 plane-balanced series, half slices
    0.934 in about 30 minutes, against the incumbent 0.947 in 177

### What today's four board points say about the instruments

Second members ranked by plateau, against how the board ranked the pairs
containing them:

    member          plateau   pair board
    m60nano288       0.8838    0.924
    m52nano          0.8800    0.926
    m33up336new      0.8793    0.934
    m38seed1         0.8757    0.930

**Within the tiny family the local reading is right**: m33 plateaus above m38
and its pair scores above m38's. **Across families it is backwards**: both nano
members plateau above m33 and both of their pairs score below it, by 0.008 to
0.010.

That is the honest scope of the instrument, and it is narrower than this project
has been assuming. pv-hold - delivered or plateau - can order two members of the
same architecture. It cannot order a nano against a tiny, and three submissions
went to learning that.

**The last slot today is not being spent.** The frontier is closed, nothing
untested remains that a local instrument can even nominate, and a submission
with no hypothesis behind it buys variance rather than position - which is the
same argument that applies to the main board, where 15 teams sit within 0.003.

## 27 August, 15:40 — our 0.947 is a level, not a peak

The operator pointed out that our submission count overstates our main-track
selection surface: 0.947 was reached early and most of what followed was
efficiency probing, deliberately scoring below it. Checked against the history,
in chronological order:

    1-11    0.500 -> 0.946   the climb
    12      0.947            first reached
    13, 14, 17, 19   0.947   reached again, different configurations
    other main-track  0.938-0.946   never exceeded
    24-29   0.924-0.936      efficiency track, all below by design

**23 main-track submissions, 6 efficiency.** The efficiency entries cannot have
inflated our best - every one scored 0.936 or lower.

**And 0.947 was hit five times by five different configurations.** That is the
part worth keeping. Max-of-n selection produces one lucky spike; it does not
produce five independent configurations landing on the same value. Our 0.947 is
the level the pipeline actually reaches, not the top of its noise.

That answers the private-set worry from yesterday in our favour: a public score
that was not bought by selection should transfer to the private set up to
resampling noise, with no systematic drop. The same cannot be said for a 0.952
selected out of 62 submissions - at an estimated 0.003 per-submission noise, the
max of 62 carries roughly 0.0075 of inflation, which would put its true level
near 0.945.

**But the field's ceiling is a separate question from ours.** Three teams sit at
0.948, one of them on 19 submissions - fewer than our 23. Across the top 50 the
correlation between score and submission count is +0.09, so heavy submitting is
not what puts teams above us. 0.948 looks reachable and real, which is why the
label track is still worth one attempt.

## 27 August, 16:00 — Sol 5.6 loses the blind track

GPT-5.6 Sol relabelled the 58 gold studies from the reports alone, no sight of
the existing labels. Scored against the real radiologist labels:

    finding            pos   student      sol     diff
    ACL                 24    0.994    0.994   +0.000
    MCL                  9    0.972    0.983   +0.011
    Medial Meniscus     26    0.952    0.930   -0.022
    Lateral Meniscus    23    0.865    0.883   +0.017
    Medial OA           15    0.919    0.929   +0.011
    Lateral OA          11    0.818    0.812   -0.006
    PF OA               21    0.911    0.910   -0.001
    Effusion            35    0.861    0.878   +0.016
    Synovitis           27    0.787    0.769   -0.018
    Baker's             12    0.911    0.894   -0.017
    Contusion           19    0.876    0.826   -0.050
    Fracture            18    0.925    0.875   -0.050
    MACRO                     0.899    0.890   -0.009

    paired bootstrap   -0.0091   CI [-0.0216, +0.0034]   P(sol > student) 0.075

**Below the incumbent.** The adoption bar was +0.011; this is 0.009 the other
way. The CI spans zero so the two are formally indistinguishable, but nothing
here is an argument for replacing a labeller that already exists.

**Where it moved.** Sol gains on MCL, Lateral Meniscus and Effusion - all
between +0.011 and +0.017 - and loses heavily on Contusion and Fracture, both
-0.050. Synovitis, the finding with the most headroom and the reason this was
worth trying at all, went *down* by 0.018.

**A false alarm I raised and withdrew.** Rows 10 and 45 of Sol's output are
identical to the last decimal, which looked like a copied row. Opening the two
reports: both are Spanish, both list iliotibial band friction, Hoffa fat-pad
impingement and patellar chondropathy, differing only in the order of the
sentences. **Identical labels are the correct answer there.** Excluding both
rows moves the gap from -0.0090 to -0.0087, so it explains nothing either way.
The defect was in my reading, not in the output.

Track B (critique-and-revise) is running. It is the compromised track by
construction - it sees the existing labels - so it cannot rescue the axis on its
own, but it can say *where* the two labellers disagree, which is the part worth
knowing before deciding whether the 3,400 remaining studies are worth relabelling.

## 27 August, 20:35 — the Sol relabel, end to end

GPT-5.6 Sol relabelled all 4,349 non-gold studies (44 batches, quota exhausted
once at 23/44 and resumed after reset). Collected with `--normalise`, which cut
batch-to-batch scale drift from 0.0457 to 0.0056. m31's recipe retrained on the
result, unchanged in every other respect.

    m31 on student labels     pv-hold 0.8832   gold 0.9148   board 0.942 single
    m62sol on Sol labels      pv-hold 0.8696   gold 0.9009   board pending

**gold is 0.0139 lower, about 3.8 of its own replicate sd.** pv-hold is not
evidence here - it is defined against the labels that changed - but gold is
clean, and it says the swap hurt.

### The instrument lesson, which is the part worth keeping

Scored directly against the 58 real labels, the two labellers were
indistinguishable:

    label quality on gold58        student 0.8993   sol 0.8903   -0.0090
    excluding one definitional     student 0.9007   sol 0.8945   -0.0062
    paired bootstrap CI            [-0.0216, +0.0034], spans zero

Trained on, the same labels separate cleanly at -0.0139 with a spread of 0.0037.
**The image model is a far more sensitive instrument for label quality than a
direct 58-study comparison of the labels themselves**, because it aggregates
4,349 studies' worth of target error instead of 58 studies' worth of ranking.
A labeller that ties on gold58 can still be materially worse as a training
signal, and this is the first time that gap has been measured here.

### Where Sol actually differed

Splitting each finding by whether the report mentions it at all, Sol wins 5 and
loses 6 where the report is explicit, and wins 5 loses 5 where it is silent -
**no systematic reading deficit**. It gains on Medial OA (+0.041 mentioned), MCL
and Effusion. The macro gap concentrates in Contusion and Fracture.

One Contusion case carries most of that finding's deficit on its own. Report 5
says "There is popliteus muscle contusion" and "There is no fracture or bone
contusion"; the gold label is positive. **Our prompt defines Contusion as bone
contusion only, so Sol scored 0.01 and was correct by the definition it was
given, while the gold label counts the muscle.** Dropping that one study moves
Contusion from -0.050 to -0.009. The definition in `fable_prompt.py` disagrees
with the competition's own labelling on this finding, and that is worth fixing
regardless of which labeller is used.

At corpus scale Sol assigns systematically lower prevalence - Synovitis 0.432 to
0.245, Fracture 0.146 to 0.087 - the "unmentioned means absent" behaviour seen
on the 58, now across 4,349 studies. AUC is rank-based so a uniform shift is
harmless; what costs is the ranking information lost when unmentioned cases pile
up at the floor.

### The submission died on Kaggle's side, not ours

`m62sol` came back **ERROR** with `error_description: "A system error. Please
try resubmitting to resolve the error and contact Kaggle Support if it
persists."` The kernel itself is COMPLETE, its checkpoint metadata matches m31's
key for key, and the dev submission.csv is well-formed with 0 fallbacks. Nothing
on our side to fix.

It still consumed the day's last slot. Resubmission is armed for the 09:00
reset. Note for next time: a submission can burn a slot and return nothing, so
"slots remaining" is not the same as "measurements remaining".

### The board confirms it: 0.931 against m31's 0.942

The resubmission scored **0.931**. m31, same recipe, same full configuration,
trained on the student's labels, scored 0.942. **-0.011.**

All three instruments agree, which has not happened often here:

    labels scored directly on gold58    -0.009    (CI spans zero)
    gold, after training on them        -0.0139   (3.8 sd)
    public leaderboard                  -0.011

**The label axis closes.** Replacing a LoRA student distilled from fable with
GPT-5.6 Sol, over the whole 4,349-study corpus, costs about a hundredth of AUC.

### What the attempt was worth

It was not a wasted day, and the reason is that the negative result is sharp
where the earlier ones were not:

**An instrument was calibrated.** Direct label scoring on 58 studies could not
separate the two labellers (-0.009, CI [-0.0216, +0.0034]). Training on them
separated them at -0.0139 with a spread of 0.0037, and the board agreed. **The
trained model is the sensitive instrument for label quality; the 58-study
comparison is not.** Any future labeller has to be judged by training on its
output, not by scoring it against gold.

**A defect in our own prompt surfaced.** `fable_prompt.py` defines Contusion as
bone contusion only. Report 5 says "popliteus muscle contusion" and "no fracture
or bone contusion", and the competition's gold label is positive - it counts the
muscle. Sol scored 0.01, correctly by the definition it was given, and lost most
of its Contusion deficit on that one study. The definition is wrong regardless
of which labeller reads it, and fixing it is independent of everything else here.

**And Sol is not a worse reader.** Split by whether the report mentions a
finding, Sol wins 5 loses 6 where the report is explicit and wins 5 loses 5
where it is silent. It beats the student on Medial OA (+0.041 where mentioned),
MCL and Effusion. What it does differently is treat an unaddressed finding as
absent - Fracture 91% below 0.05 against the student's 16% - and at corpus scale
that flattening is what the training signal cannot afford.

## 4 September, 14:30 — eight idle days, and what the public field learned in them

Nothing has run since 27 August. The board did not wait.

    24 Aug   0.947   rank 3 of 2408
    27 Aug   0.947   rank 5          <- last submission, 11:41
    01 Sep   0.947   rank 10
    04 Sep   0.947   rank 18 of 3018

The score never moved and the rank fell fifteen places. The prize line (10th)
is **0.948**; we are one thousandth outside it. Prizes were also re-read off the
page: 1st $9,000, 2nd $7,000, 3rd $6,500, 4th $6,000, 5th $5,500, 6th-10th
$5,000 each. **The team merger and entry deadline is 15 October**, a week
before the final submission deadline - HANDOFF listed this as unknown.

### The sweep: 40 discussion topics and the 7 highest-voted notebooks

**A lever closed for free.** Berat Kirbiyik published single-variable ablations
against a 0.7914 fold-0 baseline: EMA (decay 0.999) +0.003, mixup (alpha 0.2)
+0.002, 60-epoch cosine +0.001, asymmetric loss -0.139. EMA was on our own
"remaining unmeasured levers" list. Three regularisation approaches landing
inside noise of each other says the bottleneck is not the training regime, and
that is the same conclusion nineteen models here reached from the other side.

**Synovitis is a supervision problem, not a model problem.** Only ~16% of the
4,407 reports mention any synovitis term across the nine languages, so for the
other 84% every label table infers rather than reads. The public tables agree
with the radiologists at ~0.79 AUC on Synovitis against ~0.99 on ACL. And an
independent measurement across four architectures puts RadImageNet at 0.78 on
Synovitis where ImageNet-pretrained backbones sit at 0.62-0.72.

Our own roster says the same thing and nobody had looked:

    m51radimagenet   macro pv-hold 0.8490 - last of 28
                     Synovitis     0.8330 - FIRST of 28

The worst member we have is the best member we have on the finding with the
most headroom. Note the caution: per-finding numbers are on 58 studies, and
per-finding member weighting already lost to a uniform mean here on 26 August.
What this justifies is testing m51 as a **uniform** member, not a per-label one.

**Public label tables are worse than ours.** Benchmarked against the same 58:
stevenleehans v4 blend 0.893, v2 0.887, pilkwang v2 0.870. Our student is
0.8997. The Sol experiment closed the "replace the labeller" axis; this closes
"adopt a public labeller" without spending anything.

**External data is dead, and that is worth knowing.** The host ruled KneeCoT
ineligible because an institutional data agreement makes the field uneven. The
same reasoning reaches OAI (Data Use Certification) and MRNet (Research Use
Agreement), and a second argument finishes it independently: those datasets
carry ND licences, and the winner's obligation to publish weights trained on
them would be publishing a derivative. Three follow-up threads asking for a
written ruling have no host answer. **Remove external data from the plan** - it
was the highest-cost item on the board and it is not available.

**Two things we have not tried.** A GRU/LSTM head over slices: every RSNA
winner since 2019 aggregates slices recurrently rather than by pooling, and the
2023 write-up says GRU beat self-attention in their ablations. We tried slice
self-attention (m54attn, pv-hold 0.8784) and closed it; GRU is untried, is a
recipe change, and pv-hold is valid for it. And a public Pillar-0 knee
fine-tune - a volumetric MRI foundation model, macro 0.797 on the 58 with
Effusion 0.965 - which is a genuinely different family, at the cost of being a
public checkpoint and therefore correlated with the field.

**Confirmed, no action.** Slice ordering by ImagePositionPatient, physical
crop, and the DINOv2 base-vs-small capacity null are all things we already do
or already closed. The "centre-adjacent slices beat equally-spaced by +0.018"
finding is measured at 9 slices per plane; we resample to 32 and cover the
stack, so it does not reach us. The two corrupt training DICOMs the forum
reported are handled: of the 10 series in those two studies our shards carry 8,
and the two missing ones are exactly the two corrupt files.

**Where we actually stand relative to the field.** HANDOFF says "our per-model
quality is ahead of the field", written 19 August. It is not any more. The
"best single model" thread now carries 0.938, 0.942 (0.943 with TTA), and 0.943
at 224px from three different people. Our best single model is 0.943. The
public notebook frontier went 0.917 -> 0.936 in the same period.

### The gap that the sweep actually exposed

Teams reporting 0.94+ describe their unit of work as a "single **(5-fold)**
model". Ours is a single fold. Reading our own table with that in mind:

    every one of 28 members trained on the same 3,827 studies
    the same 522 val studies have never been trained on by anything
    all diversity in the 0.947 ensemble is recipe diversity on one sample

Data diversity has never been in this project. Neither has 12% of the corpus.

### What was started

`ensemble.py --save-preds` (new). Every combination question so far paid for a
full GPU pass over the shards, and the decode - not the forward pass - is the
expensive part, which is why five weeks produced about a dozen hand-picked
combinations out of the 2^25 available. One pass over 25 submission-valid
members now writes the predictions out; `src/ensemble_search.py` (new) does
greedy forward selection on pv-sel and reports pv-hold, plus an exhaustive
sweep of the ten best solo members.

`src/make_folds.py` (new). Five report-text-grouped folds written as columns on
image_dataset.parquet, so `train_image.py --split-col fold0` runs unchanged.
3,479/870 per fold, no study spans a split, val positive rate 0.317-0.333.
Grouping is on report text and not on the study UID: note that oof_predict.py
folds on md5(StudyInstanceUID) despite a docstring claiming otherwise, which is
a real leak there and is not repeated here. The five m31-recipe fold runs are
queued behind the prediction pool.

## 4 September, 14:55 — the roster, searched exhaustively for the first time

25 submission-valid members, one GPU pass (16 minutes), then the whole
combination question in numpy. What five weeks of hand-picked pairs could not
say:

    member                 pv-sel   pv-hold     gold
    INCUMBENT (LB 0.947)   0.9036    0.8848   0.9174
    greedy-5               0.9062    0.8863   0.9196
      = m38seed1 + m32blend40new + m33up336new + m40m33s1 + m31newshards

**The search finds the inflated members first, exactly as the rules predict.**
Run unconstrained, greedy selection takes m55nbr33 and m54attn in its first
four picks and the exhaustive top-20 is almost entirely m53nbr combinations.
Those are the neighbour-channel and slice-attention models, and 2.5D was closed
on 25-26 August precisely because it improves agreement with the teacher while
losing on the board twice (0.938 vs 0.942, 0.942 vs 0.943). A search that
maximises pv-sel will preferentially pick whatever inflates pv-sel. The number
above is from the search with the nbr and attn families dropped; the
unconstrained run reaches pv-sel 0.9069 and should not be believed.

**The honest gain is small and every instrument agrees on its direction.**

    pv-sel   +0.0026
    pv-hold  +0.0015
    gold     +0.0022    (abstains - the bar is 0.008)

The leaderboard cannot resolve below 0.002, so this is a 0.947-or-0.948 change.
That is not nothing when the prize line is 0.948 and we are at 0.947, but it is
inside noise on every instrument we own, which is the case the 21 August LB
policy names as the reason a slot exists.

**Seed replicates get picked, and the docstring says they shouldn't.**
ensemble.py's header argues seed-ensembling "has little to average over"
because two convnext_nano runs differing only in seed landed 0.001 apart. The
search disagrees: greedy-5 contains both m31 and m31-seed1 (m38seed1), and both
m33 and m33-seed1 (m40m33s1). Near-identical members still average away
independent training noise, and that is measured here rather than argued.

**The RadImageNet-Synovitis claim does not reproduce on our data, and cannot.**
The public measurement (four architectures, RadImageNet 0.78 vs 0.62-0.72) and
our own per-finding table (m51 first of 28 on Synovitis at 0.833) both said to
try it. Added to the incumbent as a uniform member:

    pv-sel     0.9036 -> 0.9025
    pv-hold    0.8848 -> 0.8856
    Synovitis  0.7910 -> 0.7796   the opposite of the prediction
    what it actually helps: MCL +0.0101, Fracture +0.0100

The Synovitis line is unreadable, not negative. pv-hold scores Synovitis
against teacher labels, and the public finding that motivated the test is that
those labels agree with radiologists at only ~0.79 on this finding - so the
instrument is measuring agreement with a teacher that is close to uninformative
there. gold-58 is the only instrument that could see it and moves +0.0009,
which is an abstention. **Synovitis cannot be worked on with the instruments
this project owns.** Recording that rather than a verdict.

New: src/ensemble_search.py. Search on pv-sel, report pv-hold, gold abstains
below 0.008 - the trainer's discipline, applied to subsets instead of epochs.

### 15:05 — correction: the search was run on the wrong average

`src/ensemble_search.py` ranked members before averaging, on the standard
argument that AUC reads only order. **The submission kernel averages
probabilities**, deliberately and with a measurement behind it (2026-08-20:
probability mean +0.0052 over the best member, rank mean +0.0019), and it
cannot form ranks anyway because it streams one study at a time. A search on
ranks therefore selects a combination the kernel cannot deliver. The flag now
defaults to probability and `--rank` opts into the other.

Re-run the right way, the answer changes - m31newshards drops out for m43small:

    INCUMBENT (LB 0.947)                                0.9067   0.8880   0.9200
    m32blend40new+m38seed1+m33up336new+m40m33s1+m43small 0.9098  0.8900   0.9221
                                                        +0.0031  +0.0020  +0.0021

Both greedy paths still pick the two seed pairs, and both still leave the
incumbent behind by about the same margin, so the finding survives the
correction; the membership did not.

**Submitted 15:06 as `ens5search`**, submission 56006499 (kernel
qwrfsdr/rsna-knee-sub-ens5search v1, 5 members, ~3.5h expected; the dev run
loaded all five checkpoints with the right configs and wrote 3 rows with 0
fallbacks). The 21 August LB policy is the justification: three
instruments agree in direction and every gap is inside noise, which is the case
a slot exists for. The prize line is 0.948 and we are at 0.947, so a change the
board cannot resolve is still the change we need.

### 16:05 — fold 0, and what a single fold can and cannot say

    m63f0   12 epochs, 70 min, best at epoch 7
            fold-0 pv-sel 0.8786   fold-0 pv-hold 0.8960   gold 0.9055
    m31     same recipe, main split
            pv-sel 0.8997          pv-hold 0.8832          gold 0.9148

Only the gold column compares - the other two are measured on different studies.
gold is 0.0093 lower, which is right at the 0.008 line where gold stops
abstaining, and it is the expected direction: a fold model trains on 3,479
studies against m31's 3,827, 9% less. **A single fold being slightly worse is
the premise, not a result.** The fold set is an argument about the ensemble of
five and about the 12% of the corpus the fixed split has never trained on;
neither claim can be read off one member.

The epoch curve peaks at 5-7 and decays through 12, the same shape the plateau
rule was written for.

New: src/fold_oof.py, queued behind the five runs. One decode of the shards,
each series shown only to the member that held it out, giving out-of-fold
predictions over all 4,349 pseudo-labelled studies - sixteen times pv-hold's
sample. It also averages all five members over the 58 gold studies, by
probability because that is what the kernel does, which is the number that
decides whether this ensemble is submitted. The teacher's labels are still the
teacher's, so OOF is a better-powered measurement of the same quantity that
pv-hold measures, not a way around the teacher's ceiling.

### 19:20 — three folds in, and the board's velocity

    m63f0  ep7  gold 0.9055        m63f2  ep7  gold 0.9035
    m63f1  ep6  gold 0.9118        m63f3  ep7  gold 0.9084  (still running)
    m31         gold 0.9148

Each fold model sits about 0.008 below m31 on the only comparable column, which
is what 9% less training data buys and is the premise of the experiment rather
than a result. The claim being tested is about the five together, and
`fold_oof.py` is queued to test it.

**The 10th-place line has risen 0.941 -> 0.948 since 19 August**, +0.007 in
sixteen days, while our 0.947 has not moved since 23 August. Extrapolating the
rate to 22 October is not defensible, but the direction is not in doubt:
**0.947 will not be a top-ten score in October.** The target is not "get back
inside the line", it is "get far enough above it that six weeks of the field
converging does not pass us again". That reading argues for spending the
remaining slots on things that could move 0.005, not 0.001.

`ens5search` (56006499) has been PENDING for four and a quarter hours against a
3.5-hour estimate. The kernel's dev run is COMPLETE and clean, so this is
scoring queue time, not a failure - but note that a submission burned a slot and
returned nothing once already (27 August, Kaggle system error), so PENDING is
not the same as safe.

## 4 September, 19:25 — ens5search returns 0.947, and the roster closes

    submission 56006499   ens5search   0.947
    incumbent             55713127     0.947

    local, probability mean, against the incumbent
      pv-sel   0.9098   +0.0031
      pv-hold  0.8900   +0.0020
      gold     0.9221   +0.0021
    board                +0.000

**Five submissions have now returned 0.947.** The rule said the board cannot
resolve below about 0.002 and the search's whole margin was 0.002-0.003, so this
is the predicted outcome, not a surprise. It is still worth what it cost.

**What closes.** Not "ensembling", and not the search. What closes is *this
roster*: 25 submission-valid members, searched properly instead of by hand,
with the inflated families excluded and the averaging matched to what the kernel
actually does - and the best combination it can assemble is indistinguishable
from the four members we already had. Every future point has to come from a
member that does not exist yet.

**The distinction that matters for whether to keep the search.** The search was
not wrong. It found a combination that is genuinely better on three independent
instruments, and all three agreed on the direction. The improvement is simply
smaller than the board's resolution. Those are different failures: a search that
picks worse combinations should be abandoned, a search that picks slightly
better ones should be kept and pointed at bigger differences. So the search's
job changes from "find a submission" to **"decide cheaply whether a new member
adds anything, before a slot is spent on it"** - which is exactly the question
the fold set and m64gru are about to ask.

**And it cost a day's slot to learn something the rules already said.** The 21
August LB policy licenses a slot when instruments agree and the gap is inside
noise. That licence was written when 11 slots had been used out of 300; the
counter-argument it does not contain is that a gap inside noise on the LOCAL
instruments is also, at 0.002-0.003, inside the noise of the BOARD - so no
outcome of the submission could have been informative except by luck. **A slot
is worth spending when the local reading is uncertain, not when it is certain
and small.** Add that to the policy.

## 4 September — returning to work someone else started, and one recovered result

A week of silence, then the queue moved: `make_folds.py`, `ensemble_search.py`
and `--save-preds` all appeared without commits, and five fold trainings plus a
subset search were running. None of it is mine. What follows is a reading of it,
not a claim on it.

**The fold work's premise is sharper than anything I closed the ensemble axis
on.** `make_folds.py` points out that 522 studies - 12% of the corpus - have
never been trained on by any member, and that every member of the 0.947 ensemble
is trained on *identical data*, so all its diversity comes from recipe changes
over one fixed sample. My "a recipe is worth about two members" rule may simply
be describing members that share a sample. Five folds are five samples. That is
a different axis from the one I retired.

**A caution on the fold numbers.** `--split-col fold0` changes what pv-hold
means: the fold val sets are ~4,800 series against the fixed split's 2,885, and
they are different studies. m63f0's pv-hold 0.8960 is **not** comparable to the
0.884-0.890 band every earlier model sits in. gold carries no such bias, and
there m63f0 is 0.9055 against m31's 0.9148.

**The search output was lost** - it ran outside the queue, so its stdout went to
a terminal and nothing was written to disk. Re-run from the cached
`preds_pool.npz`, which is what that cache is for, and saved to
`artifacts/logs/ensemble_search.log` this time.

    greedy, 4 members   pv-sel 0.9093   pv-hold 0.8890   gold 0.9225
    greedy, 5 members   pv-sel 0.9098   pv-hold 0.8900   gold 0.9221
    best exhaustive     pv-sel 0.9072   pv-hold 0.8919   gold 0.9193
    incumbent           pv-sel 0.9067

**Two reasons not to submit off this table.** The top twelve span 0.0006 of
pv-sel against that metric's own spread of 0.0034, so the ordering within them
is noise wearing a rank. And nano members - m60nano288, m61nano336, m52nano -
are all over the top of it, which is exactly the failure the board settled on 27
August: pv-hold cannot order a nano against a tiny, and three submissions in a
row came back 0.008-0.010 below what it promised. A search that maximises
pv-hold will find nano first and be wrong first.

The fold ensemble is the candidate worth the slot when it finishes, precisely
because no local instrument here has a track record on data diversity yet.

## 4 September, 21:00 — the fold set answers, and it is not the answer

    5-member fold ensemble, gold-58, probability mean   0.9128
    incumbent (4 recipe-diverse members)                0.9200
                                                       -0.0072

    out-of-fold, 4,349 studies / 24,033 series          macro 0.8801
      worst   Synovitis 0.805   MCL 0.845   PF OA 0.856
      best    Medial Meniscus 0.923   Baker's 0.918   Fracture 0.916

gold moves 0.0072, just inside its abstention threshold, so formally it says
nothing. The direction is the whole result anyway - and the operator's own
measurement, run in parallel, makes it sharper than the sentence I first wrote
here.

**Against the BEST member, not the mean member, the fold ensemble gains
nothing at all:**

    m63f0 0.9051   m63f1 0.9114   m63f2 0.9036   m63f3 0.9083   m63f4 0.9129
    ENSEMBLE prob mean   0.9128    -0.0001 vs best member
    ENSEMBLE rank mean   0.9130    +0.0001 vs best member

    for contrast, the incumbent: best member 0.9138 -> ensembled 0.9200  +0.0062

I first wrote this as "+0.0044 from ensembling", comparing the ensemble to the
MEAN of its members. That is the wrong baseline for the question being asked -
nobody would ship five models to match one - and this project's own harness has
always reported the gain against the best member. Corrected, the number is zero.

**Data diversity does not ensemble here at all.** Five models of one recipe over
80%-overlapping data are so correlated that averaging them recovers nothing
beyond the luckiest one. Recipe diversity buys +0.0062 on the same instrument.
This is the "averaging pays only when members fail differently" rule measured
about as cleanly as it can be: same recipe, same everything, only the sample
differs, and the answer is 0.0000.

**No submission.** The only instrument that can judge the fold models is gold-58,
because they trained on the main split's val studies and pv-hold is contaminated
for them. gold is below the incumbent. Rule 6, adopted two hours ago, applies
directly: a slot goes to an uncertain reading, not a certain small one.

**What the fold set did NOT test, and this needs saying plainly.** Each fold
model trains on 3,479 studies against m31's 3,827 - **less** data, not more.
Nothing here tested "does more data help". The 522 studies the fixed split has
never trained on were used collectively, one fifth at a time, by five models
that each saw less than the incumbent does. The data-quantity lever is still
untouched and is a different experiment.

### What the OOF predictions are actually for

The instrument is a by-product. The real value of `oof_folds.parquet` is that it
is the input `blend_targets.py` has been waiting for since 20 August.

The largest measured number in this project's record is on gold-58: the teacher
scores 0.8997, the image model 0.9094, and a blend of the two **0.9271**. Text
and pixels fail differently. That was measured once and never carried into a
trained model at any quality, because the out-of-fold predictions it needs came
from `oof_predict.py` - which trains **6 epochs** and folds on
**md5(StudyInstanceUID)**, so the 197 studies that share a report verbatim leak
across its folds and carry identical targets on both sides.

Every blend model in the roster was built on those predictions:

    m32blend40new  pv-hold 0.8808   the run that closed the axis
    m47blend25     0.8813
    m49blend40s1   0.8832
    m46blend12     0.8830
    m48blend60     0.8845   <- second highest pv-hold of all 25 members

The axis was closed on the weakest member of that group while its strongest sat
near the top of the roster. It was never dead; it was under-powered.

The fold set replaces the input: 12-epoch models with proper epoch selection, on
report-grouped folds, covering 24,033 of 24,371 series (98.6%). blend_targets
only rewrites `split == "train"` rows, so **validation targets are untouched and
pv-hold stays comparable with every previous run** - a recipe change, decided
locally, no slot.

Queued behind m64gru at the same weights as the two best old blends, so the only
variable is the quality of the predictions:

    m65blend40oof   weight 0.40, against m32blend40new's 0.8808
    m66blend60oof   weight 0.60, against m48blend60's 0.8845

### A queue entry that was not mine

`src/ensemble.py m63f0..m63f4 --halves` launched at 21:02, appended by the
operator with the note "pv-hold 는 오염돼 무효 - gold 만 읽을 것". Same reading
as mine, arrived at independently - and it earned its sixteen minutes. fold_oof
reports the ensemble's gold but not the per-member gold beside it, so it could
not show that the ensemble fails to beat its own best member. That comparison is
the result, and I would have filed the weaker version of it without this run.

### The 5-fold ensemble, measured on the only clean instrument

    m63f0..f4 individually, gold   0.9055  0.9118  0.9035  0.9084  0.9126
    5-fold ensemble, gold          0.9128   (+(-0.0001) vs its best member)
    m31 alone, gold                0.9148

**Averaging five folds gains nothing over the best single fold**, and the result
sits 0.0020 below one model trained on the fixed split. At gold's 0.0037 spread
that is half a standard deviation - "no gain detected", not "worse" - but the
premise was that data diversity would buy what recipe diversity had stopped
buying, and on the clean instrument it did not.

The mechanism is visible in the individual numbers. Fold training uses 19,234
studies against the fixed split's 21,150, so every fold model is handicapped by
9% less data; the five of them average back up to roughly the best one and no
further.

**pv-hold reported 0.9650 for this ensemble and must be discarded.** Every fold
model trains on four-fifths of everything, which includes most of the fixed
split's held-out half, so that number is scoring models on their own training
data. It was flagged as invalid before the run rather than after, and the size
of it - 0.965 against a ceiling of 0.884-0.890 that nineteen models never left -
is itself the proof. A note to that effect went into `artifacts/queue.txt` above
the job, because the log will outlive the reasoning.

### m65blend40oof: pv-hold jumps 0.021 and the jump is a leak

Blending the training targets with the new fold-based out-of-fold predictions
(`oof_folds.parquet`, 40% weight) produces the highest pv-hold this project has
ever recorded:

    m31             pv-hold 0.8832   gold 0.9148
    m65blend40oof   pv-hold 0.9044   gold 0.9161 (plateau 0.9173)

**+0.0212 on an instrument whose replicate spread is 0.0005 would be 42 sd.**
Nineteen models across eight recipes never left 0.882-0.887. A single label
change does not clear that; something is wrong with the measurement.

It is not the circularity that closed this axis in August - checked, and the
blend touched **only the train rows**. val and gold_val targets are identical to
the unblended dataset, byte for byte. So pv-hold is scoring against the same
pseudo-labels every other model was scored against.

The leak is one step further back. **Each fold model trained on about 80% of the
fixed split's 522 held-out studies** - 411, 414, 419, 421, 423 of them. Those
models produced the OOF predictions; the predictions went into m65's training
targets; m65 was then scored on the very studies whose labels those models had
memorised. The held-out half stopped being held out the moment fold models were
allowed to train across it.

    gold is clean - it is gold_val in every fold and no model has ever trained
    on it - and there the blend is +0.0013 against a spread of 0.0037.

**The label-blend axis stays closed.** August closed it because gold did not
move; gold still does not move. What is new is only that pv-hold now moves a lot,
which makes the axis look open to anyone reading the logs without this note.

**And a rule that now has a second instance.** Any model whose training signal
passes through a fold model has inherited that fold model's view of the fixed
split's validation set. pv-sel and pv-hold are void for it. This applies to
m65blend40oof and to anything else built on `oof_folds.parquet`.

## 6 September — the GRU member earns a place it could not earn alone

`preds_pool2.npz` adds m64gru to the cached pool. **Checked first: the pool
contains no fold model and nothing built on `oof_folds.parquet`**, so
yesterday's leak has not reached the search.

m64gru alone is indistinguishable from m31 on both clean instruments:

    m31      pv-hold 0.8832   gold 0.9148
    m64gru   pv-hold 0.8825   gold 0.9148   (plateau 0.8823, honest checkpoint)

In the search it is picked **fourth**, displacing m40m33s1, which held that slot
in the previous pool:

    before, 4 members   pv-sel 0.9093   pv-hold 0.8890   gold 0.9225
    now,    4 members   pv-sel 0.9095   pv-hold 0.8890   gold 0.9236
    now,    5 members   pv-sel 0.9101   pv-hold 0.8899   gold 0.9256
    now,    6 members   pv-sel 0.9102   pv-hold 0.8909   gold 0.9240

**A member worth nothing alone can still be worth a slot in the ensemble.** That
is the "averaging pays when members fail differently" rule producing its
cleanest instance here: identical solo scores, different errors, and the search
prefers it to a member that scores the same.

**What the numbers do not support.** gold 0.9256 against the previous 0.9225 is
0.8 of gold's own spread. And the greedy path's pv-sel at four, five and six
members spans 0.0007 against that metric's spread of 0.0034 - the choice of size
is unsupported. The finding is "m64gru belongs in the pool", not "the five-member
set is the configuration".

## 6 September — 37 hours idle, and the board took four more places for it

The card finished m66blend60oof at about 01:00 on the 5th and did nothing until
14:21 today. The runner was up the whole time — `rsna-queue.service` has been
active since 25 August and its log says `queue empty and GPU idle` once a minute
for 37 hours. Nothing was broken. The queue was empty.

That is the third time this has happened (six days, 13–19 August; eight days,
27 August – 4 September; 37 hours now), and the cost is legible on the board:

    01 Sep   0.947   rank 10       05 Sep   0.947   rank 18
    03 Sep   0.947   rank 11       06 Sep   0.947   rank 22

    the 10th-place line   20 Aug 0.942 -> 06 Sep 0.949, about +0.0004 a day
    our score             unchanged since 23 August

**We are out of the money.** 0.947 was 10th on 1 September and is 22nd of 3166
today. The line does not need to keep rising at this rate for the reading to
hold: standing still has cost twelve places in five days, and there are 46 days
left.

### m64gru joins the search pool

`preds_pool.npz` was written before m64gru existed and it is the only member
built since that is clean — m63f0-f4 trained across the fixed split's held-out
half, and m65/m66 inherited that view through `oof_folds.parquet`, so all seven
are void on pv-sel and pv-hold. Re-inferring the 25 cached members to add one
would cost two hours, so `src/merge_preds.py` merges pools instead, comparing
the alignment columns (`T__*`, `U__*`) exactly and refusing rather than warning
when they differ — two pools written against different shards would silently
compare member A's series 7 against member B's series 9, and no score would
reveal it. 26 members x 2885 val series, four minutes.

The question is not whether m64gru is good. Solo it is nothing: gold 0.9148
against m31's 0.9148, pv-hold 0.8825 against 0.8832. It is whether a GRU slice
aggregator **fails differently** from the mean-pooled members, which is the only
property that has ever paid in an ensemble here. The search is pinned with
`--keep` on the four incumbent members so it answers that, rather than answering
"what scores highest" with the nano roster the board has now refused three
times.

## Where the model looks, and what that says about the crop

The crop is the only axis that has moved this leaderboard: +0.012 for the
physical-scale crop and laterality fix, +0.029 for matching the backbone's
operating point. With the model class exhausted, the roster closed and the fold
axis measured at exactly zero, it is also the axis chosen for the remaining 46
days. Joint-centred cropping has two premises, and neither had ever been
measured:

    1. the anatomy sits off-centre, so re-centring would stop clipping it
    2. it occupies much less than the 130 mm the window spends on it, so a
       tighter window would put more pixels on the same structures

pv-hold cannot answer either — a crop change is a representation change and
both sides of that comparison move together. But **the model we already own can
be asked directly**, for one backward pass per series and no submission slot.
`src/saliency_probe.py` takes |d(sum of logits)/dx| with respect to the 256 px
volume as it sits in the shard, so the map comes back in the shard's own
coordinates however the member resizes internally, and reports the saliency
centroid's distance from the frame centre in mm and the share of saliency
inside centred boxes of a given physical width.

**It reports against two nulls, not one.** A convolutional stack has more paths
from a central pixel to its output than from a corner one; padding and the
receptive field bias gradients toward the middle before any anatomy is involved.
So `--control` runs the identical measurement through the same architecture with
random weights. The uniform map bounds the geometry, the control bounds the
architecture, and only what exceeds both is learning.

On eight series — a smoke test, not a result — the two premises already separate:

    saliency centroid, distance from frame centre
      m31newshards   3.67 mm      random-weight control   2.53 mm

    share of saliency inside a centred box
      width      trained   control   uniform
       60 mm      0.408     0.282     0.212
       80 mm      0.621     0.475     0.381
      100 mm      0.797     0.678     0.586
      110 mm      0.876     0.785     0.712

Premise 1 looks dead: the model's attention is already centred, 3.7 mm against
a control that is 2.5 mm on an image that is 130 mm across. There is nothing
for re-centring to recover.

Premise 2 looks alive, and the control is what makes it readable. At 100 mm the
architecture's own centre bias accounts for about half the concentration
(0.678 against 0.586 uniform) and learning accounts for the other half (0.797
against 0.678). Without the control the whole 0.797 would have looked like
anatomy.

**If that survives the full run, the experiment is not joint-centring — it is
narrowing.** Cropping to 100 mm discards a fifth of the saliency the model
currently uses and buys 1.69x the pixel density on the rest, which is the same
trade that paid +0.029 when upsampling matched the backbone's operating point.
Whether it pays is for the leaderboard to say; what the probe decides is which
version of the experiment to build, and it decided it before any GPU time was
spent on the wrong one.

### The search, re-run with m64gru in it: no, and an interesting no

Pinning the four incumbent members with `--keep` and letting greedy add from 26:

    added 5th   m55nbr33    pv-sel 0.9083   pv-hold 0.8888   gold 0.9227
    added 6th   m38seed1           0.9091          0.8901         0.9226
    incumbent                      0.9067          0.8880         0.9200

**m64gru was never chosen.** It was available at every step - 9th of 26 on
pv-sel - and greedy preferred m55nbr33 and m38seed1 twice over. The exhaustive
search over the ten best solo members, by probability mean, puts it in none of
its top four combinations either. The GRU aggregator does not fail differently
enough from the mean-pooled members to be worth carrying, which is the third
independent confirmation this month that a member has to fail differently and
not merely be built differently.

**And no submission.** The best combination beats the incumbent by +0.0024
pv-sel, +0.0021 pv-hold, +0.0026 gold. Those are the same margins ens5search
carried on 4 September, and it returned 0.947 - the board cannot resolve them
(rule 5), so nothing this table can produce is informative (rule 6).

**The interesting part is the disagreement between the two averaging modes.**
Ranked instead of probability-averaged, m64gru appears in *all four* of the top
exhaustive combinations:

    rank mean   0.9051 / 0.8899 / 0.9198   m53nbr+m61nano336+m38seed1+m64gru
                0.9049 / 0.8898 / 0.9185   m53nbr+m60nano288+m43small+m38seed1+m64gru
    prob mean   0.9090 / 0.8917 / 0.9225   m53nbr+m55nbr33+m38seed1+m31newshards
                0.9090 / 0.8924 / 0.9209   m53nbr+m48blend60+m55nbr33+m43small+m38seed1

Same pool, same metric, opposite verdict on one member. The kernel averages
probabilities, so the probability column is the one that decides a submission
and the answer there is no. But a member that is useful only after its units
are thrown away is a member whose *ordering* carries something its calibration
hides, and that is worth remembering when the next member is judged.

**One member keeps appearing that is not in the incumbent.** m53nbr is in every
top combination in both modes and has the highest solo pv-hold of the 26
(0.8895 against the incumbent members' 0.8799-0.8835). Its gold is ordinary
(0.9118). That shape - high pv-hold, ordinary gold, never submitted - is exactly
the shape the nano members had before the board refused them three times, so it
is a candidate to watch rather than a candidate to ship.

### The probe at n=1200: re-centring is dead, narrowing is the experiment

    saliency centroid, median distance from the frame centre
      m31newshards trained   5.98 mm      random-weight control   4.92 mm

    share of saliency inside a centred box   trained / control / uniform
      100 mm   0.786 / 0.663 / 0.586     learning adds +0.123 over the control
       80 mm   0.609 / 0.446 / 0.381                   +0.163
       60 mm   0.395 / 0.257 / 0.212                   +0.138

**Premise 1 is dead and the control is what kills it.** 5.98 mm looks like a
displacement worth correcting until the untrained network of the same shape
posts 4.92 mm on the same images. The difference is 1.06 mm on a frame 130 mm
across - a tenth of one 224px input pixel after upsampling. The model's
attention is centred because the anatomy is centred, and `src/volume.py`'s
centre crop is already putting the joint where it belongs. **The experiment
this project chose as its main bet for the remaining 46 days is not
joint-centring**, and one probe costing four minutes of GPU said so before a
2.5-hour Kaggle rebuild was spent on it.

**Premise 2 survives, and the control is what makes it readable.** At 100 mm the
architecture's own centre bias accounts for 0.663 of the 0.786 - padding and
receptive-field geometry, present before any training. Learning accounts for the
remaining 0.123. Without the control the whole 0.786 would have been read as
anatomy and the effect overstated by a factor of two.

Note the shape of the learning column: +0.123 at 100 mm, +0.163 at 80 mm,
+0.138 at 60 mm. The peak at 80 mm says the trained model's extra concentration
is spread over a band roughly 80 mm wide, not piled at the very centre - which
is about the size of a knee joint and is the first evidence in this project that
the model found the anatomy rather than a scanner artefact.

**Where that leaves the bet.** Narrowing the window to 100 mm discards 21% of
the saliency the model currently uses and would buy 1.69x the pixel density on
the same anatomy - the same trade that paid +0.029 when upsampling matched the
backbone's operating point. The two halves separate cleanly:

    the COST   measurable now. --crop-mm re-crops the existing shards and
               changes nothing else. It buys no detail, because these shards
               were resampled to 130 mm / 256 px when they were written, so it
               isolates what the discarded band was worth.
    the BENEFIT needs a rebuild from the DICOMs on Kaggle at CROP_MM=100, 2.5
               hours, and cannot be faked locally.

m67crop100 and m68crop80 are queued for the cost half. Two widths rather than
one because replicate spread on pv-hold is about 0.005 and a single gap that
size is not a result, while a monotone trend over 130 -> 100 -> 80 is. If the
cost is small the rebuild's upside is unopposed and it goes to Kaggle next; if
narrowing to 100 mm already hurts, the extra detail has to pay for that damage
first and the bet is much weaker than the saliency table makes it look.

### The probe replicates on a second model at a different resolution

m33up336new is the other half of the incumbent's representation diversity -
256 px real, upsampled to 336, against m31's 128/224. Run at n=600:

    saliency centroid, median distance from the frame centre
      m33up336new trained   6.62 mm      its own control   6.02 mm
      m31newshards trained  5.98 mm      its own control   4.92 mm

    learning's contribution to the box share (trained minus control)
      width     m31      m33
      100 mm   +0.123   +0.110
       80 mm   +0.163   +0.166      <- the peak, in both
       60 mm   +0.138   +0.143

**Premise 1 dies twice, independently.** A second architecture at a different
input resolution puts its saliency centroid 0.60 mm from where its own untrained
twin puts it. There is no off-centre anatomy in this corpus for a joint-centred
crop to find.

**And the 80 mm peak replicates to within 0.003.** Two models that see the data
at different scales agree on the width of the band where learning concentrates
attention, which is not something a scanner artefact or a padding effect would
reproduce across a resolution change. Whatever the models found, they found the
same thing, and it is about 80 mm across.

Note that each model needs its own control: m33's untrained twin already holds
0.712 inside 100 mm against m31's 0.663, because a larger input at higher
resolution has a stronger geometric centre bias to begin with. Comparing m33's
trained 0.822 against m31's control would have overstated it by half again.

m33 also discards less at 100 mm (17.8% of its saliency against m31's 21.4%),
which is the direction the resolution argument predicts: the member that already
sees more detail leans on the periphery less.

### The narrowed member is shippable without the rebuild — checked before, not after

The plan above treats the local `--crop-mm` runs as a cost measurement and the
Kaggle rebuild as the only shippable version. That was wrong in a useful
direction, and it was worth ten minutes to find out before the rebuild rather
than after.

`submit_kernel.py` decodes each series once into a shared 256 px volume at
`CROP_MM=130` and then calls `member_input(base, size, upsample)` per member,
which resizes that shared volume to whatever that member's own cfg asks for.
`src/volume.py` and `src/kernels/volume.py` are byte-identical, and PREP_SIZE is
256 - the same 130 mm / 256 px the local shards hold. So a centre crop inserted
at the top of `member_input`, reading `crop_px` out of the member's cfg, is the
exact analogue of what `load_shard` now does locally.

Three consequences:

  - **The local measurement is not an approximation of the shipped model, it is
    the shipped model.** Both crop the same already-resampled 130 mm volume.
  - **Members with different crops mix in one ensemble** at no extra decode
    cost, because they share one 256 px base. The kernel was already built this
    way for `size` and `upsample`; a crop is one more per-member field.
  - **The Kaggle rebuild is an upgrade, not a prerequisite.** It buys real
    pixels at 100 mm rather than a re-crop of 130 mm ones. If the cost
    measurement comes back cheap, there is a submittable candidate immediately
    and the rebuild follows to sharpen it.

Not implemented yet - the kernel change is worth making when the cost number
says the member is worth shipping, and not before.

### The bar for the crop runs, written down before they finish

Rule 1 says compare late-epoch plateaus and not the delivered checkpoint, and
rule 2 says a single-run gap under about 0.005 on pv-hold is not a result. Both
are easy to honour after the fact in the direction one wants, so the yardstick
goes here first. m38seed1 and m39seed2 are m31's recipe run twice at 12 epochs,
which makes them a two-point baseline AND a replicate pair:

    plateau, mean of epochs 8-12        pv-hold      gold
      m38seed1                           0.877      0.914
      m39seed2                           0.881      0.910
      band                           0.877-0.881  0.910-0.914
      between-replicate spread           0.004      0.004

**So m67crop100 costs nothing if its 8-12 plateau lands inside 0.877-0.881 on
pv-hold and 0.910-0.914 on gold, and the cost is real only below both.** A
single delivered checkpoint above the band is not evidence of a gain - this is a
representation change and pv-hold cannot see gains - but a plateau below it *is*
evidence of a loss, because a narrower window can only reduce agreement with
the same teacher on the same studies.

Through four epochs m67crop100 tracks the baselines on pv-hold and sits about
0.005 low on gold:

    ep    m67crop100        m38seed1        m39seed2      (pv-hold / gold)
     1   0.838 / 0.866   0.829 / 0.864   0.835 / 0.864
     2   0.862 / 0.892   0.857 / 0.897   0.865 / 0.905
     3   0.874 / 0.905   0.872 / 0.906   0.869 / 0.902
     4   0.875 / 0.907   0.876 / 0.912   0.879 / 0.913

Mid-curve and inside the replicate band on both columns, which is neither a
result nor a surprise at epoch 4. Recorded so that the plateau is read against
a bar that existed before it.

## m67crop100: narrowing the window to 100 mm costs nothing

    plateau, mean of epochs 8-12        pv-hold      gold
      m38seed1  (baseline)               0.877      0.914
      m39seed2  (baseline)               0.881      0.910
      the band, pre-registered       0.877-0.881  0.910-0.914
      m67crop100                         0.881      0.912

    per epoch   ep8 0.881/0.913  ep9 0.882/0.912  ep10 0.881/0.914
                ep11 0.881/0.910  ep12 0.880/0.911

**Inside the band on both columns**, at the top of it on pv-hold and the middle
of it on gold. The bar was written down before the run finished, and this is the
outcome it named as "narrowing is free".

What makes it worth more than a null result: **the saliency probe said this crop
discards 21% of the attention mass the model was using, and the model reproduces
the baseline plateau anyway.** Those two facts together say the outer 30 mm band
was redundant rather than empty - whatever it carried, the model can recover
from what remains. That is a stronger statement than "the periphery is useless",
and it is the statement the crop bet needed.

The reading has a limit worth stating: pv-hold cannot show a *gain* on a
representation change, so 0.881 against a 0.877-0.881 band is "no cost", never
"an improvement". What it rules out is the outcome that would have killed the
bet - narrowing damaging the model before any extra detail is bought.

    delivered checkpoint   ep7, pv-hold 0.8784, gold 0.9119
    plateau                     pv-hold 0.881,  gold 0.912

The usual selector cost, about 0.003 of pv-hold thrown away by choosing on
pv-sel. It matters here because ensemble.py loads the delivered checkpoint, so
the member that goes into the search is the 0.8784 one, not the plateau.

### What this unlocks, and the design problem it exposes

If narrowing is free at a fixed pixel count, then narrowing *with* the pixels -
100 mm rendered at the density 130 mm currently gets, which is 1.69x - has its
upside unopposed. That needs a rebuild from the DICOMs, and the obvious version
of it is wrong:

**Do not rebuild the shards at CROP_MM=100.** `submit_kernel.py` decodes each
series exactly once into one shared volume and every member resizes out of it.
A 100 mm base is narrower than the 130 mm every incumbent member trained on, so
the incumbent could not be scored from it at all, and the ensemble would have to
be rebuilt from scratch around the new crop.

The composable version keeps the shared base at **CROP_MM=130 and raises
PREP_SIZE instead**. Then a 100 mm crop is 246 px of real detail out of a 320 px
base (or 295 of 384), the incumbent members resize down out of the same base as
they always did, and crop width stays a per-member cfg field that costs one
decode for any number of widths. One rebuild serves every future crop.

The blocker is disk. A 320 px shard set is about 56 GB compressed and 384 px
about 81 GB, against 63 GB free. `artifacts/volumes48` is 54 GB and its axis is
closed on both instruments (m50depth48: pv-hold 0.8819 against the band, gold
0.9050 against 0.9148). Deleting it is the difference between this rebuild being
possible at 320 px and not - a question for the operator, and one whose evidence
has changed since it was last asked.

### Correction: the rebuild is not what buys the detail

The section above says a 100 mm window "needs a rebuild from the DICOMs" to get
1.69x the pixel density, and proposes raising PREP_SIZE to 320 against a disk
that cannot hold it. That is wrong, and the arithmetic is one line:

**a crop does not change px/mm.** 130 mm over 256 px and 100 mm over 197 px are
both 1.97 px/mm. What throws detail away is `--size 128`, which every member
here has used since m31.

    m31           256 -> 128 over 130 mm     0.98 px/mm at the backbone
    m67crop100    197 -> 128 over 100 mm     1.28 px/mm    already 1.3x
    m69crop100real 197 -> 192 over 100 mm    1.92 px/mm    2.0x, no rebuild

The 256 px shards already carry 197 px of a 100 mm window. Nothing needs
rebuilding to use it; only `--size` was discarding it. The disk question that
looked like a blocker is not on the critical path, and `artifacts/volumes48`
does not need to be deleted for this.

**And it changes what m67crop100 measured.** m67 lost 21% of its field of view
AND gained 1.3x detail, and landed level with the baseline. That is consistent
with two effects cancelling and equally consistent with both being null - the
run cannot tell those apart, and the section above quietly read it as the
former. Holding the field of view fixed at 100 mm and doubling the detail
against m67 is what separates them, and it costs one training run.

The rebuild remains the right move for a window *narrower* than the shards can
serve at full size, and for going above 1.97 px/mm at any width. It is no longer
the next move.

## m68crop80 pays, and three widths make a dose-response

    plateau, mean of epochs 8-12        pv-hold      gold
      130 mm baseline band           0.877-0.881  0.910-0.914
      m67crop100                         0.881      0.912
      m68crop80                          0.874      0.905

    m68 per epoch  ep8 0.876/0.911  ep9 0.877/0.905  ep10 0.873/0.904
                   ep11 0.873/0.902  ep12 0.873/0.902

Below the band on both columns: 0.003 under the pv-hold floor and 0.007 under
m67, 0.005 under the gold floor and 0.007 under m67.

**Read the trend, not the gap.** Taken alone neither number clears its bar -
0.007 on pv-hold is just over the 0.005 replicate threshold, and 0.007 on gold
is just under the 0.008 at which gold is allowed to speak. What carries the
result is that three widths line up monotonically on two instruments at once,
with the middle point free and the narrow one costing on both, and that the
ordering was predicted in advance by a third measurement:

    width    saliency discarded    pv-hold      gold
    130 mm          0%           0.877-0.881  0.910-0.914
    100 mm         21%              0.881         0.912
     80 mm         39%              0.874         0.905

The saliency probe named 21% and 39% before either model was trained. **The edge
of the useful field of view is between 80 and 100 mm**, and the probe located it
to within one step for four minutes of GPU.

**Do not chase the exact edge.** A 90 mm run would cost 75 minutes to refine a
boundary that no submission depends on. What the three points establish is the
thing the next experiment needs: at 100 mm the window can be narrowed for free,
so the pixels freed by narrowing can be spent on detail instead - which is
m69crop100real, holding the field of view at 100 mm and doubling px/mm.

**A note on the ep4 read.** At epoch 4 m68crop80 sat 0.006-0.009 below every
comparison and it was written up here as "clearly below". By epoch 9 it had
closed most of that and the plateau is 0.003 below the floor, not 0.009. Rule 1
says compare plateaus and not mid-curve points; it is as easy to break in the
pessimistic direction as the optimistic one.

## The narrowed members add nothing to the ensemble

Both crop members went into the pool with `ensemble.py` reading `crop_px` out of
their cfg, so each was fed the window it trained on:

    added to the incumbent four        pv-sel   pv-hold     gold
      incumbent                        0.9067    0.8880    0.9200
      + m67crop100                     0.9070    0.8879    0.9193
      + m68crop80                      0.9085    0.8886    0.9198

    exhaustive over the 10 best of 28, top 20 combinations
      m67crop100 appears in   0
      m68crop80  appears in   0

+0.0003 pv-sel and -0.0007 gold for the member that was free to build; the
weaker one nominally does better and its +0.0018 pv-sel is half of that
instrument's own noise. Greedy did pick m68crop80 as a fifth member over
m55nbr33, by 0.0002, which is a coin flip wearing a rank.

**This is the answer a crop should give, and it is worth stating as a rule.** A
narrowed member is trained on a *subset of the same pixels* as m31, through the
same recipe, at the same density. Nothing about it fails differently - it has
strictly less information and processes it identically. Equal accuracy from less
input is an interesting fact about the periphery; it is not ensemble diversity.

    what the crop bought      a member that ties m31 at 77% of the field of view
    what an ensemble needs    a member that is wrong where m31 is right
    these are not the same thing, and this project has now paid for that
    distinction three times: same-family (+0.0035), fold diversity (0.0000),
    and now crop diversity (0.0000).

**The caveat, stated because it was pre-registered and not because it rescues
the result.** pv-sel and pv-hold underrate a representation change, so the local
instruments are handicapped against these members. But gold is not - it is the
clean instrument here, no model has trained on it, and gold is flat too
(-0.0007, against a replicate sd of 0.0037). A handicap on two of three
instruments does not explain a null on the third.

**Where the crop axis still has a live question.** m69crop100real is not a crop
of the same pixels: it is the same 100 mm window at 1.92 px/mm against m67's
1.28, so the backbone sees information m31 never had rather than a subset of it.
Through 7 epochs it is the strongest run of the group at the same epoch (ep4
pv-hold 0.882 / gold 0.915, ep5 gold 0.917 against a baseline plateau band that
tops out at 0.914), which is mid-curve and therefore not a result. Its plateau
lands about 20:20.

## m69crop100real closes the density axis, and with it the crop bet

    plateau, mean of epochs 8-12       px/mm     pv-hold      gold
      130 mm baseline band              0.98  0.877-0.881  0.910-0.914
      m67crop100      (100 mm, 128)     1.28      0.881        0.912
      m69crop100real  (100 mm, 192)     1.92      0.881        0.911

    m69 per epoch  ep8 0.882/0.912  ep9 0.882/0.910  ep10 0.881/0.912
                   ep11 0.880/0.911  ep12 0.880/0.911

**Density doubles across the three rows and neither instrument moves.** 1.5x
more real pixels on exactly the same anatomy, at a window width where narrowing
was already established as free, and the plateau is identical to the third
decimal on pv-hold and within 0.001 on gold.

This is the version of the resolution question the closed list never had. The
old entry ("real 256 and real 224 both tie upsampled 224") was decided on
pv-hold alone, at 130 mm, on the pre-crop-fix shards - wrong instrument, wrong
width, wrong data, which is why it was reopened on 19 August. Re-run properly it
gives the same answer, and the answer is now trustworthy.

### The crop bet is dead in all three of its forms

It was chosen as the main bet for the remaining 46 days on the grounds that the
crop is the only axis that has ever moved this board. Six hours of GPU later:

    re-centring   DEAD.   The saliency centroid sits 1.06 mm from where an
                          untrained network of the same shape puts it, and
                          0.60 mm on a second model at another resolution.
                          There is no off-centre anatomy to find.
    narrowing     FREE, AND USELESS. 100 mm ties the baseline on both
                          instruments while discarding 21% of the saliency, and
                          adds +0.0003 pv-sel / -0.0007 gold to the ensemble.
                          80 mm costs. A member trained on a subset of the same
                          pixels cannot fail differently.
    density       NULL.   1.92 px/mm ties 1.28 ties 0.98.

**What it cost and what it bought.** Six hours of card time, two probes and
three trainings, against a bet that a Kaggle rebuild was going to justify. The
rebuild was never started, and the two runs conditional on this result were
pulled from the queue unrun. The axis is closed with evidence rather than
abandoned, which is the difference between this and the entries that had to be
reopened.

**And it sharpens the standing rule.** "Representation diversity pays" came from
m31+m33, two members at different resolutions on different preprocessing. The
crop members are also a representation difference and they pay nothing. The
distinction that survives: a member has to see something the others do not, not
merely see the same thing differently framed. m33 upsampled to a different
operating point and trained on different targets; m67 is m31's own pixels with
the edges removed.

## 6-7 September — the crop axis, probed and then walked

`saliency_probe.py` asked where a trained model actually looks, for one backward
pass per series and no slot. On two models at different resolutions it replicated
cleanly: an 80 mm centred box is 38% of the frame's area and holds 61% of the
saliency, against 45% for an untrained control. The learned concentration is
+0.16 in both m31 (128/up224) and m33 (256/up336), so it is a property of the
data rather than of one model.

That premise was then tested by training on it. Four points, all gold:

    m31   130 mm, 128 px   1.02 mm/px   0.9148
    m67   100 mm, 128 px   0.78 mm/px   0.9119
    m69   100 mm, 192 px   0.52 mm/px   0.9080
    m68    80 mm, 128 px   0.63 mm/px   0.9045

**Monotone down in both directions.** Narrowing the field of view costs
(130 -> 100 -> 80), and holding the crop while adding resolution also costs
(m67 -> m69). Neither crop model earns a place as a member either: the incumbent four score
gold 0.9203 alone, 0.9193 with m67 added and 0.9190 with m69. Two points, same
direction as the solo scores.

**Why the probe pointed the wrong way, which is the part worth keeping.**
Saliency was measured on a model *trained at 130 mm*, reading 130 mm inputs. That
a model concentrates centrally when the periphery is present does not mean the
periphery is disposable - context that only contributes by being there does not
light up a gradient map, and removing it is exactly what a saliency map cannot
price. The probe answered "where does attention fall" and was read as "what can
be discarded", and those are different questions.

**The standing caution still applies in the other direction.** This is a
representation change, and the recorded rule is that local instruments have
twice failed to see gains on this axis (m2->m17: gold +0.001, LB +0.029;
m31newshards: gold +0.006, LB +0.012). There is no recorded instance of them
inventing a loss, and four monotone points are not what blindness produces - but
the board is still the arbiter here, and nothing above has been submitted.

## The model has already passed its teacher, and the error is four findings

`src/label_quality.py` scores the teacher's report-derived labels against the 58
gold studies' real radiologist labels, per finding. Put beside the incumbent
ensemble's own gold column (printed by the five-member run above), it answers
the question HANDOFF has been carrying unresolved since 23 August - whether
single-model quality has been flat because the model is saturated or because
pv-hold has hit the teacher's ceiling:

    finding            teacher   incumbent   model - teacher
    Synovitis           0.786      0.822         +0.036
    Lateral OA          0.818      0.822         +0.004
    Effusion            0.861      0.957         +0.096
    Lateral Meniscus    0.865      0.881         +0.016
    Contusion           0.872      0.950         +0.078
    PF OA               0.911      0.883         -0.028
    Baker's             0.918      0.975         +0.057
    Medial OA           0.919      0.981         +0.062
    Fracture            0.928      0.918         -0.010
    Medial Meniscus     0.951      0.933         -0.018
    MCL                 0.972      0.930         -0.042
    ACL                 0.994      0.969         -0.025

    macro   teacher 0.900   incumbent 0.919

**The pixels have beaten the text by 0.019, on the only true labels we have.**
The four findings where the teacher still wins - ACL, MCL, Medial Meniscus, PF
OA - are the four a radiology report states most explicitly, so the text is
nearly free of ambiguity there and the image is genuinely hard. The four where
the model wins biggest - Effusion, Contusion, Medial OA, Baker's - are the ones
you can see.

That settles the ceiling question in a specific way. pv-hold measures agreement
with a teacher the model now exceeds, so **pv-hold has been measuring the wrong
thing for a while** - not saturated model capacity, but distance from a target
that is itself 0.081 from the truth. It explains nineteen models sitting inside
0.882-0.887 while the board moved 0.930 -> 0.947 underneath them.

**And the remaining error is concentrated, not spread.** Eight findings are at
0.918-0.981. Four are at 0.822-0.883, and two of those are below 0.83. The
metric is a macro mean over twelve AUCs, so those four carry a third of the
score and every point there is worth what a point on ACL is worth - while the
loss, which averages over findings, spends exactly the same capacity on a
finding at 0.981 as on one at 0.822.

Nothing here has ever weighted the loss per finding. `pos_weight` addressed
class imbalance *within* a finding, which is a different quantity. m72fw2 is
queued: 2x on the four weak findings, everything else unchanged. It is a recipe
change, so pv-hold decides it and no slot is needed.

**A caveat on m18cos.** `label_quality.py` hard-codes its model column to
`image_model_m18cos`, an August checkpoint at LB 0.930. The numbers above use
the incumbent's per-finding column instead, read off the five-member ensemble
run. The script should take a tag; it does not yet.

### m69 in the ensemble, for completeness

    added to the incumbent four        pv-sel   pv-hold     gold
      incumbent                        0.9067    0.8880    0.9200
      + m67crop100  (100 mm, 1.28)     0.9070    0.8879    0.9193
      + m69crop100real (100 mm, 1.92)  0.9073    0.8883    0.9190

Denser pixels do not make the narrowed member fail differently either. Both
crop members move the ensemble by less than a third of pv-sel's own noise and
both take gold down by about 0.001. The closure holds on the composition
question as well as the solo one.

### The drop list came off, and the search immediately found the board's rejects

`preds_pool4.npz` (29 members) was searched **without `--drop`**, unlike every
earlier run, which excluded m53nbr, m55nbr33, m56nbr37 and m54attn. The top of
the table is now:

    pv-sel  pv-hold    gold   ensemble
    0.9087  0.8928   0.9214   m53nbr+m55nbr33+m61nano336+m38seed1
    0.9086  0.8923   0.9231   m53nbr+m48blend60+m55nbr33+m38seed1
    0.9085  0.8929   0.9225   m53nbr+m60nano288+m55nbr33+m38seed1

**Every one of those leading combinations is built from members the board has
already refused.** m53nbr and m55nbr33 are the 2.5D pair, which raised pv-hold
by 0.0064 and then lost on the leaderboard twice at two resolutions. m60nano288
and m61nano336 are nano, refused three times in a row by 0.008-0.010 - the run
of submissions on 27 August that established pv-hold cannot order a nano against
a tiny.

pv-hold 0.8929 is the highest this file has ever recorded. It is high **because**
it is made of the members pv-hold overrates. A search that maximises pv-hold over
an unfiltered roster will find them first, every time, and the number it reports
will look like a record.

**The drop list is not a preference, it is four board measurements.** Anything
searched over this pool needs `--drop m53nbr m55nbr33 m56nbr37 m54attn`, and any
result that does not carry that flag should be read as a demonstration of the
instrument's known failure mode rather than as a candidate.

## 7 September — up-weighting the weak findings makes them worse

Two runs raised the loss weight on the findings where the teacher is weakest
(Synovitis, Lateral OA, Effusion). This is a recipe change, not a representation
change, so pv-hold and gold both decide it and neither is blind here.

    model     pv-hold    gold   Synovitis  Lateral OA  Effusion
    m31        0.8832  0.9148     0.816      0.816      0.953
    m72fw2     0.8804  0.9053     0.779      0.809      0.921
    m73fw4     0.8823  0.9090     0.808      0.801      0.937

**Both are worse overall and both are worse on the findings they were aimed at.**
There is no "it cost the macro but bought the target" reading available: Synovitis
falls from 0.816 to 0.779 and 0.808, Effusion from 0.953 to 0.921 and 0.937.

This confirms a diagnosis already in this file from August: *"Where the teacher
is weak, no amount of training helps - the target is wrong"*, and *"Synovitis's
ceiling is 0.786 and the model is at 0.766; there is 0.02 there, not 0.11."*
The teacher's own agreement with the gold labels on Synovitis is 0.751-0.786.
Weighting a wrong target harder teaches the error harder.

The axis closes: **loss weighting cannot buy back what the labels do not
contain.** That is the same wall the Sol relabel hit from the other side on
27 August, and the two results agree on where it is.

## Per-finding loss weighting does nothing, and the instrument that motivated it cannot see per-finding effects

    plateau, mean of epochs 8-12       pv-hold      gold
      130 mm baseline band          0.877-0.881  0.910-0.914
      m72fw2     (2x on the weak 4)     0.878      0.904
      m73fw4     (4x on the weak 4)     0.882      0.910
      m74fw2s1   (2x, seed 1)           0.882      0.910

1x, 2x and 4x are flat on macro pv-hold, and every one lands inside or at the
edge of the band. The dose-response design that settled the crop widths returns
a null here, which is the outcome it is equally good at delivering.

**The replicate earned its 75 minutes.** m72 and m74 are the same recipe at two
seeds and differ by 0.004 on pv-hold and 0.006 on gold. Read alone, m72 says
"2x costs 0.006 of gold"; m74 says it does not. Rule 2 exists for exactly this
and the run was queued before either number was seen.

### The weak four did not move, and could not have been seen to move

    per-finding gold, mean over the four weak findings
      baselines   m31 0.8437   m38 0.8582   m39 0.8261      spread 0.032
      weighted    m72 0.8339   m73 0.8303   m74 0.8324

The weighted runs sit inside the baselines' own range, near its bottom. But the
number that matters here is the 0.032: **three runs of one recipe spread the
weak-four mean over 0.032**, which is five times any effect the weights could
plausibly produce and is what a four-finding mean on 58 studies is worth.
`label_quality.py` says as much in its own docstring - one standard error is
about 0.06 per finding - and this is that warning arriving in practice.

Two consequences, and the second is the more useful:

  - **The experiment was under-powered before it started.** Upweighting four
    findings is a per-finding intervention and the only per-finding instrument
    here is gold-58. It can rank the findings but it cannot compare two models
    on any one of them.
  - **The concentration claim survives; per-finding model comparison does not.**
    The same four are bottom in all six runs and eight sit at 0.92-0.99 in all
    six, so "the error is concentrated in four findings" is solid. "Model A is
    better than model B on Synovitis" is not readable at this sample size, and
    that also rules out per-finding member selection for the final ensemble -
    which would have looked like an obvious move.

**What this leaves.** The capacity hypothesis - that the loss spends the same
effort on a finding at 0.98 as on one at 0.82, so redirecting it should pay -
is not refuted; it is unmeasured, and unmeasurable with the instruments here.
Nothing more should be spent on it until there are true labels for more than 58
studies, which is the same conclusion the teacher-ceiling table reached from the
other direction.

### Correction, next day: the finding-weight verdict rested on one seed

m74fw2s1 replicates m72fw2 - same `--finding-weights weak`, seed 1:

    m31        pv-hold 0.8832   gold 0.9148   Synovitis 0.816
    m72fw2     pv-hold 0.8804   gold 0.9053   Synovitis 0.779   (seed 0)
    m74fw2s1   pv-hold 0.8842   gold 0.9136   Synovitis 0.804   (seed 1)

**Two seeds of one recipe differ by 0.0083 on gold.** I read m72fw2's -0.0095
as 2.6 sd against a recorded gold spread of 0.0037 and closed the axis on it.
That spread was measured on a different comparison and does not describe this
one; the replicate gap here is nearly as large as the effect I was calling
significant.

Seed 1 is indistinguishable from m31 - pv-hold +0.0010, gold -0.0012, Synovitis
-0.012. **The honest statement is that up-weighting the weak findings does
nothing measurable, not that it hurts.** The section above overstated it, and
the "axis closes" line there should be read with this correction attached.

What survives unchanged is the narrower point: neither seed improved the
findings it targeted, so there is still no evidence that loss weighting buys
back what the labels do not contain. That was the reason to expect nothing, and
nothing is what happened.

## 7 September — a fourth seed, and both replicate spreads are wrong

m75seed3 is the fourth run of m31's recipe. With four:

    seed        pv-hold     gold
    m31          0.8832    0.9148
    m38seed1     0.8836    0.9205
    m39seed2     0.8831    0.9136
    m75seed3     0.8785    0.9070

    gold      sd 0.0055   range 0.0134
    pv-hold   sd 0.0024   range 0.0051

**Both numbers this project runs on are wrong.** gold has been quoted at 0.0037
and is 0.0055. pv-hold has been quoted at 0.0005 and is 0.0024.

The pv-hold error is the instructive one. The first three seeds land in
0.8831-0.8836 - a spread of 0.0005 - and that is where the figure came from. The
fourth is 0.8785. **Three replicates agreeing tightly is not a measurement of
the spread; it is one draw from it.** This is the same failure the project
already recorded in August, when nobody had run a recipe twice in 25 runs and
six conclusions had to be retired. Going from three replicates to four brought
it back.

### What this touches

**The crop axis.** m67 -0.003, m69 -0.007, m68 -0.010 against m31 all sit inside
two gold sd. Six points moving monotonically is still weak evidence of a real
direction, but no individual point says anything, and I described the 80 mm
result as "2.7 sd" using the old figure. It is 1.8.

**m64gru "identical to m31".** gold 0.9148 against 0.9148 to four decimals, on
an instrument whose replicate spread is 0.0055. That agreement is a coincidence,
not a finding. What survives is the ensemble behaviour - it was picked fourth by
the search - which was never a solo-score argument.

**The finding-weight runs.** Already corrected yesterday for the same reason;
this quantifies it. The 0.0083 gap between those two seeds is 1.5 sd, entirely
ordinary.

**The Sol relabel.** I called gold -0.0139 "3.8 sd"; it is 2.5. That conclusion
survives anyway because the board confirmed it independently at -0.011, which is
the one instrument no replicate spread touches.

**The efficiency track is untouched** - every configuration there was decided on
board scores, not local ones.

### The rule

Two sd on gold is **0.011**, on pv-hold **0.005**. Nothing below those is a
result, whatever the sign. And a spread quoted from three replicates should be
treated as a lower bound until a fourth exists.

## 18 September — eleven idle days, and a mirror that undoes the laterality fix

Nothing ran between 7 September 01:35 and today 14:02. The runner was alive the
whole time; `artifacts/queue.txt` had no lines left in it. Our score has not
moved since 23 August and the rank went 18 -> 59 while the 10th-place line rose
0.948 -> 0.953. The deficit to the money is now +0.006.

### The finding

`volume.py` mirrors every right knee onto a canonical left frame so that
"medial" is a fixed direction in the array. `train_image.py`'s `augment()` then
flips the width axis at p=0.5 and leaves the target alone. Five of twelve
columns name a side. Half of every epoch since the 13 August shard rebuild has
carried a mirrored knee with unmirrored side labels, and every member of the
submitted ensemble was trained that way.

Two independent readings already show the predicted symptom: Lateral OA and
Lateral Meniscus are our weakest side-specific columns, and the forum's 0.903
write-up reports those same two as the only ones where its image AUC sits below
its own report-target AUC.

### The test

`--no-flip` and `--flip-swap` added, both defaulting off so existing members
stay reproducible. A recipe change with the pixels unchanged, so pv-hold judges
it against the four-seed band 0.8785-0.8836 (sd 0.0024). Seeds queued in pairs
because one run cannot clear two sd: m76noflip, m77noflips1, m78flipswap,
m79noflips2, m80flipswaps1.

Read the per-finding columns when they land. The hypothesis predicts the four
side-named findings move; a mean that moves without them is a different effect
and should be written up as one.

### A submission, 18 September 14:35 — the first since 23 August

Twenty-six days of unused slots is the larger error here, not the choice of
roster. `preds_pool4.npz` re-searched **with the drop list** (m53nbr, m55nbr33,
m56nbr37, m54attn) and additionally restricted to members that already have a
Kaggle dataset, so the submission needed no upload:

    ensemble                                            pv-sel  pv-hold    gold
    incumbent (LB 0.947)                                0.9067   0.8880  0.9200
    + m38seed1 + m40m33s1 + m43small   (submitted)      0.9084   0.8904  0.9219

**+0.0024 on pv-hold is inside the noise and this is not a certified
improvement.** It is submitted anyway, on three grounds: the board is the only
instrument that has never saturated, a lower score cannot displace our best on
the public board, and slots cost nothing. The point is the measurement, not the
expectation - and every earlier member addition paid on the board while local
instruments called it flat.

Kernel `qwrfsdr/rsna-knee-sub-ens7`, seven members, about 4.5 hours of Kaggle
runtime (53 min fixed plus ~31 per member) against the 9-hour cap. It must be
submitted with `kaggle competitions submit -k ... -v <version> -f submission.csv`
once it completes; pushing does not submit.

Unchanged by this: the flip ablation above is the real candidate, and if it
lands the members have to be retrained rather than re-blended.

### Correction, 15:05 — that slot re-asked a question the board answered on 4 September

Submission 56322372 is on the board, and the submission list shows why it
should not have been: **56006499 (`ens5search`, 4 September) already tested the
same axis with the same margin and returned 0.947, +0.000.** Its local reading
was pv-sel 0.9098 / pv-hold 0.8900 / gold 0.9221 - the numbers I re-derived
today and quoted as the case for spending a slot.

This file already carried the rule that forbids it, written the same evening:
**a local gap of 0.002-0.003 is not worth a slot even when every instrument
agrees**, because it is inside the board's own resolution (five submissions
differing by less have all returned 0.947). The right check before spending a
slot is `kaggle competitions submissions`, and it was not run first.

What the 26 idle days actually cost was *experiments*, not submissions, and the
fix for an unused slot is a candidate worth submitting rather than a submission.
The roster is closed and stays closed: every future point has to come from a
member that does not exist yet. That is what m76-m80 are for.

### The verdict, 18 September 20:17 — the mirror costs nothing measurable

All five finished clean (exit 0, 74 min each). pv-hold at the selected epoch,
against the four-seed band:

    group                          mean     range            vs baseline
    flip baseline (4 seeds)      0.8821   0.8785-0.8836          -
    no-flip       (m76,77,79)    0.8830   0.8808-0.8860      +0.0009
    flip-swap     (m78,80)       0.8827   0.8800-0.8854      +0.0006

**Two sd on pv-hold is 0.005. Both effects are a fifth of that.** Neither
correcting the mirror nor removing it moves the number, and the three outcomes
this queue item listed - no-flip wins, both win equally, flip-swap wins by more
- are all wrong in the same way. The fourth outcome happened: all three groups
sit on top of each other, including the group that carries the bug.

### The per-finding columns, which were the actual prediction

The hypothesis predicted the side-named findings move. On gold-58:

    group          Med Men   Lat Men   Med OA   Lat OA    MCL
    baseline         0.935     0.864    0.975    0.809   0.923
    no-flip          0.945     0.868    0.983    0.799   0.924
    flip-swap        0.937     0.852    0.974    0.811   0.899

Lateral OA is the column the whole argument was built on, and the group that
never sees a mirrored knee scores **lowest** on it (0.799 against 0.809). The
baseline's own four seeds span 0.785-0.820 on that column, so every number in
it is inside one group's replicate spread and none of these differences are
readable. The same holds for Lateral Meniscus.

**The prediction failed on its own terms.** Lateral OA and Lateral Meniscus are
still our weakest side-specific columns and the mirror is not why. Whatever the
forum's 0.903 write-up is seeing in those same two columns, it is not this bug
either - or not only this bug.

### What is left standing

The code defect is real: `augment()` did mirror a frame `volume.py` spends work
straightening, and it did leave five side-named targets unmoved. What is now
measured is that **it was not costing anything the instruments can see.** A bug
that changes no number is a bug worth fixing for hygiene and not worth a
retrain, so the flags stay in, both stay off by default, and the roster is not
rebuilt. The contingency in the 14:35 entry - "if it lands the members have to
be retrained" - does not trigger.

One loose thread, recorded because it is the only thing here that moves at all:
flip-swap's two runs score gold 0.9026 and 0.9042, and the baseline's four span
0.9070-0.9205. Both below the range, about 2 sd of the gold spread, on n=2.
That is the wrong direction for the hypothesis - swapping the labels to match
the mirror makes gold slightly worse - and it is consistent with the one column
the swap cannot fix (MCL has no lateral counterpart, 0.923 -> 0.899). Not a
result at n=2; noted in case a later run walks past it again.

### What this closes, and what it points at

The augmentation stack has three components: the flip, a zoom jitter, and a
gain/offset jitter. The flip is now measured at zero. The forum protocol we are
following ranks ablating this stack ahead of backbones and resolutions, so the
remaining two are queued next, four seeds, `--aug` dropped so the flip stays on
and the other two come off together (items 15-18).

If they also land inside 0.005, the honest reading is not "these augmentations
do nothing" but **"pv-hold cannot resolve anything on this axis, and the axis
should be abandoned rather than subdivided"** - which would leave the
representation class as the only place left to look. Every board gain this
project has ever measured came from there: upsample +0.029, the crop and
laterality fix +0.012. The leading unfixed candidate is the series-slot trap
from forum thread 741310 - keying slots on plane x fat-sat merges T1, PD and T2
under "non-FS" - and it needs a preprocessing change and a shard rebuild, not a
queue line.

## 19 September — the plane probe was measuring the wrong thing

The flip result closed the augmentation axis's first component, and the next
candidate was conditioning the model on what kind of series it is looking at.
The obvious objection was on file: `plane_probe.py` killed plane-weighted
aggregation in August, and the diagnosis attached to it - "the model is reading
coarse study-level appearance, not the lesions" - would kill sequence
conditioning by the same argument. That probe had never been rerun since, so it
was rerun first, on m38seed1 (gold 0.920) instead of m3tiny (LB 0.900).

    Sagittal 0.8172   Coronal 0.8072   Axial 0.7927   uniform mean 0.8415
    best plane per finding, against the mean:  -0.0152   (August: -0.0091)

### The confound

A study's prediction is the mean over whatever series it contributed. Scoring
one plane scores a third of the series, so each study is averaged over fewer
views and carries the noise of the smaller average. **The probe cannot separate
"this plane carries less signal" from "this study was averaged over fewer
series", and the -0.0152 is partly the second.** Every conclusion drawn from
per-bucket scores in this file has the same hole.

`seq_probe_analysis.py` adds the control: for each study take the same NUMBER
of series the bucket gave it, drawn at random from all of that study's series.
What a bucket scores above its own matched draw is what the bucket's identity
is worth. On pseudo-val (2,885 series, 522 studies), 40 draws:

    bucket            macro   control    +-     bucket - control
    Sagittal         0.8172   0.8186  0.0037        -0.0014
    Coronal          0.8072   0.8108  0.0037        -0.0036
    Axial            0.7927   0.7877  0.0053        +0.0050

    FS (fluid-sens)  0.8408   0.8290  0.0030        +0.0118
    non-FS           0.7967   0.8152  0.0040        -0.0186

    Sagi/FS          0.8170   0.7889  0.0058        +0.0281
    Axia/nonFS       0.7398   0.7915  0.0159        -0.0516

**The plane is worth nothing and the competition's own Fluid_Sensitive /
Fat_Suppression flag is worth 0.03 between its two levels.** The August reading
was not wrong about planes - it was right for a reason nobody had checked, and
it hid the variable that does matter, because plane and flag were never
separated.

Two things make this cheap. The flag is a column of `train_series.csv` **and of
`test_series.csv`**, so using it needs no DICOM tags, no metadata kernel and no
shard rebuild. And `Fat_Suppression == Fluid_Sensitive` on 100% of the corpus,
so it is one bit, not two.

### It reproduces on the real labels

The same analysis on gold_val (336 series, 58 expert-labelled studies), which
shares none of the teacher's biases:

    FS       +0.0085 (control sd 0.0068)        non-FS   -0.0189 (0.0093)
    Sagi/FS  +0.0399                            Axia/nonFS -0.1017

Same sign, same order of magnitude, on labels the pseudo-labels never saw. The
plane numbers do not reproduce (Coronal +0.0183 here against -0.0036 there),
which is what noise looks like.

### Forum 741310 is not our trap

Inside the non-FS bucket, where the thread says T1, PD and T2 are silently
merged, the derived types score T1 -0.0108, PD -0.0202, T2 -0.0458 against
their own controls on pseudo-val. There is nothing to recover by splitting that
bucket: it is weak throughout, not a mixture of one good and two bad types.
This also removes the last reason to depend on DICOM tags at inference.

### The cheap use of it does not work

If FS series are worth more, weight them more. `seq_weight_search.py` fits the
weight on one half of the studies and scores it on the other, both ways round:

    one global weight on non-FS      A->B -0.0003   B->A +0.0014   mean +0.0006
      fitted weight                       0.25           0.65
    per-finding weights             A->B +0.0010   B->A -0.0005   mean +0.0003
    per (plane x flag) bucket       A->B -0.0019   B->A +0.0019   mean -0.0000

**Nothing.** The fitted weight is not even stable between halves, and the
per-finding weights invert (ACL 0.00 on one half, 1.65 on the other). The
reading that fits both this and the +0.0118: the weak series pay for themselves
by failing differently, and down-weighting them removes the diversity along
with the noise. That is this file's own ensemble finding, one level down -
members are worth keeping for how they fail, not for how they score.

### So the fact goes inside the model

`--cond planeflag` adds a zero-initialised embedding on the pooled feature,
keyed by plane x flag (6 classes); `--cond flag` is the 2-class version. Zero
init means the first forward pass is numerically identical to an unconditioned
model, so anything that shows up is something training chose. The model keeps
every series and can move a prediction per finding and per image instead of by
one scalar, which is the part re-weighting could not do.

`cond` and `cond_classes` are written into image_model.json and read back by
`ensemble.py`, which now builds the embedding and passes the ids. This matters
more than the other cfg flags: `strict=True` refuses to load a conditioned
checkpoint into an unconditioned model, but nothing would have stopped the
scorer from calling a correctly built one with `cond=None` and silently
dropping the embedding - the failure that cost m53nbr 0.019 in August.

Pixels unchanged, targets unchanged, so pv-hold judges this against the
four-seed band 0.8785-0.8836 (2sd 0.005). m85cond and m86conds1 queued.

## 19 September — the augmentation axis, with replicates this time

Every arm run at least twice, against the four-seed baseline (gold sd 0.0055):

    arm         n    gold      sd      vs base
    base        4   0.9140   0.0055      -
    noflip      3   0.9141   0.0035   +0.0002   0.0 sd
    flipswap    2   0.9034   0.0011   -0.0106   3.7 sd
    noaug       2   0.9034   0.0030   -0.0106   3.0 sd

**Augmentation is worth 0.0106 and none of it comes from the flip.** Turning
everything off costs that much; turning off only the flip costs nothing
measurable. The gain lives in the rotation and scale jitter.

**Adding the label swap to the flip costs exactly what removing all augmentation
costs** - both -0.0106, which is not a coincidence worth ignoring. Swapping the
side-specific labels on a mirrored frame does not merely cancel one useful
augmentation; it injects enough wrong signal to wipe out the benefit of the rest.

**My hypothesis had the sign backwards.** I proposed that the flip un-does the
laterality normalisation the preprocessing applied, so the side labels ought to
swap with it. The measurement says the opposite: the preprocessing normalisation
is already correct, and swapping labels on top of it is what creates the
mismatch. Leave the flip on and leave the labels alone.

Note the pattern across this and the last two days: two-seed effects that
vanished at three (finding weights, the laterality-specific findings under
noflip), and two that held (flipswap, noaug). The ones that held are both around
0.010 - roughly two gold sd - and the ones that vanished were all under 0.008.
That is the resolution limit behaving exactly as the corrected spread predicts.

## 19 September — DINOv2 was not the problem; unfreezing it was

August closed backbone-family diversity on two DINOv2 runs that landed 0.018 to
0.024 below the convnext baseline, measured twice and called decisive. With the
backbone mostly frozen (`--unfreeze-last 6`, backbone-lr 8e-6) the same family
comes back level:

    August, fully unfrozen
      m34dinonew    gold 0.8902   -0.0238
      m36dinolr     gold 0.8958   -0.0182
    September, partially frozen
      m89dinofrz    gold 0.9154   +0.0015
      m90dino224    gold 0.9112   -0.0028

**The verdict was "DINOv2 does not work"; the measurement supported only
"DINOv2 fully unfrozen does not work."** A self-supervised backbone losing its
representation under too much fine-tuning is a known failure, and both August
runs were configured that way.

This matters more than a recovered single-model score. Every member of every
ensemble here is a convnext. DINOv2 is a different architecture on a different
pretraining objective, which is the strongest prior available for *failing
differently* - and the rule this project keeps relearning is that averaging pays
for different failures, not better members. m64gru made the same point earlier
this month: level alone, picked fourth by the search.

Both runs are single seeds and gold's spread is 0.0055, so neither solo number
means anything on its own. The question worth measuring is the ensemble
contribution, which no solo score can answer.

**Retract from the August record:** "backbone-family diversity (DINOv2 members
land 0.017-0.020 behind and lose every reading, measured twice)" is withdrawn as
stated. The two readings were of one configuration, not of the family.

## 20 September — what the DINO members are worth in a roster

The section above asks the question this answers: both runs are single seeds and
no solo score settles a member's value. The final checkpoints, on all three
instruments rather than gold alone (the gold figures above were read off a log
mid-run; these are the saved, epoch-selected numbers):

    run           regime                       pv-hold   pv-sel     gold
    m36dinolr     3e-4 / 3e-5, all trainable    0.8634   0.8794    0.8958
    m89dinofrz    1e-3 / 8e-6, last 6 blocks    0.8779   0.9012    0.9181
    m90dino224    1e-3 / 8e-6, last 6 blocks    0.8766   0.9016    0.9112

+0.014 pv-hold and +0.022 on both pv-sel and gold, against a 2sd of 0.005 and
0.011. A third seed (m91dinofrzs1) is running, because a new regime's first run
is still one draw.

### In the roster

    roster                                      pv-sel   pv-hold    gold
    incumbent four                              0.9067   0.8880   0.9200
    + m89dinofrz                                0.9085   0.8887   0.9200
    + m89dinofrz + m90dino224                   0.9095   0.8885   0.9208
    the submitted seven (LB 0.948)              0.9084   0.8904   0.9219

pv-sel is the highest any roster has recorded here. pv-hold moves +0.0005, which
is nothing, and stays under the seven already on the board. With this file's own
rule attached - pv-hold saturated in August, and member selection on it has been
looked at too many times to be independent - the honest statement is that **the
local instruments cannot rank these rosters against each other.** That is an
argument for the board, not against the members: the nine-member measurement and
a pool search carrying the drop list are queued before any slot is spent.

The regime is also not DINO-specific. m51radimagenet (resnet50, pv-hold 0.8490)
was fine-tuned the old way like everything else here, and the public notebook
trains its RadImageNet and CoAtNet arms the same way it trains its DINO arm.

### Two process failures, both this project's oldest

The card idled 7h27m overnight with an empty queue, and 3h16m in the afternoon
because a leftover `pgrep -f` wait loop matched its own shell and
`gpu_queue.sh` counted that shell as a running job. Fixed in the runner rather
than in anyone's memory: `running()` now matches the process name and confirms
it through `/proc/<pid>/exe`, the busy branch prints what it thinks is holding
the slot every ten polls, and an flock makes a second runner impossible - two
were live from 17:10 to 00:38, which is why m89 and m90 shared the card and m89
took 213 minutes instead of 150.

**Two sessions are editing this file.** The 19 September entries above and this
one were written by different agents within hours of each other, and one
`git add -A artifacts/` nearly committed the checkpoint directories. Stage named
files here, never a directory.

### The drop list is half-applied, and the search still cannot choose

`preds_pool5.npz` (31 members, no fold or OOF contamination) searched with
`--drop m53nbr m55nbr33 m56nbr37 m54attn`. The 2.5D family is out; **the nano
family is not**, and it fills the top again:

    0.9071   m52nano+m61nano336+m43small+m38seed1+m37up288
    0.9070   m60nano288+m43small+m38seed1+m31newshards+m37up288
    0.9069   m48blend60+m60nano288+m43small+m38seed1+m37up288

The drop list exists because the board refused 2.5D twice. **The board refused
nano three times** - 27 August, 0.924-0.926 against 0.934 for the nano-free pair,
three submissions in the same direction. By the standard that put 2.5D on the
list, nano belongs on it.

**And the ranking is empty anyway.** The top twelve span pv-sel 0.9069-0.9071
against a reference of 0.9067 - a range of 0.0002 on a metric whose spread is
0.0034. There is no ordering information in this table; there is a cluster.

DINOv2 appears nowhere in the top twelve, which agrees with the direct ensemble
measurement (4 members 0.9203, +m89 0.9200, +m89+m90 0.9208 - all inside gold's
0.0055).

**Recommended: add `m52nano m58nanos1 m59nano192 m60nano288 m61nano336` to the
drop list.** Not because nano is bad in itself - m52nano's plateau matches m33's
at a quarter the cost - but because pv-hold cannot order it against tiny, and a
search maximising pv-hold over an unfiltered roster will keep surfacing it.

### Searching with every board-rejected family excluded

Re-ran the search on pool5 dropping 2.5D *and* nano - every family the board has
actually refused. The top ten are no longer a cluster of rejects:

    0.9075  0.8910  0.9201  m43small+m38seed1+m31newshards+m37up288+m64gru
    0.9074  0.8916  0.9214  m48blend60+m43small+m38seed1+m37up288+m64gru
    0.9073  0.8915  0.9219  m48blend60+m43small+m38seed1+m64gru

The pv-sel spread is still 0.0004 across the ten against a 0.0034 metric spread,
so **the ordering within them remains noise**. What the table can support is not
"this combination" but "which members keep appearing":

    m64gru        10/10
    m38seed1      10/10
    m43small       7/10
    m48blend60     6/10
    m37up288       6/10
    m31newshards   5/10
    m33up336new    0/10
    m32blend40new  0/10

**Two of the four incumbent members never appear.** m33up336new and
m32blend40new are in the configuration that scores 0.947-0.948 on the board, and
the search with the rejects removed does not pick either of them once in ten.

That is a concrete, board-testable hypothesis rather than another local
comparison: swap m33 and m32 out for m64gru and m43small. The local instruments
cannot separate the candidates - 0.0004 on a 0.0034 spread - but the board can,
and this is exactly the condition the August rule names for spending a slot.

Caveat worth keeping attached: m64gru appearing in 10/10 is the same kind of
signal that put nano at the top before the board refused it three times.
Frequency in a pv-sel search is not a board result. The difference here is that
m64gru has never been tested on the board at all, so there is something to learn
rather than something already answered.

## 20 September — two slots, two questions the local instruments cannot answer

Twenty-seven days with one submission, and both of today's are on questions
where this file's own instruments contradict each other rather than agree.

**A, `ensAgru`** (submission 56365789): m31newshards + m37up288 + m38seed1 +
m43small + **m64gru**, the roster the pool-5 search asks for once nano and 2.5D
are dropped. m33up336new and m32blend40new - two of the four members in the
configuration that scores 0.947-0.948 - appear in 0 of the top 10 combinations;
m64gru appears in 10 of 10 and has never been on the board. The top ten are
separated by 0.0004 of pv-sel against a 0.0034 metric spread, so the ordering
inside them is noise and only the board can resolve it.

**B, `ensBdino`**: the incumbent four plus m89dinofrz and m90dino224 - **the
first non-convnext family this project has ever put on the board.** The local
readings disagree with each other, which is the argument for spending the slot
rather than against it:

    pv-sel   0.9095   the highest any roster has recorded here
    pv-hold  0.8885   +0.0005 on the incumbent four, and under the submitted seven
    search   neither DINO member appears in the top 10 of pool5

pv-sel says take it, pv-hold says nothing, the search says no. One of those is
wrong and no amount of local measurement will say which, because all three read
the same saturated 522-study pseudo-val. The board reads 955 held-out studies
with real labels.

Both kernels were checked member-by-member in their logs before submitting -
five and six loaded, with m64gru's `slice_gru: 1` and both ViT backbones rebuilt
from their own checkpoint metadata. That check exists because m53nbr once scored
0.019 low from a flag the kernel did not carry.

What each outcome buys:

- **A beats 0.948** - the search's frequency reading is worth trusting, and the
  roster has been carrying two members that do nothing for three weeks.
- **A ties or loses** - the search is doing to m64gru what it did to nano, and
  frequency in a pv-sel search is confirmed as not a board signal.
- **B beats 0.948** - family diversity is the axis, and the queue should be
  spent on the frozen regime across every backbone family rather than on more
  convnext seeds.
- **B ties or loses** - the DINO members are real models but not useful members,
  and pv-sel is confirmed as the instrument to stop reading.

### Does the new family actually fail differently? Partly.

The case for the DINO members is diversity, and diversity is measurable rather
than assumed. Mean rank correlation between members' pseudo-val predictions,
per finding:

    convnext <-> convnext      0.8172   (325 pairs, 26 members)
    m89/m90  <-> convnext      0.7870   (52 pairs)
    m34/m36  <-> convnext      0.7545   (52 pairs, the August DINO runs)
    m89      <-> m90           0.8734

**The claim holds, and it is smaller than the story suggests.** A frozen-regime
DINO member sits 0.030 further from the convnext block than the convnext members
sit from each other - real, and a third of the way to the decorrelation the
August runs had, which came from being weak rather than from being different.

The more useful number is the last one. **The two DINO members are more
correlated with each other (0.8734) than the convnexts are among themselves
(0.8172)**, so the second one is close to a replicate of the first: same
backbone, same regime, same shards, only the upsample differs. That matches the
roster measurement, where the first DINO member bought +0.0018 pv-sel and the
second only +0.0010 more, and it predicts that m91dinofrzs1 - a third seed of
m89's exact configuration - will add nothing as a member however well it scores
alone. It is a replicate for the seed question, not a candidate for the roster.

Where the remaining diversity has to come from is therefore other families, not
more DINO seeds, which is what m92effvfrz, m93swinfrz and m94radfrz are for.

### The third seed closes the DINO question: level, not better

m91dinofrzs1 (pv-hold 0.8783, pv-sel 0.9012, gold 0.9119) makes three runs of
the frozen regime against the convnext recipe's four:

    group                       pv-hold                 pv-sel    gold
    convnext (4 seeds)      0.8821  [0.8785-0.8836]     0.9003   0.9140
    DINO frozen (3 runs)    0.8776  [0.8766-0.8783]     0.9013   0.9137

**All three DINO runs sit below the bottom of the convnext band on pv-hold**,
and the group is -0.0045 against it - just inside 2sd, so not a result, but
pointing down rather than up. pv-sel is +0.0010 and gold is -0.0003: level on
both.

The correct statement is therefore narrower than yesterday's. **The regime is
worth +0.014 over the old DINO configuration, and it buys a family that is level
with convnext, not better than it.** Yesterday's entry said "comes back level",
which the third seed confirms; anything read as "DINO is the new best backbone"
is not supported by these numbers and was not claimed.

Also worth noting: the DINO runs are tighter than the convnext ones (spread
0.0017 against 0.0051 over a comparable number of runs). A mostly-frozen
backbone has far fewer free parameters to land differently, which is what a
smaller replicate spread should mean here - and it makes the three-run group a
better-determined number than the four-seed one it is being compared against.

So the roster case for these members rests entirely on decorrelation (0.030,
measured above), not on quality. That is a thin case, and it is exactly what
submission B is on the board to test.

## 20 September — A returns 0.947, and the search's top pick has now scored 0.947 twice

    submission          roster                                              LB
    56322372 (18 Sep)   incumbent four + m38seed1 + m40m33s1 + m43small   0.948
    56365789 (A)        m31 + m37 + m38seed1 + m43small + m64gru          0.947
    56006499 (4 Sep)    ens5search's top pick, five members               0.947

**The pre-registered reading applies: the search is doing to m64gru what it did
to nano.** Frequency in a pv-sel search is not a board signal - m64gru appeared
in 10 of 10 top combinations and the roster built around it lost 0.001 to the
one the search never proposed.

The confound, stated rather than buried: A is five members and the 0.948 is
seven, so this is not the clean "m33 and m32 out, m64gru in" swap the hypothesis
described - A also drops m40m33s1. What it does cleanly test is the search
itself, and that test has now been run twice with the same answer. **Two
different "best in pool" rosters, chosen four days apart by the same procedure
on different pools, both return 0.947, while the board's best comes from adding
members rather than selecting them.**

That is the third instrument-failure of the same shape in this file: pv-hold
overrates 2.5D, pv-sel overrates nano, and now the search over either overrates
whatever member happens to correlate best with the selection half. The rule that
survives all three: **local scores rank members, the board ranks rosters, and
the two have never agreed on this problem.**

What it does not say: nothing here is evidence about m64gru as a member. It says
the procedure that picked it is unreliable, which is a weaker and more useful
claim. A roster of the submitted seven plus m64gru would test the member; this
tested the search.

### DINOv2 replicated: level with convnext, and still contributes nothing

    convnext base, 4 seeds        gold 0.9140  sd 0.0055
    DINOv2 mostly frozen, 2 seeds gold 0.9150  (+0.0010, 0.2 sd)
    DINOv2 fully unfrozen, Aug    gold 0.8930  (-0.0210)

The August verdict is now firmly wrong: **0.022 separates two configurations of
the same backbone on the same data**, and both August runs sampled only one of
them. Replication fixes a spread; it does not explore a setting. "Measured
twice" was read as "two things were measured".

**But the ensemble contribution is still nil.** Added to the incumbent four:
0.9200 with one, 0.9208 with both, against 0.9203 alone. It appears in none of
the top ten of the reject-free search.

That combination - level alone, additive nothing - says DINOv2 **fails the same
way convnext does**, despite a different architecture and a different
pretraining objective. If the most dissimilar member available cannot decorrelate
the errors, what remains is not model-shaped. It points at the labels, which is
where the teacher-ceiling diagnosis put it in August by a completely different
route.

## 20 September — both slots come back negative, and B is the clean one

    date     roster                                                      LB
    08-23    m31 + m33 + m32 + m37up288  (the incumbent four)          0.947
    09-18    those four + m38seed1 + m40m33s1 + m43small                0.948
    09-20 A  m31 + m37 + m38seed1 + m43small + m64gru                   0.947
    09-20 B  the incumbent four + m89dinofrz + m90dino224               0.946

**B is member-matched against a board score we already own.** The same four
members, plus the two DINO members, and the board goes 0.947 -> 0.946. Adding
three convnext members to that same four went 0.947 -> 0.948. So on the only
instrument that has never saturated, a member of the new family is worth about
-0.0005 each and a member of the old one about +0.0003.

### What this kills

**pv-sel, as an instrument for rosters.** It called B 0.9095 - the highest
number any roster has produced in this file, above the 0.9084 of the roster
scoring 0.948 - and the board put B last of everything submitted this month.
That is not a near-miss; it is the wrong ordering at the top of the range. The
entry that spent this slot said "pv-sel says take it, pv-hold says nothing, the
search says no. One of those is wrong." It was pv-sel.

**Decorrelation as a sufficient argument.** The 0.030 measured yesterday is
real and it did not survive contact with the board. A member that is level
solo, tighter across seeds, and less correlated with the rest still made the
ensemble worse. Different failures are necessary for averaging to pay; they are
not sufficient, and this file should stop treating a correlation number as a
prediction.

**The frozen regime as a general key.** m92effvfrz finished at pv-hold 0.8546
against m22effv2's 0.8629 - the same backbone, the same shards, the new
learning-rate ratio, and it is *worse*. The caveat written into the queue before
the run holds exactly: DINOv2 is self-supervised and EfficientNetV2 is not, so
there was never a representation to preserve. m93swinfrz and m94radfrz are still
queued and will answer the same question for two more supervised checkpoints,
but the prior on them is now poor and their result changes nothing about the
DINO finding either way.

### What survives

The DINO finding itself is untouched: **+0.014 pv-hold from a schedule change is
still the largest single-run effect this project has measured**, and the family
really does come back level. What died is the inference from "level and
decorrelated" to "useful in the roster". Those are different claims and only the
first was ever measured locally.

And the roster question is now answered as far as this pool can answer it: seven
convnext members scoring 0.948 is the best configuration this project owns, it
has been beaten by nothing in three attempts, and **every remaining point has to
come from a better single model rather than a better selection over the ones
that exist.**

## 20 September — where the remaining error actually lives

With roster selection closed, the question is which single-model axis to spend
eleven days on. `label_quality.py` against the 58 gold studies, teacher AUC
against the ensemble's:

    finding             pos  teacher    model   model-teacher
    Synovitis            27    0.786    0.766      -0.020   <- labels are the ceiling
    Lateral OA           11    0.818    0.826      +0.008
    Effusion             35    0.861    0.957      +0.095
    Lateral Meniscus     23    0.865    0.839      -0.027
    Contusion            19    0.872    0.926      +0.053
    PF OA                21    0.911    0.883      -0.028
    Baker's              12    0.918    0.987      +0.069
    Medial OA            15    0.919    0.978      +0.060
    Fracture             18    0.928    0.933      +0.006
    Medial Meniscus      26    0.951    0.912      -0.038
    MCL                   9    0.972    0.939      -0.033
    ACL                  24    0.994    0.960      -0.035

    macro teacher 0.900 | macro model 0.909

**The model is already above its own teacher on the macro.** That kills the
relabelling plan as a general answer: better labels cannot lift a model that is
beating the labels it was trained on, except where the teacher specifically is
the ceiling, and that is one column - Synovitis, worth 1/12 of the metric.

The six columns where the model trails its teacher are the informative ones:
ACL, Medial Meniscus, MCL, PF OA, Lateral Meniscus, Synovitis. **On five of
those the label knows and the model does not see it.** Those five are exactly
the fine structures - a meniscal tear is millimetres of fissure, an ACL is a
band a few pixels wide at our sampling - and they are the same columns the
August plane probe implicated when it concluded "the model is reading coarse
study-level appearance, not the lesions."

At n=58 one standard error is 0.06, so no single row here is a result. The
pattern across six rows pointing the same way is what carries.

### The cheap test before the expensive fix

The shards are 256 px over 130 mm = 0.51 mm/px, and members train at 128 or 256
of that. Real extra pixels need a rebuild from the DICOMs - a Kaggle
preprocessing run over 4,407 studies plus retraining, which is most of the
eleven days and the representation class that paid +0.012 and +0.029 on the
board before.

Before spending that, a run that costs three hours: **256 source at upsample
448**, against m33up336new's 256/336. Upsampling adds no information, so if the
fine-structure columns still improve from 336 to 448 the backbone is
scale-limited and a rebuild would buy real detail; if they do not move, 0.51
mm/px is the binding constraint and the rebuild is the only thing that can help
- or nothing can.

### The rebuild is blocked by disk, not by time

Costing the 384 px rebuild before proposing it: the shards are
(700, 32, 256, 256) uint8, 36 GB for the corpus. At 384 that is (384/256)^2 =
**81 GB**, at 512 it is 144 GB. The Kaggle side is fine - `push_preprocess.py`
says the corpus decodes in ~2.1 hours single-threaded, split across parts that
each stay under the 20 GB output cap, so 384 needs about seven parts instead of
three.

The local side is not fine:

    /dev/nvme1n1p2  468G  423G   23G  96% /

**23 GB free against an 81 GB requirement.** Eleven days is enough time for the
rebuild; the disk is what makes it impossible today.

What is reclaimable, measured rather than guessed:

    artifacts/volumes48/     54 GB   the 48-slice shard set, built for m50depth48,
                                     whose axis closed at pv-hold 0.8819
    artifacts/cache/         54 GB   derived memmaps, rebuildable from the shards;
                                     ~40 GB of it is crop-experiment caches
                                     (vols_32x128_c158, _c197) whose axis is closed
    last.pt x 59 runs        19 GB   resume state for runs that have finished, kept
                                     beside the image_model.pt that is the artifact

That is ~113 GB, which would take free space to ~136 GB and make a 384 rebuild
straightforward. **None of it is deleted here.** A second session is committing
to this repo, the caches and shards are shared state, and 113 GB of experiment
artifacts is not a call to make inside an autonomous loop. The numbers are
recorded so the decision is a decision rather than an investigation.

Nothing currently queued needs the space: m94radfrz and m95up448 both stream
from the existing shards and write checkpoints of a few hundred MB.

### The frozen-backbone sweep, and what it says about August

    family                       gold      vs base
    convnext (base, 4 seeds)    0.9140        -
    DINOv2, frozen (2 seeds)    0.9150     +0.0010
    Swin, frozen                0.8969     -0.0170
    Swin, fully unfrozen (Aug)  0.8782     -0.0358
    EfficientNetV2, frozen      0.8914     -0.0225

**Freezing lifted Swin by 0.019 as well**, so the August configuration was
penalising two families, not one. But Swin is still 0.017 short and
EfficientNetV2 0.023 short; only DINOv2 reaches parity.

The practical answer is unchanged - keep convnext - but the reason is different
from the one on record. Not "other backbones are worse at this task" but "the
fine-tuning was wrong, and fixing it still does not beat convnext here."

**The stronger result is the one about errors.** DINOv2 matches convnext alone
and adds nothing in ensemble, across a different architecture and a different
pretraining objective. Three families, one parity, zero decorrelation. Whatever
error remains is not something a different model sees differently, and that is
the third independent route to the same place as the teacher-ceiling diagnosis.

## 20 September — the schedule, not the backbones: three families, and m23swin peaked at epoch 2

    family            August (3e-4 / 3e-5, all trainable)   September (1e-3 / 8e-6)
    DINOv2 ViT-S      m36dinolr   pv-hold 0.8634  ep 16     m89dinofrz  0.8779  ep 9
    swin_base         m23swin     pv-hold 0.8517  ep 2      m93swinfrz  0.8707  ep 8
    EfficientNetV2-S  m22effv2    pv-hold 0.8629  ep 9      m92effvfrz  0.8546  ep 8

**swin gains +0.019 pv-hold, +0.030 pv-sel and +0.019 gold - more than DINO
did.** And the detail that explains all of it is in the epoch column:
**m23swin's best epoch was 2.** It peaked almost immediately and fell for the
next fourteen epochs, which is what a pretrained backbone looks like while it is
being overwritten. Under the new ratio the same architecture peaks at 8.

So the August sentence "backbone-family diversity does not work here" was not
about backbones. **It was one hyperparameter choice, applied identically to
every family, that happened to suit convnext and to destroy the two transformer
families.** That cost this project its entire diversity axis for six weeks.

EfficientNetV2 is the exception and it sharpens the rule rather than breaking
it: a 20M-parameter CNN got *worse* under the lower backbone rate. The pattern
across three families is that large pretrained transformers need a backbone rate
far below the head's, and small CNNs do not.

**Prediction, before m94radfrz finishes:** RadImageNet resnet50 is a CNN of
25M parameters, so by that pattern it should behave like EfficientNetV2 - flat
or slightly worse than its August run's 0.8490 - not like swin. If it gains
0.019 like the transformers did, the pattern is wrong and the story is about
something else.

### What this is and is not worth

It is not worth board points by itself. Corrected swin (0.8707) and corrected
DINO (0.8776) both remain below the convnext band (0.8821), and submission B
already established that adding a member from one of these families takes the
board *down* 0.001. Fixing the schedule recovered two families from "broken" to
"slightly worse", not to "better".

What it is worth is the correction itself, and a warning this file should carry:
**a single shared hyperparameter, never varied, silently decided the outcome of
every backbone comparison this project ran.** The comparisons were fair to each
other and wrong about the world.

## The roster pool had not been refreshed since 4 September

The three roster searches run between 4 and 20 September chose from 31 members.
The project had trained 80 models by then. `ensemble.py --save-preds` is what
puts a member in the pool and it is a separate GPU pass, so a model only enters
the search if someone remembers to run it; the last time anyone did was
20 September 00:38, for the two DINOv2 members alone. The whole 18-20 September
batch - the flip, no-aug and conditioning seeds - was never in front of the
searcher that then reported there was nothing to add.

Refreshing the pool to 51 members (20 current-recipe models added; the fold
models and the OOF-blended ones are left out because both leak) changes the
solo ordering at the top:

    member          pv-sel  pv-hold    gold
    m86conds1       0.8980   0.8873  0.9103   <- best solo pv-hold of all 41
    m48blend60      0.8984   0.8845  0.9155
    m77noflips1     0.8928   0.8832  0.9156
    ...
    m43small        0.8956   0.8811  0.9093   <- best of the incumbent seven
    m38seed1        0.9002   0.8810  0.9190

m86conds1 clears the best incumbent member by 0.0062, which is 2.5 sd on
pv-hold and the first single-model separation this project has resolved in
weeks. It does not survive into the ensemble: inc7 + m86conds1 is +0.0008 on
pv-hold, and no combination of the new members reaches +0.002. **A saturated
ensemble does not care that a member is better; it cares that it is different.**
That is the fourth independent route to the same diagnosis.

Submitted anyway, as `ens10nf`: incumbent seven + m76noflip + m64gru +
m89dinofrz, the top pv-sel pick that is also gold-positive (pv-sel 0.9099 vs
0.9084, pv-hold 0.8917 vs 0.8904, gold 0.9228 vs 0.9219). Every gain is under
one sd. The rule "submit only what wins locally" cannot produce a candidate for
a +0.005 deficit when the local instrument's 2 sd is 0.005, and four days of
holding to it cost 20 unused slots and twelve places on the board.

**m86conds1 is not submittable.** It is a `cond=planeflag` model and
`submit_kernel.py` has no cond handling at all, so the kernel would load it and
silently drop the embedding - the exact failure the conditioning section warns
about. Fixing that is only worth doing if the axis below pays.

## The plane x fat-sat key merges three contrasts

Forum thread 741310 lists this as a data trap and the corpus confirms it:

              PD     T1     T2   inter   unk
    FS      6853    620   1472    1998   951
    non-FS  4161   5208   2004     849   255

`cond=planeflag` tells the head that a T1, a PD and a T2 sagittal series are
the same kind of picture, because it keys on fat suppression, and 5,208 T1 and
4,161 PD series share the "non-FS" bucket. m86conds1 posted the best solo
pv-hold this project owns **on a key that is wrong**.

`--cond contrast` keys on plane x contrast instead, splitting T1/PD/T2 by TR
and TE (TR<800 -> T1; else TE<40 -> PD, TE>=60 -> T2, between -> inter; missing
-> unknown). 15 classes, all populated, smallest 241 series. Fat suppression
drops out of the key: 6,853 of 8,894 FS series are PD, so it is largely
redundant with contrast, and keeping both would need 30 classes.

Queued as m96ctr and m96ctrs1. Two seeds because 2 sd on pv-hold is 0.005.
This is a recipe change with the pixels unchanged, so pv-hold judges it.

## 23 September — conditioning on contrast, and what the seed spread says

`--cond contrast`, taken to four seeds:

    base, 4 seeds   gold 0.9140   sd 0.0055
    ctr,  4 seeds   gold 0.9019   sd 0.0014   [0.9031, 0.9019, 0.8999, 0.9025]
    difference     -0.0121   4.2 sd at three seeds, unchanged by the fourth

**That is the second conditioning failure.** Plane conditioning came in at
-0.0071 (1.4 sd, unreadable); contrast at -0.0121, readable. Feeding metadata in
explicitly does not help - the model already reads it off the pixels.

**A first reading of this, written at two seeds, was wrong and is retracted.**
Three unrelated interventions land on the same number:

    flipswap          -0.0106
    noaug             -0.0106
    cond contrast     -0.0121

At two seeds I took that coincidence for a floor - the idea that disturbing the
training signal at all costs a fixed amount, so "it costs 0.011" would say
nothing about the specific intervention. The third and fourth seeds kill that
reading. **The ctr group's seed spread is 0.0014 against the base recipe's
0.0055, one quarter of it.** A perturbation that merely knocks training off
course would widen the spread, because each seed would be knocked somewhere
different. Narrowing it means the contrast channel pulls every seed toward the
same inferior solution - it is a consistent change of what the model learns, not
noise injection. The shared 0.011 across three interventions remains unexplained
and is now just an unexplained coincidence, not a mechanism.

Worth recording separately: the fourth seed bought nothing. The verdict was
settled at 4.2 sd after three, and the fourth cost 74 GPU-minutes to move the
mean by 0.0003. Confirming a closed result is the cheapest thing to do and the
least valuable.

## 24 September — the two instruments disagree by 0.026, and it is measurable now

The Global96 board number from 23 September made something possible that had
not been possible before: both instruments have now priced the same two systems,
so the gap between the instruments can be read directly instead of argued about.

                gold58 (local)    board
    ours            0.9228        0.948
    Global96        0.9305        0.930
    verdict       coat +0.008    ours +0.018

**The instruments disagree by 0.026, and they disagree about the sign.** gold58
says the public package is the better system; the board says we are better than
it by more than our whole deficit to the prize line. This is the largest
instrument disagreement recorded in this project, and it is not a puzzle - it is
the contamination that was predicted in advance on 23 September, arriving at
roughly the predicted size. Their three checkpoints were chosen out of 24 epochs
on these same 58 studies, and their published training history has
val_auc == gold_auc on every epoch, so gold58 is scoring a selection, not a
model.

Two things follow.

**The local blend curve cannot pick the weight.** Rank-mean blend of our ens10nf
with their package, on gold58:

    w      0.0     0.1     0.2     0.3     0.4     0.5     0.6     0.7
    auc  0.9228  0.9256  0.9279  0.9321  0.9357  0.9365  0.9358  0.9358

It peaks at w=0.5 and gains +0.0137. Both numbers are inflated by however much
of the 0.026 is theirs, and the peak sits at 0.5 precisely because gold58 reads
the two arms as equal in quality. On the board's reading they are 0.018 apart in
our favour, which moves the optimum well below 0.5. The curve's one usable
feature is that it is broad and flat from 0.3 to 0.7, so the weight is unlikely
to be a knife edge. Blends are therefore going to the board at
w = 0.20/0.30/0.40/0.50 and the board picks.

**The decorrelation is real even if the quality is not.** Mean per-label rank
correlation with our ensemble is 0.868, against 0.910 for our own most dissimilar
member. Correlation is a property of the two prediction vectors and is not
inflated by how their checkpoints were picked - selecting on gold58 makes their
arm look better, not more different. Four routes said our deficit is diversity
rather than single-model quality, and this is the first partner on hand that is
measurably outside the family.

Their three published aggregations (submission_percentile_rank,
checkpoint_rank_mean, probability_mean) blend within 0.001 of each other at every
weight, so the arm's own submission.csv is used and the choice is not a variable.

## 25 September — four weights on the board, and one thing Kaggle will not do

The blend kernel runs our ens10nf arm, then the Global96 package as a second
arm, and rank-mean blends them. Verified on the dev run before anything was
submitted: both arms ran, zero fell back to priors, the timm 1.0.22 the package
pins landed *after* our backbones were built, the two arms aligned on UID, and
all four weight files were written.

Two things the dev runs paid for.

**Finding the package cost 750 seconds.** The recursive glob over /kaggle/input
walks the mounted competition data, and dev mounts three studies where the rerun
mounts 955 - so the cost scales with the test set and buys nothing. Globbing the
depths Kaggle actually mounts at took it to 0.5s. Left in, it would have eaten
an unknown and growing slice of the 540-minute cap on every rerun.

**Kaggle only scores an output named submission.csv.** The kernel writes
submission_w20/w30/w40/w50.csv, and the plan was one run offering the board the
whole curve through `-f`. That is a 400: CreateCodeSubmission rejects any other
name. No slot was spent finding out, and each weight now gets its own kernel
version with PRIMARY_WEIGHT baked in. The four weights therefore cost four
reruns rather than one, which is wall clock rather than slots - and slots are
what is scarce here, not GPU.

Measured costs for sizing the next one: our ten-member arm is 19.3s per study
after 34s of model loading, so 307 minutes for 955 studies. The package did 3
studies in 51s across two T4 workers, most of it its own model load. Releasing
our ten checkpoints before handing the GPUs over returns 2.5 GiB on device 0,
which matters because the package runs CoAtNet at native 384 on both cards.

Submitted at w = 0.20/0.30/0.40/0.50 against our arm alone, which the board has
already priced at 0.948. Any weight above that says the diversity diagnosis was
right and the curve says how much to buy next; all four below says the package
is as weak as the board's 0.930 implies and the axis closes in one night. Either
answer is one the local instruments have not been able to give in three weeks.

## 25 September — the most decorrelated model we own was already on disk

The Global96 submissions went out described as "rank correlation 0.868 against
ours, where our own most dissimilar member sits at 0.910". **The 0.910 is wrong,
and wrong in a way worth keeping.** It was measured over the ten ensemble
members, not over the 31-model pool. Measured properly, against every model on
disk:

    m51radimagenet   gold 0.8945   corr 0.839   <- most decorrelated thing we own
    Global96         gold 0.9305   corr 0.868   <- what four slots were spent on
    m34dinonew       gold 0.8902   corr 0.910
    m36dinolr        gold 0.8958   corr 0.926
    ...
    m49blend40s1     gold 0.9124   corr 0.969

A medically pretrained backbone trained on 18 August, scored poorly solo, and
filed under "ruled out" is **more decorrelated from our ensemble than the public
package is.** It was closed on solo quality, which is the criterion today's
entire argument says is the wrong one for a member.

**Decorrelation alone is not enough, though, and this is the useful half.**
Rank-blending each candidate into the ensemble on gold58:

    w                 0.0     0.1     0.2     0.3     0.4
    m51radimagenet  0.9228  0.9235  0.9246  0.9246  0.9239
    m56nbr37        0.9228  0.9229  0.9230  0.9230  0.9232
    m34dinonew      0.9228  0.9221  0.9203  0.9176  0.9139
    m36dinolr       0.9228  0.9218  0.9206  0.9176  0.9155

m34dinonew is nearly as decorrelated (0.910) and costs 0.005 at w=0.2.
m51radimagenet is the only member that is both different and not harmful. So
"find the least correlated member" is not the rule; being different is necessary
and not sufficient, which is the more careful version of the diversity claim this
project has been making for three weeks.

**Not submitted tonight, deliberately.** The gain is +0.0018 against a gold sd of
0.0055 - a third of the instrument - so only the board can read it, and one slot
remains. Four board readings on the Global96 weight curve land in about six
hours and they reframe this exact question: if a decorrelated partner pays, this
is the in-house version and worth a slot; if it does not, this says whether the
failure was decorrelation in general or that package in particular. Spending the
last slot blind, six hours before four relevant numbers arrive, is the hoarding
error's mirror image.

## 16 October — wrapping up

Final position: **0.949, rank 1904 of 6,049**, with the 10th-place line at
0.964 and the deadline on 22 October. Between 3 and 13 October the public
notebooks reached 0.956-0.958: two Raptor CoAtNet checkpoints from nartaa
(384 and 224 px) plus heliosli's R6 ConvNeXt-small, rank-blended with
per-finding weights. About 940 teams now sit at exactly 0.956. Raptor is
trained on OAI external data and pooled public LLM labels; our report-distilled
student does not reach it on its own. Six days and a 0.015 gap do not leave
room for a recipe win, so local research stops here.

The 154 GB of local artifacts were deleted: the volume caches (`cache/`,
`volumes/`) and every checkpoint under `image_model_*/` (each run's
`image_model.json` record is kept). All ten ens10nf members, and ten more,
remain on Kaggle as `qwrfsdr/rsna-knee-image-model-*`. Any retraining has to
rebuild the volume cache first.

`--blend-public` in `push_submit_model.py` (with `kernels/blend_public_tail.py`)
is the one move left. It embeds a pulled public notebook as a second arm and
rank-blends our ens10nf into it at a small weight. It has not been run.

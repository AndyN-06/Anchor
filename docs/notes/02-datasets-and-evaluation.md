# Step 2 — Datasets and evaluation

The measuring apparatus, built before the thing it measures. Nothing here
estimates a pose. It loads real data, decides which ground-truth pose belongs
to which image, and turns an estimated trajectory into numbers. Every later
claim ("relocalizes to 3 cm") goes through this code. A bug here means a wrong
number with nothing to flag it, so the loader and metrics were checked against
the real fr1/xyz sequence as well as against fixtures.

## The TUM RGB-D loader

### Three clocks, none shared

A TUM sequence has three independent streams: colour (~30 Hz), depth (~30 Hz)
and motion-capture ground truth (~100 Hz). No timestamp appears in more than
one file, so loading a sequence is mainly an association problem.

`associate` uses the standard TUM rule. It considers every pair within
`max_diff`, takes the closest pair, removes both entries, and repeats. The
greedy removal makes the matching **one-to-one**. Nearest-neighbour lookup in
one direction can map two frames onto the same ground-truth pose and silently
count it twice in the error statistics.

`max_diff = 0.02 s` is TUM's default, 60% of a 30 Hz frame interval. Pairing
across a larger gap matches a frame to ground truth from a different instant.
During motion that timing error becomes position error, and the metric reports
it as estimator error.

Measured on fr1/xyz:

| | |
|---|---|
| Colour frames associated | 790 of 798 |
| Duration / path length | 26.57 s / 8.01 m (0.30 m/s mean, 0.60 m/s peak) |
| Colour→depth offset | mean 6.3 ms, max 17.2 ms |

The colour and depth images are **not simultaneous**. At this sequence's speeds
the offset is worth at most 7.7 mm of camera motion (offset × GT speed, worst
frame). That is below what Steps 3–5 will resolve, but it is not zero. It grows
on the faster fr1 sequences, and it is one reason depth-derived "ground truth"
correspondences get a pixel tolerance, not an exact test.

### Intrinsics are per camera

TUM calibrated the three Freiburg Kinects separately. fr1 has real distortion
(k1 = 0.26, k2 = −0.95, k3 = 1.16). fr3 shipped pre-rectified. The `ros`
default (fx = fy = 525, centred principal point) is a placeholder. Using the
wrong set gives a systematic reprojection error of several pixels, which shows
up as pose bias that no matcher can fix.

The values come from ORB-SLAM3's TUM1/2/3 configs. That was deliberate:
ORB-SLAM3 is the baseline, and identical intrinsics keep the comparison fair.
TUM's own calibration page did not render machine-readably when checked, so
these values have not been compared against it directly.

### Depth: scale, holes, and z vs. range

- **Scale is 5000 units/m for fr1 and fr3, but 5208 for fr2.** TUM published a
  per-camera depth correction for fr2, and ORB-SLAM3 folds it into the scale.
  Using 5000 there makes every fr2 depth ~4% too large. Downstream that looks
  like the system has a 4% scale error.
- **0 means "no measurement", and becomes NaN on load.** Frame 0 of fr1/xyz is
  74.8% valid (0.90–3.72 m). Holes come from projector shadows, specular
  surfaces, and anything past ~4 m. Left as 0, they unproject to a cluster of
  fake points at the camera centre. NaN makes any accidental use fail visibly.
- **Depth is z, not range.** A pixel gives a ray, and the depth image stores the
  point's z coordinate along the optical axis, not its distance along the ray.
  So the ray is scaled to unit z (`pixel_to_ray(normalize=False)`), then
  multiplied by depth. A unit-length ray times depth puts each point at
  `cos(angle off axis)` of its true depth, i.e. too close. For fr1, the
  unit-z ray norms are 1.11 at the top/bottom edge centres, 1.17 at the
  left/right edge centres and 1.25 at the corners, so the error would be 10%,
  14% and 20%. (The original docstring had this backwards, "too far, up to
  11%". It is corrected now.)
- Depth is undistorted on the way in: the ray comes from the Brown-Conrady
  inverse from Step 1, so fr1's distortion doesn't bend the point cloud.

### Ground-truth convention: measured, not assumed

TUM documents ground truth as the colour camera's optical centre relative to
the world, i.e. `T_world_cam`. Secondary sources say otherwise often enough
that I didn't trust the docs. If the direction is backwards, the trajectory
still looks plausible and is wrong.

`verify_groundtruth_convention` decides it from the data:

1. Unproject frame *i*'s depth into camera *i*.
2. Push the points into the world with the candidate convention, then into
   camera *j*, and project them.
3. Compare the depth this predicts at each pixel of *j* with the depth *j*
   actually measured there.

On fr1/xyz with a 30-frame gap:

| Hypothesis | Median depth residual |
|---|---|
| `T_world_cam` | **0.87 cm** (6374 points) |
| `T_cam_world` | 10.67 cm (5996 points) |

That is a 12.3× separation. The same residual also validates the fr1
intrinsics, the depth scale, the distortion model and the unprojection
together. If any of them were wrong, the correct hypothesis would not reach
sub-centimetre.

**The first version of this check was useless.** It compared point-cloud
centroids, got 0.128 m vs 0.116 m, and picked the wrong convention. A centroid
summarises thousands of points in 3 numbers, and both hypotheses moved it
about the same amount. A check that can't tell the hypotheses apart is worse
than no check, because it gives a confident wrong answer. The fix compares
per pixel, and the function returns the ratio so a weak result is visible.
It also warns below 2×. With a near-stationary camera, both hypotheses give
nearly the identity relative transform and the test loses its power.

## Alignment: Umeyama

An estimator recovers geometry in **its own** frame, usually anchored at the
first camera. Ground truth lives in the mocap frame. Comparing them directly
measures the difference between two arbitrary origins. So ATE first finds the
best transform between the trajectories, then measures what's left.

The free parameters of that transform are a claim about the system:

| Alignment | DoF | When |
|---|---|---|
| SE(3) | 6 (3 rotation + 3 translation) | Metric systems: RGB-D, stereo, VIO, relocalization into a metric map |
| Sim(3) | 7 (+ scale) | Monocular, where scale is unobservable |
| none | 0 | Estimate and GT already share a frame, e.g. **relocalization into a GT-built map** |

Umeyama solves the least-squares problem in closed form:

1. **Centre both point sets.** For any R and s, the best t aligns the
   centroids, so subtracting the means removes t from the problem.
2. **Rotation from the SVD of the cross-covariance** `Σ = (1/n) Yᶜᵀ Xᶜ = U D Vᵀ`,
   giving `R = U S Vᵀ`. This is orthogonal Procrustes.
3. **Scale** `s = tr(D S) / var(X)` (Sim(3) only).

The `S` matters. `U Vᵀ` is orthogonal but can have det = −1, which is a
reflection. That happens when the data is close to degenerate and noise
decides the sign. A reflected trajectory has the wrong handedness, which is
physically impossible, and it can fit *better* than the truth, so it yields a
suspiciously small error. `S = diag(1, 1, −1)` when `det(U)·det(V) < 0` forces
the best proper rotation instead.

**Minimum data: 3 non-collinear points.** Two points fix an axis, but rotation
about that axis is free. `alignment_is_degenerate` flags collinear input (the
second singular value of the centred cloud ≈ 0). A camera pushed straight down
a corridor is that case. Umeyama still returns a rotation, but one degree of
freedom is arbitrary, so the ATE can't be compared to anything.

## ATE and RPE

Both follow Sturm et al., IROS 2012, so the numbers are comparable to the
literature.

- **ATE** aligns once, then takes per-frame camera-centre distance. It
  measures **global consistency**: does the whole trajectory sit where it
  should?
- **RPE** compares the motion over an interval Δ, `E = (Q_i⁻¹Q_j)⁻¹ (P_i⁻¹P_j)`.
  Both sides are relative motions, so the arbitrary world frames cancel and no
  alignment is needed. It measures **local accuracy**: drift per second.

They disagree on purpose. Slow, smooth drift gives bad ATE and good RPE.
Frame-to-frame jitter with no drift gives good ATE and bad RPE. The test suite
builds both cases and asserts the disagreement in both directions.

Each reports RMSE together with the median, max and std. RMSE is dominated by
the tail: a run that is perfect for 95% of frames and lost for 5% can have the
same RMSE as one that is uniformly mediocre. Those are different problems, and
only the full distribution tells them apart.

### Worked check: does RPE return what was injected?

`scripts/check_tum_sequence.py` corrupts fr1/xyz ground truth with 2 cm/s
linear drift along x, plus 0.5 cm Gaussian noise per axis per frame. Then it
evaluates the result against the original.

RPE over 1 s reported **2.34 cm RMSE**. Predicted:

- The drift contributes exactly 2.0 cm per 1-second interval.
- The noise enters twice (once at each end of the interval), so the per-axis
  std of the difference is 0.5·√2 cm. Over 3 axes the RMS norm is
  `√(3·2)·0.5 = 1.22 cm`.
- The two are independent, so they add in quadrature: `√(2.0² + 1.22²) = 2.34 cm`.

It matches to the reported precision. That only works if the association,
interval pairing and relative-pose algebra are all right.

## Three evaluation traps, each now a test

### 1. Alignment can absorb the drift it is supposed to measure

SE(3) alignment has 6 free parameters. If the drift looks like something a
rigid transform can produce, the aligner removes it.

The test gives ground truth a steady climb in z, so its position has a
component linear in time. Then it adds a drift along x that is also linear in
time. A small rotation that mixes z into x produces exactly that displacement,
so the aligner applies it. About **80% of the drift vanishes from ATE**. On a
trajectory without the linear climb, 99% of the same drift survives, and
matches the closed form for linear drift after mean-removal: `d·T/√12` (the RMS
of a value spread uniformly over ±dT/2).

RPE reported the true 2 cm/s in both cases.

**Consequence:** an ATE number describes the estimate *and* the trajectory
shape it was measured on. It is not a lower bound on drift. Always report RPE
next to it.

### 2. ATE rotation error is mostly the alignment rotation

Umeyama fits **positions only**. The resulting rotation is then applied to the
estimate's orientations too, because a trajectory has to move rigidly. So
whenever the aligner rotates to fit positions, it adds an orientation offset
that has nothing to do with how well the orientations were estimated.

On fr1/xyz with **zero injected rotation error** and 2 cm/s drift (plus 0.5 cm
noise), the run gives:

```
ATE rotation  rmse 14.255 deg  median 14.255 deg  max 14.255 deg
NOTE: alignment itself rotated 14.26 deg
RPE rotation  rmse  0.000 deg/1s
```

The reported orientation error is the aligner's rotation and nothing else.
`ATEResult` now exposes `alignment_rotation_deg` and prints a warning when the
two agree. RPE rotation is the honest orientation metric. This is also why the
standard TUM ATE is translation-only.

### 3. Sim(3) hides a real scale error

The test scales a trajectory by 1.10 (every distance 10% too long):

| Alignment | ATE RMSE | Recovered scale |
|---|---|---|
| SE(3) | > 2 cm | 1 (fixed) |
| Sim(3) | < 1e-9 m | 1/1.10 |

For an RGB-D system, that 10% is a real bug, and Sim(3) makes it look perfect.
So the alignment mode is printed next to every ATE number, and the default is
SE(3).

## Pose error for relocalization (fixed after Step 2)

Relocalization doesn't produce a trajectory to align. It produces one pose per
query, compared directly to ground truth in the map frame. The per-query
metric is `pose_error`. Its original version had a silent trap, fixed now.

The original computed `T_gt⁻¹ · T_est` and reported the norm of its
translation. Whether that is the right number depends on the direction of the
inputs:

- **Poses (`T_world_cam`).** The translation is `R_gtᵀ (c_est − c_gt)`. Its norm
  is the camera-centre distance, which is correct.
- **Extrinsics (`T_cam_world`).** The norm is ‖t_est − t_gt‖. With `t = −R c`, this
  is not zero even when the centres coincide.
  - Worked case: centre exact, rotation off by 1°, camera 3 m from the world
    origin. The function reports `2 · 3 m · sin(0.5°) = 5.2 cm` of
    "translation error", which is rotation error times a lever arm.
  - The size depends on where the world origin happens to be.

PnP returns extrinsics, so Step 4 would have hit this on its first evaluation.
The existing PnP tests passed only because their rotation errors were tiny
(0.036°) and the camera was close to the origin. On the 1 px noise test the
old and new numbers are 1.130 mm and 1.120 mm. The bug was present, just too
small to see there.

The fix:

- `pose_error(T_est, T_gt, *, convention)` makes `convention` **required and
  keyword-only** (`"T_world_cam"` or `"T_cam_world"`). Every call site now
  states it.
- Translation is the camera-centre distance from the new
  `camera_center(T, convention)`, which returns `t` for a pose and `−Rᵀt` for
  extrinsics.
- Rotation is `‖log(R_estᵀ R_gt)‖`. That angle is the same in both conventions
  (transposing both rotations conjugates the residual, which preserves its
  angle). `log_so3` is used instead of `arccos((tr R − 1)/2)`, which loses
  precision near zero and returns NaN when rounding pushes the trace above 3.
- If both inputs carry frame labels and the labels differ, the function raises.

Tests:

- The same perturbation expressed both ways gives identical errors (1e-12).
- The 1°, 3 m case gives 0 translation and 1° rotation, while the naive formula
  gives the predicted 5.2 cm.
- Calling without `convention` is a `TypeError`.

## Bugs found while building this

- **`.gitignore` excluded the loader.** A bare `data/` pattern matches every
  directory named `data`, including `anchor/data/`. So the whole loader module
  was missing from every commit until Step 2, while tests passed locally.
  Fixed by anchoring the pattern to the repo root (`/data/`).
- **fr2 depth scale**, above. Not hit, because only fr1 is downloaded, but it
  would have appeared later as scale drift.
- **Centroid convention check**, above. It picked the wrong answer.
- **Unprojection docstring**, above. It had the direction and the magnitude
  wrong. The code was right.

## What's tested

126 tests in total for Steps 1–2. For this step:

- TUM file parsing: comments, malformed lines, quaternion order (`xyzw`), and
  a non-unit quaternion rejected as a likely column-order error. Round trip
  through `to_tum_file`.
- Association: one-to-one under contention, `max_diff` respected, clock
  `offset` applied.
- Umeyama: exact recovery of a random Sim(3), the reflection case, and
  collinear-input detection.
- ATE/RPE: invariance to a global rigid transform, the drift vs. jitter
  disagreement, ATE linear in drift rate, RPE equal to the injected drift
  rate, and the three traps above.
- Real data, via `scripts/check_tum_sequence.py`: association counts, the
  convention check, and the simulated-estimate evaluation.

## Not built yet

These are in ARCHITECTURE.md §3 but not built:

- **Relocalization aggregates:** recall at (2 cm, 2°), (5 cm, 5°) and
  (10 cm, 10°), median errors, failure rate counted as a miss, and runtime
  per stage. `pose_error` is the per-query building block.
- **7-Scenes loader.** The primary relocalization benchmark. Open questions:
  intrinsics (f ≈ 525 is a community convention, not a shipped calibration)
  and which GT version (depth-SLAM vs. the SfM pseudo-GT from Brachmann et al.
  2021). These loaders are needed for Steps 6–7, not for 3–5.
- **EuRoC loader.** GT is `T_world_body`, so it needs `T_body_cam0` from
  `cam0/sensor.yaml`.
- **Confidence-calibration plot** (inlier count / covariance vs. true error).
  It needs Step 5's output first.

## Known gaps, deliberate

- **Kinect colour and depth are not synchronised in hardware.** The offset is
  up to 17 ms, worth under 8 mm of motion on fr1/xyz. It is not compensated.
  For TUM-derived correspondence tests, a pixel tolerance absorbs it. For
  faster sequences it should be measured, not assumed small.
- **Ground truth is treated as exact.** TUM's mocap is good to roughly mm level,
  but the hand-eye calibration between the mocap markers and the optical centre
  adds its own error. Sub-centimetre differences between methods on TUM are
  near the floor of what this GT can resolve.
- **Intrinsics are from ORB-SLAM3's configs,** not checked against TUM's
  calibration page.

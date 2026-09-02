# Step 1 — Geometry foundations

The pieces every later stage depends on: a camera model, a rigid-transform type,
and a synthetic scene with exact ground truth. Nothing here is relocalization
yet. The point is that when the relocalization core is written, a wrong answer
can only come from the algorithm, not from a convention or a numerical bug
underneath it.

## Conventions, fixed once

| Quantity | Convention | Why it matters |
|---|---|---|
| Camera frame | +x right, +y **down**, +z **forward** (OpenCV) | ARCore/OpenGL use +y up, −z forward. Mixing them renders the overlay upside down and behind the camera. |
| Transform naming | `T_A_B` maps points **from** frame B **into** frame A: `p_A = T_A_B · p_B` | Adjacent frames cancel in a chain: `T_A_C = T_A_B @ T_B_C`. |
| Camera pose vs extrinsics | `T_world_cam` is the pose (its translation is the camera centre). `T_cam_world` is what projection needs. | TUM ground truth publishes `T_world_cam`; OpenCV's `rvec/tvec` **is** `T_cam_world`. |
| Points | `(N, 3)` rows; rotation applied as `P @ R.T` | NumPy is row-major, so a point is one contiguous row. |
| Units | metres, radians | No function in this repo takes degrees. |
| Pixels | `(0, 0)` is the **centre** of the top-left pixel | The half-pixel alternative silently biases calibration. |
| Quaternions | `(w, x, y, z)` internally; `xyzw` helpers named explicitly | TUM and ROS use `xyzw`. Getting this backwards produces a valid-looking wrong rotation. |

`SE3` carries optional `to_frame` / `from_frame` labels, and composing
mismatched frames raises. `T_world_cam @ T_world_cam` — the forgot-to-invert
bug — now fails at the line where it happens instead of 200 lines later as a
plausible pose.

## The math, briefly

**Projection.** `(X,Y,Z) → (X/Z, Y/Z) → distort → (fx·x_d + cx, fy·y_d + cy)`.
The perspective divide is the only nonlinearity in the ideal model, and it is
why pose from points needs a solver rather than a matrix inverse. It is also
why depth is unobservable from one view: scaling a point along its viewing ray
does not move its pixel. There's a test asserting exactly that.

**Intrinsics: 4 DoF** — `fx, fy` (focal length in pixels per axis; they differ
when pixels aren't square) and `cx, cy` (principal point). Skew is assumed zero.
Distortion adds 5 more: `k1, k2, k3` radial (slides points along rays from the
principal point — radially symmetric, so it cannot rotate anything) and
`p1, p2` tangential (lens not parallel to the sensor).

**Rotations: 3 DoF, 9 numbers.** A rotation matrix over-parameterises SO(3) with
6 constraints to maintain. The axis-angle vector `ω` — direction is the axis,
norm is the angle — has exactly 3 free parameters and no constraints, which is
why optimisers work in it. `exp_so3` (Rodrigues) goes `ω → R`; `log_so3` comes
back.

**Rigid transforms: 6 DoF.** `SE3.exp([ρ, ω])` maps a 6-vector to a transform.
The translation block is `V(ω)·ρ`, not `ρ`: when there's rotation, a tangent-space
translation traces an arc, and `V` accounts for the curvature. This is the update
an iterative pose refinement applies — `T ← SE3.exp(ξ) @ T` — which stays exactly
on the manifold, unlike adding numbers to a rotation matrix and hoping it stays
orthonormal.

## Two numerical bugs found while writing this

Both were caught by testing across scales rather than at one "typical" value.

**1. `SE3.log` lost 13 digits at small rotations.** The closed form for `V⁻¹`
contains `1 − θ·sin θ / (2(1 − cos θ))`, which subtracts two nearly equal numbers
and then divides by `θ²`. At θ = 1e-7 the true bracket value is ~8e-16 while the
cancellation error is ~1e-16 — and `1/θ² = 1e14` multiplies that noise straight
into the answer. **Recovered translation was off by 2 mm of pure numerical
error**, larger than the accuracy this project is trying to measure. Fixed by
switching to the Taylor series `1/12 + θ²/720 + θ⁴/30240` below θ = 1e-2, which
has no cancellation. Worst-case round-trip error went from 2e-3 m to 8e-13 m.

This matters in practice because a converging pose optimiser produces exactly
these tiny increments — the bug would only appear in the last iterations, as a
refinement step that stops improving or starts wandering.

**2. My distortion inverse claim was wrong, and I checked instead of assuming.**
I first wrote that fixed-point iteration diverges at the image corners for the
TUM freiburg1 coefficients. Measured: it doesn't. It reaches 0.02 px at the
corners in 5 iterations and machine precision by 20. It only breaks *outside*
the image (normalised radius ≳ 1.3), where the radial polynomial stops being
monotonic and the mapping isn't injective at all — no solver can invert that.
Newton on the analytic 2×2 Jacobian is used anyway: quadratic convergence (4
iterations vs 20), it still works in that outer region, and deriving the
Jacobian by hand is the same exercise as the pose Jacobian in refinement.

## What is tested, and against what

67 tests. Where an external reference exists, the test compares against it
rather than against my own algebra — a shared misunderstanding would otherwise
pass silently.

- `exp_so3` / `log_so3` vs `cv2.Rodrigues` — agree to 1e-12.
- `project` vs `cv2.projectPoints`, ideal and TUM-distorted — agree to 1e-9 px.
- `pixel_to_normalized` vs `cv2.undistortPoints` — agree to 1e-4 normalised
  (OpenCV's default 5 fixed-point iterations is the looser of the two).
- Distortion Jacobian vs central finite differences.
- Exp/log round trips parameterised across θ ∈ {0, 1e-12, …, π−1e-9}.
- `cv2.solvePnP` on the synthetic scene recovers ground truth to < 1e-6 m and
  < 1e-4°. This is the end-to-end proof that frames, intrinsics and distortion
  are mutually consistent — and it's the baseline the hand-written PnP gets
  measured against in a later step.

Negative tests matter as much: points behind the camera *do* produce
finite-looking pixels (the perspective divide flips their sign), so there's a
test asserting the validity mask catches them; and feeding a `wxyz` quaternion
to the `xyzw` reader must produce a *different* rotation, or the two orderings
would be indistinguishable and the TUM loader could silently use the wrong one.

## Known gaps, deliberate

- No fisheye/equidistant model. Brown-Conrady only.
- Outliers are drawn uniformly over the image. Real mismatches cluster on
  repeated structure (tiles, windows, brick), so this is the optimistic case and
  RANSAC results on it will look better than reality.
- `look_at` fails when the view direction is parallel to world-up. Correct
  behaviour, but a real trajectory looking straight down needs a different
  parameterisation.
- Rolling shutter is not modelled. Phone cameras have it, and it biases pose
  during fast motion.

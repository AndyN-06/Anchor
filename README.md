# Anchor

Persistent AR relocalization: map a space once, then recover the camera's 6-DoF
pose inside that map from a single image — so virtual content stays anchored
across sessions and devices.

This is the problem behind Niantic's Lightship VPS and Google's Visual
Positioning System. Place a cube in your living room, close the app, reopen it:
the cube is gone, because the phone has no idea where it is. Anchor fixes that.

## Status

Step 1 of N. Geometry foundations only — no relocalization yet.

| Step | Contents | State |
|---|---|---|
| 1 | Camera model, SE(3), synthetic ground truth | done |
| 2 | Dataset loaders (TUM RGB-D, EuRoC) + ATE/RPE evaluation | next |
| 3 | Feature extraction and matching | |
| 4 | PnP + RANSAC | |
| 5 | Pose refinement | |
| 6 | Map construction and persistence | |
| 7 | End-to-end relocalization + benchmark vs ORB-SLAM3 | |
| 8 | Phone AR rendering | |

## Design commitments

- **NumPy/OpenCV, no wrapper libraries that hide the geometry.** The core —
  matching, PnP, RANSAC, refinement — is implemented directly.
- **Every geometric quantity states its frame and units.** `T_world_cam`,
  metres, SE(3). "Pose" on its own is ambiguous and ambiguity is where the bugs
  live.
- **Evaluation against public ground truth, not demo videos.** ATE/RPE on TUM
  RGB-D and EuRoC, benchmarked against ORB-SLAM3, with relocalization success
  rate under lighting change, viewpoint change and occlusion.
- **Failure analysis is part of the deliverable.** Where it breaks and why.

## Layout

```
anchor/geometry/   camera model, rotations, SE(3), synthetic scenes
tests/             pytest suite, cross-checked against OpenCV
docs/notes/        what each step does and why
```

## Setup

```bash
pip install -e ".[dev]"
pytest -q
```

## Notes

- [01 — Geometry foundations](docs/notes/01-geometry-foundations.md)

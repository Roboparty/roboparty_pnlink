# GMR runtime used by PN→RP1

This directory vendors the runtime subset of the GMR fork used by this project.
The fork reports package version **0.2.0** and is based on commit
`d918bde83c5d72b0dbf979018793f6a74e529f10` from
[`zerojuhao/GMR`](https://github.com/zerojuhao/GMR), whose package metadata points
to [`YanjieZe/GMR`](https://github.com/YanjieZe/GMR).
The source checkout has local changes; it is not described as a clean official
upstream release. See `SOURCE_METADATA.json` for exact file fingerprints.

## Included

- `general_motion_retargeting/motion_retarget.py`: copied byte for byte from the
  current source checkout. Its task construction, offsets, gains and numerical
  operations are unchanged.
- `general_motion_retargeting/__init__.py`: a project-specific import surface
  loading only the task construction runtime.
- `general_motion_retargeting/params.py`: project-specific empty registries.
  `PNToRP1Retargeter` registers the selected RP1.3.2 model and generated PN
  calibration temporarily, constructs the GMR tasks, and removes the entry.
- `LICENSE`: the complete license text found in that fork, including its
  `Copyright 2026 Roboparty` notice. It applies to this GMR subset; it does not
  choose the license of this repository's own code or of the robot assets.

The project uses `GeneralMotionRetargeting` for model and mink task construction.
The project's `pn_rp1_retarget.py` performs its existing single QP per source
frame. It does not invoke GMR's legacy iterative `retarget()` loop.

## Excluded

Other robot models and mappings, SMPL-X, FBX/BVH/OptiTrack/XRobot data loaders,
GMR's viewer, torch utilities, sample recordings and calibration tools are not
part of this runtime. Use the project's Viser display and PN calibration tools.
This directory is deliberately not a complete replacement for general-purpose
GMR. No standalone GMR `setup.py` is included: the project resolves this bundled
directory directly and installs only the runtime dependencies.

## Runtime dependencies

`motion_retarget.py` imports `mink`, `mujoco`, `numpy`, `scipy`, `rich`, and an
unused `Cython.ptrdiff_t`. Cython remains required so this file can remain
identical to the currently used version. A supported QP backend (`daqp`) is
also required by this project's single-step solver.

Torch, SMPL-X, imageio, OpenCV and loop-rate-limiters are not dependencies of
this subset. Removing their eager imports changes package loading only, not
the retained task construction code or the project's retarget algorithm.

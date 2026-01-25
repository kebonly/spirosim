# Changelog - spiro_cfd

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-01-06
### Added
- `src/spiro_cfd/faxen/` module added
- `spiro_cfd.kernels.u_inf_poiseuille`
- `spiro_cfd.simulation` module with `simple_active_particle` simulation method
- `spiro_cfd.viz.plot_oriented_trajectory`
- Added `u_inf_poiseuille_2d` in `kernels` module

### Changed
- Updated docstrings and type annotations to `viz` functions `plot_quiver_field` and `plot_streamlines`
- 
### Removed
- `src/spiro_cfd/faxen/operators.py` removed and placed in the `../faxen/__init__.py`

## [0.1.0] - 2026-01-01
Starting to properly keep track of changes in this repository

# Changelog - spiro_analysis

## [Unreleased]
### Added
- `spiro_analysis.preprocess.merge_strobe_triplet` function added
- `spiro_analysis.preprocess.merge_strobe_directory` function added
- `spiro_analysis.preprocess.save_frames` added to decouple from `Experiment` class
- `spiro_analysis.preprocess.get_crop_coordinates` added to decouple from `Experiment` class
- `spiro_analysis.preprocess.CircleSelector` new feature to visualize crop preview

### Changed
- Got rid of the imports in the `spiro_analysis/__init__.py` file for now
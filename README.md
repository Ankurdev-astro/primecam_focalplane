# Prime-Cam Focal Plane

Tools and data for constructing and visualizing the Prime-Cam detector focal
plane for the Fred Young Submillimeter Telescope (FYST). The repository builds
configuration-driven, TOAST-compatible detector tables for individual
Prime-Cam modules and can place those modules in the telescope boresight frame.

## Repository contents

| Path | Description |
| --- | --- |
| `fp_scripts/fp_config.yaml` | Detector, wafer, module, and output settings for each module. |
| `fp_scripts/fp_primecam.py` | Builds an individual module detector table. |
| `fp_scripts/module_to_boresight.py` | Converts module-relative detector quaternions to boresight-relative quaternions. |
| `fp_scripts/visualize_fp.py` | Plotting helper for TOAST `Focalplane` objects. |
| `fp_files/` | Generated HDF5 detector tables. |
| `notebooks/visualize_fp_primecam.ipynb` | Example workflow for assembling and plotting the full focal plane. |
| `plots/fp_primecam_tube.pdf` | Example full focal-plane plot. |

The configured module keys are `f280`, `f350`, `f420`, `f850`, `f_fake1`,
`f_fake2`, and `f_fake3`.

Note: Only the `f280` noise parameters are set now. All other modules are replicas of the `f280`.

## Requirements

The scripts require Python 3.10 or newer and the following packages:

- NumPy
- Astropy
- PyYAML
- TOAST
- Matplotlib (for plots)
- `fyst-trajectories` (for module-to-boresight offsets)
- Jupyter (to run the example notebook)

This repository does not currently pin dependency versions. Install these
packages in the Python environment of your choice before running the scripts or
notebook.

## Build a detector table

Run `fp_scripts/fp_primecam.py` from the repository root, pass
`fp_scripts/fp_config.yaml` as the configuration, and select a module with the
required `--module` option. For example, use `--module f280` to build the 280
GHz module.

```bash
python fp_scripts/fp_primecam.py fp_scripts/fp_config.yaml --module f280
```

With the supplied configuration, this writes
`fp_files/primecam_f280_dettable.h5`. The Astropy `QTable` is stored at the
HDF5 path `dettable_stack`.

Run the command once for each module key to rebuild the complete set of
configured tables.

The output table contains detector identity and placement fields (`name`,
`quat`, `module`, `wafer_slot`, `band`, `pixel`, `pol`, and `index`) together
with polarization, beam, noise, band-center, and bandwidth metadata. Astropy
units are serialized with the table.

## Configuration

Each entry under `modules` in `fp_scripts/fp_config.yaml` has three sections:

- `detector` defines the band, pixel layout, polarization pattern, beam,
  sampling rate, and noise properties.
- `module` defines the module geometry and its three wafer placements.
- `output` defines the output directory, filename, HDF5 table path, and
  overwrite behavior.

Relative output directories are resolved from the directory containing the
configuration file. The output filename may use `{name}`, which is replaced by
the configured `module.name`.

## Convert modules to the boresight frame

The generated tables describe detectors relative to each module center.
`ModuleToBoresightOffset` in `fp_scripts/module_to_boresight.py` applies the
module's static Prime-Cam position and returns a copy of the table. Construct it
with a configured module name, such as `primecam_f280`, then call `apply()`.

```python
from fp_scripts.module_to_boresight import ModuleToBoresightOffset

f280 = ModuleToBoresightOffset("primecam_f280")
f280_boresight = f280.apply()
```

In the returned table:

- `module_quat` preserves the original module-relative quaternion.
- `quat` contains the boresight-relative quaternion.

The current position assignments are defined in `MODULE_POSITIONS` in
`fp_scripts/module_to_boresight.py`. Module clocking is currently assumed to be
zero for every module and is kept explicitly in `MODULE_CLOCKING_DEG`; either
update that mapping or pass `clocking_deg` to the constructor when measured
values are available.

To assemble all seven modules, convert each module table and combine the
results with Astropy's `vstack`. The notebook demonstrates this complete
workflow.

## Visualize the focal plane

Pass a detector table to TOAST's `Focalplane`, then call
`plot_focalplane_primecam` from `fp_scripts/visualize_fp.py`.

Set `outfile=None` for an interactive Matplotlib window. Optional arguments can
show detector labels, pixel centers, polarization directions, and per-module
face colors. See `notebooks/visualize_fp_primecam.ipynb` for the complete
multi-module conversion and visualization workflow.

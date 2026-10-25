# CM1 Sounding Editor

View, diagnose and edit CM1 `input_sounding` files (`isnd = 7` format), then save the edited profile under a new name.

## Install (macOS, Linux, Windows)

Requires Python 3.9+.

```bash
cd cm1-sounding-editor
python3 -m venv .venv
source .venv/bin/activate          # windows: .venv\Scripts\activate
pip install -e .
```

With conda instead: `conda create -n sounding python=3.12 && conda activate sounding && pip install -e .`

## Run

```bash
cm1-sounding-editor path/to/input_sounding
# or
python -m sounding_editor path/to/input_sounding
```

You can also start it with no file and use **File ▸ Open**, or drop a file on the window.

## Editing

- **Drag points** on the Skew-T (red = T, green = Td) or the hodograph. Dragging empty space pans; the scroll wheel zooms.
- **Smooth edits** (on by default) spreads a drag over neighbouring levels with a Gaussian of the chosen half-width. Turn it off to move exactly one level and keep the kink.
- **T drags keep** either the mixing ratio or the relative humidity at the edited levels.
- **Layer operations** act on a height range:
  - thermodynamics: set lapse rate, add to θ, well-mix θ and qv, scale qv, set RH, cap at saturation, remove superadiabatic layers, smooth θ/qv
  - winds: rotate shear, scale shear, shift winds, smooth winds
  - With **Smooth edits** on, the thermodynamic operations blend their edges over the smoothing half-width; off, they stop sharply at the layer edges.
  - **Carry wind change above the layer** shifts the winds above the layer with the layer top, so the shear above is unchanged.
- **Data table** tab: edit z, θ, qv, u, v directly (and surface pressure on the first row).
- Undo / redo (⌘Z / ⇧⌘Z) and **Edit ▸ Revert to original**.

## Comparing soundings

**File ▸ Add Comparison Sounding…** (⇧⌘O) or **Add…** in the **Compare** panel loads other input soundings read-only, alongside the one you are editing. Each gets its own color:

- Skew-T: T solid, Td dashed. Hodograph: a dashed line.
- Parameters: one extra column per comparison, computed with the same diagnostic settings (surface wind, storm motion). When files share a name (e.g. several `input_sounding`), columns are headed by their folder; hover for the full path.
- Untick a comparison to hide it from the plots (its column stays); select it and **Remove** to unload it.

Dropping several files on the window opens the first for editing and adds the rest as comparisons; hold ⌥ while dropping to add them all as comparisons. Comparisons stay loaded when you open a different sounding to edit.

## Saving

**Save As** opens your system's save dialog, defaulting to `<original>_edited` next to the original. Save is enabled only once the profile differs from the original. Overwriting the original requires an extra confirmation.

The output is written in the same format as CM1 reads it (5 columns, 3 decimals, same levels). By default a sidecar `<name>.edits.json` is also written, listing every operation and the original vs. edited parameters (toggle under **File**).

Before saving, the tool blocks on non-increasing heights, negative qv or non-finite values, and warns about supersaturation.

## How things are computed

- **Base state:** pressure is integrated hydrostatically in θv from the surface pressure, with CM1's constants (Rd = 287.04, Rv = 461.5, cp = 1005.7, g = 9.81). Height is the vertical coordinate and stays fixed; editing θ changes the pressure above it.
- **Surface wind:** the file has no surface wind. The 0 m wind used for shear, SRH and the hodograph is set under Diagnostics (same as lowest level, linear extrapolation, or zero). It is never written to the file.
- **CAPE / CIN:** pseudoadiabatic parcel (Bolton 1980 θe conserved), virtual-temperature correction, buoyancy integrated in height.
  - SB: surface parcel.
  - ML: pressure-weighted mean of the lowest 100 hPa.
  - MU: highest θe in the lowest 300 hPa.
- **Effective inflow layer:** Thompson et al. (2007), CAPE ≥ 100 J/kg and CIN ≥ −250 J/kg.
- **Storm motion:** Bunkers et al. (2000), with a 0–6 km mean wind and D = 7.5 m/s. You can also pick the left mover, the mean wind, or enter a custom motion.
- **Composites:**
  - STP: effective-layer version (with the CIN term, Thompson et al. 2012) and fixed-layer version.
  - SCP: effective version.
  - SHERBS3 and SHERBE: Sherburn and Parker (2014).
- **Critical angle** (Esterheld and Giuliano 2008) and the 0–500 m streamwise vorticity and streamwise fraction use the chosen storm motion.

The parameters are computed with NumPy rather than MetPy so they can update while you drag. The tests cross-check the parcel temperatures and SRH against MetPy.

## Scripting

The core works without the GUI, e.g. for building perturbed ensembles:

```python
from sounding_editor.sounding import Sounding
from sounding_editor import edits, params

snd = Sounding.read("input_sounding")
for f in (0.9, 1.0, 1.1):
    new = edits.scale_qv(snd, 0, 1500, f, sigma=250)
    new.write(f"input_sounding_qv{f:.1f}")
    print(f, params.compute(new).values["mlcape"])
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

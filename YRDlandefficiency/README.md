# Yangtze River Delta urban land-use efficiency

This directory contains the data, reproducible analysis code, and results used
for the Yangtze River Delta urban construction-land efficiency study.

## Directory structure

```text
YRDlandefficiency/
├── data/
│   ├── raw/          # City-level input workbook, 41 cities × 4 years
│   └── boundaries/   # Spatial boundary inputs used to construct weights
├── scripts/          # Efficiency, spatial, sensitivity, and audit code
└── results/
    ├── tables/       # Final six-sheet Excel results workbook
    ├── derived/      # Machine-readable intermediate and robustness results
    └── figures/      # Final publication figures (PNG and SVG)
```

## Data and methods

- Years: 1990, 2000, 2010, and 2020.
- Decision-making units: 41 prefecture-level cities.
- Input: construction-land area.
- Desirable outputs: resident population and real GDP.
- Undesirable output: city-level CO2 emissions.
- Static efficiency: global undesirable-output SBM under CRS and VRS;
  `OE`, `PTE`, and `SE = OE / PTE`.
- Dynamic efficiency: global Malmquist–Luenberger index with
  `GML = TC × EC = TC × PEC × SEC`.
- Spatial analysis: row-standardized Queen contiguity weights with the
  documented Zhoushan–Ningbo island correction; 999 permutations.

## Reproduction

Python 3.11+ and Node.js 20+ are recommended.

```bash
python -m pip install -r requirements.txt
python scripts/compute_metrics.py
node scripts/spatial_analysis.mjs
node scripts/spatial_change_analysis.mjs
node scripts/robustness_checks.mjs
python scripts/audit_efficiency.py
```

The frontier/grid sensitivity analysis is more computationally intensive:

```bash
python scripts/frontier_grid_sensitivity.py
```

All scripts resolve paths relative to this directory; no machine-specific
absolute paths are required.

## Primary outputs

- `results/tables/YRD_land_efficiency_results.xlsx`: formatted final results.
- `results/derived/computed_metrics.json`: city-year efficiency estimates and
  dynamic decomposition.
- `results/derived/spatial_results*.json`: global and local spatial statistics.
- `results/derived/robustness_results.json`: alternative spatial weights and
  robust convergence checks.
- `results/derived/frontier_grid_sensitivity.json`: frontier and grid
  sensitivity results.

Raw source measurements are preserved in `data/raw`; scripts write only to
`results/derived`.

# Validity-aware MOSFET HI and warning reproducibility package

This repository contains the reproducibility materials for the manuscript:

**Validity-aware Rds(on) health indicator and physics-guided residual GRU for MOSFET failure warning**

The package is organized for public archiving on GitHub and, preferably, Zenodo. It is intended to support the manuscript's main claims on measurement-validity-aware health indicator construction, train/test-level evaluation, warning coverage, calibrated warning probability, and Python-to-Simulink consistency checks.

## Repository contents

- `data/source_data/`: source CSV files used for the manuscript figures and tables.
- `data/supplementary_tables/`: supplementary CSV tables, including excluded-test criteria, seed-repeat metrics, warning-probability calibration, and EOL/RUL error summaries.
- `data/processed_nasa/`: processed NASA MOSFET run-level/window-level CSV files used by the scripts.
- `scripts/`: canonical Python scripts for data loading, model training/evaluation, seed repeats, sensitivity checks, warning probability, and Simulink export.
- `figure_scripts/`: Python scripts used to regenerate manuscript-style figures from source CSV files.
- `figures/vector_reference/`: editable/vector reference material for the circuit figure.

## Data scope

The repository includes processed CSV files and manuscript source data. It does not redistribute the original NASA raw dataset archive. The original data should be obtained from the NASA Prognostics Center of Excellence data repository:

https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Local low-voltage hardware records are represented through processed source data used in the manuscript. Raw oscilloscope or LabVIEW waveform files are not included in this public package because they contain local instrument/export metadata and are large. They can be made available from the author upon reasonable request.

## Suggested environment

The scripts were developed with Python 3.10 or later. Install the Python dependencies from:

```bash
pip install -r scripts/requirements.txt
```

Main scripts:

```bash
python scripts/run_nasa_mosfet_experiments.py
python scripts/run_pg_rgru_seed_repeats.py
python scripts/run_calibrated_warning_probability_64hidden_20260522.py
python scripts/run_rds_delta_sensitivity.py
```

Some scripts may require updating local path variables if the repository is moved to another machine. The source CSV files in `data/source_data/` are included so that manuscript figure/table values can be checked without re-running every model training step.

## Reproducibility notes

- Train/test splitting is performed at complete test-trajectory level rather than random window level.
- Excluded tests and objective exclusion criteria are listed in `data/supplementary_tables/Supplementary_Table_S1_excluded_tests.csv`.
- The low-current measurement-validity boundary is treated as a measurement feasibility condition before constructing corrected Rds(on)-based HI labels and valid training windows.
- Warning-probability summaries and seed-repeat results are provided as CSV files to support robustness checks.

## Archive DOI

GitHub repository: https://github.com/Tang-rui-wq/MR_MOSFET_validity_warning_reproducibility

Zenodo DOI: to be added after public archive deposition.

## License

No open-source license has been selected yet. Before making the repository public, choose a code/data license that matches the manuscript and institution requirements.

# BNetworks

Course assignment for **Bayesian Networks** (Radboud University, B2 / PER3). Implements three classic inference and learning algorithms for discrete Bayesian networks in Python.

## Contents

| Part | File | Algorithm |
| --- | --- | --- |
| 1 | [VE.py](Code/VE.py) | Variable Elimination (with `min_fill` / `min_neighbors` heuristics) |
| 2 | [MAP.py](Code/MAP.py) | MAP / MPE inference via max-product VE |
| 3 | [EM.py](Code/EM.py) | Expectation-Maximisation for CPT learning with hidden variables |

Supporting files:

- [run.py](Code/run.py) — entry point that runs all three parts and writes results to `log_endorisk.txt`
- [read_bayesnet.py](Code/read_bayesnet.py) — BIF parser (provided by course staff)
- `*.bif` — networks in Bayesian Interchange Format (`earthquake`, `endomcancer`, `endorisk_new`, `learned_endorisk`)
- `simulation_data_hid_names.dat` — simulated dataset for the endometrial-cancer network, with hidden variables
- [Code Report.pdf](Code/Code%20Report.pdf) / [Code_Report_Overleaf.tex](Code_Report_Overleaf.tex) — written report

## Running

```bash
cd Code
python run.py
```

Behaviour can be tuned via environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `VE_HEURISTIC` | `min_fill` | Elimination ordering heuristic |
| `EM_MODE` | `final` | EM run mode |
| `EM_SAMPLE_ROWS` | `500` | Rows sampled from the dataset |
| `EM_RESTARTS` | `5` | Number of random restarts |
| `EM_MAX_ITER` | `10` | Max EM iterations per restart |

Results are written to `Code/log_endorisk.txt`.

## Credits

Skeleton code and assignment design by Joris van Vugt, Moira Berens, and Leonieke van den Bulk (Radboud University).

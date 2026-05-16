# Replication Package: Conglomerate Mergers: Effects on Growth and Competition

This repository contains the replication code for the paper "Conglomerate Mergers: Effects on Growth and Competition" by Simon Blöthner.

Find the article [here](latex/collaboration.pdf).

## Overview

This project investigates the effects of profit-sharing arrangements on market concentration using an agent-based simulation model. The analysis examines how different levels of collaboration (profit sharing) between firms affect market outcomes such as the Gini coefficient, market share distribution, and conglomerate formation.

## Repository Structure

```
IOxEE/
├── programs/                    # Simulation and analysis code
│   ├── collaborative_growth.py  # Core simulation model
│   ├── parallel_counterfactuals.py
│   ├── generate_robustness_scenarios.py
│   ├── merge_counterfactual_results.py
│   ├── compare_parametrizations.py
│   ├── visualize_counterfactuals.py
│   └── requirements.txt
├── latex/                       # Paper and figures
│   ├── collaboration.tex        # Main manuscript
│   ├── collaboration.pdf        # Compiled paper
│   └── figures/                 # Generated figures
└── README.md
```

## Requirements

- Python 3.9+
- SLURM cluster (for full replication) or local machine (for testing)

Install dependencies:
```bash
cd programs
pip install -r requirements.txt
```

## Replication Instructions

The analysis uses a robustness design that tests model sensitivity across 54 parameter scenarios (4 cost function types × ~13 parameter variations each).

### Step 1: Generate SLURM Submission Scripts

```bash
cd programs
python generate_robustness_scenarios.py
```

This creates:
- `robustness_slurm_scripts/` - Individual SLURM job scripts for each scenario
- `submit_all_robustness.sh` - Master submission script

### Step 2: Submit Jobs to SLURM Cluster

```bash
./submit_all_robustness.sh
```

This submits 54 scenarios as SLURM job arrays. Each scenario runs:
- 50 counterfactual experiments
- 26 share values (0.00 to 0.50 in 0.02 increments)
- 10,000 simulation steps per experiment

**Resource requirements per job:**
- 10 CPU cores
- 10GB memory
- Up to 24 hours runtime (varies by scenario)

Monitor progress:
```bash
squeue -u $USER
```

### Step 3: Merge Results

After all jobs complete:
```bash
python merge_counterfactual_results.py --results_dir robustness_results/<scenario_name>
```

Or merge all scenarios:
```bash
for dir in robustness_results/*/; do
    python merge_counterfactual_results.py --results_dir "$dir"
done
```

### Step 4: Run Sensitivity Analysis

```bash
python compare_parametrizations.py --results_dir results --dashboard_dir sensitivity_plots
```

This generates:
- Controlled regression results across all scenarios
- Sensitivity analysis plots
- Cross-parametrization comparisons

### Step 5: Generate Publication Figures

```bash
python visualize_counterfactuals.py
```

## Model Parameters

### Baseline Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `markets` | 100 | Number of independent markets |
| `firms_per_market` | 100 | Firms per market |
| `steps` | 10,000 | Simulation timesteps |
| `merge_thresh` | 0.05 | Probability of merger attempt |
| `lookback` | 50 | Exit decision lookback window |
| `share` | 0.00-0.50 | Profit sharing parameter (varied) |

### Cost Function Types
The model supports four management cost functions:
- `linear`: $c(n) = c_0 + c_1 \cdot n$
- `quadratic`: $c(n) = c_0 + c_1 \cdot n + c_2 \cdot n^2$
- `exponential`: $c(n) = c_0 \cdot e^{c_1 \cdot n}$
- `power_law`: $c(n) = c_0 \cdot n^{c_1}$

Where $n$ is the conglomerate size.

### Robustness Variations
Each parameter is varied one-at-a-time from baseline:
- `markets`: 50, 100, 150
- `firms_per_market`: 50, 100, 150
- `merge_thresh`: 0.01, 0.05, 0.1
- `lookback`: 10, 50, 100
- Cost parameters: function-specific variations

## Output Files

### Intermediate Results
- `robustness_results/<scenario>/share_X.XX_chunk_*.pkl` - Per-chunk experiment results
- `robustness_results/<scenario>/counterfactual_results_final.pkl` - Merged scenario results

### Final Outputs
- `sensitivity_plots/` - Sensitivity analysis figures
- `latex/figures/` - Publication-ready figures

## Local Testing

For testing without a SLURM cluster:
```bash
python parallel_counterfactuals.py \
    --share 0.1 \
    --counterfactuals 5 \
    --n_cores 4 \
    --markets 50 \
    --firms_per_market 50 \
    --steps 1000 \
    --scenario_name local_test
```

## File Descriptions

| File | Description |
|------|-------------|
| `collaborative_growth.py` | Core agent-based simulation model with Numba-optimized merger and exit logic |
| `parallel_counterfactuals.py` | Parallelized experiment runner for counterfactual analysis |
| `generate_robustness_scenarios.py` | Generates all 54 robustness scenarios and SLURM submission scripts |
| `merge_counterfactual_results.py` | Aggregates results from parallel job chunks |
| `compare_parametrizations.py` | Cross-scenario sensitivity analysis with controlled regressions |
| `visualize_counterfactuals.py` | Generates publication-quality figures |

## Citation

If you use this code, please cite:

```bibtex
@article{blothner2025conglomerate,
  title={Conglomerate Mergers: Effects on Growth and Competition},
  author={Bl{\"o}thner, Simon},
  year={2026}
}
```

## Contact

Simon Blöthner
Austrian Institute of Economics and Social Philosophy
Operngasse 6/1, 1010 Wien

Department of Law, Business Administration & Economics
University of Bayreuth
Universitätsstraße 30, 95447 Bayreuth, Germany

Email: s.bloethner@austrian-institute.org

## License

MIT License

Copyright (c) 2026 Simon Blöthner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

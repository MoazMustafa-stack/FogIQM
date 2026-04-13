# FogIQM: Fog-Assisted Internet Quality Monitoring System

> **Paper:** *"FogIQM: A Fog-Assisted Internet Quality Monitoring System for Latency-Optimized IoT Networks"*
> **Institute:** Vellore Institute of Technology, Vellore, India

---

## Overview

FogIQM is a simulation codebase for evaluating performance and latency optimization in fog-assisted IoT network quality monitoring.

---

## Requirements

```bash
pip install numpy matplotlib scipy tabulate
```

---

## Usage

```bash
python FogIQM_Simulation_Code.py
```

---

## Outputs

The simulation produces the following **printed tables** and **saved plots**:

| Output | Type |
|--------|------|
| Per-method latency / jitter / accuracy (Table II) | Printed Table |
| Ablation study results | Printed Table |
| `fogiqm_latency_comparison.png` | Plot |
| `fogiqm_jitter_cdf.png` | Plot |
| `fogiqm_mos_timeline.png` | Plot |
| `fogiqm_utilisation.png` | Plot |

---

## Citation

If you use this code, please cite the associated paper *"FogIQM: A Fog-Assisted Internet Quality Monitoring System for Latency-Optimized IoT Networks"*.

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

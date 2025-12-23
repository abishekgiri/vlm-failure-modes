# The Confidence Trap: Perturbation-Induced Miscalibration in Vision–Language Models

**Author:** Abishek Kumar Giri (Stockton University)

## 🚨 Overview
This repository contains the official code and data for the research paper **"The Confidence Trap"**. We identify a critical failure mode in Vision-Language Models (VLMs) like LLaVA-1.5: **Perturbation-Induced Confidence Bias**.

Contrary to the expectation that degrading an input image (via adversarial noise) should *increase* model uncertainty, we find that the model often becomes **more confident** (lower entropy) as the visual signal is corrupted. This "Confidence Trap" suggests that VLMs may revert to strong language priors when grounding fails, hallucinating with high certainty.

## 📊 Key Findings

| $\epsilon$ (Perturbation) | Mean $\Delta$ Entropy | Behavior |
|:-------------------------:|:---------------------:|----------|
| 0.1                       | $\approx 0.000$       | Unstable / Random |
| 0.2                       | $-0.018$              | Emerging Bias |
| **0.3**                   | **$-0.028$**          | **Consistent Overconfidence** |
| 0.5                       | $-0.050$              | Strong Confidence Amplification |

*Negative $\Delta$ indicates the model is MORE confident on the adversarial input.*

![Entropy Shift](results/figures/figure_1_epsilon_delta.png)

## 🚀 Quick Start: Reproduce in Colab

The easiest way to reproduce our results (including the OOM-safe adversarial proxy) is to use the self-contained notebook:

1.  Download `vlm_failure_modes.ipynb`.
2.  Upload it to **Google Colab**.
3.  Set the runtime to **T4 GPU**.
4.  Run all cells.

The notebook will:
- Clone the LLaVA repository.
- Install all dependencies.
- Run the experiment with $\epsilon=0.1$ and $\epsilon=0.3$.
- Generate the entropy report.

## 📂 Repository Structure

```
vlm-failure-modes/
├── paper/                  # Research Paper Artifacts
│   └── vlm_failure_modes_paper.md   # Final Paper Draft
├── vlm_failure_modes.ipynb # Self-contained reproduction notebook
├── experiments/            # Experiment Orchestration
│   └── entropy_analysis.py # Main script: Model loading + Entropy probing
├── attacks/                # Adversarial Methods
│   └── pgd_visual.py       # PGD and Safe Proxy implementations
├── probes/                 # Measurement Tools
│   └── entropy.py          # Shannon entropy calculation
├── results/                # Data and Visuals
│   ├── entropy_summary.md  # Tabulated experiment results
│   └── figures/            # Generated plots
└── requirements.txt        # Python dependencies
```

## 🛠️ Local Installation

If you prefer to run locally (requires ~16GB VRAM for LLaVA-1.5-7B):

```bash
# 1. Clone LLaVA (Required dependency)
git clone https://github.com/haotian-liu/LLaVA.git
pip install -e LLaVA

# 2. Install Project Dependencies
pip install -r requirements.txt

# 3. Run Experiments using the Python Path to include local modules
PYTHONPATH=.:LLaVA python experiments/entropy_analysis.py
```

## 🛡️ License
This project is open-source and available for academic and research use.

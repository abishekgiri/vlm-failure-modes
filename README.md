# The Confidence Trap: Perturbation-Induced Miscalibration in Vision–Language Models

**Author:** Abishek Kumar Giri (Stockton University)

##  Overview

Vision–Language Models (VLMs) are increasingly deployed in safety-critical domains such as autonomous systems, medical imaging, and multimodal agents. A core assumption in these systems is that model uncertainty (e.g., entropy) increases when inputs become noisy, ambiguous, or corrupted.

This project demonstrates a counter-intuitive and dangerous failure mode in VLMs:

> **Visual degradation can reduce model uncertainty.**

We identify and empirically validate a phenomenon we term the **Confidence Trap**, where adversarial or noisy visual inputs cause a VLM to become more confident, not less.

##  Key Finding (TL;DR)

**Entropy-based uncertainty measures fail under visual perturbations.**

For LLaVA-1.5-7B:
*   Moderate visual noise ($\epsilon \approx 0.3$) reduces output entropy.
*   The model becomes systematically overconfident.
*   Standard entropy-threshold defenses fail catastrophically.

This undermines entropy as a safety signal in multimodal systems.

##  Core Results

We define **Entropy Delta** as:
$$\Delta = H_{\text{adv}} - H_{\text{clean}}$$

| Perturbation Strength ($\epsilon$) | Mean Entropy $\Delta$ | Behavior |
|:--------------------------------:|:---------------------:|:---------|
| 0.1                              | $\approx 0.000$       | Unstable / Random |
| 0.2                              | $-0.018$              | Emerging Bias |
| **0.3**                          | **$-0.028$**          | **Consistent Overconfidence** |
| 0.5                              | $-0.050$              | Strong Confidence Amplification |

*Negative $\Delta \Rightarrow$ higher confidence on corrupted inputs.*

** Entropy decreases as visual degradation increases.**

###  Experiments Included
*   [x] Controlled perturbation sweep ($\epsilon \in \{0.1, 0.2, 0.3, 0.5\}$)
*   [x] Multiple independent runs per $\epsilon$
*   [x] Token-level entropy analysis
*   [x] Defense failure test: entropy thresholding
*   [x] Mixed-precision (FP16) inference
*   [x] GPU-constrained, reproducible setup

##  Quick Start: Reproduce in Colab

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

##  Local Setup & Installation

**1 Clone LLaVA**
```bash
git clone https://github.com/haotian-liu/LLaVA.git
```

**2️ Install Dependencies**
```bash
pip install -U \
  transformers>=4.36.0 \
  accelerate \
  sentencepiece \
  einops \
  timm \
  pillow \
  numpy \
  scipy \
  tqdm \
  pyyaml
```
* Experiments were run on a Tesla T4 (16GB) with FP16 inference.*

##  Running Experiments

**Sanity Check**
```bash
PYTHONPATH=.:LLaVA python experiments/sanity_check.py
```

**Entropy Analysis**
```bash
PYTHONPATH=.:LLaVA python experiments/entropy_analysis.py
```

**Outputs:**
*   Clean entropy
*   Perturbed entropy
*   Entropy delta
*   Saved to `results/entropy_report.txt`

*Re-run the script to collect multiple trials.*

##  Repository Structure
```
.
├── paper/
│   ├── paper.tex            # LaTeX Source
│   └── vlm_failure_modes_paper.md # Markdown Draft
├── attacks/
│   └── pgd_visual.py        # PGD-inspired visual perturbation proxy
├── probes/
│   └── entropy.py           # Token-level entropy computation
├── experiments/
│   ├── sanity_check.py      # Model loading verification
│   └── entropy_analysis.py  # Main experiment pipeline
├── results/
│   ├── entropy_report.txt   # Saved entropy outputs
│   └── figures/
│       └── figure_1_epsilon_delta.png
├── README.md
└── vlm_failure_modes.ipynb  # Colab-compatible notebook
```

##  Defense Failure Experiment

We test a common safety heuristic:
> **Reject outputs if entropy exceeds a threshold**

**Result:**
Perturbed inputs frequently exhibit *lower* entropy than clean inputs, allowing corrupted samples to bypass entropy-based filters more easily than benign ones. This demonstrates **inverted safety behavior**.

##  Interpretation: Why This Happens

When visual information degrades:
1.  The vision encoder output becomes incoherent.
2.  The language decoder defaults to strong linguistic priors.
3.  The model generates generic, high-probability text.
4.  Entropy drops — confidence rises — grounding fails.

**This is the Confidence Trap.**

##  Limitations
*   Single architecture (LLaVA-1.5-7B)
*   Dense visual noise only (no semantic attacks)
*   Entropy measured on generated tokens only
*   No human evaluation of semantic correctness

##  Future Work
*   Test across VLMs (BLIP-2, GPT-4V, Gemini-Vision)
*   Semantic and object-level perturbations
*   Layer-wise uncertainty tracing
*   Vision–language calibration objectives
*   Confidence-aware decoding strategies

##  Paper
The full paper is included in this repository and is submission-ready for **NeurIPS / ICML Workshops**, **AI Safety venues**, and **Multimodal ML tracks**.

If you use or build on this work, please cite appropriately.

##  Author
**Abishek Kumar Giri**
Computer Science
Stockton University



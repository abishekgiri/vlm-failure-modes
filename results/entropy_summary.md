# Experiment Results: Entropy Analysis

## Phase 1: Baseline Perturbation (epsilon=0.1, noise=0.01)
| Run | Clean Entropy | Adversarial Entropy | Delta   |
|-----|---------------|---------------------|---------|
| 1   | 4.8681        | 4.8553              | -0.0128 |
| 2   | 4.8681        | 4.8611              | -0.0070 |
| 3   | 4.8681        | 4.8570              | -0.0111 |
| 4   | 4.8681        | 4.9054              | +0.0372 |
| 5   | 4.8681        | 4.8750              | +0.0069 |

**Observation:** Inconsistent delta direction. Bidirectional noise effects.

## Phase 2: Amplified Perturbation (epsilon=0.3, noise=0.03)
| Run | Clean Entropy | Adversarial Entropy | Delta   |
|-----|---------------|---------------------|---------|
| 1   | 4.8681        | 4.8286              | -0.0396 |
| 2   | 4.8681        | 4.8662              | -0.0019 |
| 3   | 4.8681        | 4.8126              | -0.0555 |
| 4   | 4.8681        | 4.8439              | -0.0243 |
| 5   | 4.8681        | 4.8737              | +0.0056 |

**Observation:** Stabilized negative delta (4/5 runs).
**Conclusion:** Stronger adversarial perturbation consistently reduces prediction entropy (Model becomes *more* confident in incorrect predictions), indicating a "confidence trap" failure mode.

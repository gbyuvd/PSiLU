# PSiLU: Parametric Sigmoid Linear Unit

**Technical Report**  
*Version 1.0 — April 2026*

---

<img width="500" height="300" alt="PSiLUsweep" src="https://github.com/user-attachments/assets/85cb7598-411d-4cc1-87ad-c7d554da9678" />
<img width="500" height="300" alt="PSiLUFullSweep" src="https://github.com/user-attachments/assets/350c06e9-8af3-4ed3-9657-28bb1d4654f7" />

## Abstract

We introduce **PSiLU** (Parametric Sigmoid Linear Unit), a learnable activation function that extends the Swish/SiLU family with per-channel adaptive parameters. PSiLU is motivated by the desire to navigate a middle ground between discrete activation selection (e.g., Mixture-of-Experts approaches) and unbounded function approximation (e.g., KANs, PAUs). By bounding the function search within the family of smooth gated activations and introducing learnable steepness and shift parameters, PSiLU provides per-neuron adaptability with minimal computational overhead. We present the formulation, implementation, and preliminary results on FashionMNIST and a constrained parity task. This report serves as a technical description to facilitate reproducibility and invites further evaluation across diverse architectures and domains.

---

## 1. Introduction

### 1.1 Motivation

Modern neural networks predominantly employ fixed, homogeneous activation functions (e.g., ReLU, GELU, Swish) applied uniformly across all neurons. While effective, this design imposes a rigid structural prior: every neuron must adapt its weights to work within the same nonlinearity.

Recent research has explored two divergent directions to increase activation flexibility:

1.  **Discrete Selection (MoE-style):** Approaches that select from a menu of activation functions per neuron or layer. These can introduce routing complexity and training instability due to discrete decisions.
2.  **Unbounded Approximation (KAN/PAU):** Approaches that learn arbitrary functions via splines or rational approximations. These offer high expressivity but can incur significant computational cost and optimization challenges.

PSiLU is designed as a **pragmatic middle ground**. Rather than searching over arbitrary functions or making discrete selections, PSiLU bounds the search within the family of **smooth gated activations**, which have demonstrated strong empirical performance in modern architectures. By introducing two learnable parameters per channel, PSiLU allows each neuron to adapt its activation shape (steepness, shift, and non-monotonicity) while maintaining:
- **Differentiability:** Fully smooth gradients for stable optimization.
- **Efficiency:** Negligible parameter overhead and hardware-friendly operations.
- **Simplicity:** Drop-in replacement for standard activations.

### 1.2 Scope

This report provides:
- A formal description of PSiLU and its relationship to existing activations.
- Implementation details for reproducibility.
- Preliminary empirical results on classification and parity tasks.
- A call for community evaluation on broader benchmarks.

We present results as observed without claims of universal superiority. Further validation across architectures, datasets, and training regimes is encouraged.

---

## 2. Method

### 2.1 Formulation

For an input $x \in \mathbb{R}$ to a neuron in channel $c$, PSiLU is defined as:

$$
\text{PSiLU}_c(x) = x \cdot \sigma(\beta_c x + \delta_c)
$$

where:
- $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid function.
- $\beta_c \in \mathbb{R}$ is a learnable **steepness** parameter for channel $c$.
- $\delta_c \in \mathbb{R}$ is a learnable **shift** parameter for channel $c$.

For a tensor input $\mathbf{X} \in \mathbb{R}^{B \times C \times \dots}$, the parameters $\beta_c$ and $\delta_c$ are broadcast across all non-channel dimensions.

### 2.2 Parameter Interpretation

| Parameter | Effect |
|-----------|--------|
| $\beta_c$ | Controls the **steepness** of the sigmoidal gate. Larger positive values produce sharper transitions. When $\beta_c < 0$, the activation becomes **non-monotonic**, enabling **inhibitory gating** where a signal is suppressed as its magnitude increases—a behavior that fixed monotonic functions like ReLU or GELU cannot replicate without additional layers. |
| $\delta_c$ | Controls the **lateral shift** of the gate, effectively learning a per-channel threshold for activation. |

### 2.3 Relationship to Existing Activations

PSiLU generalizes several common activations:

- **Swish/SiLU:** Recovered when $\beta_c = 1, \delta_c = 0$.
- **ReLU:** Approximated as $\beta_c \to \infty, \delta_c = 0$ (sigmoid approaches step function).
- **Linear:** Recovered when $\beta_c = 0, \delta_c \to \infty$ (sigmoid approaches 1).

Unlike fixed activations, PSiLU allows each channel to discover its own operating point within this function family during training.

### 2.4 Initialization

Parameters are initialized from uniform distributions to encourage initial diversity:

$$
\beta_c \sim \mathcal{U}(\beta_{\min}, \beta_{\max}), \quad
\delta_c \sim \mathcal{U}(\delta_{\min}, \delta_{\max})
$$

Default ranges used in our experiments:
- $\beta_{\min} = -1.0, \beta_{\max} = 5.0$
- $\delta_{\min} = -5.0, \delta_{\max} = 5.0$

By employing stochastic initialization for $\beta$ and $\delta$, PSiLU performs **structural symmetry breaking** at initialization. This ensures that gradient descent begins with a heterogeneous population of feature extractors, potentially mitigating neuron co-adaptation in early training phases.

### 2.5 PyTorch Implementation

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class PSiLU(nn.Module):
    """
    Parametric Sigmoid Linear Unit (PSiLU)
    Formula: f(x, v) = x * sigmoid(beta * v + delta)
    
    This implementation supports:
    1. Stochastic Init: Uniform distribution to break symmetry.
    2. Dynamic Broadcasting: Works for (B, C), (B, C, H, W), and (B, C, D, H, W).
    3. External Gating: Can accept a separate gate_input (GLU-style).
    """
    def __init__(self, channels: int, beta_range: Tuple[float, float] = (-1.0, 5.0), 
                 delta_range: Tuple[float, float] = (-5.0, 5.0)):
        super().__init__()
        self.channels = channels
        self.beta = nn.Parameter(torch.empty(channels))
        self.delta = nn.Parameter(torch.empty(channels))
        
        # Stochastic initialization: Breaking symmetry from the start
        nn.init.uniform_(self.beta, *beta_range)
        nn.init.uniform_(self.delta, *delta_range)
        
    def forward(self, x: torch.Tensor, gate_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        # If no gate_input is provided, it acts as a self-gated unit (Swish-like)
        v = gate_input if gate_input is not None else x
        
        # Handle broadcasting for multidimensional tensors (CNNs)
        if v.dim() > 2:
            view_shape = (1, self.channels) + (1,) * (v.dim() - 2)
            beta = self.beta.view(*view_shape)
            delta = self.delta.view(*view_shape)
        else:
            beta = self.beta
            delta = self.delta

        return x * torch.sigmoid(beta * v + delta)

    @torch.no_grad()
    def get_params(self):
        """Helper for visualization and migration analysis."""
        return self.beta.detach().cpu().numpy().copy(), self.delta.detach().cpu().numpy().copy()

    def extra_repr(self) -> str:
        return f'channels={self.channels}'
```

---

## 3. Related Work & Context

PSiLU draws inspiration from several lines of research:

| Category | Related Work | PSiLU Position |
|----------|--------------|----------------|
| **Gated Activations** | Swish (Ramachandran et al., 2017), GELU (Hendrycks & Gimpel, 2016), GLU | PSiLU extends Swish/SiLU with per-channel learnable $\beta$ and $\delta$. |
| **Parametric Activations** | PReLU (He et al., 2015) | PSiLU adds both steepness and shift parameters, enabling richer shape adaptation including non-monotonic regimes. |
| **Initialization Diversity** | Stochastic Depth (Huang et al., 2016) | PSiLU uses diverse initialization for structural symmetry breaking, similar in spirit to stochastic architectural methods. |
| **Adaptive Function Search** | KAN (Liu et al., 2024), PAU | PSiLU bounds the search to the smooth gated family, prioritizing efficiency and stability over unbounded expressivity. |
| **Mixture of Activations** | MoE-style activation selection | PSiLU provides continuous adaptation without discrete routing or selection overhead. |

---

## 4. Experiments

### 4.1 Experimental Setup

- **Optimizer:** Adam with learning rate $0.001$.
- **Loss:** Cross-entropy for classification tasks.
- **Hardware:** CPU-based training for reproducibility.
- **Reporting:** Single-run results; multiple seeds recommended for rigorous comparison to account for stochastic initialization variance.

### 4.2 FashionMNIST Classification

**Task:** 10-class image classification.  
**Architecture:** MLP `[784 → 256 → 128 → 10]`.  
**Training:** 30 epochs, batch size 64.

**Results:**

<img width="2300" height="1438" alt="psilu_analysis" src="https://github.com/user-attachments/assets/af16cc80-d065-4cb3-ab92-2d8a216dbbe6" />

| Activation | Final Loss | Final Accuracy | Best Accuracy | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **PSiLU** | **0.5091** | 89.05% | **89.28%** (E29) | +0.3% |
| GELU | 0.5088 | 88.65% | 89.22% (E13) | baseline |
| Swish | 0.5548 | **89.13%** | 89.27% (E26) | baseline |

### 4.3 Preliminary Observations

Based on the 30-epoch sweep:

* **Competitive Performance:** PSiLU achieves a final accuracy of **89.05%**, placing it within **0.08%** of the Swish baseline and outperforming GELU by **0.40%**.
* **Loss Calibration:** Despite the marginal accuracy lead of Swish, PSiLU maintained a significantly lower **Final Test Loss** (0.5091 vs. 0.5548). This suggests that the parametric adaptation may lead to better-calibrated probability outputs.
* **Convergence Dynamics:** While the fixed-prior models (Swish/GELU) show slightly higher performance in the initial 5 epochs, PSiLU amortizes its "learning period" by Epoch 10, demonstrating stable upward trends in the latter half of training.
* **Structural Heterogeneity:** Visual analysis of parameter migration confirms that neurons deviate from their stochastic starting points to form distinct functional groups. Specifically, the emergence of **negative $\beta$ values** indicates the model utilizes non-monotonic "inhibitory" gates for certain feature channels.
* **Efficiency:** The performance gains are achieved with a negligible parameter increase of approximately **0.3%** for the tested architecture.

### 4.4 Mechanistic Analysis: Learned Activation Diversity

The primary strength of PSiLU lies in its ability to break architectural symmetry through stochastic initialization and subsequent parametric adaptation. Visual inspection of the learned curves in Layer 1 reveals several distinct functional archetypes:

* **Inhibitory "Negative Dip" Logic:** Neurons such as **N2 ($\beta \approx -0.41, \delta \approx 0.36$)** and **N4 ($\beta \approx -0.61, \delta \approx -1.20$)** exhibit non-monotonic behavior. These curves allow the neuron to respond to specific input ranges while suppressing others, a feature entirely absent in monotonic functions like ReLU or standard Swish.
* **Sharpened Gates:** Neurons like **N0 ($\beta \approx 4.57, \delta \approx 4.30$)** move toward a high-steepness regime. This approximates a "hard" threshold gate, enabling the network to implement discrete decision logic where necessary.
* **Stochastic Migration:** As seen in the **Parameter Migration (Beta vs. Delta)** scatter plots, the initial uniform "cloud" of parameters (the blue mass) migrates into a structured, learned distribution (the green mass). This drift indicates that the model is actively optimizing the activation shapes to fit the data manifold.

### 4.3 15-Bit Parity with Constrained Coverage

**Task:** Learn the parity function on 15-bit binary inputs. Parity outputs 1 if the count of 1s is odd, 0 otherwise.

**Constraints:**
- **Coverage:** Only 9% of the $2^{15} = 32,768$ possible inputs observed during training. This prevents memorization and requires generalization.
- **Bottleneck:** Hidden dimension limited to 64, inducing capacity pressure and neuron competition.
- **Architecture:** MLP `[15 → 64 → 64 → 1]`.

**Motivation:** The parity task serves as a probe for **Boolean complexity**. Because parity requires sensitive signal inversion (XOR-like logic), it tests whether PSiLU's learnable parameters can synthesize non-linear decision boundaries more efficiently than fixed-prior activations under severe data and capacity constraints.

**Results:**

| Activation | Train Accuracy | Test Accuracy | Generalization Gap |
|------------|:--------------:|:-------------:|:------------------:|
| PSiLU      | `[TODO]`%      | `[TODO]`%     | `[TODO]`%          |
| GELU       | `[TODO]`%      | `[TODO]`%     | `[TODO]`%          |
| Swish      | `[TODO]`%      | `[TODO]`%     | `[TODO]`%          |
| ReLU       | `[TODO]`%      | `[TODO]`%     | `[TODO]`%          |

*Results to be populated. This task is designed to stress-test activation flexibility.*

---

## 5. Future Work

The following evaluations are planned or encouraged:

| Domain | Task | Notes |
|--------|------|-------|
| **Vision** | CIFAR-10 / CIFAR-100 | CNN architectures (ResNet, VGG) |
| **Tabular** | MONK's Problems | Classic symbolic learning benchmarks |
| **NLP** | BERT fine-tuning on SST-2 | Evaluate PSiLU in transformer layers |
| **NLP** | GLUE benchmark | Broader language understanding |
| **Analysis** | Parameter distribution study | Correlation between $(\beta, \delta)$ and neuron function |
| **Ablation** | Initialization sensitivity | Effect of $\beta$, $\delta$ ranges on convergence |

We invite researchers to test PSiLU on these and other tasks. Both positive and negative results are valuable for understanding the applicability and limitations of adaptive activations.

---

## 6. Discussion

### 6.1 Observations

- **Low Overhead:** PSiLU introduces minimal parameter cost ($2C$ per layer) while enabling per-channel adaptation.
- **Bounded Flexibility:** By constraining the search to the smooth gated family, PSiLU avoids the complexity of unbounded function approximation while still offering meaningful shape variation.
- **Initial Diversity:** Diverse initialization creates heterogeneous activation shapes at the start of training, which may help exploration in early optimization phases.
- **Negative $\beta$ Regime:** The function space includes non-monotonic shapes when $\beta < 0$, enabling inhibitory gating. Whether networks exploit this regime depends on the task.

### 6.2 Limitations

- Results are preliminary and limited to small-scale tasks.
- No systematic hyperparameter sweep has been conducted for initialization ranges.
- Interaction with normalization layers (BatchNorm, LayerNorm) requires further study.
- Behavior in very deep networks and transformer architectures is untested.

### 6.3 Call for Evaluation

We release this technical report to encourage independent evaluation. We are particularly interested in:
- Results on diverse architectures (CNNs, Transformers, GNNs).
- Comparisons with other learnable activations (PReLU, APL, PAU).
- Analysis of learned parameter distributions across layers.
- Failure cases and limitations.

Please share results via the repository issues or discussions.

---

## 7. Citation

**BibTeX Citation:**

```bibtex
@misc{psilu2026,
  title = {PSiLU: Parametric Sigmoid Linear Unit},
  author = {[GP Bayu]},
  year = {2026},
  howpublished = {\url{https://github.com/gbyuvd/PSiLU}},
  note = {Technical Report, Version 1.0}
}
```

---

## Appendix A: Activation Shape Examples

The PSiLU function can express diverse shapes depending on $(\beta, \delta)$:

| $(\beta, \delta)$ | Shape Description |
|-------------------|-------------------|
| $(1, 0)$ | Standard Swish/SiLU |
| $(5, 0)$ | Sharper gate, ReLU-like |
| $(0.5, -2)$ | Gentle slope, shifted threshold |
| $(-2, 0)$ | Non-monotonic with inhibitory dip |

Learned parameters after training show migration toward task-appropriate shapes.

---

## Appendix B: Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| $\beta$ initialization | $\mathcal{U}(-1.0, 5.0)$ |
| $\delta$ initialization | $\mathcal{U}(-5.0, 5.0)$ |
| Weight decay | 0 (default) |

---

*This document will be updated as additional results become available.*  
*Last updated: April 15, 2026*

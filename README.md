# PSiLU: Parametric Sigmoid Linear Unit  
*A Characterization Report (DRAFT)*

<img width="500" height="300" alt="PSiLUsweep" src="https://github.com/user-attachments/assets/85cb7598-411d-4cc1-87ad-c7d554da9678" />
<img width="500" height="300" alt="PSiLUFullSweep" src="https://github.com/user-attachments/assets/350c06e9-8af3-4ed3-9657-28bb1d4654f7" />

> **Transparency Note:** Mathematical derivations and symbolic analysis were assisted by large language models. Empirical experiments, data collection, and implementation were conducted directly by the author. Readers are encouraged to verify derivations and report any discrepancies via the repository issue tracker.

## Abstract

We introduce **PSiLU** (Parametric Sigmoid Linear Unit), a continuously differentiable activation function extending the Swish/SiLU family via two learnable parameters: $\beta$ (slope/gating sharpness) and $\delta$ (shift/threshold). The formulation is defined as:

$$
f(x; \beta, \delta) = x \cdot \sigma(\beta x + \delta), \quad \sigma(z) = \frac{1}{1 + e^{-z}}.
$$

This report provides a mathematical characterization of PSiLU, including domain, range, derivatives, gradient bounds, fixed points, and behavioral regimes. Empirical analysis identifies a stable subspace ("manifold") within the $(\beta, \delta)$ parameter space that balances gradient stability and gating activity. Gain analysis indicates PSiLU is inherently variance-shrinking ($g \approx 0.56$ for standard configurations), suggesting implications for weight initialization scaling. Approximation analysis quantifies PSiLU's capacity to recover existing activations (SiLU, GELU, ReLU) with low error. Preliminary synthetic experiments illustrate parameter dynamics, noting minimal parameter drift (< 0.03 mean absolute change) under standard optimization, and compare performance against fixed activations in shallow MLP/ToyTransformer settings. This document serves as a technical description to facilitate reproducibility and invites further investigation into mechanisms for functional specialization.

## 1. Introduction

### 1.1 Motivation

Modern neural networks predominantly employ fixed, homogeneous activation functions (e.g., ReLU, GELU, Swish). While effective, this design imposes a structural prior where every neuron must adapt its weights to work within the same nonlinearity. Recent research has explored increasing activation flexibility via discrete selection (MoE-style) or unbounded approximation (splines-KAN, PAU), though these introduce routing complexity or computational cost.

PSiLU is designed as a continuous extension of the smooth gated activation family. By introducing two learnable parameters per unit, PSiLU allows each neuron to adapt its activation shape while maintaining differentiability and hardware efficiency. This report characterizes the mathematical properties and empirical behavior of PSiLU without asserting universal superiority.

### 1.2 Scope and Contributions

This report provides:
- A formal description of PSiLU and its relationship to existing activations.
- Mathematical characterization: domain, range, derivatives, gradient bounds, fixed points, monotonicity.
- Behavioral taxonomy in $(\beta, \delta)$ parameter space.
- Approximation capacity analysis for ReLU, SiLU, and GELU.
- Empirical manifold sweep methodology for identifying stable initialization regions.
- Signal propagation analysis: gain factors and variance dynamics in MLPs and ResNets.
- Descriptive statistics on parameter dynamics and preliminary task performance.

All results are presented as observed. The question of how to encourage full utilization of PSiLU's behavioral diversity remains open.

## 2. Method

### 2.1 Formulation

For input $x \in \mathbb{R}$ and dynamic parameters $\beta, \delta \in \mathbb{R}$, PSiLU is defined as:

$$
f(x; \beta, \delta) = x \cdot \sigma(\beta x + \delta).
$$

This formulation factors the shift $\delta$ as an additive bias within the sigmoid argument, enabling intuitive interpretation: $\delta$ translates the activation threshold, while $\beta$ scales the gating sharpness.

### 2.2 Minimal PyTorch Implementation

```python
import torch
import torch.nn as nn

class PSiLU(nn.Module):
    """
    Parametric Sigmoid Linear Unit.
    Formula: f(x; β, δ) = x · σ(βx + δ)
    Parameters β and δ are provided externally (e.g., learned per-channel).
    """
    def forward(self, x: torch.Tensor, beta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(beta * x + delta)
```

Parameters $\beta$ and $\delta$ may be learned per-channel, per-token, or provided externally. This implementation emphasizes flexibility: the activation function itself contains no learned state.

A more elaborate implementation:
```python
_stratified_init(param: nn.Parameter, low: float, high: float):
    n = param.numel()
    grid = torch.linspace(low, high, n)
    idx = torch.randperm(n)
    with torch.no_grad():
        param.copy_(grid[idx])


class PSiLU(nn.Module):
    """
    f(x; β, δ) = x · σ(β·x + δ)      [self-gated, default]
    f(x, v; β, δ) = x · σ(β·v + δ)   [external gate]
    """
    def __init__(
        self,
        channels: int,
        beta_range:  Tuple[float, float] = (-1.0, 5.0),
        delta_range: Tuple[float, float] = (-5.0, 5.0),
        init: str = "stratified",       # "stratified" | "normal" | "uniform" | "silu"
    ):
        super().__init__()
        self.channels    = channels
        self.beta_range  = beta_range
        self.delta_range = delta_range
        self.init_method = init
        self.beta  = nn.Parameter(torch.empty(channels))
        self.delta = nn.Parameter(torch.empty(channels))
        self._init_params()

    def _init_params(self):
        m = self.init_method
        if m == "stratified":
            _stratified_init(self.beta,  *self.beta_range)
            _stratified_init(self.delta, *self.delta_range)
        elif m == "normal":
            nn.init.normal_(self.beta,  mean=1.0, std=1.0)
            nn.init.normal_(self.delta, mean=0.0, std=1.0)
        elif m == "uniform":
            nn.init.uniform_(self.beta,  *self.beta_range)
            nn.init.uniform_(self.delta, *self.delta_range)
        elif m == "silu":
            nn.init.constant_(self.beta,  1.0)
            nn.init.constant_(self.delta, 0.0)
        else:
            raise ValueError(f"Unknown init: {m}")

    def forward(
        self,
        x: torch.Tensor,
        gate_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v = gate_input if gate_input is not None else x
        if v.dim() >= 2:
            shape = (1,) * (v.dim() - 1) + (self.channels,)
            beta  = self.beta.view(*shape)
            delta = self.delta.view(*shape)
        else:
            beta, delta = self.beta, self.delta
        return x * torch.sigmoid(beta * v + delta)

    @torch.no_grad()
    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.beta .detach().cpu().numpy().copy(),
            self.delta.detach().cpu().numpy().copy(),
        )

    def extra_repr(self):
        return f"channels={self.channels}, init={self.init_method}"
```
### 2.3 Relationship to Existing Activations

| Activation | PSiLU Parameterization | Notes |
|------------|----------------------|-------|
| SiLU / Swish | $\beta=1, \delta=0$ | Exact recovery |
| Linear (slope $c$) | $\beta=0, \delta=\text{logit}(c)$ | Exact recovery |
| ReLU | $\beta \to \infty, \delta=0$ | Limit case (sigmoid → step) |
| GELU | $\beta \approx 1.78, \delta \approx 0$ | Approximate |
| Inverse SiLU | $\beta=-1, \delta=0$ | Exact; non-monotonic regime |
| Zero function | $\delta \to -\infty$ | Limit case |

## 3. Mathematical Characterization

### 3.1 Domain, Range, and Asymptotics

| Property | Statement |
|----------|-----------|
| **Domain** | $f: \mathbb{R}^3 \to \mathbb{R}$ (unrestricted $x, \beta, \delta$) |
| **Range ($\beta > 0$)** | $(-\frac{1}{\beta e}, \infty)$ — mild negative dip, unbounded positive |
| **Range ($\beta = 0$)** | $\mathbb{R}$ (linear with slope $\sigma(\delta)$) |
| **Range ($\beta < 0$)** | $(-\infty, \frac{1}{\|\beta\| e})$ — bounded above, unbounded below |
| **Gate factor** | $\sigma(\beta x + \delta) \in (0, 1) \; \forall x \in \mathbb{R}$ |
| **Asymptotic ($\beta > 0, x \to +\infty$)** | $f(x) \to x$, $f'(x) \to 1$ |
| **Asymptotic ($\beta > 0, x \to -\infty$)** | $f(x) \to 0$, $f'(x) \to 0$ |
| **Asymptotic ($\beta < 0, x \to +\infty$)** | $f(x) \to 0$, $f'(x) \to 0$ |
| **Asymptotic ($\beta < 0, x \to -\infty$)** | $f(x) \to x$, $f'(x) \to 1$ |

Unlike ReLU (range $[0,\infty)$) or tanh (range $(-1,1)$), PSiLU's range depends on $\beta$. For $\beta > 0$ it is SiLU-like: mildly negative dip then unbounded positive.

### 3.2 Continuity and Differentiability

Since $\sigma(z)$ is $C^\infty$ and $x$ is a polynomial, their product is $C^\infty$ everywhere on $\mathbb{R}$.
- **Continuity:** $f$ is continuous $\forall x \in \mathbb{R}, \forall (\beta, \delta) \in \mathbb{R}^2$.
- **Differentiability:** $f$ is differentiable to all orders $\forall x \in \mathbb{R}$.
- **Contrast with ReLU:** ReLU is not differentiable at $x = 0$. PSiLU has no such kink — the gate smoothly interpolates around any threshold.

### 3.3 Derivatives

Let $s = \sigma(\beta x + \delta)$. The first derivative (backpropagated gradient) admits a closed form:

$$
\begin{aligned}
f'(x) &= \frac{d}{dx}\big[x \cdot s\big] \\
      &= s + x \cdot \frac{ds}{dx} \\
      &= s + x \cdot \beta \cdot s(1-s) \\
      &= s \cdot \big[1 + \beta x (1 - s)\big] \\
      &= \sigma(\beta x + \delta) \cdot \big[1 + \beta x (1 - \sigma(\beta x + \delta))\big].
\end{aligned}
$$

This decomposes into the gate value $s$ times a correction term scaling with $\beta$ and $x$. At $\beta = 0$ this collapses to $f'(x) = \sigma(\delta)$, a constant.

The second derivative (curvature) is:

$$
\begin{aligned}
f''(x) &= \beta \cdot s(1-s) \cdot \big[2 + \beta x (1 - 2s)\big] \\
       &= \beta \cdot \sigma(\beta x + \delta)(1-\sigma(\beta x + \delta)) \cdot \big[2 + \beta x (1 - 2\sigma(\beta x + \delta))\big].
\end{aligned}
$$

Setting $f''(x)=0$ yields a transcendental equation for inflection points; no closed-form solution exists in $x$, but numerical evaluation per $(\beta, \delta)$ is straightforward.

### 3.4 Gradient Bounds and Stability

We analyze tight bounds on $f'(x) = s \cdot [1 + \beta x (1-s)]$.

**Asymptotic analysis ($\beta > 0$ case):**
- As $x \to +\infty$: $s \to 1$, $\beta x (1-s) \to 0$ → $f'(x) \to 1$
- As $x \to -\infty$: $s \to 0$, $\beta x \cdot s \to 0$ → $f'(x) \to 0$

**Global minimum of $f'(x)$ — where dying gradient risk lives:**
$f'(x)$ can go below 0 for intermediate $x$ when $\beta > 0$. Setting $f'(x) = 0$ yields $x^* \approx -1 / [\beta(1-s^*)]$.

Numerical evaluation ($\delta=0$):

| $\beta$ | $x^*$ (approx) | $f'(x^*)$ (approx) |
|---------|---------------|-------------------|
| 1.0 | -1.278 | -0.0854 |
| 2.0 | -0.760 | -0.101 |
| 5.0 | -0.400 | -0.068 |

**Summary bounds:**

| Condition | Gradient Range | Practical Implication |
|-----------|---------------|---------------------|
| $\beta > 0$ | $f'(x) \in [f'(x^*), 1)$ where $f'(x^*) \approx -0.09$ | Mild negative dip; stable in practice (matches SiLU) |
| $\beta = 0$ | $f'(x) = \sigma(\delta) \in (0,1)$ | Constant gradient; no saturation, no vanishing |
| $\beta < 0$ | $f'(x)$ can exceed 1 in magnitude | Potential gradient amplification; monitor during training |
| Large $\|\delta\|$ | $f'(x) \to 0$ or $1$ uniformly | Risk of vanishing gradients or dead units |

### 3.5 Fixed Points and Monotonicity

**Fixed points.** Solving $f(x) = x$:
$$
x \cdot \sigma(\beta x + \delta) = x \quad \Rightarrow \quad x \cdot [\sigma(\beta x + \delta) - 1] = 0
$$
Case 1: $x = 0$ → $f(0) = 0$.
Case 2: $\sigma(\beta x + \delta) = 1$ → $\beta x + \delta \to +\infty$ → $x \to +\infty$ (not finite).

∴ $x = 0$ is the **unique finite fixed point** of PSiLU for all $(\beta, \delta)$.
This implies PSiLU is not self-normalizing in the SELU sense; pre-activations near zero are always suppressed.

**Monotonicity.** PSiLU is *not* monotone in general.

| Condition | Monotone? | Why |
|-----------|-----------|-----|
| $\beta = 0$ | Yes, strictly | $f(x) = \sigma(\delta) \cdot x$, linear with positive slope |
| $\beta > 0$, small | No | Slight negative dip near $x \approx -1/\beta$ before recovering |
| $\beta \to \infty$ | Approximately yes | Approaches ReLU |
| $\beta < 0$ | No | Gate inverts; $f$ rises then falls; can have a maximum |

## 3. Empirical Characterization

### 3.1 Approximation Capacity

PSiLU's two-parameter formulation enables it to approximate several widely-used activation functions through appropriate choice of $(\beta, \delta)$. Approximation quality was quantified via mean squared error (MSE) over $x \in [-5, 5]$.

<img width="1789" height="989" alt="image" src="https://github.com/user-attachments/assets/3b88a396-41e8-458c-b335-7b11463fd831" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/e524df33-8e9a-4851-abf9-90da12934369" />


| Target Activation | Optimized $(\beta, \delta)$ | Approximation Loss (MSE) | Notes |
|------------------|-----------------------------|-------------------------|-------|
| **SiLU / Swish** | $(1.0000, -0.0000)$ | $0.000000$ | Exact recovery by definition |
| **GELU** | $(1.7801, 0.0000)$ | $0.000072$ | Close approximation |
| **ReLU** | $(3.2170, 0.0000)$ | $0.000950$ | Sharp gating regime; residual error from sigmoid smoothness |
| **Linear** ($y=x$) | $(\beta \to 0, \delta \gg 0)$ | Limit case | Approaches identity as gate saturates to 1 |
| **Zero function** | $(\text{any}, \delta \ll 0)$ | Limit case | Gate saturates to 0 |

*Observation:* PSiLU recovers SiLU exactly and approximates GELU and ReLU with low error using only two scalar parameters. This suggests the smooth gated family is sufficiently expressive to capture key behaviors of common activations within the tested domain.

## 3.2 Activation Landscape Analysis
Visual analysis of the forward function and its derivatives reveals distinct behavioral regimes:

<img width="1937" height="516" alt="Pasted image 20260416175243" src="https://github.com/user-attachments/assets/b17b698e-ca39-4cff-b0cf-be6b6d41d82f" />

| Regime | Parameters | Forward Shape $f(x)$ | 1st Derivative $f'(x)$ | 2nd Derivative $f''(x)$ | Interpretation |
|--------|------------|----------------------|------------------------|-------------------------|----------------|
| **SiLU** | $\beta=1, \delta=0$ | Smooth rectification | Peak $\approx 1.1$ at $x \approx 1.5$ | Smooth curvature | Standard baseline |
| **Hard Gate** | $\beta=10, \delta=0$ | Step-like (ReLU-ish) | Sharp spike (Dirac-like) | High frequency oscillation | Feature selection |
| **Linear** | $\beta=0, \delta=0$ | Straight line ($0.5x$) | Constant $0.5$ | Zero | Identity pass-through |
| **Always-On** | $\beta=1, \delta=5$ | Near-linear ($\approx x$) | Constant $\approx 1.0$ | Near Zero | Signal preservation |
| **Always-Off** | $\beta=1, \delta=-5$ | Flat ($\approx 0$) | Near Zero | Near Zero | Channel pruning |
| **Inverse** | $\beta=-1, \delta=0$ | Non-monotonic (dip) | **Negative** region | High curvature | Inhibitory dynamics |

### 3.3 Behavioral Zone Map
<img width="1655" height="1280" alt="Pasted image 20260416175257" src="https://github.com/user-attachments/assets/b2c61f6c-3cb9-4d62-b6bb-e261d4a7082c" />

Heatmaps over the $(\beta, \delta)$ space define safe and unsafe operating regions:
- **Rectification Ratio:** High $\beta$ + Low $\delta$ $\rightarrow$ High rectification (ReLU-like). Low $\beta$ $\rightarrow$ Symmetric/Linear.
- **Linearity Index:** High $\delta$ ("Always-On") creates linear regions ($R^2 \approx 1$). Low $\delta$ creates non-linear regions.
- **Gate Saturation:** Low $\beta$ + Low $\delta$ leads to saturation (output $\approx 0$).
- **Negative Output:** ReLU-like behavior (No negative output) occurs at **High $\beta$ and Low $\delta$**. Standard SiLU/GeLU reside in the "Has negative output" zone.


### 3.4 Gradient Statistics 
Analysis of gradient statistics across the $(\beta, \delta)$ parameter space reveals critical stability constraints:
<img width="1027" height="506" alt="Pasted image 20260416175317" src="https://github.com/user-attachments/assets/e4fd09f0-3bc6-4c58-b01b-7bdb7d02ea0c" />
<img width="1642" height="1280" alt="Pasted image 20260416175323" src="https://github.com/user-attachments/assets/956fb624-933b-43a2-ad48-e486dba84994" />

1.  **Min Gradient (Dead/Dying Risk):**
    - **Red Zones (Near 0):** Occur at extreme $\delta$ values (far from 0).
    - **Implication:** Channels with large $|\delta|$ saturate, leading to vanishing gradients ("dead neurons").
2.  **Max Gradient (Explosion Risk):**
    - **Green Zones (> 1.5):** Occur primarily when **$\beta < 0$ and $\delta < -2$**.
    - **Implication:** "Inverse" configurations with negative shifts are dangerous and can cause gradient explosion. High $\beta$ alone is less risky than negative $\beta$.
3.  **Mean Gradient:**
    - Generally stable ($\approx 0.4 - 0.8$) in the central region ($\beta \in [0, 3], \delta \in [-1, 1]$).
4.  **Fraction of Negative Gradients:**
    - Non-zero only when $\beta < 0$ or $\delta$ is extreme.
    - **Implication:** Standard positive $\beta$ configurations are monotonic ($f'(x) \ge 0$), ensuring stable optimization.

### 3.5 Empirical Gain Factor Analysis
The empirical gain factor $g = \sqrt{\text{Var}[f(z)]/\text{Var}[z]}$ determines signal propagation health. For unit-variance input $z \sim \mathcal{N}(0, 1)$:
<img width="1676" height="643" alt="Pasted image 20260416175342" src="https://github.com/user-attachments/assets/d03f73d7-0274-4fec-9216-1836b71b797f" />
**Visual Gain Surface:**
- **Low Gain Valley:** A broad valley of low gain ($g < 0.4$) exists for $\beta < 0$ and $\delta < 0$.
- **High Gain Ridge:** Gain $> 1.0$ is only achieved in specific regions (e.g., $\beta \approx -1, \delta \approx 3$), but these often coincide with instability risks.
- **Standard Region:** For $\beta \in [1, 2], \delta=0$, gain is consistently $\approx 0.56 - 0.60$.

**Numerical Gain Table:**

| $\beta$ | $\delta$ | Gain $g$ | $1/g^2$ (Weight Scale Factor) | Regime |
|---------|----------|----------|-------------------------------|--------|
| -1.00 | 0.00 | 0.5615 | 3.17 | Inverse gate |
| 0.00 | 0.00 | 0.5000 | 4.00 | Linear |
| **1.00** | **0.00** | **0.5609** | **3.18** | **SiLU (baseline)** |
| 1.78 | 0.00 | 0.5857 | 2.91 | GELU-like |
| 10.00 | 0.00 | 0.5864 | 2.91 | Hard gate |
| 1.00 | 3.00 | 0.8939 | 1.25 | Always-on (high gain) |
| 1.00 | -3.00 | 0.1642 | 37.10 | Always-off (low gain) |

**Observations:**
1.  PSiLU is inherently variance-shrinking for standard configurations ($g \approx 0.56$).
2.  Standard He/Xavier initialization may lead to signal collapse without compensation.

## 3.6 Initialization Strategies & Parameter Diversity
Empirical analysis compares initialization strategies in shallow MLPs:
<img width="2067" height="1407" alt="D_init_comparison" src="https://github.com/user-attachments/assets/1591fb6e-f05e-4b43-9527-64e4c826b1e7" />

| Strategy | Diversity Score | Median Var Ratio | Visual Characteristic |
|----------|----------------|------------------|-----------------------|
| **Manifold/Stratified** | **0.212** | **0.350** | Wide spread of $\beta, \delta$; mixes regimes |
| Normal | 0.190 | 0.329 | Clustered around mean |
| Uniform | 0.175 | 0.351 | Random spread; may hit unstable zones |
| Static SiLU | 0.000 | 0.313 | All channels identical |

**Observations:**
- Static initialization (all channels = SiLU) leads to negligible parameter drift; the model stays stuck in the "SiLU basin".
- Stratified/Manifold initialization preserves parameter diversity throughout training, suggesting heterogeneous activation behaviors are maintained and  suggesting each behavior serves a purpose.

### 3.7 Dead Neuron Recovery
<img width="1677" height="643" alt="Pasted image 20260416175431" src="https://github.com/user-attachments/assets/26c472ce-bb2a-44fc-a2c1-ae6b65c9b8a7" />

- **Stratified PSiLU:** Starts with ~27% dead channels (due to aggressive $\delta$ sampling). Recovers to ~10% after training.
- **ReLU:** Starts with ~27% dead channels. Recovers to ~7.5%.
- **Static SiLU:** Starts with ~10% dead channels. Stays at 0% (stable but less flexible).
- **Insight:** Stratified PSiLU allows the network to **dynamically prune and activate channels** during training, acting as a form of learnable capacity allocation.

### 3.8 Signal Propagation in Deep Networks

**3.8.1 Plain MLP (No Residual Connections)**
**Setup:** 20 layers, width=256.
<img width="1677" height="643" alt="Pasted image 20260416175439" src="https://github.com/user-attachments/assets/732b2c7d-1538-40c9-ad60-4f1465690f36" />

| Activation | Final Variance | Max Grad Norm | Min Grad Norm | Final Loss |
|------------|---------------|---------------|---------------|------------|
| ReLU | $10^{-6}$ | **0.0002** | **0.000125** | **0.148** |
| PSiLU-stratified | $10^{-8}$ | 0.0000 | 0.000013 | 0.160 |
| PSiLU-normal | $10^{-10}$ | 0.0000 | 0.000001 | 0.179 |
| PSiLU-silu-init | $10^{-11}$ | 0.0000 | ~0 | 0.205 |
| SiLU / GELU | $10^{-11}$ | 0.0000 | ~0 | 0.175–0.211 |

**Plain MLP (No Residuals):** All activations suffer exponential variance decay in plain MLPs >10 layers. PSiLU mitigates decay slightly by including high-gain channels but remains susceptible to vanishing signals without normalization.

**Residual MLP (With $y = x + f(x)$)**
<img width="1027" height="643" alt="Pasted image 20260416175453" src="https://github.com/user-attachments/assets/0fc4735b-b7a5-49ad-97ea-1843570f1dd7" />

| Activation           | Final Variance (Layer 20) | Variance Growth Factor |
| -------------------- | ------------------------- | ---------------------- |
| ReLU                 | **803,650**               | ~2.0× per layer        |
| PSiLU-silu-init      | 408,607                   | ~1.8× per layer        |
| SiLU                 | 281,306                   | ~1.7× per layer        |
| GELU                 | 419,049                   | ~1.8× per layer        |
| **PSiLU-stratified** | **32,974**                | **~1.4× per layer**    |
| PSiLU-normal         | 30,712                    | ~1.4× per layer        |

**Residual MLP:** Residual connections cause variance explosion ($Var(y) \approx Var(x) + Var(f(x))$).
- ReLU explodes fastest (all neurons active).
- PSiLU-stratified explodes slowest because it includes "Always-off" channels where $f(x) \approx 0$, making those residual blocks act closer to identity maps.

### 3.9 Manifold Sweep: Identifying a Stable Subspace

We perform a high-resolution grid sweep over $(\beta, \delta) \in [-2, 8] \times [-6, 6]$, evaluating per-point metrics over $x \in [-3, 3]$. Validity constraints are defined as:
1.  **Gradient Stability:** $\min f'(x) > -0.5$ and $\max f'(x) < 2.2$.
2.  **Active Gating:** $\text{std}_x[\sigma(\cdot)] > 0.08$ (ensures gate switches).
3.  **Dynamic Range:** $\text{mean}_x[\sigma(\cdot)] \in (0.05, 0.95)$ (avoids saturation).

<img width="2589" height="590" alt="image" src="https://github.com/user-attachments/assets/13d4c026-2a1b-4b9f-9271-f73cc50dae75" />
Total valid coordinates: 163441

**Observations:**
- The valid region is non-convex and excludes extreme $\beta < 0$ (explosion risk) and large $|\delta|$ (saturation).
- SiLU and GELU reference points lie within the manifold, confirming they are stable starting points.
- The manifold includes diverse regimes: linear, hard gating, and shifted thresholds.

---
## 4. Preliminary Task Results across Init Methods
*Note: These results are presented for illustrative purposes only. Tasks are synthetic and shallow; no claims of generalizability or superiority are made.*


### 4.1 Training Dynamics (Synthetic Regression)
<img width="2063" height="643" alt="Pasted image 20260416175506" src="https://github.com/user-attachments/assets/65309df9-ed3f-4641-85bd-3c0ffe6aa4dd" />


| Activation | Final Loss | Final Grad Norm | Final Residual Norm |
|------------|-----------|-----------------|---------------------|
| ReLU | **0.148** | 0.288 (unstable) | **7.85** (high) |
| PSiLU-stratified | 0.160 | **0.128** (stable) | 4.38 |
| PSiLU-silu-init | 0.205 | 0.095 | **3.29** (lowest) |
| SiLU | 0.211 | 0.094 | 3.17 |

PSiLU variants produce **more stable gradient norms** than ReLU (less spiky), at the cost of slightly higher final loss in this simple task. The trade-off favors PSiLU in deeper, more complex networks where gradient stability is critical.

#### 4.2 Toy Transformer Training (MHA + FFN)
<img width="1675" height="643" alt="Pasted image 20260416175521" src="https://github.com/user-attachments/assets/1161b71e-0aa9-4585-9787-b7adca3eeebe" />

| Activation | Final Loss | $\beta_{mean}$ | $\delta_{mean}$ |
|------------|-----------|----------------|-----------------|
| PSiLU-stratified | 4.1616 | 2.002 | 0.006 |
| PSiLU-silu-init | 4.1607 | **1.004** | 0.007 |
| ReLU / SiLU / GELU | ~4.161 | NaN | NaN |

**Findings:**
- All activations converge to nearly identical loss in this small-scale setting.
- PSiLU-silu-init parameters remain fixed near $(1, 0)$, confirming the "SiLU basin" stability.
- PSiLU-stratified parameters drift to $\beta \approx 2.0$, suggesting the model finds value in sharper gating for attention/FFN layers.

#### 4.3 Parameter Drift & Diversity Retention
<img width="1546" height="1025" alt="Pasted image 20260416175533" src="https://github.com/user-attachments/assets/ebb57821-4b4f-4a0a-aca4-2689c0338e7e" />

- **Static Init (SiLU):** Parameters show negligible drift. The model does not learn to deviate from the initial SiLU shape.
- **Stratified Init:** Parameters remain diverse throughout training. Channels specialize: some become linear ($\beta \approx 0$), some become hard gates ($\beta \gg 1$), some shift ($\delta \neq 0$).

---

## 5. Preliminary Task Results with Manifold Init

*Note: These results are presented for illustrative purposes only. Tasks are synthetic and shallow; no claims of generalizability or superiority are made.*

### 5.1 Synthetic Regression Task

**Setup:** 2-layer MLP (width=128) trained on $y = \sin(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, 0.1)$, 100 epochs. Manifold initialization used for PSiLU.
<img width="1789" height="1190" alt="Pasted image 20260419020035" src="https://github.com/user-attachments/assets/bd94736f-c100-4062-8ce2-905456109cfb" />

| Activation | Validation Loss | Final Grad Norm | Dead Neuron % |
|------------|----------------|-----------------|---------------|
| PSiLU (Manifold) | **0.0146** | 0.0120 | 2.98% |
| ReLU | 0.4526 | 0.0040 | N/A |
| SiLU | 0.7914 | 0.0037 | N/A |

**Parameter Dynamics (PSiLU):**
- Mean Absolute Beta Drift: 0.0273
- Mean Absolute Delta Drift: 0.0314
- Observation: Parameters barely moved, suggesting the initialization manifold itself provided a favorable starting point.

### 5.2 Simple Classification Task

**Setup:** MLP on 10-class synthetic data with low-dimensional features, 100 epochs.

<img width="1790" height="1190" alt="Pasted image 20260419020049" src="https://github.com/user-attachments/assets/e46d5049-289e-43ab-8941-1325705c4f58" />

| Activation | Validation Loss | Final Grad Norm | Dead Neuron % |
|------------|----------------|-----------------|---------------|
| PSiLU (Manifold) | **0.0971** | 0.00017 | 13.47% |
| ReLU | 0.3254 | 0.00033 | N/A |
| SiLU | 0.2628 | 0.00169 | N/A |

**Parameter Dynamics (PSiLU):**
- Mean Absolute Beta Drift: 0.0193
- Mean Absolute Delta Drift: 0.0195
- Dead Neuron Change: +5.38% (Initial 8.09% → Final 13.47%)
- Observation: The increase in dead neurons suggests adaptive capacity allocation, where the network prunes unnecessary channels via $\delta$ shifts.

### 5.3 Gradient Norm Observations

- **Regression:** PSiLU exhibited higher gradient norms than ReLU/SiLU, correlating with lower loss.
- **Classification:** PSiLU exhibited lower gradient norms than ReLU/SiLU but achieved lower loss, suggesting more efficient gradient flow (less wasted gradient).

## 6. Conclusion

PSiLU offers a minimal, flexible extension of the smooth gated activation family, accepting dynamic parameters $\beta$ and $\delta$ to enable per-unit adaptability. Mathematical analysis clarifies its domain, derivatives, gradient bounds, fixed points, monotonicity conditions, and behavioral regimes. PSiLU exactly recovers SiLU and closely approximates ReLU and GELU with two scalar parameters.

Empirical manifold sweep methodology identifies a subspace of $(\beta, \delta)$ values that balance stability and expressivity. Descriptive statistics characterize gain behavior ($g \approx 0.56$) and output distributions across regimes. A key observation is that standard optimization yields minimal parameter drift (< 0.03 mean absolute change) even with manifold-aware initialization, suggesting gradient signals alone may not suffice to encourage functional specialization. Preliminary synthetic results illustrate potential performance gains in shallow networks but are explicitly labeled as illustrative.

We invite further investigation into:
1.  Mechanisms that promote meaningful adaptation in learnable activations (e.g., specialized regularization, meta-learning).
2.  The relationship between PSiLU's output statistics and layer-wise normalization requirements.
3.  Scalability to large-scale architectures and real-world benchmarks.
4.  Theoretical analysis of the optimization landscape for $(\beta, \delta)$ parameters.

This report serves as a technical description to facilitate reproducibility and discussion. All observations are preliminary; no claims of superiority or universal applicability are made.

## 7. Related Work & Context

PSiLU sits at the intersection of several research directions focused on activation flexibility, signal propagation, and functional diversity.

| Category | Related Work | PSiLU Position |
| :--- | :--- | :--- |
| **Gated Activations** | Swish (Ramachandran et al., 2017), GELU (Hendrycks & Gimpel, 2023) | PSiLU extends the Swish/SiLU family with per-channel learnable $\beta$ (steepness) and $\delta$ (shift). |
| **Parametric Activations** | PReLU (He et al., 2015), ACON (Ma et al., 2021) | PSiLU adds both steepness and shift parameters, enabling richer shape adaptation including non-monotonic regimes. |
| **Activation Diversity Strategies** | Stochastic Activations (Lomeli et al., 2025) | PSiLU employs diverse parameter initialization to encourage functional heterogeneity. While Stochastic Activations use runtime randomness to select between discrete functions (e.g., SiLU/ReLU), PSiLU achieves diversity through continuous parameter space exploration. |
| **Adaptive Function Search** | KAN (Liu et al., 2025), PAU (Molina et al., 2020) | PSiLU bounds the search to the smooth gated family, prioritizing efficiency and stability over unbounded expressivity. |

**Contextual Notes:**
- **Vs. Stochastic Activations:** Lomeli et al. (2025) propose randomly selecting between non-linear functions during training to circumvent optimization issues (e.g., dead ReLU neurons) and increase generation diversity. PSiLU shares the goal of introducing functional diversity but achieves this via deterministic, learnable continuous parameters rather than runtime stochastic selection.
- **Vs. PReLU:** While PReLU learns a slope for negative inputs, PSiLU learns the gating shape itself, allowing channels to transition between linear, gated, and suppressed regimes.
- **Vs. KAN/PAU:** Kolmogorov-Arnold Networks and Padé Activation Units explore broader function spaces (splines, rational functions). PSiLU restricts the search space to smooth gated linear units to maintain hardware efficiency and optimization stability.

## 7. Citation & Acknowledgments

This document is provided as a technical resource for the community. If the characterization, empirical findings, or implementation details presented here contribute to your research or development, we welcome attribution via the citation below. We also welcome feedback, corrections, and collaboration regarding further characterization of PSiLU.

**BibTeX Citation:**

```bibtex
@misc{psilu2026,
  title = {PSiLU: Parametric Sigmoid Linear Unit},
  author = {[GP Bayu]},
  year = {2026},
  howpublished = {\url{https://github.com/gbyuvd/PSiLU}},
  note = {Technical Report Draft, Version 1.1}
}
```
References:
```bibtex
@misc{ramachandran2017searchingactivationfunctions,
      title={Searching for Activation Functions}, 
      author={Prajit Ramachandran and Barret Zoph and Quoc V. Le},
      year={2017},
      eprint={1710.05941},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/1710.05941}, 
}

@misc{hendrycks2023gaussianerrorlinearunits,
      title={Gaussian Error Linear Units (GELUs)}, 
      author={Dan Hendrycks and Kevin Gimpel},
      year={2023},
      eprint={1606.08415},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1606.08415}, 
}

@misc{he2015delvingdeeprectifierssurpassing,
      title={Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification}, 
      author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
      year={2015},
      eprint={1502.01852},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1502.01852}, 
}

@misc{ma2021activatenotlearningcustomized,
      title={Activate or Not: Learning Customized Activation}, 
      author={Ningning Ma and Xiangyu Zhang and Ming Liu and Jian Sun},
      year={2021},
      eprint={2009.04759},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2009.04759}, 
}

@misc{lomeli2025stochasticactivations,
      title={Stochastic activations}, 
      author={Maria Lomeli and Matthijs Douze and Gergely Szilvasy and Loic Cabannes and Jade Copet and Sainbayar Sukhbaatar and Jason Weston and Gabriel Synnaeve and Pierre-Emmanuel Mazaré and Hervé Jégou},
      year={2025},
      eprint={2509.22358},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.22358}, 
}

@misc{molina2020padeactivationunitsendtoend,
      title={Pad\'e Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks}, 
      author={Alejandro Molina and Patrick Schramowski and Kristian Kersting},
      year={2020},
      eprint={1907.06732},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1907.06732}, 
}

@misc{liu2025kankolmogorovarnoldnetworks,
      title={KAN: Kolmogorov-Arnold Networks}, 
      author={Ziming Liu and Yixuan Wang and Sachin Vaidya and Fabian Ruehle and James Halverson and Marin Soljačić and Thomas Y. Hou and Max Tegmark},
      year={2025},
      eprint={2404.19756},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.19756}, 
}
```


---
## Appendix A: Manifold Sweep Code Reference

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def empirical_psilu_sweep(res_beta=500, res_delta=500, res_x=300):
    """
    High-resolution sweep with refined constraints to identify stable, diverse (β,δ) subspace.
    Returns valid beta/delta coordinates and boolean validity mask.
    """
    print(f"Starting Refined High-Res Sweep ({res_beta}x{res_delta})...")
    
    # 1. Setup Grid
    beta_vals = np.linspace(-2.0, 8.0, res_beta, dtype=np.float32)
    delta_vals = np.linspace(-6.0, 6.0, res_delta, dtype=np.float32)
    B, D = np.meshgrid(beta_vals, delta_vals)
    x_vals = np.linspace(-3.0, 3.0, res_x, dtype=np.float32)
    
    # 2. Pre-allocate 2D Metric Arrays
    max_grad = np.zeros_like(B)
    min_grad = np.zeros_like(B)
    gate_std = np.zeros_like(B)
    mean_gate = np.zeros_like(B)
    
    # 3. Chunked Processing (memory-efficient)
    chunk_size = 50 
    for start_row in range(0, res_delta, chunk_size):
        end_row = min(start_row + chunk_size, res_delta)
        b_chunk = B[start_row:end_row, :, np.newaxis]
        d_chunk = D[start_row:end_row, :, np.newaxis]
        x_slice = x_vals[np.newaxis, np.newaxis, :]

        z = b_chunk * x_slice + d_chunk
        sig_z = 1.0 / (1.0 + np.exp(-z))
        g = sig_z + x_slice * b_chunk * sig_z * (1.0 - sig_z)
        
        max_grad[start_row:end_row, :] = np.max(g, axis=2)
        min_grad[start_row:end_row, :] = np.min(g, axis=2)
        gate_std[start_row:end_row, :] = np.std(sig_z, axis=2)
        mean_gate[start_row:end_row, :] = np.mean(sig_z, axis=2)

    # 4. Refined Constraints 
    safe_flow = (max_grad < 2.2) & (min_grad > -0.5)      # Gradient stability
    active_gating = gate_std > 0.08                         # Gate switches across x
    dynamic_range = (mean_gate > 0.05) & (mean_gate < 0.95) # Avoid saturation
    validity_mask = safe_flow & active_gating & dynamic_range
    
    # Apply smoothing for clean visual contour
    validity_mask_smoothed = gaussian_filter(validity_mask.astype(np.float32), sigma=1.5) > 0.5
    
    # 5. Visualization (optional)
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    extent = [-2, 8, -6, 6]
    
    def add_overlay(ax):
        ax.contour(B, D, validity_mask_smoothed, levels=[0.5], colors='white', linewidths=1.2)

    im1 = axes[0].imshow(min_grad, extent=extent, origin='lower', aspect='auto', cmap='RdYlGn')
    add_overlay(axes[0]); axes[0].set_title("Min Gradient Field"); plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(gate_std, extent=extent, origin='lower', aspect='auto', cmap='plasma')
    add_overlay(axes[1]); axes[1].set_title("Gate Std Dev"); plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(mean_gate, extent=extent, origin='lower', aspect='auto', cmap='coolwarm')
    add_overlay(axes[2]); axes[2].set_title("Mean Gate (Bias)"); plt.colorbar(im3, ax=axes[2])

    axes[3].imshow(validity_mask_smoothed, extent=extent, origin='lower', aspect='auto', cmap='gray')
    axes[3].scatter([1.0], [0.0], color='cyan', marker='*', s=200, label="SiLU", edgecolors='black')
    axes[3].scatter([1.702], [0.0], color='lime', marker='*', s=200, label="GeLU", edgecolors='black')
    axes[3].set_title("FINAL SAMPLING MANIFOLD"); axes[3].legend()

    for ax in axes: ax.set_xlabel("Beta (β)"); ax.set_ylabel("Delta (δ)")
    plt.tight_layout(); plt.show()

    return B[validity_mask_smoothed], D[validity_mask_smoothed], validity_mask_smoothed

if __name__ == "__main__":
    valid_b, valid_d, mask = empirical_psilu_sweep()
    np.savez("psilu_manifold.npz", betas=valid_b, deltas=valid_d)
    print(f"Manifold analysis complete. Total valid coordinates: {len(valid_b)}")
```

---

*This report is a technical description intended to facilitate reproducibility and discussion. All mathematical derivations, empirical observations, and code snippets are provided as-is. No claims of superiority, universal applicability, or production readiness are made. Further validation across diverse architectures, datasets, and training regimes is encouraged.*

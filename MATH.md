# Statistical Manifold Dynamics (SMD) — Math & Implementation Guide

> **Who this is for:** no background in differential geometry or quant finance required. Every symbol is defined before it's used, every formula gets a plain-English translation right after it, and there's a fully worked numeric example near the end. If you already know the math, skim the notation table and jump to Section 4.

---

## 0. The one-sentence idea

> Markets don't just move on a flat map — they move on a **curved surface**, and the curvature itself (not just how fast prices wiggle) tells you when a crash or breakout is coming.

**Analogy:** imagine walking on a landscape. On flat, gentle ground, you can wander wherever you like and nothing dramatic happens. But near a cliff edge (a region of sharp curvature), the ground itself constrains you — you get squeezed into a narrow path, tension builds, and a small nudge can send you tumbling. SMD's claim is that markets have "cliff-edge" regions like this, and a model that can *see the shape of the ground*, not just how fast you're walking, can spot the cliff before you fall off it.

---

## 1. Notation glossary — read this before anything else

| Symbol | Plain-English meaning | Type |
|---|---|---|
| $t$ | A point in time | index |
| $X_t$ | Observable market data at time $t$ (prices, volumes, order book, etc.) | vector of numbers |
| $\theta_t$ | The market's hidden ("latent") state at time $t$ — the model's internal summary of "where the market is" | point on a manifold $M$ |
| $M$ | A **manifold** — think "a curved surface that $\theta_t$ lives on," like the surface of a hill, instead of a flat sheet of paper | space |
| $p(X_t \mid \theta_t)$ | The probability of seeing data $X_t$ given hidden state $\theta_t$ | probability |
| $g(\theta)$, $g_{ij}(\theta)$ | The **Fisher Information Metric (FIM)** — a matrix that measures how "sensitive" the observable data is to small changes in the hidden state, at every point on the manifold | matrix (the "ruler" of the surface) |
| $g^{-1}(\theta)$ | The inverse of that matrix — used to correct for the surface's local shape | matrix |
| $V(\theta, X_t)$ | A **potential function** — an economic "tension" the market is trying to relax (e.g. mispricing, order imbalance) | scalar function |
| $\nabla_\theta V$ | The **gradient** of $V$ — the direction of steepest increase in tension; $-\nabla_\theta V$ points toward *relaxing* it | vector |
| $R(\theta)$ | The local **scalar curvature** at state $\theta$ — a single number describing how "sharply bent" the manifold is right there | scalar |
| $dW_t$ | A tiny random nudge (Brownian motion / random noise), the mathematical building block of most random-walk models | random increment |
| $\circ$ | A symbol meaning this equation uses **Stratonovich calculus** (a technical convention needed when noise is being reshaped by a changing metric — see Section 4) | notation |
| $v_t$ | The **geodesic velocity** — how fast the hidden state is moving, measured using the manifold's own ruler $g$, not a flat ruler | scalar |
| $S_t$ | The **Alpha Signal** ("Geodesic Tension") — the model's final tradeable output | scalar |
| $\hat{Y}_t$ | The model's prediction (e.g. expected return) | scalar or vector |
| $w_t$ | Portfolio weight — how much capital to allocate | scalar |
| $\alpha, \beta$ | Tuning numbers that balance the two ingredients of the Alpha Signal | positive numbers |

**A note on "manifolds":** you don't need formal topology to follow this document. Every time you see "manifold," just picture a curved surface (like a hill, a saddle, or a valley) instead of a flat tabletop. Distances and "straight lines" on a curved surface behave differently than on a flat one — that difference is the entire point of this model.

---

## 2. The core hypothesis, formalized

**Hypothesis in words:** predictable big price moves come from **structural fragility** (geometric curvature), not just from high volatility. A market can be volatile without being fragile, and fragile without (yet) being volatile — SMD is built to tell these apart.

### 2.1 Setting up the hidden state

$$
\theta_t \in M
$$

**What this means:** at each moment, the market has some hidden condition $\theta_t$ that lives on a curved surface $M$. We never observe $\theta_t$ directly — we only observe $X_t$, which is *generated from* $\theta_t$ through a probability distribution:

$$
p(X_t \mid \theta_t)
$$

**What this means:** given a hidden state, there's a known (or learned) recipe for how likely each possible observation is. This is standard in statistics — think of $\theta_t$ as "the true bias of a coin" and $X_t$ as "the coin flips you actually see."

### 2.2 The Fisher Information Metric — the "ruler" of the surface

$$
g_{ij}(\theta) = \mathbb{E}\left[ \frac{\partial \log p(X\mid\theta)}{\partial \theta_i} \frac{\partial \log p(X\mid\theta)}{\partial \theta_j} \right]
$$

**What this means, piece by piece:**
- $\log p(X\mid\theta)$ is the *log-likelihood* — a standard statistics quantity measuring how well a given $\theta$ explains the data $X$.
- $\dfrac{\partial \log p(X\mid\theta)}{\partial \theta_i}$ is "how much does the log-likelihood change if I nudge $\theta$ slightly in direction $i$" — a sensitivity measurement.
- Multiplying two of these sensitivities together and averaging ($\mathbb{E}[\cdot]$) over the data gives a number for every pair of directions $(i,j)$ — together, all these numbers form the matrix $g(\theta)$.

**Intuition:** $g(\theta)$ tells you **how much the data would change if the hidden state moved a little**. If $g$ is large in some direction, a tiny move in $\theta$ produces a huge, easily-detectable change in the data — meaning that direction is "steep" or "risky." If $g$ is small, you can move $\theta$ a lot without the data changing much — that direction is "flat" and safe. This is exactly how a physical ruler on a curved map works: a centimeter on the map might represent a mile in a "stretched" region and only a few feet in a "compressed" region — $g(\theta)$ is that stretching factor, generalized to many dimensions.

### 2.3 The three physical assumptions

**(a) Potential energy.** The market has some quantity $V(\theta, X_t)$ representing economic "tension" it's trying to reduce — think of it as strain in a system, like mispricing between related assets. Markets are assumed to drift toward *lower* tension over time.

**(b) Natural gradient flow, not ordinary gradient descent.** A normal machine learning model would just move $\theta$ in the direction of $-\nabla_\theta V$ (straight downhill on a flat map). SMD instead uses the **natural gradient**:

$$
-g^{-1}(\theta)\,\nabla_\theta V
$$

**What this means:** before moving downhill, first *correct the direction for the shape of the surface* using $g^{-1}$. On a flat map, $g^{-1}$ is just the identity (does nothing) and this reduces to ordinary gradient descent. On a curved surface, $g^{-1}$ stretches or shrinks the step depending on the local geometry — exactly like how you'd naturally take smaller steps walking on a steep hillside than on a flat field, even if you're trying to descend at the same "logical rate."

**(c) Curvature = fragility.** The scalar curvature $R(\theta)$ measures how sharply the manifold bends at that point. High $R(\theta)$ means the state $\theta_t$ is currently in a geometrically fragile spot — small pushes can cause outsized effects, similar to standing near the edge of a cliff versus in the middle of a flat field.

---

## 3. The dynamics: how the hidden state evolves over time

Instead of a plain Neural ODE (see the EDS companion document for that simpler case), SMD uses a **Riemannian Stochastic Differential Equation (RSDE)**:

$$
d\theta_t = -g^{-1}(\theta_t)\,\nabla_\theta V(\theta_t, X_t)\,dt \;+\; g^{-1/2}(\theta_t) \circ dW_t
$$

**Breaking this into two pieces:**

- **Drift term** $-g^{-1}(\theta_t)\nabla_\theta V(\theta_t,X_t)\,dt$: the "economic force" pulling the state toward lower tension, corrected for the local geometry as explained above. This is the *deterministic* part — where the market "wants" to go.
- **Diffusion term** $g^{-1/2}(\theta_t)\circ dW_t$: random noise ($dW_t$), but *scaled by the geometry* through $g^{-1/2}$ (a matrix square root of $g^{-1}$). This means the amount of random "jitter" the state experiences also depends on where it is on the manifold.

> **Why "Stratonovich" ($\circ$) instead of the more common Itô calculus?** When noise is being multiplied by something that itself depends on the current state (here, $g^{-1/2}(\theta_t)$), there are two mathematically valid ways to define what "multiply" even means at an instant — Itô and Stratonovich. Stratonovich is the convention that behaves consistently with ordinary calculus rules when your space is curved, which is why it's required here. You don't need to compute this by hand — numerical solvers handle the distinction — but it's worth knowing the equation isn't using the "default" convention for a specific, principled reason.

**The key qualitative behavior:**
- In **flat, low-curvature regions**: $g^{-1}$ is close to "normal size," so the market drifts and jitters roughly the way a standard model would expect.
- In **highly curved regions**: the eigenvalues (roughly: "the amount of stretching") of $g$ become large in certain directions, which shrinks $g^{-1}$ in those directions. The state effectively gets "stuck" — it moves less per unit of push — and **tension builds up** until the driving force $\nabla_\theta V$ finally overcomes the resistance, producing a fast, sudden move: a phase transition. This is the model's mathematical description of a "regime change."

---

## 4. The prediction: the Alpha Signal

SMD doesn't just forecast the next value of $\theta_t$ — it forecasts the *geometric character* of the market's trajectory.

### 4.1 Two geometric quantities

$$
R(\theta_t) \qquad \text{(local curvature — "how fragile is it here?")}
$$

$$
v_t = \|\dot\theta_t\|_g \qquad \text{(geodesic velocity — "how fast is the state moving, measured with the manifold's own ruler?")}
$$

**What $\|\cdot\|_g$ means:** a normal vector length (like the Pythagorean theorem) uses a flat ruler. $\|\dot\theta_t\|_g$ instead measures length *using the local metric $g$* — so the same raw speed $\dot\theta_t$ can count as "fast" in a stretched region and "slow" in a compressed one, exactly mirroring the ruler intuition from Section 2.2.

### 4.2 The Alpha Signal formula

$$
S_t = \alpha \cdot R(\theta_t) \cdot \|\nabla_\theta V\|_g \;+\; \beta \cdot \text{Geodesic Acceleration}
$$

**What this means:** the signal fires strongly only when **both** of two things are true at once:
1. The market is in a **structurally fragile** spot (high curvature $R(\theta_t)$), **and**
2. There is a **strong economic force** pushing it ($\|\nabla_\theta V\|_g$ is large).

The second term, geodesic acceleration (how quickly the geodesic velocity itself is changing), adds sensitivity to *sudden* changes in the state's motion — catching the moment the "snap" actually starts, not just the buildup.

**Why this design matters:** fragility alone (a cliff with nobody pushing) doesn't move prices. A strong push in a perfectly stable, flat region (a shove on flat ground) also doesn't cause a dramatic move. It's the **combination** — fragility *and* force — that the model claims isolates genuine regime changes from ordinary noise.

---

## 5. Where the novelty comes from

| Question | Answer for SMD |
|---|---|
| **New neural network architecture?** | No — it uses standard building blocks (encoders, MLPs, a Neural ODE-style solver) as plumbing. |
| **New training objective?** | Partially — the metric-learning and RSDE-consistency terms are non-standard, but the headline claim is elsewhere. |
| **New inductive bias?** | **Yes — the core contribution.** Market state changes are assumed to obey the *information geometry* of the data, not a flat Euclidean space. |
| **New representation of the market?** | **Yes.** The market is represented as a point moving on a curved statistical manifold, with a locally-defined "ruler" ($g$) rather than a single global one. |

**The key conceptual distinction from ordinary risk models:**
- **Volatility (variance)** answers: *"how much is the market moving?"*
- **Curvature** answers: *"is the market's current structure about to break?"*

A market can have low volatility yet sit in a high-curvature (fragile) region — SMD's bet is that this is exactly the situation standard variance-based models miss.

---

## 6. Implementation pipeline (how this becomes actual code)

You do **not** need general infinite-dimensional differential geometry to implement this. Four practical components:

**1. The Encoder** — a neural network maps raw data to the hidden state:
$$
X_t \xrightarrow{} \theta_t
$$

**2. The Metric Learner** — a separate small network outputs the metric tensor $g_\phi(\theta_t)$ at any given point. To guarantee it's mathematically valid (a metric must be symmetric and positive-definite — loosely, "never produce a negative squared distance"), the network outputs a **Cholesky decomposition** $L$ with strictly positive diagonal entries (enforced via $\text{softplus}(\cdot) > 0$) and sets:
$$
g = L L^T + \epsilon I
$$
**What a Cholesky decomposition is, intuitively:** for any valid metric matrix $g$, there is a unique lower-triangular matrix $L$ with positive diagonal entries such that multiplying $L$ by its own transpose reconstructs $g$ exactly. Having the network output $L$ with constrained positive diagonals guarantees the metric tensor is non-singular and strictly positive-definite across the latent manifold.

**3. The Dynamics Engine & Geometric Proxy** — in full theory, the RSDE from Section 3 is integrated forward in time. In the empirical PyTorch prototype, natural gradient flow is discretized along a potential gradient step:
$$
\theta_{t+\Delta} = \theta_t - \eta \, g^{-1}(\theta_t)\nabla_\theta V(\theta_t)
$$
Furthermore, while exact scalar curvature $R(\theta)$ requires second-order spatial derivatives of the metric via Christoffel symbols, empirical implementations commonly use the **Riemannian volume density form** $\log(\det g) = 2 \sum_i \log L_{ii}$ as an efficient local metric distortion proxy.

**4. The Head** — a final MLP combines the evolved state and the geometric features into the prediction:
$$
(\theta_{t+\Delta},\ \log\det g,\ v_t) \xrightarrow{\text{MLP}} \hat{Y}_t
$$

---

## 7. From signal to trading strategy

$$
S_t \rightarrow \hat{Y}_t \rightarrow w_t
$$

**What this means:** the Alpha Signal $S_t$ is mapped to a prediction $\hat{Y}_t$ (expected return, or probability of an upward move), which then gets fed into a **standard portfolio optimizer** (e.g. mean-variance optimization with transaction costs and risk limits) to produce final position sizes $w_t$. SMD deliberately keeps this last step "boring and standard" — the novelty is entirely in generating $S_t$, not in reinventing portfolio construction.

**The self-adjusting property:** because $S_t$ is built from curvature, it behaves like an automatic regime-aware position sizer:
- **Low curvature** → the model expects ordinary mean-reversion → signal is dampened → smaller positions.
- **High curvature** → the model expects a breakout/phase-transition → signal is amplified → larger positions.

This means volatility targeting emerges naturally from the geometry, rather than needing a separate hand-tuned rule.

---

## 8. A small worked numeric example

Real manifolds can have many dimensions, but the mechanics below hold in miniature with just two dimensions, so you can see actual numbers move through the equations.

Suppose at time $t$ the hidden state is $\theta_t = (1.0,\ 1.0)$, and the metric at this point happens to be:

$$
g(\theta_t) = \begin{bmatrix} 4 & 0 \\ 0 & 1 \end{bmatrix}
$$

**Reading this:** the metric says "direction 1 is 4× as 'sensitive'/'stretched' as direction 2." A small move along direction 1 changes the observable data much more than the same-sized move along direction 2.

Its inverse (used in the natural gradient and the drift term) is:

$$
g^{-1}(\theta_t) = \begin{bmatrix} 1/4 & 0 \\ 0 & 1 \end{bmatrix}
$$

Suppose the economic tension gradient is $\nabla_\theta V = (2.0,\ 2.0)$ — i.e., ordinary gradient descent would want to push equally in both directions. The **natural gradient** direction is instead:

$$
-g^{-1}\nabla_\theta V = -\begin{bmatrix} 1/4 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 2.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} -0.5 \\ -2.0 \end{bmatrix}
$$

**Reading this:** even though the raw economic pressure was identical in both directions, the natural gradient moves the state **four times further** along direction 2 than direction 1. Direction 1 is "stiff" (high sensitivity, $g_{11}=4$), so the model deliberately takes a smaller step there — exactly the cliff-edge caution described in Section 0. If instead $g_{11}$ had been, say, $100$ (an extremely fragile direction), the correction $1/g_{11}=0.01$ would nearly freeze movement in that direction entirely — the "getting stuck" behavior from Section 3 — until enough tension built up elsewhere to force a move.

---

## 9. Evaluating whether it actually works

A geometric story is not evidence by itself. The test plan compares SMD against the same baselines and under the same conditions as any other candidate model:

| Metric | Plain-English meaning |
|---|---|
| **Baselines** | Linear regression, XGBoost, standard LSTM, standard Transformer — all trained on identical data and features. |
| **Identical conditions** | Same dataset, same train/test split, same purged walk-forward validation scheme (test data always chronologically after training data, with a gap to avoid leakage). |
| **Information Coefficient (IC)** | Does the Geodesic Tension signal $S_t$ actually rank-correlate with future returns, beyond chance? |
| **Turnover & slippage** | Does the strategy trade too frequently, eating its own edge in costs? Curvature-based signals can be jumpy, so this must be explicitly penalized, not assumed away. |
| **Regime performance** | Does SMD specifically outperform baselines during high-volatility/crash periods — the exact scenario the curvature hypothesis is built to catch? A model that only matches baselines in calm markets hasn't demonstrated its unique claim. |

---

## 10. Recap — why this counts as a real research question

The claim is **not** "a fancier neural network was built." The claim is:

> *Markets move on a statistical manifold whose local geometry — not just its volatility — encodes structural fragility, and a model that explicitly tracks this curvature can isolate genuine regime changes from ordinary noise, improving prediction specifically during high-fragility periods.*

This is falsifiable: if $S_t$ shows no meaningful IC, or if SMD's advantage over baselines disappears once turnover/slippage are accounted for, or if it fails to specifically outperform during crash regimes, the hypothesis is wrong. That built-in possibility of failure is what makes Section 9 a real test and not a formality.

---

## 11. Mini glossary

| Term | Meaning |
|---|---|
| **Manifold** | A curved space that locally looks flat close-up (like the Earth's surface looking flat from where you stand) but is globally curved. |
| **Riemannian manifold** | A manifold equipped with a metric $g$ that lets you measure lengths, angles, and "straight lines" (geodesics) consistently with its curvature. |
| **Fisher Information Metric** | A specific, statistically-motivated way of measuring distance on the space of probability distributions — how much the data changes per unit change in the hidden parameter. |
| **Natural gradient** | Gradient descent that's been corrected by $g^{-1}$ to respect the local geometry, rather than assuming a flat space. |
| **Scalar curvature** ($R$) | A single number summarizing how sharply a manifold bends at a point; zero for a flat space. |
| **Geodesic** | The "straightest possible path" on a curved surface — the curved-space analogue of a straight line. |
| **Stochastic Differential Equation (SDE)** | A differential equation (Section 2.5 of the EDS document) with an added random noise term. |
| **Itô vs. Stratonovich calculus** | Two different, both valid, conventions for handling calculus with random noise; Stratonovich is the one that stays consistent with ordinary calculus rules on curved spaces. |
| **Cholesky decomposition** | A way of writing a valid (symmetric, positive-definite) matrix $g$ as $LL^T$ for some matrix $L$ — a convenient "square root" used to guarantee validity. |
| **Eigenvalues** (informal use here) | Roughly, "how much stretching happens in each principal direction" of a matrix like $g$ — large eigenvalues mean strong local sensitivity. |
| **Mean-variance optimization** | The standard portfolio construction method that balances expected return against risk (variance) and costs. |
| **Purged walk-forward validation** | Testing only on data chronologically after training data, with a gap to prevent subtle information leakage. |

---

## 12. Notes on scope

This document explains the **mathematical structure** of SMD. It intentionally excludes: specific hyperparameter choices, exact neural network architectures for the encoder/metric learner, live execution and slippage modeling, and production risk controls — those belong in an implementation-specific document once the research prototype above is validated against the baselines in Section 9.

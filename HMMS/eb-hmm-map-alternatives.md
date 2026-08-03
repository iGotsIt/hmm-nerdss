# MAP-EM alternatives to the variational EB-HMM

Companion to [`eb-hmm-derivation.md`](eb-hmm-derivation.md), which derives the
mean-field variational treatment of the hierarchical constrained-means HMM. This
document answers a follow-up: **is there a MAP-EM approach instead, and is it
viable?**

Short answer: the naive MAP-EM is not viable, and not for the reason you'd guess
— its objective is *unbounded above*, so it has no maximizer at all. But there is
a MAP-flavored method that is viable and probably **better** than the variational
method: Laplace empirical Bayes. The organizing lesson (§5) is that MAP-vs-VB is
the wrong axis to think along.

All numerical claims below are reproduced by
[`eb_map_checks.py`](eb_map_checks.py).

---

## 0. Notation recap

From the companion document: trace $n$ has level vector
$\theta^{(n)} \in \mathbb R^L$ ($L = 9$), state means $u_i^{(n)} = M_i\theta^{(n)}$
via the fixed selection matrices $M_i \in \mathbb R^{d \times L}$, hidden path
$z^{(n)}$, and

$$
\theta^{(n)} \sim \mathcal N(m, S), \qquad
y_t^{(n)} \mid z_t^{(n)}{=}i,\theta^{(n)} \sim \mathcal N\big(M_i\theta^{(n)}, \Sigma_i\big).
$$

The per-trace sufficient statistics are
$N_i^{(n)} = \sum_t \gamma_{it}^{(n)}$ and $r_i^{(n)} = \sum_t \gamma_{it}^{(n)} y_t$,
and the block system the code already builds is

$$
\Lambda^{(n)} = \sum_i N_i^{(n)} M_i^\top \Sigma_i^{-1} M_i \;\;(\texttt{coeff}),
\qquad
b^{(n)} = \sum_i M_i^\top \Sigma_i^{-1} r_i^{(n)} \;\;(\texttt{[c1;c2;c3]}).
$$

The variational method gives $q_n(\theta) = \mathcal N(\bar\theta^{(n)}, V^{(n)})$
with $V^{(n)} = (\Lambda^{(n)} + S^{-1})^{-1}$ and
$\bar\theta^{(n)} = V^{(n)}(b^{(n)} + S^{-1}m)$.

**The point to keep in view:** every method in this document computes the *same*
$\bar\theta^{(n)}$ from the *same* linear system. They differ only in what they
propagate into the estimate of $S$ alongside it.

---

## 1. Naive joint MAP is degenerate, not merely biased

The obvious "MAP-EM" reading treats the $\theta^{(n)}$ as parameters and maximizes
jointly with the hyperparameters:

$$
J\big(\theta^{(1:N)}, m, S, A, \Sigma\big) = \sum_n \Big[\log p\big(y^{(n)}\mid\theta^{(n)},A,\Sigma\big) + \log \mathcal N\big(\theta^{(n)}\mid m,S\big)\Big].
$$

The $\theta$ update is the same ridge solve as before. The $(m,S)$ update becomes
the plain sample MLE of the $\theta$'s, with no $V^{(n)}$ term — which is exactly
the "plug-in" estimator shown to collapse in §5.1 of the companion document.

That collapse is usually described as a bias. It is worse than that.

**Claim.** $J$ has no finite maximizer: $\sup J = +\infty$.

**Proof.** Fix any $m$, set $\theta^{(n)} = m$ for all $n$, and let $S = \varepsilon I$.
The prior terms give

$$
\sum_n \log \mathcal N(m \mid m, \varepsilon I) = -\tfrac{N}{2}\log\big|2\pi\varepsilon I\big| = -\tfrac{NL}{2}\log(2\pi\varepsilon) \;\longrightarrow\; +\infty
$$

as $\varepsilon \to 0$, while the data terms
$\sum_n \log p(y^{(n)}\mid m, A, \Sigma)$ are independent of $\varepsilon$ and
finite. So $J \to +\infty$ along this path. $\;\blacksquare$

The maximizing configuration is complete pooling with zero population variance —
precisely the model the hierarchy was built to avoid.

**Numerically** ($N = 12$ traces, per-trace data precision $\lambda = 4$, true
$S = 0.25$), sweeping $S$ downward along the degenerate path:

| $S$ | $10^{-1}$ | $10^{-2}$ | $10^{-4}$ | $10^{-8}$ | $10^{-16}$ |
|---|---|---|---|---|---|
| $J$ | $-7.8$ | $+6.1$ | $+33.7$ | $+88.9$ | $+199.5$ |

growing like $-\tfrac N2\log S$, without bound.

This subsumes the fixed-point argument in the companion document. The recursion
$S \leftarrow \lambda S^2/(\lambda S + 1) \to 0$ is not a numerical artifact or an
unlucky stationary point — it is the algorithm faithfully climbing toward a
degenerate supremum. Fixing the optimizer cannot help, because the problem is the
objective.

This is the same pathology as the joint mode of $(\theta, \tau)$ in a hierarchical
normal model always sitting at $\tau = 0$; see Gelman et al., *BDA3* §5.4, where
it is the standard argument for why one marginalizes rather than jointly
maximizes in hierarchical models.

**Verdict: not viable as stated.**

---

## 2. Three variants that are viable

### 2a. Fix $S$; don't estimate it

For any fixed $S \succ 0$ the objective is bounded and MAP-EM is well-posed. It
reduces to ridge regression of each trace's levels toward $m$ at a strength you
choose. Delete the $(m,S)$ update from the current code and you have it.

The cost is conceptual: this is no longer empirical Bayes. The selling point of
the hierarchy — that the shrinkage strength is set by the data, via the ratio of
$\Lambda^{(n)}$ to $S^{-1}$ — is exactly what you have given up. You can recover
some of it by choosing $S$ on held-out marginal likelihood (leave-one-trace-out),
at $N$-fold refit cost per candidate $S$.

Useful as a **baseline for comparison**, not as the production method.

### 2b. MAP-EM with an inverse-Wishart ridge on $S$

Put $S \sim \mathrm{IW}(\Psi, \nu)$ and maximize the penalized objective. The
prior contributes

$$
-\tfrac{\nu+L+1}{2}\log|S| \;-\; \tfrac12\operatorname{tr}\big(\Psi S^{-1}\big).
$$

Along $S = \varepsilon I$ the first term grows like $|\log\varepsilon|$ but the
second falls like $-\operatorname{tr}(\Psi)/2\varepsilon$, which dominates. So the
objective is bounded above and the degeneracy is removed. Note it is
$\Psi \succ 0$ that does the work, not $\nu$.

Combining with the Gaussian terms, the maximizer is available in closed form:

$$
\boxed{\;
S = \frac{\Psi + \sum_n \big(\theta^{(n)}-m\big)\big(\theta^{(n)}-m\big)^\top}{N + \nu + L + 1}
\;}
$$

Same sweep as §1, now with an inverse-gamma $\Psi = 0.05$:

| $S$ | $10^{-1}$ | $10^{-2}$ | $10^{-4}$ | $10^{-8}$ |
|---|---|---|---|---|
| $J_{\text{reg}}$ | $-3.4$ | $+12.8$ | $-197.9$ | $-2.5\times10^{6}$ |

Bounded, with an interior maximum near $10^{-2}$.

This is a one-line change and it works. But be clear about what it costs: you have
replaced *estimating* the population spread with *asserting* it. At the trace
counts typical here ($N$ of order $10$, $L = 9$), the numerator is
$\Psi + \text{(rank-}\le N{-}1\text{ scatter)}$ and the denominator is
$N + \nu + 10$, so $\Psi$ is a large fraction of the answer. It is a patch rather
than a principle.

### 2c. Laplace empirical Bayes — the MAP-flavored method worth using

Keep $\theta$ latent, but approximate the marginal likelihood by Laplace
expansion at the per-trace MAP $\hat\theta^{(n)}$ rather than by mean field:

$$
\log p\big(y^{(n)}\mid\Phi\big) \;\approx\;
\log p\big(y^{(n)}\mid\hat\theta^{(n)}, A, \Sigma\big)
+ \log \mathcal N\big(\hat\theta^{(n)}\mid m,S\big)
+ \tfrac L2\log 2\pi - \tfrac12\log\big|H^{(n)}\big|,
$$

$$
H^{(n)} = \mathcal I_{\text{obs}}^{(n)} + S^{-1},
\qquad
\mathcal I_{\text{obs}}^{(n)} = -\nabla_\theta^2 \log p\big(y^{(n)}\mid\theta, A, \Sigma\big)\Big|_{\hat\theta^{(n)}} .
$$

Maximizing over $(m,S)$ gives updates with the exact shape of the variational
ones, with $V^{(n)}$ replaced by $(H^{(n)})^{-1}$:

$$
m = \frac1N\sum_n \hat\theta^{(n)},
\qquad
S = \frac1N\sum_n\Big[\big(\hat\theta^{(n)}-m\big)\big(\hat\theta^{(n)}-m\big)^\top + \big(H^{(n)}\big)^{-1}\Big].
$$

Structurally this is the same code path — one matrix swapped. Section 3 explains
why that swap matters.

---

## 3. Why Laplace EB beats the variational method here

The variational $V^{(n)} = (\Lambda^{(n)} + S^{-1})^{-1}$ is built from
$\Lambda^{(n)}$, which is the **complete-data** information — the information the
trace would carry about its levels *if the state path were known*. Louis's
identity relates it to the observed information of the actual marginal
likelihood:

$$
\mathcal I_{\text{obs}} \;=\; \mathcal I_{\text{complete}} \;-\; \mathcal I_{\text{missing}},
\qquad
\mathcal I_{\text{missing}} = \operatorname{Var}\big[\nabla_\theta \log p(y,z\mid\theta) \,\big|\, y\big] \succeq 0,
$$

so $\mathcal I_{\text{obs}} \preceq \Lambda^{(n)}$ always.

The variational method therefore credits each trace with **more information about
its levels than the data contains**, because it conditions on the responsibilities
as though the path were resolved. Consequences, in order:

1. $V^{(n)}$ is too small.
2. $S = \frac1N\sum[\text{scatter} + V^{(n)}]$ is too small.
3. The shrinkage $\big(\Lambda^{(n)} + S^{-1}\big)^{-1}S^{-1}m$ is too aggressive.

This is the familiar "mean-field VB underestimates posterior variance" result. It
matters more than usual here because $S$ — the thing being biased — is not a
nuisance quantity but the estimand that controls the entire hierarchy.

**Measured** on a synthetic 2-state, $T = 400$ HMM ($\sigma = 0.35$, levels $0.30$
and $0.85$, sticky transitions), comparing the two informations at the fitted
$\hat\theta$:

$$
\Lambda = \begin{pmatrix} 1091.5 & 0 \\ 0 & 2173.8\end{pmatrix},
\qquad
\mathcal I_{\text{obs}} = \begin{pmatrix} 728.3 & -211.5 \\ -211.5 & 1715.1\end{pmatrix}
$$

$\Lambda - \mathcal I_{\text{obs}}$ is PSD with eigenvalues $194$ and $628$, as
Louis requires. At $S = 0.05\,I$ the resulting posterior variances are

| | level 0 | level 1 |
|---|---|---|
| variational $V$ (uses $\Lambda$) | $9.0\times10^{-4}$ | $4.6\times10^{-4}$ |
| Laplace $V$ (uses $\mathcal I_{\text{obs}}$) | $1.38\times10^{-3}$ | $6.0\times10^{-4}$ |
| variational understates by | **35 %** | **24 %** |

There is a second, more structural point visible in those matrices. $\Lambda$ came
out exactly **diagonal**, while $\mathcal I_{\text{obs}}$ carries an off-diagonal
of $-211$. The complete-data information can only couple levels through $\Sigma$
and $M$; the true likelihood *also* couples them through frames that are ambiguous
between states, where raising one level and lowering the other leaves the fit
nearly unchanged. That negative correlation is real posterior structure, and mean
field cannot represent it at all. In your 3-channel setting, this is the coupling
between levels that a mid-FRET frame induces between the mid and high levels of
the same channel.

### Cost

You need $\mathcal I_{\text{obs}}^{(n)}$. Two routes:

- **Louis's identity directly** — accumulate the conditional variance of the
  complete-data score during forward-backward. Roughly one extra pass, but fiddly
  to derive and easy to get subtly wrong.
- **Finite differences** — much easier to trust. Fisher's identity gives the
  gradient for free from quantities you already accumulate,
  $\nabla_\theta \log p(y\mid\theta) = b^{(n)} - \Lambda^{(n)}\theta$, so
  central-differencing the *gradient* costs $2L = 18$ scaled forward passes per
  trace per iteration and is far better conditioned than differencing the
  log-likelihood twice.

At smFRET trace counts ($N \sim 10$–$100$, $T \sim 10^3$) the finite-difference
route is entirely tractable, especially since $\mathcal I_{\text{obs}}$ need not
be recomputed every iteration — refreshing it every few iterations, or only after
the parameters have roughly settled, captures most of the benefit.

---

## 4. Blocked Gibbs as the reference standard

Worth knowing that this model is **fully conjugate given $z$**, so a blocked Gibbs
sampler requires no tuning and no proposals:

| block | full conditional | draw |
|---|---|---|
| $z^{(n)} \mid \theta^{(n)}, A, \pi, \Sigma$ | HMM path posterior | forward-filter, backward-sample |
| $\theta^{(n)} \mid z^{(n)}, m, S, \Sigma$ | $\mathcal N\big(V^{(n)}(b^{(n)}+S^{-1}m),\, V^{(n)}\big)$ | the §0 solve, sampled not maximized |
| $m, S \mid \theta^{(1:N)}$ | Normal-Inverse-Wishart | conjugate draw |
| $A_{i,:} \mid z$ | Dirichlet | conjugate draw |
| $\Sigma_i \mid z, \theta$ | Inverse-Wishart | conjugate draw |

Note the $\theta$ block reuses the exact linear system you already build — the
only change is drawing from $\mathcal N(\bar\theta^{(n)}, V^{(n)})$ instead of
returning its mean. Crucially, because $z$ is sampled rather than averaged, the
sampler does **not** inherit the complete-data-information problem of §3.

Too slow for routine fitting, but this is the honest reference. Run it once on a
handful of traces and you learn how much the mean field is actually costing you
in $\hat S$, instead of assuming.

---

## 5. Summary

| approach | $\theta$ treated as | objective bounded? | bias in $\hat S$ | verdict |
|---|---|---|---|---|
| Joint MAP, estimate $(m,S)$ | parameter | **no** — $J \to +\infty$ | collapses to $0$ | not viable |
| MAP, $S$ fixed by hand | parameter | yes | n/a | viable, not EB |
| MAP + IW ridge on $S$ | parameter | yes | pulled toward $\Psi$ | viable; prior-dominated at small $N$ |
| Mean-field VB (current code) | latent | yes | biased low (Louis) | viable; over-shrinks |
| Laplace EB | latent | yes | smaller | viable; best cost/accuracy |
| Blocked Gibbs | latent | yes | none asymptotically | reference standard |

**The organizing lesson: "MAP vs. variational" is the wrong axis.** All six
methods compute the same $\bar\theta^{(n)}$ from the same linear system. What
decides whether a method works is **whether it retains second-order information
about $\theta$, and whether that curvature is honest**:

- Joint MAP discards the curvature entirely → the objective diverges.
- Mean-field VB keeps a curvature that is systematically overconfident → $S$
  biased low, over-shrinkage.
- Laplace keeps the observed-information curvature → correct to second order.
- Gibbs needs no curvature because it samples.

### Recommended order of work

1. **Fix the three gaps in the current variational implementation first**
   (companion document §10). You cannot judge whether the mean field is adequate
   until it is implemented correctly, and two of the three gaps bias $S$ in the
   same direction as the Louis effect — so they are currently confounded.
2. **Simulate with known $(S, \Sigma)$ and check recovery of both.** This is the
   test that separates the §3 bias from the identifiability problem in the
   companion document §9, and it is cheap.
3. **If $\hat S$ is still biased low and it matters for your conclusions**, add
   Laplace via finite-differenced gradients (§2c). It is a contained change.
4. **Keep MAP with fixed $S$ (§2a) as a baseline**, not as the production path.
5. **Write the Gibbs sampler (§4) only if you need to defend the approximation**
   — e.g. for a referee asking how much the mean field costs.

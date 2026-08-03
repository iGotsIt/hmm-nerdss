# Empirical-Bayes constrained-means HMM: full derivation

This derives the hierarchical / empirical-Bayes multi-trace HMM implemented in
`expectation_maximization_multi_trace_hb` from a single objective, and shows
exactly which lines of that function do and do not follow from it.

The motivating question: *the per-trace mean update looks like maximizing a
penalized likelihood with $\theta$ as a parameter, but the hyperparameter update
takes an expectation over $\theta$ as a latent variable — aren't those two
different objectives?*

Answer: no. There is one objective. The per-trace solve is an **E-step**, not an
M-step; it computes the mean of a Gaussian variational posterior, and only
*looks* like a penalized maximization because a Gaussian's mode equals its mean.
Section 3 makes this precise. Section 4 shows the place where the code actually
does break consistency.

---

## 1. Model and notation

**Indices.** $n = 1,\dots,N$ traces; $t = 1,\dots,T_n$ frames; $i,j = 1,\dots,K$
hidden states ($K$ = `Nstate`); $d = 3$ channels (green, blue, red).

**Level parameterization.** Each trace has a level vector
$\theta^{(n)} \in \mathbb{R}^L$, $L = L_g + L_b + L_r$ (typically $9$), formed by
stacking $[u_g;\, u_b;\, u_r]$. The `stateindex` array fixes a **selection
matrix** for each state,

$$
M_i = \big[\, M_i^g \;\big|\; M_i^b \;\big|\; M_i^r \,\big] \in \mathbb{R}^{d \times L},
$$

which in code is `[Mg[:,:,i] | Mb[:,:,i] | Mr[:,:,i]]`. Each row of $M_i$ has a
single $1$ selecting that channel's level for state $i$. The state mean is
*linear* in $\theta$:

$$
u_i^{(n)} = M_i\,\theta^{(n)}.
$$

This linearity is what makes every step below closed-form.

**Generative model.**

$$
\begin{aligned}
\theta^{(n)} &\sim \mathcal N(m,\,S) &&\text{i.i.d. across traces}\\
z_1^{(n)} &\sim \pi, \qquad z_{t+1}^{(n)} \mid z_t^{(n)}{=}i \sim A_{i,:} &&\text{shared dynamics}\\
y_t^{(n)} \mid z_t^{(n)}{=}i,\ \theta^{(n)} &\sim \mathcal N\big(M_i\theta^{(n)},\ \Sigma_i\big) &&\text{shared noise}
\end{aligned}
$$

**Parameters** $\Phi = (m,\, S,\, \pi,\, A,\, \{\Sigma_i\})$.
**Latent variables** $\{z^{(n)},\, \theta^{(n)}\}_{n=1}^N$.

The defining feature of empirical Bayes is the placement of that line: the
population mean and spread $(m,S)$ are *parameters* to be estimated, while the
per-trace levels $\theta^{(n)}$ are *latent variables* to be integrated out —
exactly like the state paths $z^{(n)}$.

**Objective (type-II / marginal likelihood).**

$$
\boxed{\;
\ell(\Phi) \;=\; \sum_{n=1}^N \log \int \sum_{z} p\big(y^{(n)}, z, \theta \mid \Phi\big)\, d\theta
\;}
$$

Everything that follows is coordinate ascent on this one function.

---

## 2. The variational bound and the one approximation

For any distribution $q_n(z,\theta)$,

$$
\log p(y^{(n)}\mid\Phi) \;=\; \mathcal F_n[q_n,\Phi] \;+\; \mathrm{KL}\big(q_n \,\|\, p(z,\theta\mid y^{(n)},\Phi)\big),
$$

$$
\mathcal F_n[q_n,\Phi] \;=\; \mathbb E_{q_n}\!\big[\log p(y^{(n)},z,\theta\mid\Phi)\big] \;+\; \mathbb H[q_n],
\qquad \mathcal F = \sum_n \mathcal F_n .
$$

Since $\mathrm{KL}\ge 0$, $\mathcal F$ lower-bounds $\ell$, with equality iff
$q_n$ is the exact posterior. EM is coordinate ascent on $\mathcal F$ in
$(q, \Phi)$.

**Why we cannot use the exact posterior.** $\theta^{(n)}$ is shared by *every*
frame of trace $n$. Marginalizing it,

$$
\int \prod_{t} \mathcal N\big(y_t \mid M_{z_t}\theta, \Sigma_{z_t}\big)\, \mathcal N(\theta\mid m,S)\, d\theta,
$$

is Gaussian in $y$ but with a covariance that does **not** factorize across $t$.
It couples all timesteps and destroys the Markov structure that forward-backward
relies on. So we restrict $q$ to the mean-field family

$$
\boxed{\; q_n(z,\theta) \;=\; q_n(z)\;q_n(\theta) \;}
$$

**This factorization is the only approximation in the entire derivation.**
Everything below is exact given it. It is also the direct answer to the
motivating question: the mean field is precisely what licenses one update to
treat $\theta$ as a fixed quantity and another to average over it, without
contradiction — they are updates to two different factors of one $q$.

**Complete-data log joint.** With $\delta_{it} = \mathbb 1[z_t{=}i]$,

$$
\log p(y,z,\theta\mid\Phi) =
\underbrace{\log \mathcal N(\theta\mid m,S)}_{\text{prior}}
+ \underbrace{\log \pi_{z_1} + \sum_{t\ge2}\log A_{z_{t-1}z_t}}_{\text{dynamics}}
+ \sum_t \sum_i \delta_{it}\log \mathcal N\big(y_t \mid M_i\theta, \Sigma_i\big).
$$

---

## 3. E-step for $\theta$ — the update that looks like an M-step

Mean-field stationarity gives
$\log q_n(\theta) = \mathbb E_{q_n(z)}\!\left[\log p(y,z,\theta\mid\Phi)\right] + \text{const}$.
Keeping only $\theta$-dependent terms, and writing
$\gamma_{it}^{(n)} = \mathbb E_{q_n(z)}[\delta_{it}] = q_n(z_t{=}i)$:

$$
\log q_n(\theta) = -\tfrac12(\theta-m)^\top S^{-1}(\theta-m)
\;-\;\tfrac12\sum_t\sum_i \gamma_{it}^{(n)}\,(y_t - M_i\theta)^\top \Sigma_i^{-1}(y_t - M_i\theta) + \text{const}.
$$

Expand the second sum and collect powers of $\theta$. Define the **per-trace
sufficient statistics**

$$
N_i^{(n)} = \sum_t \gamma_{it}^{(n)} \quad(\texttt{gammaSumT}),
\qquad
r_i^{(n)} = \sum_t \gamma_{it}^{(n)}\, y_t \quad(\texttt{gammaSumY[:,i]}).
$$

Then

$$
-\tfrac12\,\theta^\top \underbrace{\Big[\textstyle\sum_i N_i^{(n)} M_i^\top \Sigma_i^{-1} M_i\Big]}_{\displaystyle \Lambda^{(n)}}\theta
\;+\;\theta^\top \underbrace{\Big[\textstyle\sum_i M_i^\top \Sigma_i^{-1} r_i^{(n)}\Big]}_{\displaystyle b^{(n)}}.
$$

**Identification with the code.** Because $M_i = [M_i^g \mid M_i^b \mid M_i^r]$,
the $(\rho,c)$ block of $\Lambda^{(n)}$ is
$\sum_i N_i^{(n)} (M_i^{\rho})^\top \Sigma_i^{-1} M_i^{c}$ — exactly the
`a11 … a33` blocks — and the $\rho$-th block of $b^{(n)}$ is
$\sum_i (M_i^{\rho})^\top \Sigma_i^{-1} r_i^{(n)}$ — exactly `c1, c2, c3`.
So $\Lambda^{(n)} = $ `coeff` and $b^{(n)} = $ `b`, provided $\Sigma_i^{-1}$
(a genuine precision) sits in the middle. The `_hb` function does invert
`sigma` first, so its units are right.

Adding the prior's $-\tfrac12\theta^\top S^{-1}\theta + \theta^\top S^{-1}m$:

$$
\log q_n(\theta) = -\tfrac12 \theta^\top\big(\Lambda^{(n)} + S^{-1}\big)\theta + \theta^\top\big(b^{(n)} + S^{-1}m\big) + \text{const},
$$

which is a Gaussian log-density. Completing the square:

$$
\boxed{\;
q_n(\theta) = \mathcal N\big(\theta \mid \bar\theta^{(n)},\, V^{(n)}\big),
\qquad
V^{(n)} = \big(\Lambda^{(n)} + S^{-1}\big)^{-1},
\qquad
\bar\theta^{(n)} = V^{(n)}\big(b^{(n)} + S^{-1} m\big).
\;}
$$

### 3.1 The reconciliation, stated precisely

The linear system $(\Lambda^{(n)} + S^{-1})\theta = b^{(n)} + S^{-1}m$ arises here
as **the mean of the variational posterior over a latent variable**. It is an
E-step. The alternative derivation — "add a Gaussian penalty to the M-step
objective and maximize over $\theta$ as a parameter" — writes down the same
linear system because it is computing the **mode** of the same Gaussian, and for
a Gaussian the mode equals the mean.

So the two framings agree on $\bar\theta^{(n)}$ and disagree on nothing. What the
latent-variable derivation gives you *in addition* is $V^{(n)}$, the posterior
covariance, which the parameter framing has no way to produce. That is why
$V^{(n)}$ appears in the $S$ update: it comes from the same $q_n(\theta)$, not
from a second objective.

**When this coincidence fails.** The reconciliation is available only because
$q_n(\theta)$ is exactly Gaussian, which requires (a) a Gaussian population prior
and (b) a *linear* link $u_i = M_i\theta$. Replace the prior with a Student-$t$
for outlier-robustness, or make the link nonlinear, and mode $\neq$ mean. Then
you must commit explicitly:

- **MAP-EM**: plug in the mode, drop $V^{(n)}$, accept the collapse in §5.1.
- **Variational / Laplace EM**: keep a mean and a curvature, as here.

---

## 4. E-step for $z$ — the term the code drops

By the same stationarity condition,

$$
\log q_n(z) = \log \pi_{z_1} + \sum_{t\ge2}\log A_{z_{t-1}z_t}
+ \sum_t \mathbb E_{q_n(\theta)}\!\left[\log \mathcal N\big(y_t \mid M_{z_t}\theta,\ \Sigma_{z_t}\big)\right] + \text{const}.
$$

Evaluate the expectation. Write $e_{it} = y_t - M_i\bar\theta^{(n)}$ and
$\Delta = \theta - \bar\theta^{(n)}$, so $\mathbb E[\Delta] = 0$ and
$\mathrm{Cov}[\Delta] = V^{(n)}$:

$$
\mathbb E\big[(e_{it}-M_i\Delta)^\top \Sigma_i^{-1}(e_{it}-M_i\Delta)\big]
= e_{it}^\top \Sigma_i^{-1} e_{it} + \mathbb E\big[\Delta^\top M_i^\top \Sigma_i^{-1} M_i \Delta\big]
= e_{it}^\top \Sigma_i^{-1} e_{it} + \operatorname{tr}\!\big(\Sigma_i^{-1} M_i V^{(n)} M_i^\top\big).
$$

Hence the **effective emission**

$$
\boxed{\;
\log \tilde p_i^{(n)}(y_t) \;=\; \log \mathcal N\big(y_t \mid M_i \bar\theta^{(n)},\, \Sigma_i\big)
\;-\; \tfrac12 \operatorname{tr}\!\big(\Sigma_i^{-1} M_i V^{(n)} M_i^\top\big).
\;}
$$

Two things to notice.

1. **The correction is independent of $t$.** It is one scalar per (state, trace),
   so implementing it costs $K$ numbers per trace: multiply column $i$ of
   `obs_dist` by $\exp\!\big(-\tfrac12\operatorname{tr}(\Sigma_i^{-1}M_i V^{(n)}M_i^\top)\big)$.
2. **It is state-dependent**, so it genuinely reweights the responsibilities. It
   down-weights states whose levels are poorly determined relative to their noise
   — the model's way of saying "don't confidently assign frames to a state whose
   mean you don't know."

With $\tilde p$ in hand, $q_n(z)$ is an ordinary HMM posterior, so standard
forward-backward returns exact $\gamma$ and $\xi$. The mean field preserves the
chain structure; that is the whole point of it.

**The code omits this correction** (`gaussian_emission` is called at the point
$\bar\theta^{(n)}$), which is the one place where the implementation silently
switches back to "$\theta$ is known exactly." It is the only real inconsistency
between the two updates in the motivating question.

---

## 5. M-step for the hyperparameters $(m, S)$

Only $\mathbb E_{q(\theta)}[\log \mathcal N(\theta\mid m,S)]$ depends on $m,S$:

$$
\mathcal F(m,S) = -\tfrac N2 \log|2\pi S| - \tfrac12 \operatorname{tr}\!\Big(S^{-1}\sum_n \mathbb E_{q_n}\big[(\theta-m)(\theta-m)^\top\big]\Big).
$$

The required expectation uses the *second* moment of $q_n(\theta)$:

$$
\mathbb E_{q_n}\big[(\theta-m)(\theta-m)^\top\big] = V^{(n)} + \big(\bar\theta^{(n)}-m\big)\big(\bar\theta^{(n)}-m\big)^\top .
$$

Setting $\partial\mathcal F/\partial m = 0$ and then $\partial\mathcal F/\partial S^{-1} = 0$:

$$
\boxed{\;
m = \frac1N\sum_n \bar\theta^{(n)},
\qquad
S = \frac1N\sum_n\Big[\big(\bar\theta^{(n)}-m\big)\big(\bar\theta^{(n)}-m\big)^\top + V^{(n)}\Big].
\;}
$$

Both use only $\mathbb E_q[\theta]$ and $\mathbb E_q[\theta\theta^\top]$ under the
*same* $q_n(\theta)$ derived in §3. There is no second likelihood function
anywhere.

### 5.1 Why dropping $V^{(n)}$ collapses $S$ to zero — proof

Take the scalar case ($L=1$), $N$ traces each with data precision $\lambda$, true
population variance $S$. The shrinkage weight is $w = \lambda S/(\lambda S + 1)$,
so $\bar\theta^{(n)} = w\hat\theta^{(n)} + (1-w)m$ where $\hat\theta^{(n)}$ is the
trace's own MLE, with $\operatorname{Var}(\hat\theta^{(n)}) = S + 1/\lambda$.

The **plug-in** estimator (scatter of the shrunk point estimates only) has
expectation

$$
w^2\Big(S + \tfrac1\lambda\Big)
= \frac{\lambda^2S^2}{(\lambda S+1)^2}\cdot\frac{\lambda S+1}{\lambda}
= \frac{\lambda S^2}{\lambda S + 1} \;<\; S .
$$

The missing piece is exactly the posterior variance
$V = (\lambda + 1/S)^{-1} = S/(\lambda S + 1)$, since

$$
\frac{\lambda S^2}{\lambda S+1} + \frac{S}{\lambda S+1} = \frac{S(\lambda S + 1)}{\lambda S + 1} = S. \qquad\checkmark
$$

So the correct update is unbiased at the truth, while the plug-in update iterates
the map $g(S) = \lambda S^2/(\lambda S + 1)$. Since $g(S)/S = w < 1$ for all
$S > 0$, every iteration multiplies $S$ by a factor strictly below one — and as
$S$ shrinks, $w$ shrinks too, so the contraction accelerates. Its only fixed
point is $S = 0$. At $S = 0$ the shrinkage is total and the model degenerates to
complete pooling.

This is why $V^{(n)}$ is not a refinement but a correctness requirement. The
collapse is fast, not asymptotic: with $\lambda = 4$ and a true $S = 0.25$, the
plug-in update returns $0.125$ on the first pass and reaches $S \approx 10^{-53}$
by the eighth iteration, while the corrected update sits exactly at $0.25$.

---

## 6. M-step for $\pi$, $A$, $\Sigma$

$\theta$ does not appear in the dynamics terms, so those updates are the
familiar pooled ones:

$$
\pi_i = \frac1N \sum_n \gamma_{i1}^{(n)},
\qquad
A_{ij} = \frac{\sum_n \sum_{t=1}^{T_n-1} \xi_{ijt}^{(n)}}{\sum_n \sum_{t=1}^{T_n-1}\gamma_{it}^{(n)}} .
$$

The covariance does involve $\theta$, and needs the same second-moment expansion
as §4:

$$
\mathbb E_{q(\theta)}\big[(y_t - M_i\theta)(y_t - M_i\theta)^\top\big]
= \big(y_t - M_i\bar\theta^{(n)}\big)\big(y_t - M_i\bar\theta^{(n)}\big)^\top + M_i V^{(n)} M_i^\top .
$$

Therefore

$$
\boxed{\;
\Sigma_i = \frac{\displaystyle\sum_n\Big[\sum_t \gamma_{it}^{(n)}\big(y_t - M_i\bar\theta^{(n)}\big)\big(y_t - M_i\bar\theta^{(n)}\big)^\top \;+\; N_i^{(n)}\, M_i V^{(n)} M_i^\top\Big]}{\displaystyle\sum_n N_i^{(n)}}\;}
$$

The second term in the numerator is the second place the code drops $V^{(n)}$.
Omitting it biases $\Sigma$ **downward**, which is self-reinforcing: smaller
$\Sigma_i$ inflates $\Lambda^{(n)} = \sum_i N_i M_i^\top\Sigma_i^{-1}M_i$, which
shrinks $V^{(n)} = (\Lambda^{(n)}+S^{-1})^{-1}$, which shrinks $S$ — re-entering
the §5.1 collapse through a side door even when the $V^{(n)}$ term in the $S$
update is present.

**Optional inverse-Wishart regularization.** With $\Sigma_i \sim \mathrm{IW}(\Psi,\nu)$
the MAP update just adds the prior's pseudo-counts:

$$
\Sigma_i = \frac{\Psi + \sum_n\big[\cdots\big]}{\nu + d + 1 + \sum_n N_i^{(n)}},
$$

with $[\cdots]$ the same numerator bracket as above. This is a principled
replacement for the hard variance floor and the `sigma0` fallbacks currently in
the code.

---

## 7. The free energy to monitor

Group $\mathcal F_n$ into two pieces. The first,
$\mathbb E_{q(z)}\mathbb E_{q(\theta)}[\log p(y,z\mid\theta)] + \mathbb H[q(z)]$,
is the standard HMM free energy *with emissions* $\tilde p$; at the optimal
$q_n(z)$ it equals $\log \tilde Z^{(n)}$, the log normalizer of the
$\tilde p$-HMM, which forward-backward returns as the sum of the log scaling
constants. The second,
$\mathbb E_{q(\theta)}[\log p(\theta\mid m,S)] + \mathbb H[q(\theta)]$,
is $-\mathrm{KL}$. Hence

$$
\boxed{\;
\mathcal F = \sum_n \Big[\underbrace{\textstyle\sum_t \log \tilde c_t^{(n)}}_{\log \tilde Z^{(n)}} \;-\; \mathrm{KL}\big(\mathcal N(\bar\theta^{(n)}, V^{(n)}) \,\big\|\, \mathcal N(m,S)\big)\Big]
\;}
$$

$$
\mathrm{KL} = \tfrac12\Big[\operatorname{tr}\big(S^{-1}V^{(n)}\big) + \big(\bar\theta^{(n)}-m\big)^\top S^{-1}\big(\bar\theta^{(n)}-m\big) - L + \ln\frac{\det S}{\det V^{(n)}}\Big].
$$

Three consequences:

- $\log\tilde Z$ must come from the **corrected** emissions of §4. With the
  uncorrected ones, $\mathcal F$ is not a bound on $\ell$ and need not increase.
- $\log p(y \mid \bar\theta, A, \Sigma)$ — the conditional likelihood at a point —
  is *not* $\mathcal F$ and is not monotone. Neither is it what either update
  ascends.
- $\mathcal F$ is monotone when compared **at the same phase of each iteration**,
  since coordinate ascent increases it at every individual step. Evaluating it
  with $\gamma$'s from the E-step and parameters from the M-step mixes two points
  in the cycle and can appear non-monotone even when the implementation is
  correct.

---

## 8. The algorithm

Per iteration:

1. **Per trace $n$**, given current $\Phi$ and $(\bar\theta^{(n)}, V^{(n)})$:
   - form the $K$ emission corrections $-\tfrac12\operatorname{tr}(\Sigma_i^{-1}M_iV^{(n)}M_i^\top)$;
   - run forward-backward on $\tilde p$ → $\gamma^{(n)}, \xi^{(n)}, \log\tilde Z^{(n)}$;
   - accumulate $N_i^{(n)}, r_i^{(n)}$ → build $\Lambda^{(n)}, b^{(n)}$;
   - solve $\bar\theta^{(n)} = (\Lambda^{(n)}+S^{-1})^{-1}(b^{(n)}+S^{-1}m)$ and store $V^{(n)}$;
   - accumulate pooled $\pi, A, \Sigma$ statistics **including** $N_i^{(n)}M_iV^{(n)}M_i^\top$.
2. **M-step**: $\pi, A$ from pooled $\gamma,\xi$; $\Sigma_i$ by §6; $m, S$ by §5.
3. **Record** $\mathcal F$ by §7 and test convergence on it.

Steps 1a–1b can be alternated to convergence within each trace before moving on;
one pass of each is the usual compromise and is what the code does.

---

## 9. Practical caveats specific to this model

**$\Sigma$ and $S$ compete.** Both explain variance across traces: $\Sigma_i$ as
within-trace measurement noise, $S$ as genuine between-trace heterogeneity of the
levels. With few traces they are weakly identified, and the §6 bias tips the
solution toward *both* being too small. Simulating traces with known $(S,\Sigma)$
and checking recovery of both is the right test.

**$S$ is $L\times L$ from $N$ traces.** The scatter term has rank at most
$N-1$, so for $N < L$ (commonly $L=9$) it is singular on its own; $V^{(n)}$ and
the $10^{-6}$ ridge are carrying it. Prefer a diagonal $S$, or an inverse-Wishart
MAP update
$S = \big(\Psi_S + \sum_n[\cdots]\big) / (N + \nu_S + L + 1)$, when $N$ is small.

**The code shares $\Sigma$ across traces**; the notebook's generative model
writes $\Sigma_i^{(n)} \sim \mathrm{IW}$, i.e. per-trace covariances. These are
different models. Per-trace $\Sigma^{(n)}$ would break the conjugacy used in §3
(the $\theta$ posterior would then depend on $\mathbb E[\Sigma^{-1}]$, requiring
a further mean-field factor $q(\Sigma)$ and a full VB treatment). Worth deciding
which model is intended before extending.

**Level ordering must mean the same thing in every trace** for shrinkage toward
$m$ to be meaningful. Here it is enforced structurally, since `stateindex` and
hence $M_i$ are fixed and shared — no label-switching correction is needed.

---

## 10. Summary: derivation vs. `expectation_maximization_multi_trace_hb`

| Result | Section | Code | Status |
|---|---|---|---|
| $\Lambda^{(n)}, b^{(n)}$ block structure | §3 | `a11…a33`, `c1,c2,c3` | correct, precision units right |
| $\bar\theta^{(n)} = (\Lambda+S^{-1})^{-1}(b+S^{-1}m)$ | §3 | `hmm-scratch.py:1383` | correct |
| $V^{(n)} = (\Lambda+S^{-1})^{-1}$ | §3 | `hmm-scratch.py:1384` | correct |
| Emission correction $-\tfrac12\operatorname{tr}(\Sigma_i^{-1}M_iVM_i^\top)$ | §4 | — | **missing** |
| $m = \frac1N\sum\bar\theta$ | §5 | `hmm-scratch.py:1410` | correct |
| $S = \frac1N\sum[(\bar\theta-m)(\bar\theta-m)^\top + V]$ | §5 | `hmm-scratch.py:1413-1417` | correct |
| $\pi$, $A$ pooled | §6 | `hmm-scratch.py:1401-1406` | correct |
| $\Sigma_i$ with $N_i M_i V M_i^\top$ | §6 | `hmm-scratch.py:1388-1390` | **missing correction term** |
| $\mathcal F = \sum \log\tilde Z - \sum\mathrm{KL}$ | §7 | `hmm-scratch.py:1338-1339` | **wrong quantity entirely** |

The three gaps are all the same error — reverting to "$\theta$ known exactly" on
the $z$ side of the E-step — and all three fixes are localized.

**See also:** [`eb-hmm-map-alternatives.md`](eb-hmm-map-alternatives.md) works
through the MAP-EM alternatives to the mean-field treatment derived here — why
naive joint MAP is degenerate rather than merely biased, and why Laplace
empirical Bayes is likely a better choice than the variational method above.


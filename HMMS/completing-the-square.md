# Completing the square: from a quadratic log-density to $\mathcal N(\bar\theta^{(n)}, V^{(n)})$

This expands the single step in
[`eb-hmm-derivation.md`](eb-hmm-derivation.md) §3 that is compressed into the
two words *"Completing the square:"* — the move from

$$
\log q_n(\theta) = -\tfrac12 \theta^\top\big(\Lambda^{(n)} + S^{-1}\big)\theta + \theta^\top\big(b^{(n)} + S^{-1} m\big) + \text{const}
$$

to

$$
q_n(\theta) = \mathcal N\big(\theta \mid \bar\theta^{(n)},\, V^{(n)}\big),
\qquad
V^{(n)} = \big(\Lambda^{(n)} + S^{-1}\big)^{-1},
\qquad
\bar\theta^{(n)} = V^{(n)}\big(b^{(n)} + S^{-1} m\big).
$$

There is no approximation anywhere in this document. The step is an exact
algebraic identity plus one normalization argument, and the whole content of it
is: **a Gaussian is the only density whose log is a concave quadratic, and you
can read its mean and covariance directly off the quadratic's coefficients.**

The trace index $n$ is fixed and inert throughout, so it is dropped after the
notation table and restored only in the code map at the end.

---

## 1. Notation

### Model quantities

| Symbol | Type / shape | Role | Code |
|---|---|---|---|
| $\theta$ | vector in $\mathbb R^{L}$ | **latent variable** — the trace's level vector, integrated out | `thetas[n]` |
| $L$ | positive integer, $L = L_g + L_b + L_r$ (typically $9$) | dimension of $\theta$ | `L` |
| $L_g, L_b, L_r$ | positive integers | number of distinct levels in the green, blue, red channels | `Lg, Lb, Lr` |
| $K$ | positive integer | number of hidden states | `Nstate` |
| $d$ | positive integer, $=3$ | channels (green, blue, red) | — |
| $i$ | index, $i=1,\dots,K$ | hidden state | — |
| $t$ | index, $t=1,\dots,T$ | frame within this trace | — |
| $T$ | positive integer | number of frames in this trace (written $T_n$ in the source) | `T` |
| $M_i$ | matrix in $\mathbb R^{d\times L}$ | selection matrix; each row has a single $1$ | `M_all[i]` |
| $M_i^g, M_i^b, M_i^r$ | matrices in $\mathbb R^{d\times L_g}$ etc. | the channel blocks of $M_i = [\,M_i^g \mid M_i^b \mid M_i^r\,]$; $M_i^\rho$ denotes a generic one, $\rho, c \in \{g,b,r\}$ | `Mg[:,:,i]` etc. |
| $u_i$ | vector in $\mathbb R^{d}$ | state $i$'s mean, $u_i = M_i\theta$ | `u[:,i]` |
| $z_t$ | value in $\{1,\dots,K\}$ | **latent variable** — hidden state at frame $t$ | — |
| `stateindex` | integer array | fixes which level each state selects in each channel, hence fixes $M_i$ | `stateindex` |
| $y_t$ | vector in $\mathbb R^{d}$ | observed datum at frame $t$ | `trace[:,t]` |
| $\Sigma_i$ | symmetric positive definite, $\mathbb R^{d\times d}$ | **parameter** — emission covariance | `sigma[:,:,i]` |
| $\Sigma_i^{-1}$ | symmetric positive definite, $\mathbb R^{d\times d}$ | emission precision | `precision[:,:,i]` |
| $m$ | vector in $\mathbb R^{L}$ | **parameter** — population mean of $\theta$ | `m` |
| $S$ | symmetric positive definite, $\mathbb R^{L\times L}$ | **parameter** — population covariance of $\theta$ | `S` |
| $\gamma_{it}$ | scalar in $[0,1]$ | responsibility $q(z_t{=}i)$; held **fixed** throughout | `gamma[i,t]` |
| $N_i$ | scalar $\ge 0$ | $\sum_t \gamma_{it}$, soft count of state $i$ | `gammaSumT[i]` |
| $r_i$ | vector in $\mathbb R^{d}$ | $\sum_t \gamma_{it}\,y_t$, soft data sum in state $i$ | `gammaSumY[:,i]` |

### Quantities introduced by this step

| Symbol | Type / shape | Role | Code |
|---|---|---|---|
| $\Lambda$ | symmetric positive **semi**definite, $\mathbb R^{L\times L}$ | data precision contributed to $\theta$ | `a` |
| $b$ | vector in $\mathbb R^{L}$ | data precision-weighted mean contribution | `b` |
| $P$ | symmetric positive definite, $\mathbb R^{L\times L}$ | $\Lambda + S^{-1}$, posterior precision | `coeff_n` |
| $h$ | vector in $\mathbb R^{L}$ | $b + S^{-1}m$, posterior potential | `b + Sinv @ m` |
| $V$ | symmetric positive definite, $\mathbb R^{L\times L}$ | posterior covariance $P^{-1}$ | `V[n]` |
| $\bar\theta$ | vector in $\mathbb R^{L}$ | posterior mean $P^{-1}h$ | `thetas[n]` |
| $\mu$ | vector in $\mathbb R^{L}$ | generic Gaussian center; equals $\bar\theta$ once identified | — |
| $\hat\theta$ | vector in $\mathbb R^{L}$ | per-trace MLE $\Lambda^{-1}b$; exists only when $\Lambda$ is invertible | — |
| $C$ | positive scalar | normalization constant of $q$ | — |
| $G$ | matrix in $\mathbb R^{Kd\times L}$ | factor with $G^\top G = \Lambda$ (§6c) | — |
| $\mathcal U, \mathcal T$ | index subsets partitioning $\{1,\dots,L\}$ | levels **untouched** / **touched** by visited states (§4) | — |
| $A_{\mathcal{UT}}$, $A_{\mathcal U,:}$, $v_{\mathcal U}$ | submatrix / subvector | rows $\mathcal U$ and columns $\mathcal T$ of $A$; all rows $\mathcal U$ of $A$; entries $\mathcal U$ of $v$ | — |
| $E_{\mathcal U}$ | matrix in $\mathbb R^{L\times|\mathcal U|}$ | selects the $\mathcal U$ coordinates (§7) | — |
| $\preceq$ | relation on symmetric matrices | Loewner order: $A \preceq B$ means $B - A$ is positive semidefinite | — |
| $\lambda_{\min}(\cdot)$ | scalar | smallest eigenvalue | — |
| $\lvert A\rvert$ | scalar | **determinant** of a square matrix $A$ | — |
| $\lVert v\rVert$ | scalar | Euclidean **norm** of a vector $v$ | — |
| $\mathcal F$, $\tilde Z^{(n)}$ | scalars | variational free energy and the effective-emission HMM normalizer; used only in the §8 code map, defined in §7 of the source document | `F`, `logZ_total` |

The parameter/latent distinction in column 3 is the one that matters. $\theta$ is
a latent variable; $(m, S, \Sigma_i)$ are parameters. This step computes a
*distribution over* $\theta$, not an estimate of it.

The names $P$ (precision) and $h$ (potential) are mine, introduced to keep the
identity readable; the source document writes $\Lambda + S^{-1}$ and
$b + S^{-1}m$ inline everywhere. In §7 the scalar textbook case uses $p_0$ for a
per-observation precision and $n_0$ for a count, deliberately avoiding $\lambda$,
which is reserved for eigenvalues throughout.

---

## 2. The step before: collecting terms

The quadratic does not fall out of the sky, and the collection that produces
$\Lambda$ and $b$ is where a stray factor of $\tfrac12$ would hide. Starting from
the mean-field stationarity result (§3 of the source, taken as given here):

$$
\log q(\theta) = \underbrace{-\tfrac12(\theta-m)^\top S^{-1}(\theta-m)}_{\text{(A) prior}}
\;\underbrace{-\;\tfrac12\sum_t\sum_i \gamma_{it}\,(y_t - M_i\theta)^\top \Sigma_i^{-1}(y_t - M_i\theta)}_{\text{(B) expected likelihood}} + \text{const}.
$$

**Expanding (B).** For each $(t,i)$, expand the quadratic form:

$$
(y_t - M_i\theta)^\top \Sigma_i^{-1}(y_t - M_i\theta)
= y_t^\top\Sigma_i^{-1}y_t \;-\; y_t^\top \Sigma_i^{-1} M_i\theta \;-\; \theta^\top M_i^\top \Sigma_i^{-1} y_t \;+\; \theta^\top M_i^\top\Sigma_i^{-1}M_i\theta .
$$

The two cross terms are equal. Each is a $1\times1$ matrix, hence equal to its own
transpose, so $y_t^\top \Sigma_i^{-1} M_i\theta = (y_t^\top \Sigma_i^{-1} M_i\theta)^\top = \theta^\top M_i^\top \Sigma_i^{-\top} y_t$,
and $\Sigma_i^{-\top} = \Sigma_i^{-1}$ because $\Sigma_i$ is symmetric (so is its
inverse). Therefore

$$
(y_t - M_i\theta)^\top \Sigma_i^{-1}(y_t - M_i\theta)
= y_t^\top\Sigma_i^{-1}y_t \;-\; 2\,\theta^\top M_i^\top \Sigma_i^{-1} y_t \;+\; \theta^\top M_i^\top\Sigma_i^{-1}M_i\theta .
$$

Substituting into (B) and multiplying through by $-\tfrac12$:

$$
\text{(B)} = \underbrace{-\tfrac12\sum_t\sum_i \gamma_{it}\, y_t^\top\Sigma_i^{-1}y_t}_{\text{no }\theta\;\to\;\text{const}}
\;+\; \theta^\top \sum_i M_i^\top \Sigma_i^{-1} \Big(\sum_t \gamma_{it} y_t\Big)
\;-\; \tfrac12\,\theta^\top \Big[\sum_i \Big(\sum_t \gamma_{it}\Big) M_i^\top\Sigma_i^{-1}M_i\Big]\theta .
$$

The sums over $t$ pass through $M_i^\top\Sigma_i^{-1}$ because that factor does
not depend on $t$ — this is the only reason $N_i$ and $r_i$ are sufficient
statistics. Recognizing $\sum_t\gamma_{it} = N_i$ and $\sum_t\gamma_{it}y_t = r_i$
gives exactly the definitions

$$
\Lambda \;\equiv\; \sum_i N_i\, M_i^\top \Sigma_i^{-1} M_i \qquad(\text{*definition*}),
\qquad
b \;\equiv\; \sum_i M_i^\top \Sigma_i^{-1} r_i \qquad(\text{*definition*}),
$$

so that $\text{(B)} = -\tfrac12\theta^\top\Lambda\theta + \theta^\top b + \text{const}$.

**Expanding (A).** The same three moves — expand, merge the cross terms using
$S^{-\top} = S^{-1}$, discard the $\theta$-free piece:

$$
-\tfrac12(\theta-m)^\top S^{-1}(\theta-m) = -\tfrac12\theta^\top S^{-1}\theta + \theta^\top S^{-1}m \underbrace{-\tfrac12 m^\top S^{-1} m}_{\text{const}} .
$$

**Adding.** The two quadratic coefficients add and the two linear coefficients
add:

$$
\log q(\theta) = -\tfrac12\,\theta^\top \underbrace{\big(\Lambda + S^{-1}\big)}_{P}\,\theta \;+\; \theta^\top \underbrace{\big(b + S^{-1}m\big)}_{h} \;+\; \text{const}.
\tag{2.1}
$$

Everything absorbed into `const` is genuinely free of $\theta$, which is what
licenses discarding it: it contributes a multiplicative constant to $q(\theta)$
that normalization will fix regardless of its value. This is the reason the step
never needs $y_t^\top\Sigma_i^{-1}y_t$ evaluated.

*Numerical check.* For a random instance ($L=9$, $K=4$, $d=3$, $T=200$, random
SPD $\Sigma_i$ and $S$, random selection matrices), evaluating the raw sum (A)+(B)
and the collected form (2.1) at 50 random $\theta$ and taking the difference: the
difference is constant across $\theta$ to $2.7\times10^{-12}$. A constant
difference is exactly what "equal up to a $\theta$-free const" asserts, and a
$\theta$-*dependent* residual is what a dropped factor of $\tfrac12$ or a missing
cross term would produce.

---

## 3. The identity itself

### 3.1 Scalar warm-up

For scalars $p>0$ and $h$, the schoolbook completion is

$$
-\tfrac12 p\theta^2 + h\theta
= -\tfrac{p}{2}\Big(\theta^2 - \tfrac{2h}{p}\theta\Big)
= -\tfrac{p}{2}\Big(\theta - \tfrac{h}{p}\Big)^2 + \tfrac{h^2}{2p},
$$

where the middle equality adds and subtracts $(h/p)^2$ inside the bracket. So the
quadratic peaks at $\theta = h/p$ with curvature $-p$. Exponentiating,
$\exp(-\tfrac{p}{2}(\theta - h/p)^2)$ is an unnormalized $\mathcal N(h/p,\, 1/p)$.
The matrix case is this and nothing more; $p^{-1}$ becomes $P^{-1}$ and the
squaring becomes a quadratic form.

### 3.2 The matrix identity

> **Lemma 1 (completing the square, matrix form).** Let $P \in \mathbb R^{L\times L}$
> be **symmetric** and **invertible**, and let $h \in \mathbb R^{L}$. Then for
> every $\theta \in \mathbb R^{L}$,
> $$
> -\tfrac12\,\theta^\top P\,\theta + \theta^\top h
> \;=\; -\tfrac12\,\big(\theta - P^{-1}h\big)^\top P \big(\theta - P^{-1}h\big) \;+\; \tfrac12\, h^\top P^{-1} h .
> $$

*Proof.* Work backwards from the right-hand side, which is the direction that
requires no guessing. Write $\mu = P^{-1}h$ and expand:

$$
-\tfrac12(\theta-\mu)^\top P(\theta-\mu)
= -\tfrac12\Big[\theta^\top P\theta - \theta^\top P\mu - \mu^\top P\theta + \mu^\top P\mu\Big].
$$

The two middle terms are equal: $\mu^\top P\theta$ is $1\times1$, so it equals its
transpose $\theta^\top P^\top \mu$, and $P^\top = P$ by the symmetry hypothesis,
giving $\theta^\top P\mu$. Hence

$$
-\tfrac12(\theta-\mu)^\top P(\theta-\mu) = -\tfrac12\theta^\top P\theta + \theta^\top P\mu - \tfrac12\mu^\top P\mu .
$$

Now substitute $\mu = P^{-1}h$. In the linear term, $P\mu = PP^{-1}h = h$, so it
becomes $\theta^\top h$. In the constant, $\mu^\top P\mu = h^\top P^{-\top} P P^{-1} h = h^\top P^{-1}h$,
using $P^{-\top} = P^{-1}$ (the inverse of a symmetric matrix is symmetric).
Therefore

$$
-\tfrac12(\theta-\mu)^\top P(\theta-\mu) = -\tfrac12\theta^\top P\theta + \theta^\top h - \tfrac12 h^\top P^{-1}h ,
$$

and moving the last term to the other side gives the claim. $\blacksquare$

**Where each hypothesis is used.** Symmetry of $P$ is what merges the two cross
terms; without it the identity fails and only the symmetric part
$\tfrac12(P+P^\top)$ is identifiable from the quadratic form. Invertibility is
what lets $\mu$ exist at all. Both are checked in §4.

Applying Lemma 1 to (2.1) with $P = \Lambda + S^{-1}$ and $h = b + S^{-1}m$:

$$
\log q(\theta) = -\tfrac12\big(\theta - P^{-1}h\big)^\top P\big(\theta - P^{-1}h\big) \;+\; \underbrace{\tfrac12 h^\top P^{-1}h + \text{const}}_{\text{still }\theta\text{-free}} .
\tag{3.1}
$$

*Numerical check.* Both sides of (3.1) evaluated at 50 random $\theta$ in the
same random instance as §2 agree to $2.7\times10^{-12}$ — here pointwise, not just
up to a constant, since the lemma is an exact equality.

---

## 4. Checking the hypotheses: is $P$ symmetric and invertible?

This is the part most treatments skip, and in this model it is *not* automatic —
$\Lambda$ is genuinely singular in realistic instances.

**Symmetry.** Each $M_i^\top\Sigma_i^{-1}M_i$ is symmetric, since
$(M_i^\top\Sigma_i^{-1}M_i)^\top = M_i^\top\Sigma_i^{-\top}M_i = M_i^\top\Sigma_i^{-1}M_i$.
A nonnegative combination of symmetric matrices is symmetric, so $\Lambda$ is;
$S^{-1}$ is symmetric because $S$ is; a sum of symmetric matrices is symmetric.
So $P$ is symmetric. ✓

**$\Lambda$ is positive semidefinite but generally *not* definite.** For any
$v\in\mathbb R^L$,

$$
v^\top \Lambda v = \sum_i N_i\, v^\top M_i^\top \Sigma_i^{-1} M_i v = \sum_i N_i\, (M_i v)^\top \Sigma_i^{-1} (M_i v) \;\ge\; 0,
$$

because each $N_i \ge 0$ (a sum of responsibilities) and each $\Sigma_i^{-1}$ is
positive definite. Moreover, since every summand is individually nonnegative, the
total vanishes **if and only if** every summand does; and $N_i (M_iv)^\top\Sigma_i^{-1}(M_iv) = 0$
with $N_i > 0$ forces $M_i v = 0$, because $\Sigma_i^{-1}$ being positive definite
means the quadratic form vanishes only at the zero vector. Hence

$$
v^\top\Lambda v = 0 \iff M_i v = 0 \ \text{ for every } i \text{ with } N_i > 0 .
$$

So $\Lambda$ has a nontrivial null space exactly when some level is selected by no
visited state. Two ways that happens in practice:

- a level appears in `stateindex` for no state at all, so no $M_i$ ever selects it;
- a state $i$ is visited with $N_i \approx 0$, and it was the only state selecting
  that level.

In a random $L=9$, $K=4$ instance with randomly drawn selection matrices, only
6 of the 9 levels were touched by any state, and $\operatorname{rank}(\Lambda) = 6$
with smallest eigenvalue $-5.6\times10^{-14}$ (numerically zero). **$\Lambda$ alone
is not invertible, and $\Lambda^{-1}$ must never appear in the implementation.**

Write $\mathcal U$ for the untouched level indices and $\mathcal T$ for the rest.
The characterization above says precisely that $\Lambda_{\mathcal U,:} = 0$
(verified numerically: the untouched rows of $\Lambda$ are exactly zero).

**$P$ is positive definite, hence invertible.** For $v \ne 0$,

$$
v^\top P v = \underbrace{v^\top \Lambda v}_{\ge\, 0} + \underbrace{v^\top S^{-1} v}_{>\,0} \;>\; 0 .
$$

The strict inequality on the second term is the definition of $S^{-1}$ being
positive definite, which holds because $S$ is. A positive definite matrix has all
eigenvalues strictly positive, so its determinant is nonzero and it is invertible.
✓ In the instance above, $\lambda_{\min}(P) = 1.13 > 0$ despite $\Lambda$ having a
three-dimensional null space.

> **The prior is doing structural work, not just statistical work.** $S^{-1}$ is
> what makes the linear system solvable at all. What survives exactly on the
> untouched levels is a statement about the **precision**, not the covariance:
> since $\Lambda_{\mathcal U,:} = 0$,
> $$
> P_{\mathcal U,:} = (S^{-1})_{\mathcal U,:} \qquad\text{exactly}
> $$
> (verified numerically: discrepancy $0.0$). The same argument gives
> $b_{\mathcal U} = 0$: for an untouched level, column $\mathcal U$ of every $M_i$
> is zero, so that coordinate of $M_i^\top\Sigma_i^{-1}r_i$ vanishes for every $i$.
> Hence $h_{\mathcal U} = (S^{-1}m)_{\mathcal U}$, and $P\bar\theta = h$ restricted
> to those rows reads
> $$
> (S^{-1})_{\mathcal U,:}\,\bar\theta = (S^{-1})_{\mathcal U,:}\,m .
> $$
>
> These two facts say the **conditional** law of the untouched levels given the
> touched ones is exactly what the prior said it was — both its covariance and its
> mean function. For a Gaussian with precision $P$ and mean $\nu$, the standard
> conditioning formulas are
> $\operatorname{Cov}[\theta_{\mathcal U}\mid\theta_{\mathcal T}] = (P_{\mathcal{UU}})^{-1}$
> and
> $\mathbb E[\theta_{\mathcal U}\mid\theta_{\mathcal T}] = \nu_{\mathcal U} - (P_{\mathcal{UU}})^{-1}P_{\mathcal{UT}}(\theta_{\mathcal T}-\nu_{\mathcal T})$
> (Bishop, *PRML* §2.3.1, eqs. 2.73–2.75). The first matches because
> $P_{\mathcal{UU}} = (S^{-1})_{\mathcal{UU}}$; the second matches because the
> displayed identity makes the posterior's affine mean function agree with the
> prior's at every $\theta_{\mathcal T}$. Verified numerically at several
> $\theta_{\mathcal T}$: conditional covariances agree to $0.0$ and conditional
> means to $4.4\times10^{-16}$.
>
> It does **not** say the *marginal* is unchanged — see §6, where the naive version
> of that claim is refuted.
>
> This is also why `hmm-scratch.py:1467` can use `np.linalg.solve` on `coeff_n`
> where other paths in the same file must use `lstsq` on a bare `coeff` (§8).

---

## 5. From "log-density is a quadratic" to "the density is Gaussian"

Lemma 1 is pure algebra; it says nothing about probability yet. The remaining
move — the one that actually justifies writing $q(\theta) = \mathcal N(\bar\theta, V)$ —
is this.

Exponentiate (3.1). The $\theta$-free bracket becomes a positive multiplicative
constant $C$:

$$
q(\theta) = C \exp\!\Big(-\tfrac12\big(\theta - \mu\big)^\top P\big(\theta - \mu\big)\Big),
\qquad \mu = P^{-1}h .
\tag{5.1}
$$

Rather than argue by proportionality, compute the normalizer outright.

> **Lemma 2 (Gaussian integral, matrix form).** If $P \in \mathbb R^{L\times L}$ is
> symmetric positive definite, then
> $$
> \int_{\mathbb R^L} \exp\!\Big(-\tfrac12 (\theta-\mu)^\top P (\theta-\mu)\Big)\,d\theta \;=\; (2\pi)^{L/2}\,\big|P\big|^{-1/2} \;<\; \infty .
> $$

*Proof.* By the **spectral theorem for real symmetric matrices**, $P = Q\,D\,Q^\top$
with $Q$ orthogonal and $D = \operatorname{diag}(\lambda_1,\dots,\lambda_L)$; the
hypothesis of the theorem — $P$ symmetric — holds by §4. Positive definiteness
gives $\lambda_k > 0$ for every $k$. Substitute $w = Q^\top(\theta-\mu)$. This is a
**change of variables** whose Jacobian determinant is $|\det Q^\top| = 1$ (orthogonal
matrices have determinant $\pm1$), so $d\theta = dw$, and

$$
(\theta-\mu)^\top P(\theta-\mu) = w^\top D\, w = \sum_{k=1}^{L}\lambda_k w_k^2 .
$$

The integrand therefore factorizes across coordinates, and by **Tonelli's theorem**
(the integrand is nonnegative and measurable, so the iterated integral equals the
multiple integral regardless of finiteness) the integral is the product of $L$
one-dimensional Gaussian integrals:

$$
\prod_{k=1}^{L}\int_{\mathbb R} e^{-\lambda_k w_k^2/2}\,dw_k = \prod_{k=1}^{L}\sqrt{\frac{2\pi}{\lambda_k}} = (2\pi)^{L/2}\Big(\prod_k \lambda_k\Big)^{-1/2} = (2\pi)^{L/2}|P|^{-1/2},
$$

using $\int_{\mathbb R} e^{-a w^2/2}dw = \sqrt{2\pi/a}$ for $a>0$ and
$|P| = \prod_k\lambda_k$. Each factor is finite precisely because $\lambda_k > 0$;
if any $\lambda_k$ were $0$ that factor would be $\int dw_k = \infty$, and if any
were negative it would diverge too. So positive definiteness is exactly the
condition for convergence. $\blacksquare$

This is why §4 mattered: with $\Lambda$ alone in place of $P$, (5.1) is flat along
a three-dimensional null space and **no normalization exists at all**. The prior is
what makes $q$ a probability distribution.

Since $q$ integrates to $1$ and (5.1) holds, Lemma 2 pins the constant to
$C = (2\pi)^{-L/2}|P|^{1/2}$. This is legitimate — rather than circular — only
because §2 deliberately left the additive `const` unevaluated: $C$ is
$\exp(\tfrac12 h^\top P^{-1}h + \texttt{const})$, an unknown positive number, and
normalization is what determines it. Substituting it back into (5.1) and setting
$V = P^{-1}$ — which exists by §4, and is symmetric positive definite because the
inverse of a symmetric positive definite matrix is — gives

$$
q(\theta) = (2\pi)^{-L/2}|V|^{-1/2}\exp\!\Big(-\tfrac12(\theta-\mu)^\top V^{-1}(\theta-\mu)\Big),
$$

using $|P|^{1/2} = |V|^{-1/2}$. That is the definition of the multivariate normal
density with mean $\mu$ and covariance $V$. Therefore

$$
\boxed{\;q(\theta) = \mathcal N\big(\theta \mid \bar\theta,\, V\big),\qquad V = P^{-1} = (\Lambda + S^{-1})^{-1},\qquad \bar\theta = P^{-1}h = V(b + S^{-1}m).\;}
$$

**The named bridge.** The general statement is that the Gaussian family is an
**exponential family**, and (2.1) writes it in its **canonical (natural, information)
parameterization**: with sufficient statistic $(\theta,\, \theta\theta^\top)$, the
natural parameters are $\eta_1 = h$ and $\eta_2 = -\tfrac12 P$. Completing the
square converts these to the **mean–covariance parameterization** $(\bar\theta, V) = (P^{-1}h,\, P^{-1})$:

$$
(h, P) \;\longleftrightarrow\; (\bar\theta, V) = (P^{-1}h,\; P^{-1}),
\qquad\text{inverse}\qquad (h,P) = (V^{-1}\bar\theta,\; V^{-1}).
$$

One precision worth keeping: the exponential family's *moment* (mean) parameter is
$\mathbb E[(\theta, \theta\theta^\top)] = (\bar\theta,\; V + \bar\theta\bar\theta^\top)$,
which is a bijective reparameterization of $(\bar\theta, V)$ but not literally equal
to it. The conversion performed here is canonical $\to$ mean–covariance.

Recognizing it by name is what makes it reusable: any time a log-density turns out
to be a concave quadratic in the unknown, its mean and covariance can be read off
the linear and quadratic coefficients without integrating **again** — Lemma 2 does
the integral once, for the whole family, and every later instance just cites it. The same
conversion appears as the information-filter form of the Kalman measurement update
(Anderson & Moore, *Optimal Filtering*, §6.3) and as the Gaussian–Gaussian
conjugate posterior (Bishop, *PRML*, §2.3.3, eq. 2.116).

**Why $\bar\theta$ is simultaneously the mean and the mode** — the fact §3.1 of
the source document leans on. The mode is where the gradient of (2.1) vanishes:
$\nabla_\theta \log q = -P\theta + h = 0 \iff \theta = P^{-1}h$, and this is a
maximum because the Hessian $-P$ is negative definite. The mean is $P^{-1}h$ by
the identification just made. They coincide because the Gaussian is symmetric
about its center. **This coincidence is what makes the E-step for $\theta$ look
like a penalized M-step**, and it is a property of the Gaussian, not of EM.

*Numerical check.* Drawing $4\times10^5$ samples from $\mathcal N(\bar\theta, V)$ in
the random instance recovers the sample mean to $1.5\times10^{-3}$ and the sample
covariance to $3.7\times10^{-3}$ — consistent with Monte Carlo error at that
sample size, confirming $(\bar\theta, V)$ really are the moments of (5.1) rather
than merely its mode and curvature.

---

## 6. Reading the answer: three equivalent forms

The boxed result is the form to implement, but it is opaque about what the update
*does*. Three algebraic rearrangements, each exact.

**(a) Precision-weighted average (needs $\Lambda$ invertible — see §4).** If
$\Lambda$ happens to be nonsingular, define the trace's own maximum-likelihood
level estimate $\hat\theta = \Lambda^{-1}b$. Then

$$
\bar\theta = V\big(b + S^{-1}m\big) = V\big(\Lambda\hat\theta + S^{-1}m\big) = \big(\Lambda + S^{-1}\big)^{-1}\big(\Lambda\,\hat\theta + S^{-1}\,m\big),
$$

substituting $b = \Lambda\hat\theta$. The posterior mean is the **precision-weighted
average of the trace's own estimate and the population mean** — the two precisions
$\Lambda$ and $S^{-1}$ add, and each candidate is weighted by its own precision.
This is the sentence to remember. Verified numerically on a full-rank instance.

**(b) Shrinkage form (always valid).** Using $VS^{-1} = V(V^{-1} - \Lambda) = I - V\Lambda$:

$$
\bar\theta = Vb + VS^{-1}m = Vb + m - V\Lambda m = m + V\big(b - \Lambda m\big).
$$

The posterior mean is the **population mean plus a correction proportional to how
far the data's evidence departs from it**. The quantity $b - \Lambda m$ is the
gradient of the expected log-likelihood evaluated at $\theta = m$: an innovation.
Unlike (a) this requires no inverse of $\Lambda$, so it stays meaningful when
levels are unidentified. Verified numerically.

> **What happens on the untouched levels $\mathcal U$ — and the claim to avoid.**
> It is tempting to conclude from $\Lambda_{\mathcal U,:} = 0$ that
> $\bar\theta_{\mathcal U} = m_{\mathcal U}$ and $V_{\mathcal{UU}} = S_{\mathcal{UU}}$
> — that the untouched levels simply keep their prior. **Both are false**, because
> a matrix inverse does not act blockwise: $\Lambda_{\mathcal U,:}=0$ constrains
> $P$, and $V = P^{-1}$ mixes all coordinates.
>
> In the running instance ($\mathcal U = \{3,4,5\}$):
> $\max|V_{\mathcal{UU}} - S_{\mathcal{UU}}| = 0.223$, a **20%** relative error, and
> $\max|\bar\theta_{\mathcal U} - m_{\mathcal U}| = 0.321$ against
> $\|m_{\mathcal U}\| = 1.539$ — the untouched levels move substantially.
>
> **The error is not bounded by any such figure; its supremum is $100\%$.** From
> the Woodbury form below, $V_{\mathcal{UU}}$ is squeezed between the conditional
> prior covariance $\big((S^{-1})_{\mathcal{UU}}\big)^{-1}$ and the marginal prior
> covariance $S_{\mathcal{UU}}$, and the lower end tends to $0$ as the
> $\mathcal U$–$\mathcal T$ correlation tends to $1$. With an equicorrelated
> $S = (1-\rho)I + \rho\,\mathbf 1\mathbf 1^\top$ the relative error runs
> $88.3\%,\ 98.8\%,\ 99.9\%,\ 100.0\%$ at $\rho = 0.9,\ 0.99,\ 0.999,\ 0.9999$.
> The $20\%$ above is an artifact of the weakly-correlated $S$ that the check
> script happens to draw, not a typical magnitude.
>
> The levels move because $S$ correlates them with the touched levels: learning
> about $\theta_{\mathcal T}$ updates beliefs about $\theta_{\mathcal U}$ through
> $S_{\mathcal{UT}}$. That is **borrowing strength**, the thing the hierarchical
> model exists to do, not a defect. And $S_{\mathcal{UT}}$ is exactly the culprit:
> since $G_{:,\mathcal U} = 0$, the $\mathcal{UU}$ block of the Woodbury correction
> below is
> $$
> \big(S G^\top (I + GSG^\top)^{-1} G S\big)_{\mathcal{UU}} = S_{\mathcal{UT}}\,G_{\mathcal T}^\top\,(I+GSG^\top)^{-1}\,G_{\mathcal T}\,S_{\mathcal{TU}},
> $$
> which vanishes **if and only if** $S_{\mathcal{UT}} = 0$. So the naive claim is
> right precisely under the extra hypothesis that $S$ is block-diagonal across the
> touched/untouched split — proved here, and confirmed numerically (both
> discrepancies drop to $10^{-16}$ when $S_{\mathcal{UT}}$ is zeroed).
>
> The correct unconditional statement is the one from §4, and it needs *both*
> invariants established there — $P_{\mathcal U,:} = (S^{-1})_{\mathcal U,:}$ **and**
> $h_{\mathcal U} = (S^{-1}m)_{\mathcal U}$: the *conditional* law
> $\theta_{\mathcal U}\mid\theta_{\mathcal T}$ is unchanged from the prior. The
> *marginal* is not. Precision rows alone would fix only the conditional
> covariance, not its mean.

**(c) Woodbury / dual form.** Factor $\Lambda = G^\top G$ by stacking the $K$
blocks $\sqrt{N_i}\,\Sigma_i^{-1/2}M_i$ into $G \in \mathbb R^{Kd\times L}$. This
explicit construction is available because $N_i \ge 0$ (so $\sqrt{N_i}$ is real)
and $\Sigma_i \succ 0$ (so the symmetric square root $\Sigma_i^{-1/2}$ exists, by
the spectral theorem); expanding $G^\top G$ block by block reproduces
$\sum_i N_i M_i^\top\Sigma_i^{-1}M_i$. Verified numerically to $6.8\times10^{-13}$.
The **Woodbury matrix identity** then gives

$$
V = \big(S^{-1} + G^\top G\big)^{-1} = S - S G^\top\big(I_{Kd} + G S G^\top\big)^{-1} G S .
$$

Its hypotheses — $S$ invertible and $I_{Kd} + GSG^\top$ invertible — hold: $S$ is
positive definite by assumption, and $I + GSG^\top \succeq I$ is positive definite
since $GSG^\top \succeq 0$. Verified numerically to $7.9\times10^{-14}$.

The cost comparison is not a straight swap. Evaluating $V$ from the left-hand side
needs **two** $L\times L$ inversions — $S^{-1}$ to form $P$, then $P^{-1}$ — whereas
the right-hand side needs neither, only one $Kd\times Kd$ inversion plus the
$O(KdL^2)$ cost of forming $GS$. Avoiding $S^{-1}$ altogether is the genuine
numerical argument for it, and it matters when $S$ is ill-conditioned. As a raw
flop count it wins only when $Kd$ is comfortably below $L$; here $Kd = 12 > L = 9$,
so its value in this model is conceptual rather than computational.

Written this way, $V$ is $S$ minus a positive semidefinite matrix, which proves the
next claim in one line.

**The posterior is never more uncertain than the prior: $V \preceq S$.** Two routes.
From the Woodbury form, set $X = GS$ and $W = (I+GSG^\top)^{-1}$. Then
$SG^\top = (GS)^\top = X^\top$, using $S^\top = S$, so
$S - V = SG^\top W GS = X^\top W X$. Since $W$ is positive definite, $X^\top W X$
is positive semidefinite for any $X$, giving $S - V \succeq 0$.

Alternatively and without Woodbury: $V^{-1} = S^{-1} + \Lambda \succeq S^{-1}$
because $\Lambda \succeq 0$ (§4), and matrix inversion is **operator antitone** on
positive definite matrices — if $0 \prec A \preceq B$ then $B^{-1} \preceq A^{-1}$.
The hypotheses hold with $A = S^{-1}$ and $B = V^{-1}$: both are positive definite
(§4), and $A \preceq B$ is the displayed inequality. Confirmed numerically:
$\lambda_{\min}(S - V) = -2.3\times10^{-16}$, i.e. zero to machine precision from
below, with $\operatorname{rank}(S-V) = 6$ matching $\operatorname{rank}(\Lambda)$.
Observing a trace can only sharpen beliefs about its levels.

---

## 7. Checks

**Shapes and units.** Every term in a sum must have the same shape and units.
Write $[\theta]$ for the units of a level and $[y]$ for those of an observation.
**These are the same** — $M_i$ is a dimensionless selector, so $u_i = M_i\theta$ is
measured in the same units as the $y_t$ it is compared against. The table uses
$[y]$ throughout on that stipulation.

| Quantity | Shape | Units |
|---|---|---|
| $M_i^\top\Sigma_i^{-1}M_i$ | $L\times L$ | $[y]^{-2}$ |
| $N_i$ | scalar | dimensionless (a count) |
| $\Lambda$ | $L\times L$ | $[y]^{-2}$ — a **precision** |
| $S^{-1}$ | $L\times L$ | $[\theta]^{-2} = [y]^{-2}$ — a **precision** ✓ matches $\Lambda$ |
| $M_i^\top\Sigma_i^{-1}r_i$ | $L$ | $[y]^{-1}$ (precision $\times$ level) |
| $b$, $S^{-1}m$ | $L$ | $[y]^{-1}$ ✓ they add |
| $V = P^{-1}$ | $L\times L$ | $[y]^{2}$ — a **covariance** ✓ |
| $\bar\theta = Vh$ | $L$ | $[y]^{2}\cdot[y]^{-1} = [y] = [\theta]$ ✓ |

The load-bearing check is that $\Sigma_i^{-1}$, not $\Sigma_i$, sits inside
$\Lambda$ and $b$. Substituting $\Sigma_i$ makes $\Lambda$ a covariance being added
to a precision $S^{-1}$ — dimensionally incoherent, and silently wrong rather than
crashing. The hierarchical function inverts `sigma` first and is correct. One other
solve site in the file always uses `sigma` itself (lines 358–379), and a second does
so under one flag setting (line 867); §8 gives the details.

**Limiting case: $S \to \infty$ (vague prior).** $S^{-1}\to 0$, so
$V \to \Lambda^{-1}$ and $\bar\theta \to \Lambda^{-1}b = \hat\theta$: the
per-trace MLE, no shrinkage, each trace fit independently. Verified numerically
($\|\bar\theta - \hat\theta\| = 9.3\times10^{-15}$ at $S\cdot10^{12}$). This limit
exists only when $\Lambda$ is invertible — precisely the §4 caveat, seen from the
other side.

**Limiting case: $S \to 0$ (no population heterogeneity).** $S^{-1}$ dominates, so
$\bar\theta \to m$ and $V \to 0$: total shrinkage, complete pooling, all traces
share one level vector. Verified numerically
($\|\bar\theta - m\| = 1.8\times10^{-9}$ at $S\cdot10^{-12}$). This is the
degenerate fixed point that §5.1 of the source document proves the plug-in $S$
update converges to **in the scalar case** ($L=1$); that argument is not carried
out for $L>1$ there.

**Limiting case: $N_i \to \infty$ (infinite data).** Scale $\Lambda \mapsto c\Lambda$
and let $c\to\infty$. On the touched levels the prior is overwhelmed, as consistency
requires: $V_{\mathcal{TT}} \to 0$ and
$\bar\theta_{\mathcal T} \to (\Lambda_{\mathcal{TT}})^{-1}b_{\mathcal T}$ — note this
is *not* $\hat\theta = \Lambda^{-1}b$, which does not exist here, $\Lambda$ being
singular. Verified to $7.0\times10^{-13}$ at $c = 10^{10}$.

On the untouched levels the limit is *not* $S_{\mathcal{UU}}$ but the **conditional**
prior covariance $\big((S^{-1})_{\mathcal{UU}}\big)^{-1}$. One line: since
$\ker\Lambda = \operatorname{span}\{e_k : k\in\mathcal U\}$ (§4), the $c\to\infty$
limit of $(c\Lambda + S^{-1})^{-1}$ is supported on that kernel and equals
$E_{\mathcal U}\big((S^{-1})_{\mathcal{UU}}\big)^{-1}E_{\mathcal U}^\top$, where
$E_{\mathcal U}$ selects the $\mathcal U$ coordinates — the divergent directions are
annihilated and only the $S^{-1}$ block on the kernel survives to be inverted.
Verified to $2.5\times10^{-12}$ at $c = 10^{8}$, while
$|V_{\mathcal{UU}} - S_{\mathcal{UU}}|$ stays pinned at $0.223$. This is the §6
blockquote again: infinite data on $\mathcal T$ pins $\theta_{\mathcal T}$, and what
remains on $\mathcal U$ is the prior *conditioned* on that, not the prior marginal.

**Recovery of the textbook case.** Take $L = 1$, a single state observed $n_0$
times with per-observation precision $p_0$, so $\Lambda = p_0 n_0$ and
$b = p_0 n_0\hat\theta$. Then

$$
V = \frac{1}{p_0 n_0 + 1/S},
\qquad
\bar\theta = \frac{p_0 n_0\,\hat\theta + m/S}{p_0 n_0 + 1/S},
$$

the Gaussian-mean-with-known-variance posterior under a Gaussian prior — Bishop,
*PRML* §2.3.6, eqs. 2.141–2.142; Gelman et al., *BDA3* §2.5. Concretely with
$p_0 = 4$, $n_0 = 10$, $S = 0.25$, $m = 1.0$, $\hat\theta = 1.6$: $V = 0.022727$
and $\bar\theta = 1.545455$, with weight $0.909$ on the data and $0.091$ on the
prior. The estimate is pulled $9.1\%$ of the way from $1.6$ back toward $1.0$.

**Degenerate case: a state with no data.** If $N_i = 0$ for state $i$, that state
contributes nothing to $\Lambda$ or $b$. Its levels are then in $\mathcal U$, and
§4 gives the exact consequence: both the posterior precision rows *and* the
posterior potential block for those levels equal the prior's, so their conditional
law given the touched levels is the prior conditional. Their marginal mean and
variance still move, by the amounts quantified in §6. Either way nothing blows up —
the reason this formulation degrades gracefully where a bare least-squares solve
would not.

---

## 8. Map to the code

| Result | This doc | `hmm-scratch.py` | Status |
|---|---|---|---|
| $\Lambda = \sum_i N_i M_i^\top\Sigma_i^{-1}M_i$ | §2 | `a`, assembled at [1458–1463](hmm-scratch.py#L1458-L1463) | correct |
| $b = \sum_i M_i^\top\Sigma_i^{-1}r_i$ | §2 | `b`, same loop | correct |
| $P = \Lambda + S^{-1}$ | §2, (2.1) | [1466](hmm-scratch.py#L1466) `coeff_n = a + Sinv` | correct |
| $\bar\theta = P^{-1}(b + S^{-1}m)$ | §3, §5 | [1467](hmm-scratch.py#L1467) `np.linalg.solve(coeff_n, b + Sinv @ m)` | correct |
| $V = P^{-1}$ | §5 | [1468](hmm-scratch.py#L1468) `np.linalg.inv(coeff_n)` | correct |
| $P$ invertible via $S^{-1}$, not $\Lambda$ | §4 | `solve` on `coeff_n`, never on `a` | correct — and load-bearing |

Note the hierarchical function builds `a` directly in a loop over states rather
than by the `a11…a33` block assembly used elsewhere in the file; the comment at
line 1456–1457 records that the two are equivalent, which they are, since the
$(\rho,c)$ block of $M_i^\top\Sigma_i^{-1}M_i$ is $(M_i^\rho)^\top\Sigma_i^{-1}M_i^c$.

**Where $\Lambda$'s singularity shows up as a code decision.** Four sites solve a
system of the form $\Lambda\theta = b$ with a **bare** $\Lambda$ and must therefore
use `lstsq`: [342](hmm-scratch.py#L342), [379](hmm-scratch.py#L379),
[634](hmm-scratch.py#L634), [867](hmm-scratch.py#L867). The hierarchical
path adds $S^{-1}$ first and can use `solve`. The correspondence runs one way only:
without a prior precision `lstsq` is *required*, but its presence does not force
`solve` — `semi_pooled_expectation_maximization_multi_trace` adds a population
precision at [1138](hmm-scratch.py#L1138) (`coeff += MVGenerator.precision`)
and a potential at 1142, making `coeff` nonsingular by exactly the §4 argument, yet
still calls `lstsq` at [1145](hmm-scratch.py#L1145). That is harmless, since
`lstsq` and `solve` agree on a nonsingular system.

What is *not* safe is dropping `Sinv` from `coeff_n` while keeping `lstsq`: that
silently returns the minimum-norm solution of a singular system in place of a
posterior mean. Dropping it while keeping `solve` fails loudly instead —
`np.linalg.solve` on the singular $\Lambda$ raises `LinAlgError: Singular matrix`
(verified).

**A units issue at two of those sites.** §7 warns that $\Sigma_i^{-1}$, not
$\Sigma_i$, belongs inside $\Lambda$ and $b$. At
[373–379](hmm-scratch.py#L373-L379) the middle matrix is `sigma[:,:,i]`
itself, not its inverse. At [867](hmm-scratch.py#L867) the middle matrix is
`weight[:,:,i]`, which is `inv(sigma[:,:,i])` **only when `mean_solve == "precision"`**
and is `sigma[:,:,i]` otherwise ([827–833](hmm-scratch.py#L827-L833)). Under
the covariance branch these solve the dimensionally incoherent variant. Whether
that is intended (a robustness heuristic) or a latent bug is worth deciding
separately; it is outside this step's scope.

**Drift in the source document, noticed while reading the code.** §10 of
[`eb-hmm-derivation.md`](eb-hmm-derivation.md) marks three results as missing.
All three are now implemented:

- emission correction $-\tfrac12\operatorname{tr}(\Sigma_i^{-1}M_iVM_i^\top)$ — [1382–1397](hmm-scratch.py#L1382-L1397), gated on `apply_v_corrections` at 1390;
- $\Sigma$ correction $N_i M_i V M_i^\top$ — [1480–1481](hmm-scratch.py#L1480-L1481), gated at 1480;
- free energy $\mathcal F = \sum_n\log\tilde Z^{(n)} - \sum_n\mathrm{KL}$ — accumulated at 1402, 1414, 1418 and 1361–1367, combined at [1489](hmm-scratch.py#L1489) (`F = logZ_total - kl_total`).

The flag `apply_v_corrections` (default `True`, line 1246) gates only the first two
of these, plus the $V^{(n)}$ term in the $S$ update at 1517. The free-energy
machinery is unconditional — the flag changes the *value* of $\log\tilde Z$ through
`log_corr`, but not whether $\mathcal F$ is computed.

Every line number in that §10 table is also stale, by 84 to 151 lines
(1383 → 1467, 1413–1417 → 1513–1520, 1338–1339 → 1489), and its closing sentence
about "three gaps" is obsolete. This is outside the scope of the question asked; I
have not modified either file beyond adding this document, and the correction is
worth a separate pass.

---

## 9. Summary

The step is one algebraic identity (Lemma 1, §3.2) plus one normalization argument
(Lemma 2, §5), with one hypothesis that genuinely needs checking in this model
(§4: $P$ is invertible only because $S^{-1}$ is there; $\Lambda$ alone is routinely
singular, and without the prior the density has no normalizer at all).

Zero approximations. The mean-field factorization boxed in §2 of the source
document remains the only approximation in the whole derivation; this step
introduces none.

The reusable content: **when a log-density is a concave quadratic
$-\tfrac12\theta^\top P\theta + \theta^\top h$, the distribution is
$\mathcal N(P^{-1}h,\, P^{-1})$ — no further integration required.** That is the
canonical-to-mean-covariance change of coordinates in the Gaussian exponential
family, and it is why the mean and the mode coincide here, which is the entire
resolution of the motivating question in the source document.

The trap worth remembering: **matrix inverses do not act blockwise.** Statements
that survive exactly on the levels the data never touch are statements about the
precision $P$ and the potential $h$, and hence about *conditional* laws — never
about the covariance $V$ or about marginals (§6).

---

## Reproducing the numbers

Every numerical value quoted above comes from
[`completing_the_square_checks.py`](completing_the_square_checks.py), which builds
one seeded instance (`seed = 0`, $L=9$, $K=4$, $d=3$, $T=200$) and prints each
check tagged by section:

```
python3 completing_the_square_checks.py
```

Sections 4 and 6 describe the same matrices, so the untouched set
$\mathcal U = \{3,4,5\}$ and the discrepancies reported for it are mutually
consistent. The one number *not* from that instance is the §7 scalar textbook case,
which is closed-form arithmetic on the stated inputs.

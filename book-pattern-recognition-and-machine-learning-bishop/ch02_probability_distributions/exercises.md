# PRML Chapter 2 — Probability Distributions

Source: Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 2, exercises 2.1–2.61.

## Status

Exercise list (fixed 02.11.2026, revised 04.11.2026). Test for every exercise, read in full: does
it make me do a step that later chapters or real ML work reuse? The book's marks only break ties.

**Solve (19):**
- §2.1–2.2, discrete: 2.1 (Bernoulli moments, entropy), 2.5 (beta normalisation), 2.6 (beta mean,
  variance, mode), 2.7 (posterior mean between prior mean and ML estimate: shrinkage), 2.8 (law of total expectation and variance), 2.11 (Dirichlet E[ln μ])
- §2.3, Gaussian: 2.17 (precision can be taken symmetric), 2.20 (positive definite ⇔ positive
  eigenvalues), 2.27 (sum of independent vectors), 2.30 (Bayes for Gaussians), 2.34 (ML covariance),
  2.38 (completing the square: posterior of the mean), 2.40 (multivariate posterior mean)
- §2.3.6–2.3.7: 2.41 (gamma normalisation), 2.43 (generalised Gaussian, the density behind the
  Minkowski loss of Ex. 1.27), 2.46 (Student's t as a mixture of Gaussians)
- §2.4–2.5: 2.56 (beta, gamma, von Mises as exponential family), 2.58 (moments from ln g(η)),
  2.60 (histogram density by maximum likelihood)

**Triage, plan only (15):** 2.3, 2.12, 2.13 (KL between multivariate Gaussians: the VAE term),
2.15 (multivariate Gaussian entropy), 2.19 (eigen-expansion of Σ: PCA), 2.21, 2.22, 2.24, 2.26
(Woodbury: Ch. 3, Gaussian processes), 2.31 (sum of Gaussians: Kalman filter, predictive
distributions), 2.36, 2.47, 2.57, 2.59, 2.61.

**Skipped by choice:** 2.14 (maximum entropy by calculus of variations, like 1.34); 2.45, 2.48–2.50
(Wishart, multivariate t: the univariate cases carry the idea); 2.51–2.55 (von Mises, periodic
data: niche); 2.2 (Bernoulli on {−1, 1}: re-parametrisation); 2.4 (binomial moments: N × Ex. 2.1
by Ex. 1.10); 2.10, 2.42 (Ex. 2.6 again for Dirichlet and gamma); 2.35 (Ex. 1.12 in matrix form);
2.39 (covered by 2.36); 2.44 (Gaussian-gamma posterior); 2.18, 2.23, 2.25, 2.29, 2.33, 2.37, 2.49
(more algebra on results the list already derives).

Gaps found, to practise:

- Changing the order of integration over a non-rectangular region (Ex. 2.5: 0 ≤ x ≤ t). Sketch
  the region, then read the limits the other way. Practice: the double-integral section of
  Демидович (problems on changing the order of integration).

Solved (date):

- 2.1 (02.11.2026)
- 2.8 (04.11.2026)

---

## Exercise 2.1 (\*) www

> Verify that the Bernoulli distribution (2.2) satisfies the following properties
> Σ over x = 0, 1 of p(x|μ) = 1 (2.257), E[x] = μ (2.258), var[x] = μ(1 − μ) (2.259).
> Show that the entropy H[x] of a Bernoulli distributed random binary variable x is given by
> H[x] = −μ ln μ − (1 − μ) ln(1 − μ) (2.260).

Context: §2.1, Bernoulli distribution (2.2) p(x|μ) = μ^x (1 − μ)^(1−x); entropy (1.93) in nats.

Handwritten original: [`ex-2.1-handwritten.png`](ex-2.1-handwritten.png).

### Solution

x takes two values, so every sum has two terms: p(0|μ) = 1 − μ, p(1|μ) = μ.

- Σ p(x|μ) = (1 − μ) + μ = 1.
- E[x] = 0 · (1 − μ) + 1 · μ = μ.
- var[x] = E[x²] − E[x]² (Ex. 1.5). x² = x for x ∈ {0, 1}, so E[x²] = μ and var[x] = μ − μ² = μ(1 − μ).
- H[x] = −Σ p ln p = −(1 − μ) ln(1 − μ) − μ ln μ. ∎

### Review notes (professor)

All his own, no hints. The normalisation line is stated rather than shown (the two terms
(1 − μ) + μ). The rest is complete.

In practice: a Bernoulli variable is a binary label; logistic regression outputs μ = p(C₁|x)
(§4.3). var = μ(1 − μ) is largest at μ = ½, and so is H (= ln 2): a 50/50 label is the most
uncertain. −ln of this likelihood, summed over data, is the binary cross-entropy loss.

## Exercise 2.8 (\*)

> Consider two variables x and y with joint distribution p(x, y). Prove the following two
> results
> E[x] = E_y[E_x[x|y]] (2.270)
> var[x] = E_y[var_x[x|y]] + var_y[E_x[x|y]]. (2.271)
> Here E_x[x|y] denotes the expectation of x under the conditional distribution p(x|y), with a
> similar notation for the conditional variance.

Context: §1.2.2, expectations with a subscript (1.36) and conditional expectation (1.37); sum
rule (1.10), product rule (1.11); var = E[·²] − E[·]² (1.39).

Handwritten original: [`ex-2.8-handwritten.png`](ex-2.8-handwritten.png) (begun 02.11, finished
04.11.2026).

### Solution

**Mean.** E_x[x|y] = ∫ x p(x|y) dx is a function of y only; E_y averages it with weight p(y):

E_y[E_x[x|y]] = ∫ ( ∫ x p(x|y) dx ) p(y) dy = ∫∫ x p(x, y) dx dy (p(y) moved inside, product rule)
= ∫ x ( ∫ p(x, y) dy ) dx = ∫ x p(x) dx = E[x] (sum rule). ∎

The same steps with x² in place of x give E_y[E_x[x²|y]] = E[x²].

**Variance.** Expand both terms with var = E[·²] − E[·]²:

- E_y[var_x[x|y]] = E_y[E_x[x²|y]] − E_y[(E_x[x|y])²]
- var_y[E_x[x|y]] = E_y[(E_x[x|y])²] − (E_y[E_x[x|y]])²

The middle terms cancel. What is left is E_y[E_x[x²|y]] − (E_y[E_x[x|y]])² = E[x²] − E[x]² =
var[x]. ∎

### Review notes (professor)

His own algebra, three corrections on the way:
1. The outer average was first written with y as the weight (∫ y ∫ x p(x|y) dx). The weight is
   p(y), with its own dy.
2. var_x[E_x[x|y]] instead of var_y. E_x[x|y] has no x left in it, so only its spread over y
   exists.
3. He wrote E_x[(x|y)²]: the square belongs on x, E_x[x²|y]. "x|y" only says which distribution
   the average uses.

He found the cancellation himself, and the last step (E_y[E_x[x²|y]] = ∫∫ x² p(x, y) = E[x²]).
Linearity of expectation (E[a − b] = E[a] − E[b]) was asked and confirmed.

In practice: with x the target and y the features, (2.270) says the average prediction of the
true regression function equals the average target. That is why a calibrated classifier's mean
predicted probability equals the class base rate. (2.271) splits total variance into noise within
groups and spread of the group means. That gives explained vs unexplained variance (R²), the
regression-tree split criterion, aleatoric vs epistemic uncertainty in ensembles and Bayesian
predictions, and the variance of a mixture (Ch. 9).

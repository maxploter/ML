# PRML Chapter 2 — Probability Distributions

Source: Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 2, exercises 2.1–2.61.

## Status

Exercise list (fixed 02.11.2026). Default: every (\*) plus the (\*\*) www ones, then cut by one
test: does it make me do a step that later chapters reuse?

**Solve (18):**
- §2.1–2.2, discrete: 2.1 (Bernoulli moments, entropy), 2.5 (beta normalisation), 2.6 (beta mean,
  variance, mode), 2.8 (law of total expectation and variance), 2.11 (Dirichlet E[ln μ])
- §2.3, Gaussian: 2.17 (precision can be taken symmetric), 2.20 (positive definite ⇔ positive
  eigenvalues), 2.27 (sum of independent vectors), 2.30 (Bayes for Gaussians), 2.34 (ML covariance),
  2.38 (completing the square: posterior of the mean), 2.40 (multivariate posterior mean)
- §2.3.6–2.3.7: 2.41 (gamma normalisation), 2.43 (generalised Gaussian, the density behind the
  Minkowski loss of Ex. 1.27), 2.46 (Student's t as a mixture of Gaussians)
- §2.4–2.5: 2.56 (beta, gamma, von Mises as exponential family), 2.58 (moments from ln g(η)),
  2.60 (histogram density by maximum likelihood)

**Triage, plan only (10):** 2.3, 2.12, 2.21, 2.22, 2.24, 2.36, 2.47, 2.57, 2.59, 2.61.

**Skipped by choice:** 2.14 (maximum entropy by calculus of variations, like 1.34); 2.45, 2.48–2.50
(Wishart, multivariate t: the univariate cases carry the idea); 2.51–2.55 (von Mises, periodic
data: niche); every other (\*\*) or (\*\*\*) without www.

Solved (date):

- 2.1 (02.11.2026)

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

# PRML Chapter 2 — Probability Distributions

Source: Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 2, exercises 2.1–2.61.

## Status

Exercise list (fixed 02.11.2026, revised 04.11.2026). Test for every exercise, read in full: does
it make me do a step that later chapters or real ML work reuse? The book's marks only break ties.

**Solve (20):**
- §2.1–2.2, discrete: 2.1 (Bernoulli moments, entropy), 2.5 (beta normalisation), 2.6 (beta mean,
  variance, mode), 2.7 (posterior mean between prior mean and ML estimate: shrinkage), 2.8 (law of total expectation and variance), 2.10 (Dirichlet mean, variance, covariance; added 14.11.2026 as Γ-recursion practice), 2.11 (Dirichlet E[ln μ])
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
by Ex. 1.10); 2.42 (Ex. 2.6 again for gamma); 2.35 (Ex. 1.12 in matrix form);
2.39 (covered by 2.36); 2.44 (Gaussian-gamma posterior); 2.18, 2.23, 2.25, 2.29, 2.33, 2.37, 2.49
(more algebra on results the list already derives).

Gaps found, to practise:

- Changing the order of integration over a non-rectangular region (Ex. 2.5: 0 ≤ x ≤ t). Sketch
  the region, then read the limits the other way. Practice: the double-integral section of
  Демидович (problems on changing the order of integration).
- Substitution in an inner integral while the outer variable is held fixed (Ex. 2.5: x = tμ, so
  dx = t dμ, not μ dt, and the limits move from x ∈ [0, t] to μ ∈ [0, 1]). He wrote μ dt twice
  (04.11 and 08.11). Practice: the same Демидович section, substitutions in iterated integrals;
  then redo 2.5 from a blank page.

- Gamma recursion direction: Γ(x + 1) = xΓ(x), so Γ(a + 2) = (a + 1)aΓ(a). Wrong in Ex. 2.6
  ((a − 1)aΓ(a)) and Ex. 2.7 ((x + 1)Γ(x)). Check: Γ(n + 1) = n!, so Γ(3) = 2 = 2·Γ(2).

Solved (date):

- 2.1 (02.11.2026)
- 2.5 (08.11.2026)
- 2.6 (08.11.2026)
- 2.7 (14.11.2026)
- 2.8 (04.11.2026)
- 2.10 (14.11.2026)

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

## Exercise 2.5 (\*\*) www

> In this exercise, we prove the normalization of the beta distribution, given by (2.13) by
> showing that the integral
> ∫ from 0 to 1 of μ^(a−1) (1 − μ)^(b−1) dμ = Γ(a)Γ(b) / Γ(a + b) (2.265)
> holds. To do this, use the definition (1.141) of the gamma function to obtain
> Γ(a)Γ(b) = ∫ from 0 to ∞ of exp(−x) x^(a−1) dx · ∫ from 0 to ∞ of exp(−y) y^(b−1) dy. (2.266)
> Use this expression to prove (2.265) as follows. First bring the integral over y inside the
> integrand of the integral over x, next make the change of variable t = y + x where x is fixed,
> then interchange the order of the x and t integrations, and finally make the change of variable
> x = tμ where t is fixed.

Context: §2.1.1, beta distribution (2.13); gamma function (1.141) Γ(x) = ∫ from 0 to ∞ of
u^(x−1) e^(−u) du.

Handwritten original: [`ex-2.5-handwritten.png`](ex-2.5-handwritten.png) (begun 04.11, finished
08.11.2026).

### Solution

1. Both integrals in (2.266) together: Γ(a)Γ(b) = ∫∫ exp(−(x + y)) x^(a−1) y^(b−1) dy dx, both from
   0 to ∞.
2. Inner integral, x fixed: t = y + x, dt = dy, y from 0 to ∞ becomes t from x to ∞:
   ∫ over x from 0 to ∞ of ∫ over t from x to ∞ of exp(−t) x^(a−1) (t − x)^(b−1) dt dx.
3. Swap the order. The region is 0 ≤ x ≤ t < ∞; read with t outside: t from 0 to ∞, x from 0 to t:
   ∫ over t from 0 to ∞ of ∫ over x from 0 to t of exp(−t) x^(a−1) (t − x)^(b−1) dx dt.
4. Inner integral, t fixed: x = tμ, dx = t dμ, x from 0 to t becomes μ from 0 to 1.
   (tμ)^(a−1) = t^(a−1) μ^(a−1) and (t − tμ)^(b−1) = t^(b−1) (1 − μ)^(b−1). The powers of t:
   t^(a−1) · t^(b−1) · t = t^(a+b−1). The μ integral has no t, so the double integral splits:
   Γ(a)Γ(b) = ∫ from 0 to ∞ of t^(a+b−1) e^(−t) dt · ∫ from 0 to 1 of μ^(a−1) (1 − μ)^(b−1) dμ
   = Γ(a + b) · ∫ from 0 to 1 of μ^(a−1) (1 − μ)^(b−1) dμ.
5. Divide by Γ(a + b): (2.265). ∎ So the factor Γ(a + b) / (Γ(a)Γ(b)) in (2.13) makes the beta
   density integrate to 1.

### Review notes (professor)

His own, over two sessions, with hints at each of the three moves. Steps 1–3 were done on 04.11;
the swap of order took the longest (he drew the region 0 ≤ x ≤ t himself). On 08.11 he twice wrote
dx = μ dt for x = tμ: the inner variable is μ, t is a constant there, so dx = t dμ, and the limits
must move with the variable (x ∈ [0, t] → μ ∈ [0, 1]). He then tried to split the t integral into
Γ(a) times something; one exp(−t) means one gamma, of the summed exponent. Final page complete.
Both calculus slips are logged in the gaps list above.

In practice: this is the normaliser of the beta prior, the conjugate prior for a Bernoulli rate
(§2.1.1). It is the constant behind Bayesian A/B tests, Thompson sampling, and smoothed rates
(posterior mean (m + a)/(N + a + b)), and the same Γ(a + b)/(Γ(a)Γ(b)) appears in the Dirichlet
(2.38).

## Exercise 2.6 (\*)

> Make use of the result (2.265) to show that the mean, variance, and mode of the beta
> distribution (2.13) are given respectively by
> E[μ] = a / (a + b) (2.267), var[μ] = ab / ((a + b)²(a + b + 1)) (2.268),
> mode[μ] = (a − 1) / (a + b − 2). (2.269)

Context: §2.1.1, beta distribution (2.13); (2.265) from Ex. 2.5; Γ(x + 1) = xΓ(x) (Ex. 1.17).

Handwritten original: [page 1, mean and variance](ex-2.6-handwritten-1.png),
[page 2, mode](ex-2.6-handwritten-2.png) (08.11.2026).

### Solution

**Mean.** Multiply the density by μ: μ · μ^(a−1) = μ^((a+1)−1). The integral is (2.265) with a
replaced by a + 1:

E[μ] = Γ(a + b)/(Γ(a)Γ(b)) · Γ(a + 1)Γ(b)/Γ(a + b + 1). Cancel Γ(b); Γ(a + 1) = aΓ(a) and
Γ(a + b + 1) = (a + b)Γ(a + b), so E[μ] = a/(a + b). ∎

**Variance.** var[μ] = E[μ²] − E[μ]². The same step with a + 2:
E[μ²] = Γ(a + b)/(Γ(a)Γ(b)) · Γ(a + 2)Γ(b)/Γ(a + b + 2). Apply the recursion twice:
Γ(a + 2) = (a + 1)aΓ(a), Γ(a + b + 2) = (a + b + 1)(a + b)Γ(a + b). So
E[μ²] = a(a + 1)/((a + b)(a + b + 1)), and over the common denominator (a + b)²(a + b + 1):

numerator = a(a + 1)(a + b) − a²(a + b + 1) = a³ + a²b + a² + ab − a³ − a²b − a² = ab. ∎

**Mode.** The mode is the μ where the density peaks, not the peak height. Drop the constant (it
does not move the peak) and set the derivative of μ^(a−1)(1 − μ)^(b−1) to zero (product rule):

(a − 1)μ^(a−2)(1 − μ)^(b−1) − (b − 1)μ^(a−1)(1 − μ)^(b−2) = 0.

Divide by μ^(a−2)(1 − μ)^(b−2): (a − 1)(1 − μ) = (b − 1)μ, so μ = (a − 1)/(a + b − 2). ∎
(He wrote it as (1 − a)/(2 − a − b), the same fraction.) It is an interior maximum only for
a > 1 and b > 1; otherwise the density peaks at an end of [0, 1] or (a = b = 1) is flat.

### Review notes (professor)

His own, with one hint per step. Mean: he saw μ · μ^(a−1) = μ^((a+1)−1) and read (2.265) with
a + 1 himself. Variance: first wrote Γ(a + 2) = (a − 1)aΓ(a); the recursion only steps up,
Γ(a + 2) = (a + 1)Γ(a + 1). The sign check would have caught it: his numerator came out
−2a² − ab, and a variance cannot be negative. Mode: he first wrote "mode = derivative" (the mode
is where the derivative is zero) and got the chain-rule sign of (1 − μ)^(b−1) wrong (a double
minus). He then took logs of both sides, which is valid but longer than dividing out the common
powers. He asked what a mode is: first time computing one.

In practice: with a Beta(a, b) prior and m ones in N Bernoulli trials, the posterior is
Beta(m + a, N − m + b) (2.18). Its mean (m + a)/(N + a + b) is the smoothed rate used for
click-through rates and A/B tests (Laplace smoothing is a = b = 1). Its mode
(m + a − 1)/(N + a + b − 2) is the MAP estimate, and with a = b = 1 it is the maximum-likelihood
m/N. The variance shrinks like 1/N: more data, narrower posterior (Figure 2.3).

## Exercise 2.7 (\*\*)

> Consider a binomial random variable x given by (2.9), with prior distribution for μ given by
> the beta distribution (2.13), and suppose we have observed m occurrences of x = 1 and l
> occurrences of x = 0. Show that the posterior mean value of x lies between the prior mean and
> the maximum likelihood estimate for μ. To do this, show that the posterior mean can be written
> as λ times the prior mean plus (1 − λ) times the maximum likelihood estimate, where
> 0 ≤ λ ≤ 1. This illustrates the concept of the posterior distribution being a compromise
> between the prior distribution and the maximum likelihood solution.

Context: §2.1, μ_ML = m/N (2.8); §2.1.1, beta prior (2.13), posterior (2.18), predictive
p(x = 1 | D) = E[μ | D] (2.19)–(2.20); (2.265) from Ex. 2.5; Γ(x + 1) = xΓ(x).

Handwritten original: [page 1, E[μ | D] and p(D)](ex-2.7-handwritten-1.png) (12.11.2026),
[page 2, simplification and λ](ex-2.7-handwritten-2.png) (12.11 and 14.11.2026).

### Solution

**Three quantities.** N = m + l.
- Prior mean: E[μ] = a/(a + b) (2.15, Ex. 2.6).
- Maximum likelihood: μ_ML = m/(m + l) (2.8). The binomial likelihood is the Bernoulli
  likelihood (2.5) times (N choose m), which has no μ in it, so the maximum does not move.
- Posterior mean: E[μ | D] = ∫₀¹ μ p(D | μ) p(μ) / p(D) dμ.

**Posterior mean.** Substitute (2.9) and (2.13), pull out the constants
C = (N choose m) · Γ(a + b)/(Γ(a)Γ(b)), and collect powers: μ · μ^m · μ^(a−1) = μ^(m+a),
(1 − μ)^l (1 − μ)^(b−1) = (1 − μ)^(l+b−1). By (2.265) with A = m + a + 1, B = l + b:

E[μ | D] = C/p(D) · Γ(m + a + 1)Γ(l + b)/Γ(m + a + l + b + 1).

**Evidence.** Sum rule: p(D) = ∫₀¹ p(D | μ) p(μ) dμ = C · Γ(m + a)Γ(l + b)/Γ(m + a + l + b)
(the same integral without the extra μ: A = m + a).

Divide; C and Γ(l + b) cancel. With Γ(x + 1) = xΓ(x):

E[μ | D] = (m + a)/(m + a + l + b),

the mean of Beta(m + a, l + b), the posterior (2.18). (Shortcut: read off (2.18) and use Ex. 2.6.)

**Decomposition.** Split the numerator into a + m and give each term the denominator of its mean:

(a + m)/(a + b + m + l) = (a + b)/(a + b + m + l) · a/(a + b) + (m + l)/(a + b + m + l) · m/(m + l),

so λ = (a + b)/(a + b + m + l) and 1 − λ = (m + l)/(a + b + m + l).

**Bounds.** a, b > 0 (Appendix B: otherwise the normaliser (2.265) diverges), so λ > 0. m, l ≥ 0
are counts, so a + b ≤ a + b + m + l and λ ≤ 1. Both weights are non-negative, so E[μ | D] lies
between a/(a + b) and m/(m + l). ∎

Limits: N = 0 gives λ = 1 (no data, posterior = prior). N → ∞ gives λ → 0, 1 − λ → 1, so
E[μ | D] → μ_ML: with enough data the prior washes out (Bishop, after (2.20)).

### Review notes (professor)

His own algebra over two evenings, hints one step at a time. Slips on the way:
1. First integrand μ · p(μ, x) · p(μ): the joint already contains the prior, so it was counted
   twice; and 1/p(D) was missing. He fixed both to μ p(D | μ) p(μ)/p(D).
2. Exponents m + a + 1 and l + b + 1 instead of m + a and l + b − 1: the prior's exponents are
   a − 1 and b − 1, and the extra μ multiplies μ only.
3. Bolted the Ex. 2.6 mean (with Beta(m + a + 1, l + b), wrong parameters) onto the integral
   before p(D) was known. Crossed out.
4. Γ(m + a + 1) = (m + a + 1)Γ(m + a): the recursion is Γ(x + 1) = xΓ(x). **Same slip as Ex. 2.6.**
   Check: m = l = 0 must return the prior mean a/(a + b).
5. Bounds: wrote a ≥ 1, b ≥ 1 and |D| > 0. Neither is needed: a, b > 0 and m, l ≥ 0 suffice.
   Read "1 − λ → ∞/∞" (an indeterminate form); compute 1 − λ = 1 − (a + b)/(a + b + N) → 1.

He computed p(D) himself, found λ himself, and gave the interpretation unprompted: the posterior
is a compromise between prior and data. Questions asked: mean vs median (mean = expectation);
where Bishop names the data D (§2.1 before (2.5), (2.19)); what p(D) means (the evidence); why
Bernoulli ML applies to the binomial (constant factor); why a, b > 0 (Appendix B, the a ≥ 1 bound
there is for a finite density, not for existence).

In practice: this is **shrinkage**. a and b act as pseudo-counts, and λ is the share of prior
pseudo-counts in all counts. A new product with 2 clicks in 3 views gets a smoothed rate pulled
toward the prior (site-wide) rate; one with 2000 in 3000 keeps its own. That is the smoothed
click-through rate, Bayesian A/B testing, Laplace smoothing in naive Bayes (a = b = 1), and the
same pattern as target encoding with smoothing in tabular ML. The same "prior vs data,
weighted by amounts of information" form returns for the Gaussian mean in Ex. 2.38 / (2.141).

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

## Exercise 2.10 (\*\*)

> Using the property Γ(x + 1) = xΓ(x) of the gamma function, derive the following results for
> the mean, variance, and covariance of the Dirichlet distribution given by (2.38)
> E[μ_j] = α_j / α_0 (2.273)
> var[μ_j] = α_j(α_0 − α_j) / (α_0²(α_0 + 1)) (2.274)
> cov[μ_j μ_l] = −α_j α_l / (α_0²(α_0 + 1)), j ≠ l (2.275)
> where α_0 is defined by (2.39).

Context: §2.2.1, Dirichlet (2.38), α_0 = Σ_k α_k (2.39); the same move as the beta mean in Ex. 2.6.

Handwritten original: [page 1, mean](ex-2.10-handwritten-1.png),
[page 2, variance](ex-2.10-handwritten-2.png), [page 3, covariance](ex-2.10-handwritten-3.png)
(14.11.2026).

### Solution

One idea for all three parts. Dir(μ | α) integrates to 1 over the simplex, so for any
parameters α':

∫ ∏_k μ_k^(α'_k − 1) dμ = Γ(α'_1)⋯Γ(α'_K) / Γ(α'_0).

Extra factors of μ in front merge into their own factor of the product and raise that α by one.

**Mean.** μ_i · μ_i^(α_i − 1) = μ_i^(α_i), so α'_i = α_i + 1 and α'_0 = α_0 + 1:

E[μ_i] = Γ(α_0)/(Γ(α_1)⋯Γ(α_K)) · Γ(α_1)⋯Γ(α_i + 1)⋯Γ(α_K)/Γ(α_0 + 1)
= Γ(α_i + 1)/Γ(α_i) · Γ(α_0)/Γ(α_0 + 1) = α_i/α_0. ∎

**Variance.** μ_j² gives α'_j = α_j + 2, α'_0 = α_0 + 2. Recursion twice:
E[μ_j²] = (α_j + 1)α_j / ((α_0 + 1)α_0). Then over α_0²(α_0 + 1):

var[μ_j] = (α_j + 1)α_j/((α_0 + 1)α_0) − α_j²/α_0²
= α_j(α_j α_0 + α_0 − α_0 α_j − α_j) / (α_0²(α_0 + 1)) = α_j(α_0 − α_j)/(α_0²(α_0 + 1)). ∎

**Covariance** (j ≠ l). Still one integral over the simplex. μ_j μ_l raises two different
parameters: α'_j = α_j + 1, α'_l = α_l + 1, α'_0 = α_0 + 2. So
E[μ_j μ_l] = α_j α_l / ((α_0 + 1)α_0), and

cov[μ_j, μ_l] = α_j α_l/((α_0 + 1)α_0) − α_j α_l/α_0² = α_j α_l(α_0 − α_0 − 1)/(α_0²(α_0 + 1))
= −α_j α_l/(α_0²(α_0 + 1)). ∎

Checks: K = 2 with α_1 = a, α_2 = b gives the beta mean and variance of Ex. 2.6. The covariance
is negative because Σ_k μ_k = 1: if one share rises, the others must fall on average.

### Review notes (professor)

His own algebra, with hints on setup only. On the way:
1. First attempt wrote E[μ_i] as a sum Σ_i μ_i (μ is continuous: an integral over the simplex),
   Γ(α_i) on top of the normaliser instead of Γ(α_0), and ∏ α_i^(α_i − 1) instead of
   ∏ μ_k^(α_k − 1). Fixed after comparing with (2.38).
2. Thought the other μ_k drop out when averaging μ_i (confusing μ with the one-hot x, whose other
   entries are 0). They are integrated out, as y in Ex. 2.8.
3. Wrote ∫₀¹ … dμ_i: the integral is over the whole simplex, dμ.
4. Labelled E[μ_j²] as var[μ_j] for one line; then subtracted E[μ_j]².
5. **Γ recursion correct throughout** (after slips in Ex. 2.6 and 2.7).

Asked along the way (reading §2.2): what bold x is (one-hot), why Σ_x in (2.27) runs over the K
one-hot vectors, sufficient statistics (2.30), the Lagrange multiplier in (2.31) (still not clear:
revisit with Appendix E at §3.1.4), conjugate prior, simplex.

In practice: a Dirichlet over class or topic proportions has mean α/α_0 and spread controlled by
α_0 (bigger α_0, tighter around the mean: α_0 is the total pseudo-count). That is the prior in LDA
topic models, Dirichlet smoothing of multinomial counts (naive Bayes over words, language-model
smoothing), and Dirichlet-based uncertainty for classifiers. The negative covariance is why the
components of any probability vector (softmax outputs too) cannot be modelled as independent.

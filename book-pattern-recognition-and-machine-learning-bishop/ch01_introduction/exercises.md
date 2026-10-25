# PRML Chapter 1 — Introduction

Source: Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 1, exercises 1.1–1.41.

## Status

Solved (date):

- 1.1 (30.09.2026)
- 1.3 (14.10.2026)
- 1.4 (17.10.2026)
- 1.5 (17.10.2026)
- 1.6 (17.10.2026)
- 1.7 (18.10.2026)
- 1.8 (19.10.2026)
- 1.10 (20.10.2026)
- 1.11 (20.10.2026)
- 1.12 (21.10.2026)
- 1.14 (21.10.2026)
- 1.17 (21.10.2026)
- 1.18 (22.10.2026)
- 1.19 (23.10.2026)
- 1.20 (24.10.2026)
- 1.21 (25.10.2026)
- 1.22 (25.10.2026)
- 1.25 (25.10.2026)
- 1.26 (25.10.2026)

Open: 1.2 (regularised version of 1.1).

Skipped, plan correct (approach checked, algebra not written out): 1.9 (mode of the Gaussian:
dN/dx = 0 gives x = μ); 1.13 (variance estimate around the true μ is unbiased:
(1/N)Σₙ E[(xₙ − μ)²] = σ²; contrast with 1.12); 1.23 (general loss + priors: choose j minimizing
Σ_k L_kj p(x|C_k) p(C_k)); 1.24 (reject option: θ = 1 − λ).

Skipped by choice: 1.15, 1.16 (exact parameter count for order M, induction/combinatorics; the
D^M message is in §1.4).

---

## Exercise 1.1 (\*) www

> Consider the sum-of-squares error function given by (1.2) in which the function y(x, **w**) is
> given by the polynomial (1.1). Show that the coefficients **w** = {wᵢ} that minimize this error
> function are given by the solution to the following set of linear equations
>
> Σⱼ₌₀ᴹ Aᵢⱼ wⱼ = Tᵢ          (1.122)
>
> where
>
> Aᵢⱼ = Σₙ₌₁ᴺ (xₙ)ⁱ⁺ʲ,   Tᵢ = Σₙ₌₁ᴺ (xₙ)ⁱ tₙ.          (1.123)

Handwritten original: [`ex-1.1-handwritten.png`](ex-1.1-handwritten.png). Below is a transcription
of that page.

### Setup

Model and error function:

- y(x, **w**) = Σⱼ₌₀ᴹ wⱼ xʲ
- E(**w**) = ½ Σₙ₌₁ᴺ { y(xₙ, **w**) − tₙ }²

Only **w** is free. The xₙ and tₙ are data. **w** has M+1 components (w₀ … w_M), so E has M+1
partial derivatives, and at the minimum all of them are 0.

### Gradient

∇_w E(**w**) = [ Σₙ (y − tₙ)·1 ,  Σₙ (y − tₙ)·xₙ¹ ,  … ]

The component for wᵢ is Σₙ (y(xₙ, **w**) − tₙ) · xₙⁱ. The ½ cancels the 2 from the chain rule.

### Matrix form of the residual

Stack the data into X (N × (M+1)). Row n is (xₙ⁰, xₙ¹, …, xₙᴹ).

X · **w** − **t** = 0      shapes: N×(M+1) · (M+1)×1 − N×1

X**w** − **t** is the vector of residuals, one per data point. It is **not** the derivative itself.
Each partial derivative is a row times this residual vector:

- i = 0: the row is [1, 1, …, 1] (1×N), because xₙ⁰ = 1
- i = 1: the row is (x₁¹, x₂¹, …, x_N¹)
- general i: the row is (x₁ⁱ, …, x_Nⁱ), which is column i of X laid on its side

### Distribute and move t to the right

For each i:

(x₁ⁱ, …, x_Nⁱ) · X · **w** = (x₁ⁱ, …, x_Nⁱ) · **t**
  (1×N)(N×(M+1))((M+1)×1)   (1×N)(N×1)

### Back to sums

Σⱼ₌₀ᴹ ( Σₙ₌₁ᴺ xₙⁱ · xₙʲ ) wⱼ = Σₙ₌₁ᴺ xₙⁱ tₙ

- Left bracket: xₙⁱ · xₙʲ = xₙⁱ⁺ʲ, so it is **Aᵢⱼ**.
- Right side: **Tᵢ**.
- **i = index of the partial derivative** (fixed in each equation), j runs with the weights, and
  n runs over the data.

Hence Σⱼ Aᵢⱼ wⱼ = Tᵢ for i = 0 … M, which is (1.122)–(1.123). ∎

### Review notes (professor)

The final result is correct. Points to tighten on the page:

1. The line `w X − t` with w 1×M and X M×N is a dead end: a row minus a column. You replaced it
   correctly with X·**w** − **t** (N×1). Also count M+1, not M.
2. The second gradient entry is written `(X·w − T)·X₁`. The order is wrong: (N×1)·(1×N) is not what
   you want. The row goes **on the left**, as you did in the next lines.
3. Notation: capital T is used for the target vector **t**, but Tᵢ in (1.123) means something else.
   Keep **t** lowercase for the targets. The middle lines write the row as `xᵢʲ` (letters swapped).
   The final line uses the book's n/i/j correctly.
4. Make "set each ∂E/∂wᵢ = 0" and "swap Σₙ and Σⱼ" explicit steps. They are the heart of the proof,
   and on the page they are implicit.

Open bonus questions:

- Why is this stationary point a minimum, not a maximum?
- Name A and T in terms of X and **t**. Cross-check with MML §9.2.

---

## Exercise 1.3 (\*\*)

> Suppose that we have three coloured boxes r (red), b (blue), and g (green). Box r contains
> 3 apples, 4 oranges, and 3 limes, box b contains 1 apple, 1 orange, and 0 limes, and box g
> contains 3 apples, 3 oranges, and 4 limes. If a box is chosen at random with probabilities
> p(r) = 0.2, p(b) = 0.2, p(g) = 0.6, and a piece of fruit is removed from the box (with equal
> probability of selecting any of the items in the box), then what is the probability of
> selecting an apple? If we observe that the selected fruit is in fact an orange, what is the
> probability that it came from the green box?

Handwritten originals: [`ex-1.3-handwritten-1.png`](ex-1.3-handwritten-1.png),
[`ex-1.3-handwritten-2.png`](ex-1.3-handwritten-2.png). Below is a transcription of those pages.

### Data

| box | apples | oranges | limes | total | p(box) |
|---|---|---|---|---|---|
| r | 3 | 4 | 3 | 10 | 0.2 |
| b | 1 | 1 | 0 | 2 | 0.2 |
| g | 3 | 3 | 4 | 10 | 0.6 |

Drawn as a tree: first the box (branch weight p(box)), then the fruit (branch weight p(fruit|box)).

### Probability of an apple

Use the product rule (1.11) for each joint, p(a, box) = p(a|box)·p(box). Then the sum rule
(1.10) over boxes:

p(a) = p(a|r)·p(r) + p(a|b)·p(b) + p(a|g)·p(g)
   = 3/10 · 0.2 + 1/2 · 0.2 + 3/10 · 0.6
   = 0.2 · (3 + 5 + 9) / 10 = 0.2 · 17 / 10 = **0.34**

### Probability of the green box given an orange

Picture it as area: the orange outcomes make up a total area p(o). The part of that area lying in
the green box is p(o|g)·p(g). So, by Bayes (1.12):

p(g|o) = p(o|g)·p(g) / p(o)

p(o) = p(o|r)·p(r) + p(o|b)·p(b) + p(o|g)·p(g)
   = 4/10 · 0.2 + 1/2 · 0.2 + 3/10 · 0.6
   = 0.08 + 0.10 + 0.18 = 0.36

p(g|o) = 0.18 / 0.36 = **0.5**

### Review notes (professor)

Method correct throughout: sum rule, product rule, Bayes. Points from the pages:

1. On page 2, p(o) leaves out the blue term p(o|b)·p(b). Box b has 1 orange; its 0 is the limes.
   That gave p(o) = 0.26 and p(g|o) = 9/13 ≈ 0.69. With blue included it's 0.36 and 0.5 (above).
2. The numerator is computed as p(o) − p(o|r)·p(r). That works, but the direct green term
   p(o|g)·p(g) is simpler and doesn't depend on every other box being subtracted.
3. The last line is labelled p(o|g) but computes p(g|o). Bayes is exactly the step between those
   two, so keep the labels straight.
4. The crossed-out p(r|a)·p(a) on page 1 isn't wrong. It's the other factorisation of the same
   joint: p(a, r) = p(a|r)·p(r) = p(r|a)·p(a). That equality *is* Bayes' theorem. It just can't be
   used to compute p(a), because it contains p(a). The tree labels p(r)·p(a) should read p(a|r).

In practice: p(g|o) is a posterior over a hidden cause (which box) given an observation (which
fruit). The same computation shows up in naive Bayes classifiers and in the E-step of a mixture
model (PRML Ch. 9).

---

## Exercise 1.4 (\*\*) www

> Consider a probability density p_x(x) defined over a continuous variable x, and suppose that we
> make a nonlinear change of variable using x = g(y), so that the density transforms according to
> (1.27). By differentiating (1.27), show that the location ŷ of the maximum of the density in y
> is not in general related to the location x̂ of the maximum of the density over x by the simple
> functional relation x̂ = g(ŷ) as a consequence of the Jacobian factor. This shows that the
> maximum of a probability density (in contrast to a simple function) is dependent on the choice
> of variable. Verify that, in the case of a linear transformation, the location of the maximum
> transforms in the same way as the variable itself.

Handwritten original: [`ex-1.4-handwritten.png`](ex-1.4-handwritten.png). Below is a transcription
of that page.

### Setup

x = g(y), with g invertible (strictly monotonic). That is what (1.27) requires: each x comes from
exactly one y. Then

p_y(y) = p_x(g(y)) · |g′(y)|

x̂ is the maximum of p_x, so p_x′(x̂) = 0.

### Differentiate (1.27)

Product rule plus chain rule:

d p_y / dy = p_x′(g(y)) · g′(y) · |g′(y)| + p_x(g(y)) · ( |g′(y)| )′

g is monotonic, so g′ has a fixed sign s = ±1, |g′| = s·g′, and ( |g′(y)| )′ = s · g″(y).

### Test the naive guess

Take y\* with g(y\*) = x̂: x̂ carried over by the transformation. The question is whether y\* = ŷ.

p_y′(y\*) = p_x′(x̂) · g′(y\*) · |g′(y\*)| + p_x(x̂) · s · g″(y\*) = p_x(x̂) · s · g″(y\*)

The first term vanishes because p_x′(x̂) = 0. p_x(x̂) > 0: a density integrates to 1, so it's
positive somewhere, and the maximum is at least that. So **the slope of p_y at y\* is zero only if
g″(y\*) = 0.**

### Cases

- **Linear, g(y) = a·y + b:** g″ ≡ 0, so y\* is stationary. In fact
  p_y(y) = |a| · p_x(a·y + b) is a shifted, rescaled copy of p_x, so its peak sits exactly at
  a·ŷ + b = x̂. The maximum transforms like the variable.
- **Nonlinear, e.g. g(y) = eʸ:** g″ = eʸ > 0 for every real y (it only tends to 0 as y → −∞). So
  p_y′(y\*) ≠ 0, and y\* is not the maximum: ŷ ≠ g⁻¹(x̂).

**Conclusion.** The relation x̂ = g(ŷ) holds only when g″(y\*) = 0. That's always true for linear
g and generally false for nonlinear g. The cause is the Jacobian factor |g′(y)|: its derivative
produces the g″ term. So the maximum of a density depends on the choice of variable. ∎

### Review notes (professor)

1. The page writes ( |g′(y)| )′ as |g″(y)|. That's wrong in general: with g(y) = y³/3 + y,
   ( |g′| )′ = 2y but |g″| = 2|y|. The correct form is s·g″(y). Here it doesn't change the
   conclusion, because only whether the term is zero matters.
2. The page's last line, "g nonlinear ⇒ g″ ≠ 0 ⇒ y\* ≠ ŷ", overstates it. A nonlinear g can have
   g″ = 0 at isolated points: g(y) = y³ + y has g″(0) = 0. The claim is "not *in general*", and
   one counterexample (eʸ) is enough to prove that. The conclusion above is the corrected
   statement, and this doubt was his own.
3. The page writes the naive point as ŷ before it's known to be the maximum. The write-up names it
   y\* and asks whether y\* = ŷ.
4. The monotonicity of g was an unstated assumption. It's now explicit (needed for (1.27)).

Open bonus: for an ordinary function f with h(y) = f(g(y)), show h′(y\*) = f′(x̂)·g′(y\*) = 0. The
maximum moves with the variable, because there's no Jacobian.

In practice: MAP estimates depend on parameterisation (maximising over σ² vs log σ gives different
answers) for exactly this reason. Normalizing flows train on the same formula: their
log-det-Jacobian term is log |g′|.

---

## Exercise 1.5 (\*)

> Using the definition (1.38) show that var[f(x)] satisfies (1.39).

(1.38): var[f] = E[(f(x) − E[f(x)])²]   (1.39): var[f] = E[f(x)²] − E[f(x)]²

Handwritten original: [`ex-1.5-handwritten.png`](ex-1.5-handwritten.png). Below is a transcription
of that page.

### Proof

Write μ = E[f(x)]. It's a number, not a function of x.

var[f] = E[(f(x) − μ)²]
   = E[f(x)² − 2 f(x) μ + μ²]       (expand the square)
   = E[f(x)²] − 2 E[f(x) μ] + E[μ²]   (linearity of E)
   = E[f(x)²] − 2 μ·E[f(x)] + μ²     (μ is a constant: E[c·f] = c·E[f], E[c] = c)
   = E[f(x)²] − 2μ² + μ²
   = E[f(x)²] − E[f(x)]²  ∎

### Review notes (professor)

Correct and complete on the first attempt. One thing to make explicit in your own writing:

1. The step from E[f(x)·E[f(x)]] to E[f(x)]·E[f(x)] uses the fact that E[f(x)] is a **constant**,
   so it can come out of the outer expectation. Say so on the line where you use it. It's the only
   non-mechanical step in the proof.
2. Brackets: E[f(x)²] (mean of the square) and E[f(x)]² (square of the mean) differ only in
   where the ² sits, and the page shows a couple of exponents being moved. Their difference *is*
   the variance, so it's worth keeping them visually distinct, e.g. E[f(x)²] vs (E[f(x)])².

In practice: this identity is how variance is computed in one pass over data (keep running sums of
f and f²), e.g. in BatchNorm statistics. It's also numerically unstable when the mean is large
compared with the spread (catastrophic cancellation), which is why Welford's algorithm exists.

---

## Exercise 1.6 (\*)

> Show that if two variables x and y are independent, then their covariance is zero.

Handwritten original: [`ex-1.6-handwritten.png`](ex-1.6-handwritten.png). Below is a transcription
of that page.

### Proof

Covariance (1.41): cov[x, y] = E_{x,y}[xy] − E[x]·E[y].

Independence means the joint factorises: p(x, y) = p(x)·p(y). So

E_{x,y}[xy] = Σ_x Σ_y x·y·p(x, y)
     = Σ_x Σ_y x·p(x) · y·p(y)
     = ( Σ_x x·p(x) ) · ( Σ_y y·p(y) )
     = E[x]·E[y]

and therefore cov[x, y] = E[x]·E[y] − E[x]·E[y] = 0. ∎

(Continuous case: the same, with ∬ … dx dy in place of the double sum.)

### Review notes (professor)

Correct, first attempt. He remembered the result E[xy] = E[x]E[y] but not why. After a pointer to
the definitions, he derived it himself.

1. The split of the double sum into a product is one jump on the page. The reason is that inside
   the inner sum over y, the factor x·p(x) doesn't depend on y, so it comes out:
   Σ_x x·p(x) · (Σ_y y·p(y)), and the bracket is then a constant E[y] for the outer sum. Say so.
2. The page writes E_{xy}[x, y] (with a comma). It's the expectation of the product xy.

Worth knowing: the converse is **false**. Zero covariance doesn't imply independence. Example:
x uniform on {−1, 0, 1} and y = x². Then y is completely determined by x, yet
cov = E[x³] − E[x]·E[x²] = 0 − 0 = 0. Covariance only sees linear dependence.

In practice: this is the gap between decorrelating and making independent. PCA (Ch. 12) only
removes correlation; ICA goes further and looks for independent components.

---

## Exercise 1.7 (\*\*) www

> In this exercise, we prove the normalization condition (1.48) for the univariate Gaussian. To do
> this consider the integral
>
> I = ∫₋∞^∞ exp(−x²/2σ²) dx   (1.124)
>
> which we can evaluate by first writing its square in the form
>
> I² = ∫₋∞^∞ ∫₋∞^∞ exp(−x²/2σ² − y²/2σ²) dx dy.   (1.125)
>
> Now make the transformation from Cartesian coordinates (x, y) to polar coordinates (r, θ) and
> then substitute u = r². Show that, by performing the integrals over θ and u, and then taking the
> square root of both sides, we obtain
>
> I = (2πσ²)^{1/2}.   (1.126)
>
> Finally, use this result to show that the Gaussian distribution N(x|μ, σ²) is normalized.

Handwritten original, three pages: [`ex-1.7-handwritten-1.png`](ex-1.7-handwritten-1.png) (polar
setup), [`ex-1.7-handwritten-2.png`](ex-1.7-handwritten-2.png) (the u-integral),
[`ex-1.7-handwritten-3.png`](ex-1.7-handwritten-3.png) (normalisation of N). Below is a
transcription.

### What I is

I is the area under the **unnormalised** bell exp(−x²/2σ²): one number, not a function of x, and
not a CDF. The Gaussian's front factor is 1/I, so the exercise is "where does 1/√(2πσ²) come
from?". exp(−x²) has no elementary antiderivative, which is why I is computed via I².

### Computing I

I² is the product of two copies of I; the integration variable is a dummy, so the second copy is
written in y, and e^a·e^b = e^(a+b) merges them into (1.125).

Polar coordinates x = r cosθ, y = r sinθ:
- exponent: −(r²cos²θ + r²sin²θ)/2σ² = −r²/2σ², since cos²θ + sin²θ = 1
- area element: dx dy = r dr dθ (a small polar "rectangle" has sides dr and r dθ)
- limits: r from 0 to ∞, θ from 0 to 2π

I² = ∫₀^{2π} ∫₀^∞ exp(−r²/2σ²) r dr dθ

u = r², du = 2r dr, so r dr = du/2:

I² = ∫₀^{2π} ∫₀^∞ exp(−u/2σ²) · ½ du dθ

Inner integral, with v = −u/2σ², dv = −du/2σ², i.e. du = −2σ² dv:

∫₀^∞ exp(−u/2σ²) du = [−2σ² exp(−u/2σ²)]₀^∞ = −2σ²(0 − 1) = 2σ²

So I² = 2π · ½ · 2σ² = 2πσ², and I = (2πσ²)^{1/2}. ∎ (1.126)

### N(x|μ, σ²) is normalised

∫₋∞^∞ N(x|μ, σ²) dx = (1/√(2πσ²)) ∫₋∞^∞ exp(−(x−μ)²/2σ²) dx

z = x − μ, dz = dx, limits stay ±∞:

= (1/√(2πσ²)) ∫₋∞^∞ exp(−z²/2σ²) dz = (1/√(2πσ²)) · I = 1. ∎

### Review notes (professor)

Solved, **heavily guided** (no answer handed over, but most steps needed a hint). The idea is now
his: square the integral, go polar, and the Jacobian's r is what makes u = r² work.

1. Calculus technique is the real gap, not the probability. He needed help with: dx dy → r dr dθ
   (first tried d(r cosθ)·d(r sinθ)); the antiderivative of e^(ku) (first wrote ½e^(−u/2σ²),
   missing the 1/k); limits after substitution (first used u from −∞, which made exp blow up; he
   noticed "something odd" himself, which was the right reaction); and carrying all factors (he
   dropped ½ and the θ integral, and lost the π).
2. In the last part he first substituted the whole exponent, u = −(x−μ)²/2σ², and was left with a
   stray (x−μ). That's exactly the wall that makes the 1-D integral impossible, and the reason for
   the I² trick: in 1-D there's no Jacobian r to pair with dr. The fix is to shift only, z = x − μ,
   and *recognise* I instead of computing it.
3. On the page: write the limits after each substitution, and I = √(2πσ²) with π inside the root
   (page 3 has "√(2σ²) π?" at the top and "√(2πσ)" without the square under the brace).

In practice: −½ log(2πσ²) in every Gaussian log-likelihood is the log of this 1/I. With a learned
σ (heteroscedastic regression, VAEs' decoder variance) it's the term that stops σ from running off
to infinity. The I² / polar trick reappears for the multivariate Gaussian in Ch. 2, where the
Jacobian becomes a determinant.

---

## Exercise 1.8 (\*\*) www

> By using a change of variables, verify that the univariate Gaussian distribution given by (1.46)
> satisfies (1.49). Next, by differentiating both sides of the normalization condition
>
> ∫₋∞^∞ N(x|μ, σ²) dx = 1   (1.127)
>
> with respect to σ², verify that the Gaussian satisfies (1.50). Finally, show that (1.51) holds.

(1.49): E[x] = μ. (1.50): E[x²] = μ² + σ². (1.51): var[x] = E[x²] − E[x]² = σ².

Handwritten original, two pages: [`ex-1.8-handwritten-1.png`](ex-1.8-handwritten-1.png) (E[x]) and
[`ex-1.8-handwritten-2.png`](ex-1.8-handwritten-2.png) (E[x²], var). Below is a transcription.

### Part 1: E[x] = μ

E[x] = (1/√(2πσ²)) ∫₋∞^∞ exp(−(x−μ)²/2σ²) · x dx

z = x − μ, dz = dx, x = z + μ, limits stay ±∞:

E[x] = (1/√(2πσ²)) [ ∫₋∞^∞ exp(−z²/2σ²) z dz + μ ∫₋∞^∞ exp(−z²/2σ²) dz ]

The second integral is I from Ex. 1.7, √(2πσ²).

The first: k = −z²/2σ², dk = −(z/σ²) dz, so z dz = −σ² dk. k = −z² folds at z = 0, so split
there: z ∈ (−∞, 0] gives k from −∞ to 0, and z ∈ [0, ∞) gives k from 0 back to −∞:

∫₋∞^∞ exp(−z²/2σ²) z dz = −σ² ( ∫₋∞^0 eᵏ dk + ∫₀^{−∞} eᵏ dk ) = −σ² (1 − 1) = 0

So E[x] = (1/√(2πσ²)) · (0 + μ√(2πσ²)) = μ. ∎

### Review notes (professor), part 1

Correct, guided. He split x = z + μ into two integrals himself. Gaps, all calculus technique again:
first attached the "by Ex. 1.7" brace to the z-term (the μ-term is the one that is I); needed the
substitution k = −z²/2σ² that absorbs the z in z dz; asked how limits transform (rule: plug the
old limits into the substitution). His instinct that a 0 belongs in the new limits was right: the
substitution isn't monotonic, and splitting at the fold z = 0 is the rigorous way to do it.
Shortcut to know: z·exp(−z²/2σ²) is odd, so its integral over a symmetric range is 0 without any
computation.

### Part 2: E[x²] = μ² + σ², by differentiating the normalisation

Move the constant across and write s = σ²:

∫₋∞^∞ exp(−(x−μ)²/2s) dx = (2πs)^{1/2}

This holds for every s, so the derivatives in s of both sides are equal too. The derivative goes
inside the integral (an integral is a sum; the derivative of a sum is the sum of derivatives):

∫₋∞^∞ exp(−(x−μ)²/2s) · (x−μ)²/2s² dx = ½(2πs)^{−1/2} · 2π = π/√(2πs)

Multiply both sides by 2s²/√(2πs), so that N appears under the integral:

∫₋∞^∞ N(x|μ, σ²) · (x−μ)² dx = 2πs²/(2πs) = s

So E[(x−μ)²] = σ², and since E[x] = μ (part 1) this is var[x] = σ², i.e. (1.51).

With the identity from Ex. 1.5, var[x] = E[x²] − E[x]²:

E[x²] = var[x] + E[x]² = σ² + μ². ∎ (1.50)

### Review notes (professor), part 2

Correct idea, guided. The differentiation step was the hard part conceptually: his first reaction
was that differentiating inside an integral "sounds almost like a mistake". Resolved with
"integral = sum, derivative of a sum = sum of derivatives" and the check ∫₀¹ a x dx = a/2 done
both ways. After that, both derivatives on the page were right first time.

1. First attempt replaced the integral by its value, √(2πσ²) = 1, which is false (true for one σ
   only). Lesson: the trick only works if the integral stays an integral; otherwise it collapses to
   1 = 1 and the derivative says 0 = 0.
2. **Sign slip in the last line:** the page has E[x²] = σ² − μ². From E[x²] − μ² = σ² it is
   E[x²] = σ² **+** μ². Sanity check: E[x²] ≥ 0 always, but σ² − μ² is negative for μ = 10, σ = 1.
3. He went from E[(x−μ)²] = σ² to E[x²] via the 1.5 identity rather than expanding the square.
   That's a fine route, and it reuses a result instead of redoing it.
4. Notation: N(x|μ, σ²)·(x−μ)², not N((x−μ)²|μ, σ²); and (x−μ)², not (x−μ²) in the exponent.

In practice: differentiating log Z with respect to a parameter to get moments is the standard
exponential-family result (PRML §2.4): the first derivative of the log-normaliser is E[sufficient
statistic], the second is its variance. It's why the gradients of logistic regression and softmax
cross-entropy come out as "prediction minus target".

---

## Exercise 1.10 (\*)

> Suppose that the two variables x and z are statistically independent. Show that the mean and
> variance of their sum satisfies
>
> E[x + z] = E[x] + E[z]   (1.128)
>
> var[x + z] = var[x] + var[z].   (1.129)

Handwritten original, two pages: [`ex-1.10-handwritten-1.png`](ex-1.10-handwritten-1.png) (mean) and
[`ex-1.10-handwritten-2.png`](ex-1.10-handwritten-2.png) (variance; written with y instead of z).
Below is a transcription.

### Mean

E[x + z] = Σ_x Σ_z (x + z) · p(x, z), with p(x, z) = p(x)·p(z)

= Σ_z p(z) · Σ_x x·p(x) + Σ_z z·p(z) · Σ_x p(x) = 1·E[x] + E[z]·1 = E[x] + E[z]. ∎

### Variance

var[x + z] = E[(x + z)²] − (E[x + z])²

= E[x²] + 2E[xz] + E[z²] − E[x]² − 2E[x]E[z] − E[z]²

Independence gives E[xz] = E[x]E[z] (Ex. 1.6), so the cross terms cancel:

= (E[x²] − E[x]²) + (E[z²] − E[z]²) = var[x] + var[z]. ∎

### Review notes (professor)

Correct. One misconception cleared before starting: he first guessed the weight would be
p(x) + p(z) because independent variables "don't overlap". That confuses independence (joint
factorises, p(x, z) = p(x)p(z)) with mutual exclusivity (probabilities of disjoint events add).
Two exclusive events of non-zero probability are in fact dependent.

1. He asked whether Σ_z p(z) = 1 needs independence. It doesn't: it's the normalisation of z's
   own distribution.
2. He asked why the joint p(x, z) is the weight both for x + z and for xz. Because
   E[f(x, z)] = ΣΣ f(x, z) p(x, z) for any f: f is what is averaged, p is how often each outcome
   happens.
3. Worth knowing: (1.128) holds **without** independence (sum rule: Σ_z p(x, z) = p(x)). Only
   (1.129) needs it, and only via E[xz] = E[x]E[z]. In general
   var[x + z] = var[x] + var[z] + 2cov[x, z].

In practice: this is why averaging N independent noisy estimates divides the variance by N
(var of the mean is σ²/N), the basis of ensembles, bagging and minibatch gradient noise. When the
estimates are correlated the 2cov term stays, which is why bagged trees are decorrelated by
feature subsampling (random forests).

---

## Exercise 1.11 (\*)

> By setting the derivatives of the log likelihood function (1.54) with respect to μ and σ² equal
> to zero, verify the results (1.55) and (1.56).

Handwritten original: [`ex-1.11-handwritten.png`](ex-1.11-handwritten.png). Below is a
transcription.

### Solution

The N points are independent, so the likelihood is a product over n and its log is a sum:

ln p(**x**|μ, σ²) = −(1/2σ²) Σₙ (xₙ − μ)² − (N/2) ln σ² − (N/2) ln 2π   (1.54)

**μ:** ∂/∂μ = (1/σ²) Σₙ (xₙ − μ) = 0 ⇒ Σₙ xₙ − Nμ = 0 ⇒ μ_ML = (1/N) Σₙ xₙ   (1.55)

**σ²:** ∂/∂σ² = (1/2σ⁴) Σₙ (xₙ − μ)² − N/2σ² = 0. Multiply by 2σ⁴ (σ² ≠ 0):

σ²_ML = (1/N) Σₙ (xₙ − μ_ML)²   (1.56) ∎

### Review notes (professor)

Correct, independently, first attempt. Differentiating with respect to σ² as a single variable
was done right.

1. Say why μ_ML can be plugged into (1.56): the μ equation doesn't involve σ², so it is solved
   first, on its own.
2. Write the intermediate step Σxₙ − Nμ = 0, and name the results μ_ML, σ²_ML. On the page they
   are just μ and σ², which blurs "parameter" vs "estimate".
3. Not asked, but worth one line: it's a maximum, not a minimum. ∂²/∂μ² = −N/σ² < 0.

Before this he asked why a joint of independent variables is a product when ∫p(x) + ∫p(z) = 2.
Resolved: the joint lives on the space of pairs, ∬p(x)p(z) dx dz = 1·1 = 1; probabilities add
only for mutually exclusive outcomes of one variable.

In practice: the μ part is exactly why minimising MSE = maximum likelihood under Gaussian noise
(§1.2.5). σ²_ML divides by N and is biased low (that's Ex. 1.12); this is numpy's `np.var`
default (`ddof=0`) vs the unbiased `ddof=1`, and pandas' `.var()` which defaults to `ddof=1`.

---

## Exercise 1.12 (\*\*) www

> Using the results (1.49) and (1.50), show that
>
> E[xₙxₘ] = μ² + Iₙₘσ²   (1.130)
>
> where xₙ and xₘ denote data points sampled from a Gaussian distribution with mean μ and
> variance σ², and Iₙₘ satisfies Iₙₘ = 1 if n = m and Iₙₘ = 0 otherwise. Hence prove the results
> (1.57) and (1.58).

(1.57): E[μ_ML] = μ. (1.58): E[σ²_ML] = ((N−1)/N)·σ².

Handwritten original: [`ex-1.12-handwritten-1.png`](ex-1.12-handwritten-1.png) (setup),
[`ex-1.12-handwritten-2.png`](ex-1.12-handwritten-2.png) (E[μ_ML]),
[`ex-1.12-handwritten-3.png`](ex-1.12-handwritten-3.png) and
[`ex-1.12-handwritten-4.png`](ex-1.12-handwritten-4.png) (E[σ²_ML]). Below is a transcription.

### What is being averaged

μ_ML is computed from a random data set **X** = (x₁, …, x_N), so it's itself a random variable;
E[μ_ML] is its average over all data sets (Figure 1.15). The density of a data set is the product
(independence), so

E[μ_ML] = ∫ μ_ML(**X**) · ∏ₙ N(xₙ|μ, σ²) d**X**,  d**X** = dx₁…dx_N.

### Part (1.57): E[μ_ML] = μ

N = 2 first: E[μ_ML] = ∬ ½(x₁ + x₂) p(x₁)p(x₂) dx₁dx₂ = ½ E[x₁ + x₂].

General N, by linearity (Ex. 1.10): E[μ_ML] = (1/N) E[Σₙ xₙ] = (1/N) Σₙ E[xₙ] = (N/N)·μ = μ,
since each xₙ ~ N(μ, σ²) has E[xₙ] = μ (Ex. 1.8). ∎

### Review notes (professor), part (1.57)

Guided. The conceptual step was the hard one: "the expectation of μ_ML over what?". He got there
himself ("we iterate datasets"), then wrote the N-fold integral over **X** with the product
density correctly. Slips: first replaced ½E[x₁ + x₂] with ½(μ² + I₁₂σ²), mixing up the sum with
the product E[xₙxₘ] (the comma in "E[x₁, x₂]" hid it; a units check catches it: μ² is in x²).
Also asked why E[xₙ] = μ: "mean" is overloaded (parameter μ vs sample mean μ_ML).
His crossed-out ∫p(μ|x)μ dμ is the Bayesian view (μ random, data fixed); here it's the reverse.

### Part (1.58): E[σ²_ML] = ((N−1)/N)σ²

σ²_ML = (1/N) Σₙ (xₙ − μ_ML)², with μ_ML = (1/N) Σₘ xₘ: the **sample** mean, not μ. Expand the
square and take the expectation over **X** (same integral as above, linearity of E):

E[σ²_ML] = (1/N) ( Σₙ E[xₙ²] − 2 Σₙ E[xₙ μ_ML] + N·E[μ_ML²] ).

μ_ML depends on **X**, so it stays inside E and is expanded:

- E[xₙ μ_ML] = (1/N) Σₘ E[xₙxₘ]
- μ_ML² = (1/N²) Σₙ Σₘ xₙxₘ (squaring a sum gives every pair, hence two indices), so
  N·E[μ_ML²] = (1/N) Σₙ Σₘ E[xₙxₘ]

The second and third terms carry the same double sum, −2/N + 1/N = −1/N:

E[σ²_ML] = (1/N) ( Σₙ E[xₙ²] − (1/N) Σₙ Σₘ (μ² + Iₙₘσ²) )   by (1.130)

Count: Σₙ Σₘ μ² = N²μ²; Iₙₘ is non-zero on the N diagonal pairs only, Σₙ Σₘ Iₙₘσ² = Nσ².
With E[xₙ²] = μ² + σ² (Ex. 1.8):

E[σ²_ML] = (1/N) ( Nμ² + Nσ² − Nμ² − σ² ) = ((N−1)/N) σ². ∎

(1.130) itself: for n ≠ m, independence factorises the integral, E[xₙxₘ] = E[xₙ]E[xₘ] = μ²
(Ex. 1.10); for n = m it's E[xₙ²] = μ² + σ² (Ex. 1.8).

Cleaner route (shown afterwards): Σₙ(xₙ − μ_ML)² = Σₙ(xₙ − μ)² − N(μ_ML − μ)², so
E[σ²_ML] = (1/N)(Nσ² − N·var[μ_ML]) = (1/N)(Nσ² − σ²), using var[μ_ML] = σ²/N (Ex. 1.10).

### Review notes (professor), part (1.58)

Guided, finished himself. Main error at the start: used the true μ inside σ²_ML; with μ the
estimator is exactly unbiased, so the whole bias comes from replacing μ by μ_ML. Then pulled
μ_ML out of the integral as a constant (caught and crossed out himself). Third term: lost the
square, then the factor N from Σₙ μ_ML², and wrote a single sum where squaring needs a double
one. Final arithmetic: the minus wasn't distributed over σ². Good question on the way: "μ_ML
depends on xₘ but the product is over p(xₙ)": n in ∏ₙ is a dummy index, the product is the
density of the whole data set, every xₘ is in it.

In practice: fitting the mean to the same points makes them look closer than they are; the
shortfall is exactly var[μ_ML] = σ²/N. That is train error < test error in miniature, and why
unbiased variance divides by N − 1 (`np.var(x, ddof=1)`, pandas default).

## Exercise 1.13 (\*) (skipped, plan correct)

> Suppose that the variance of a Gaussian is estimated using the result (1.56) but with the
> maximum likelihood estimate μ_ML replaced with the true value μ of the mean. Show that this
> estimator has the property that its expectation is given by the true variance σ².

Plan: same start as 1.12 (E over data sets, linearity), but with the true μ nothing depends on
**X** except xₙ: (1/N)Σₙ E[(xₙ − μ)²] = (1/N)·Nσ² = σ². To practise: expand the square and use
E[xₙ²] = μ² + σ², E[xₙ] = μ. Compare with 1.12: the bias comes only from using μ_ML.

## Exercise 1.14 (\*\*)

> Show that an arbitrary square matrix with elements wᵢⱼ can be written in the form
> wᵢⱼ = wᵢⱼˢ + wᵢⱼᴬ (1.131) where wˢ and wᴬ are symmetric and anti-symmetric matrices,
> respectively, satisfying wᵢⱼˢ = wⱼᵢˢ and wᵢⱼᴬ = −wⱼᵢᴬ for all i and j. Now consider the
> second order term in a higher order polynomial in D dimensions, given by
> Σᵢ Σⱼ wᵢⱼxᵢxⱼ (1.132). Show that Σᵢ Σⱼ wᵢⱼxᵢxⱼ = Σᵢ Σⱼ wᵢⱼˢxᵢxⱼ (1.133) so that the
> contribution from the anti-symmetric matrix vanishes. We therefore see that, without loss of
> generality, the matrix of coefficients wᵢⱼ can be chosen to be symmetric, and so not all of
> the D² elements of this matrix can be chosen independently. Show that the number of
> independent parameters in the matrix wᵢⱼˢ is given by D(D + 1)/2.

Context: §1.4 (curse of dimensionality), polynomial (1.74) in D inputs x₁…x_D. "Second order"
= products of two inputs (degree, not position in the line).

Handwritten original: [`ex-1.14-handwritten.png`](ex-1.14-handwritten.png).

### Part 1: decomposition

Found from an example (w₁₂ = 5, w₂₁ = 1): sum 6 is unchanged by the swap, difference 4 flips
sign, but 6 + 4 ≠ 5, so halve both:

wᵢⱼˢ = (wᵢⱼ + wⱼᵢ)/2,  wᵢⱼᴬ = (wᵢⱼ − wⱼᵢ)/2.

Checks: the sum is wᵢⱼ (the wⱼᵢ cancel); swapping i, j leaves wˢ unchanged and flips wᴬ.
Works for any entries, so for any square matrix. Matrix form: S = (W + Wᵀ)/2, A = (W − Wᵀ)/2.
Antisymmetric diagonal is zero (Aᵢᵢ = −Aᵢᵢ).

### Part 2: (1.133)

Σᵢ Σⱼ wᵢⱼˢxᵢxⱼ = ½ Σᵢ Σⱼ wᵢⱼxᵢxⱼ + ½ Σᵢ Σⱼ wⱼᵢxᵢxⱼ.

The second sum has the same terms as the first in a different order (rename the dummy indices
i ↔ j, xⱼxᵢ = xᵢxⱼ; summing by rows vs by columns), so the total is Σᵢ Σⱼ wᵢⱼxᵢxⱼ. ∎

Alternative: write wᵢⱼ = wᵢⱼˢ + wᵢⱼᴬ in (1.132); the wᴬ part sums to zero.

### Part 3: D(D + 1)/2 independent parameters

Independent = free to choose; the rest are then forced. In wˢ, choosing wᵢⱼ fixes wⱼᵢ, so only
the diagonal and one triangle are free: D + (D² − D)/2 = D²/2 + D/2 = D(D + 1)/2.
(D = 3: 3 + 3 = 6.) 1.15/1.16 (higher orders, combinatorics) skipped by choice.

### Review notes (professor)

Guided, with the key steps his own: the sum/difference from the 5-and-1 example, the /2, the
fold-along-the-diagonal picture, and "summing by rows or by columns gives the same". Took the
direct route for part 2 (substitute wˢ) instead of showing the A part vanishes; both are fine.
Confusion on the way: "second order" read as "third term in the line".

In practice: a quadratic form xᵀWx only sees the symmetric part of W. This is why §2.3 can take
the covariance Σ (and Σ⁻¹ in (x − μ)ᵀΣ⁻¹(x − μ)) symmetric without loss of generality.

## Exercise 1.17 (\*\*) www

> The gamma function is defined by Γ(x) ≡ ∫₀^∞ u^(x−1) e^(−u) du (1.141). Using integration by
> parts, prove the relation Γ(x + 1) = xΓ(x) (1.142). Show also that Γ(1) = 1 and hence that
> Γ(x + 1) = x! when x is an integer.

Handwritten original: [`ex-1.17-handwritten.png`](ex-1.17-handwritten.png) (headed "Ex 1.7" by
slip).

Integration by parts, ∫ f dv = f·v − ∫ v df (product rule read backwards), on Γ(x) itself:
f = u^(x−1), df = (x−1)u^(x−2) du; dv = e^(−u) du, v = −e^(−u).

Γ(x) = [−u^(x−1)e^(−u)]₀^∞ + (x−1) ∫₀^∞ u^(x−2) e^(−u) du = 0 + (x−1)Γ(x−1).

Bracket: at ∞ the limit u^(x−1)/e^u → 0 (exponential beats any power); at 0 it is 0 (x > 1).
With y = x − 1: Γ(y + 1) = yΓ(y). ∎ (1.142)

Γ(1) = ∫₀^∞ e^(−u) du = [−e^(−u)]₀^∞ = 0 − (−1) = 1.

Integer x: Γ(x + 1) = x·Γ(x) = x(x−1)Γ(x−1) = … = x(x−1)…1·Γ(1) = x!. ∎

### Review notes (professor)

Integration by parts was new: learned it mid-exercise from
[mathprofi](https://mathprofi.ru/integrirovanie_po_chastyam.html) after my explanations didn't
land (f·g' notation confused him; mathprofi's ∫u dv with differentials worked). Then applied it
correctly on his own, on Γ(x) rather than Γ(x+1), and recognised the leftover integral as
Γ(x−1). Slips: plugged ∞ in as a number (∞^(x−1)·e^(−∞)) instead of taking a limit; dropped the
minus in the boundary term (harmless, it is 0). Asked the right question: differential vs
derivative (df = f'dx).

In practice: Γ normalises the beta, gamma, Dirichlet and Student-t distributions (Ch. 2);
Γ(x+1) = xΓ(x) is the step that gives their means and variances. Next, 1.18 uses Γ for the
volume of a D-dimensional sphere (curse of dimensionality).

## Exercise 1.18 (\*\*) www

> We can use the result (1.126) to derive an expression for the surface area S_D, and the
> volume V_D, of a sphere of unit radius in D dimensions. To do this, consider the following
> result, which is obtained by transforming from Cartesian to polar coordinates
> ∏ᵢ₌₁ᴰ ∫₋∞^∞ e^(−xᵢ²) dxᵢ = S_D ∫₀^∞ e^(−r²) r^(D−1) dr. (1.142)
> Using the definition (1.141) of the Gamma function, together with (1.126), evaluate both sides
> of this equation, and hence show that S_D = 2π^(D/2)/Γ(D/2) (1.143). Next, by integrating with
> respect to radius from 0 to 1, show that the volume of the unit sphere in D dimensions is
> given by V_D = S_D/D (1.144). Finally, use the results Γ(1) = 1 and Γ(3/2) = √π/2 to show that
> (1.143) and (1.144) reduce to the usual expressions for D = 2 and D = 3.

Handwritten original: [`ex-1.18-handwritten-1.png`](ex-1.18-handwritten-1.png),
[`ex-1.18-handwritten-2.png`](ex-1.18-handwritten-2.png),
[`ex-1.18-handwritten-3.png`](ex-1.18-handwritten-3.png) (V_D, D = 2/3).

### Reading (1.142)

e^(−x²) is chosen by Bishop: it is the 1.7 Gaussian with σ² = ½ (2σ² = 1), it factorises in
Cartesian coordinates and depends only on r in polar ones. Right side = sum over shells of
(function value on the shell, e^(−r²)) × (shell volume, S_D r^(D−1) dr). S_D is the area at
radius 1 (a number for each D, like 2π in 1.7's 2πr dr); r^(D−1) carries the radius, because a
sphere's surface is (D−1)-dimensional.

### Part 1: S_D (done)

S_D = ∏ ∫ e^(−xᵢ²) dxᵢ / ∫₀^∞ e^(−r²) r^(D−1) dr.

Numerator: each factor is (1.126) with σ² = ½, √(2π·½) = √π; D factors give π^(D/2).

Denominator: u = r², du = 2r dr, limits stay 0..∞; r^(D−1)/(2r) = ½u^((D−2)/2), so
½ ∫₀^∞ e^(−u) u^(x−1) du with x − 1 = (D−2)/2, x = D/2: ½Γ(D/2).

S_D = 2π^(D/2)/Γ(D/2). ∎ (1.143)

### Part 2: V_D = S_D/D

Volume = sum of thin shells (2D: rings, ∫₀¹ 2πr dr = π; 3D: ∫₀¹ 4πr² dr = 4π/3). Shell area at
radius r is S_D r^(D−1): the sphere's surface has D−1 directions (the D−1 angles), and stretching
a k-dimensional piece by r multiplies its size by r^k. So

V_D = ∫₀¹ S_D r^(D−1) dr = S_D [r^D/D]₀¹ = S_D/D. ∎ (1.144)

(First attempt had ∫₀¹ S_D dr with r^(D−1) missing; corrected on page 3.)

### Part 3: D = 2, 3

S₂ = 2π/Γ(1) = 2π, V₂ = π (πR²). S₃ = 2π^(3/2)/(√π/2) = 4π, V₃ = 4π/3 ((4/3)πR³).
Page slip: wrote 4π² (π^(3/2)/π^(1/2) = π¹).

### Review notes (professor)

His own: the substitution u = r², dr = du/2r, the exponent bookkeeping, matching to Γ. Stuck
points: where e^(−x²) comes from (chosen, = 1.7 with σ² = ½; "no need to integrate again");
"unit radius" and why S_D is a constant while the area depends on r (dummy variables integrated
away; 2π in 1.7's 2πr dr). My explanations were tangled here; the 1.7 ring analogy was what
worked.

In practice: the r^(D−1) shell factor is what 1.20 is about. For a D-dim Gaussian,
p(r) ∝ S_D r^(D−1) e^(−r²/2σ²): the shell volume beats the density, mass sits at r ≈ √D σ, not
near the mean. High-dim data (embeddings) is all roughly equidistant; distance-based methods
degrade.

## Exercise 1.19 (\*\*)

> Consider a sphere of radius a in D-dimensions together with the concentric hypercube of side
> 2a, so that the sphere touches the hypercube at the centres of each of its sides. By using the
> results of Exercise 1.18, show that the ratio of the volume of the sphere to the volume of the
> cube is given by volume of sphere / volume of cube = π^(D/2) / (D 2^(D−1) Γ(D/2)) (1.145).
> Now make use of Stirling's formula in the form Γ(x + 1) ≃ (2π)^(1/2) e^(−x) x^(x+1/2) (1.146),
> which is valid for x ≫ 1, to show that, as D → ∞, the ratio (1.145) goes to zero. Show also
> that the ratio of the distance from the centre of the hypercube to one of the corners, divided
> by the perpendicular distance to one of the sides, is √D, which therefore goes to ∞ as D → ∞.
> From these results we see that, in a space of high dimensionality, most of the volume of a
> cube is concentrated in the large number of corners, which themselves become very long
> "spikes"!

Handwritten original: [`ex-1.19-handwritten-1.png`](ex-1.19-handwritten-1.png),
[`ex-1.19-handwritten-2.png`](ex-1.19-handwritten-2.png).

Triage exercise, solved anyway.

**Ratio.** Sphere V_D a^D = (S_D/D) a^D (volume is D-dimensional, so a^D); cube (2a)^D (side =
diameter). a cancels: S_D/(D 2^D) = π^(D/2)/(D 2^(D−1) Γ(D/2)). ∎ (1.145)

**Limit.** Numerator (√π)^D is exponential; denominator has Γ(D/2) ≈ (D/2)!, and a factorial
beats any exponential (Stirling: x^x vs c^x). Ratio → 0.

**Corner.** Corner at (a, …, a): distance √(D a²) = √D a (Pythagoras, one a² per dimension);
face at distance a. Ratio √D. ∎

### Review notes (professor)

First try: a^(D−1) (surface scaling) and cube side a; fixed after the hint (a must cancel).
Then expanded Stirling with x + 1 = D/2 and turned the product (2x+2)·2^(2x+1) into a sum —
unnecessary; the growth-rate comparison is the argument. Corner distance his own, from the 2D
sketch.

In practice: uniform samples in a box [−a, a]^D almost never land near the centre; in high D
"typical" points are near the corners, at distance ≈ √D·a/√3, not a.

## Exercise 1.20 (\*\*) www

> In this exercise, we explore the behaviour of the Gaussian distribution in high-dimensional
> spaces. Consider a Gaussian distribution in D dimensions given by
> p(**x**) = 1/(2πσ²)^(D/2) exp(−‖**x**‖²/2σ²) (1.147). We wish to find the density with respect
> to radius in polar coordinates in which the direction variables have been integrated out. To
> do this, show that the integral of the probability density over a thin shell of radius r and
> thickness ε, where ε ≪ 1, is given by p(r)ε where
> p(r) = S_D r^(D−1)/(2πσ²)^(D/2) exp(−r²/2σ²) (1.148), where S_D is the surface area of a unit
> sphere in D dimensions. Show that the function p(r) has a single stationary point located, for
> large D, at r̂ ≃ √D σ. By considering p(r̂ + ε) where ε ≪ r̂, show that for large D,
> p(r̂ + ε) = p(r̂) exp(−3ε²/2σ²) (1.149), which shows that r̂ is a maximum of the radial
> probability density and also that p(r) decays exponentially away from its maximum at r̂ with
> length scale σ. We have already seen that σ ≪ r̂ for large D, and so we see that most of the
> probability mass is concentrated in a thin shell at large radius. Finally, show that the
> probability density p(**x**) is larger at the origin than at the radius r̂ by a factor of
> exp(D/2). We therefore see that most of the probability mass in a high-dimensional Gaussian
> distribution is located at a different radius from the region of high probability density.

Context: the derivation behind Figure 1.23.

Handwritten original: [`ex-1.20-handwritten-1.png`](ex-1.20-handwritten-1.png) (shell volume),
[`ex-1.20-handwritten-2.png`](ex-1.20-handwritten-2.png) (‖x‖² = r²),
[`ex-1.20-handwritten-3.png`](ex-1.20-handwritten-3.png) (peak),
[`ex-1.20-handwritten-4.png`](ex-1.20-handwritten-4.png) (part C setup, Taylor),
[`ex-1.20-handwritten-5.png`](ex-1.20-handwritten-5.png) (part C result, part D).

### Part A: p(r) (his work, last line to fix)

Shell volume, derived himself: V(r+ε) − V(r) = (S_D/D)[(r+ε)^D − r^D] ≈ S_D r^(D−1) ε
(derivative of r^D; first try differentiated r^(D−1)).

p(**x**) depends only on r, so it is constant on the shell:
∫_shell p(**x**) d**x** = p(**x**)|_r · S_D r^(D−1) ε = p(r)·ε with p(r) as in (1.148).

Page's last line has p(r)·V instead of p(**x**)·V, so S_D r^(D−1) appears twice.

Densities: p(**x**) per unit volume (no S_D); p(r) per unit radius (= p(**x**) × shell surface);
p(r)ε = probability. 2D numbers (σ = 1, r = 1, ε = 0.01): 0.0965 × 0.0628 = 0.606 × 0.01 ≈ 0.0061.

### Part B: the peak r̂ (his own)

p'(r) = S_D/(2πσ²)^(D/2) [(D−1) r^(D−2) e^(−r²/2σ²) + r^(D−1) e^(−r²/2σ²)(−r/σ²)]
= S_D/(2πσ²)^(D/2) · r^(D−2) e^(−r²/2σ²) · [(D−1) − r²/σ²].

r^(D−2) = 0 only at r = 0 (where p = 0); the exponential is never 0. So (D−1) − r²/σ² = 0:
r̂ = σ√(D−1) ≈ √D σ for large D. ∎ Single stationary point for r > 0.

Slip: after factoring, wrote (1 − r²/σ²): dropped the D−1 (and first r in r^D/r^(D−2) = r²).

### Part C: width of the peak

Substitute r = r̂ + ε, expand (r̂ + ε)² = r̂² + 2r̂ε + ε², and pull r̂ out of the power:
(r̂ + ε)^(D−1) = r̂^(D−1)(1 + ε/r̂)^(D−1). The factors S_D r̂^(D−1) e^(−r̂²/2σ²)/(2πσ²)^(D/2)
make exactly p(r̂):

p(r̂ + ε) = p(r̂) · (1 + ε/r̂)^(D−1) · e^(−r̂ε/σ²) · e^(−ε²/2σ²).

Write (1 + ε/r̂)^(D−1) = exp((D−1) ln(1 + ε/r̂)) and use Taylor at 0:
ln(1 + x) ≈ x − x²/2 (f(0) = 0, f'(0) = 1, f''(0) = −1). With D − 1 = r̂²/σ² (part B):

(D−1)(ε/r̂ − ε²/2r̂²) − r̂ε/σ² − ε²/2σ² = r̂ε/σ² − ε²/2σ² − r̂ε/σ² − ε²/2σ² = −ε²/σ².

The linear terms cancel (r̂ is the peak). So p(r̂ + ε) = p(r̂) exp(−ε²/σ²). ∎

**Book erratum:** (1.149) prints exp(−3ε²/2σ²); the official errata correct it to exp(−ε²/σ²)
(see README). Check: d²/dr² ln p at r̂ = −(D−1)/r̂² − 1/σ² = −2/σ².

Meaning: e^(−ε²/σ²) is 0.37 at ε = σ, 0.02 at 2σ: the shell is a few σ wide at radius √D σ, so
thin relative to its radius (σ ≪ r̂ because r̂/σ ≈ √D).

### Part D: density at the centre vs at r̂

p(**x**) depends only on r: p(0) = 1/(2πσ²)^(D/2), p at r̂ = p(0)·exp(−r̂²/2σ²). With r̂² ≈ Dσ²:
p(0)/p(r̂) = exp(D/2). ∎ (D = 100: e^50 ≈ 5·10²¹.)

So the region of highest density (the centre) is not where the mass is (the shell at √D σ).

### Review notes (professor)

Hard exercise for him; parts B–D mostly his own. Part A: confused p(**x**) (per volume) with p(r)
(per radius) and multiplied p(r) by the shell volume again; resolved by stating the task ("derive
(1.148) from (1.147)") and ‖x‖² = r². Taylor series was new: learned it here (value, slope,
curvature at a point; why expand at 0). Slips: dropped D−1 when factoring in part B; sign in
(r̂ + ε)² and a stray (2πσ²)^(D/2) in C. His C result (2 instead of the book's 3) is correct.
Gap to fill later: change of variables for densities (Гнеденко, functions of random variables;
Bishop §1.2.1 (1.27); his own Ex. 1.4).

What part A taught: probability of distance r = density × amount of space at r. Density peaks
at the centre, space (S_D r^(D−1)) is zero there, so p(r = 0) = 0: the most likely point is not
the most likely distance.

ML: latent vectors in VAEs/diffusion (D ~ 100–1000) live on the shell ‖z‖ ≈ √D; linear
interpolation between two latents passes inside the shell, hence slerp.

## Exercise 1.21 (\*\*)

> Consider two nonnegative numbers a and b, and show that, if a ≤ b, then a ≤ (ab)^(1/2). Use
> this result to show that, if the decision regions of a two-class classification problem are
> chosen to minimize the probability of misclassification, this probability will satisfy
> p(mistake) ≤ ∫ {p(x, C₁) p(x, C₂)}^(1/2) dx (1.150).

Context: §1.5.1, minimizing the misclassification rate.

Handwritten original: [`ex-1.21-handwritten.png`](ex-1.21-handwritten.png).

### Part 1: a ≤ (ab)^(1/2)

Multiply a ≤ b by a ≥ 0: a² ≤ ab. Take the square root of both sides (√ is increasing on
non-negative numbers): a ≤ (ab)^(1/2). ∎ (He also did it by cases: b = a gives equality, b > a
gives ab > a².)

### Part 2: the bound (1.150)

1. By (1.78): p(mistake) = ∫_{R₁} p(x, C₂) dx + ∫_{R₂} p(x, C₁) dx.
2. Minimizing p(mistake) puts each x in the class with the larger p(x, C_k). So in R₁ the
   integrand p(x, C₂) is the smaller of the two, and likewise in R₂:
   p(mistake) = ∫ min(p(x, C₁), p(x, C₂)) dx.
3. At every x take a = min, b = max (both ≥ 0, densities). Part 1 gives
   min ≤ (min · max)^(1/2) = (p(x, C₁) p(x, C₂))^(1/2).
4. If f(x) ≤ g(x) at every x then ∫ f dx ≤ ∫ g dx, which gives (1.150). ∎

### Review notes (professor)

Correct, standard structure, his own. Fixed after review: a crossed-out line
max(p₁, p₂) = 1 − min(p₁, p₂). That holds for posteriors p(C_k|x), which sum to 1 at each x; for
joint densities p(x, C₁) + p(x, C₂) = p(x) (sum rule), and only the double sum/integral over
classes *and* x equals 1. Also added: a, b ≥ 0 (not > 0) and the "integrate a pointwise
inequality" step.

ML: ∫ (p₁ p₂)^(1/2) dx is the Bhattacharyya coefficient, a measure of how much the two classes
overlap. It bounds the Bayes error: the error no classifier can beat.

## Exercise 1.22 (\*) www

> Given a loss matrix with elements L_kj, the expected risk is minimized if, for each x, we
> choose the class that minimizes (1.81). Verify that, when the loss matrix is given by
> L_kj = 1 − I_kj, where I_kj are the elements of the identity matrix, this reduces to the
> criterion of choosing the class having the largest posterior probability. What is the
> interpretation of this form of loss matrix?

Context: §1.5.2, minimizing the expected loss. (1.80): E[L] = Σ_k Σ_j ∫_{R_j} L_kj p(x, C_k) dx;
(1.81): choose the j that minimizes Σ_k L_kj p(C_k|x). "Risk" = expected loss.

Handwritten original: [`ex-1.22-handwritten-1.png`](ex-1.22-handwritten-1.png) (loss matrix,
(1.80) in his words), [`ex-1.22-handwritten-2.png`](ex-1.22-handwritten-2.png) (solution).

### Solution

L = 1 − I: zeros on the diagonal, ones everywhere else. For a candidate class j:

Σ_k L_kj p(C_k|x) = p(C_1|x) + … + 0 · p(C_j|x) + … + p(C_K|x) = Σ_{k≠j} p(C_k|x) = 1 − p(C_j|x),

because the posteriors at a fixed x sum to 1. The larger p(C_j|x), the smaller 1 − p(C_j|x),
so minimizing (1.81) picks the class with the largest posterior. ∎

Interpretation: every mistake costs the same (1), a correct decision costs 0 (the "0-1 loss").
Then E[L] = p(mistake), so §1.5.2 reduces to §1.5.1, minimizing the misclassification rate
(the quantity bounded in Ex. 1.21).

### Review notes (professor)

His own after two hints (the expansion; "the posteriors sum to 1"). Early confusion: thought
minimizing E[L] changes the probabilities. It doesn't: the model is fixed after the inference
stage (which includes training); the decision stage only chooses the regions R_j. His own
point: x ∈ R_j is a decision, not the true label, which is why (1.80) sums over every true
class k. The last line on the page reads "≤" overwritten; it is an equality.

In practice: argmax of a classifier's softmax output is the optimal decision only under 0-1
loss. With unequal costs (fraud, medical) the decision threshold moves away from 0.5.

## Exercise 1.23 (\*) (skipped, plan correct)

> Derive the criterion for minimizing the expected loss when there is a general loss matrix
> and general prior probabilities for the classes.

"General" = any L_kj and any p(C_k), kept as symbols.

Plan (his, with one fix; written down from his spoken explanation on 25.10.2026, no
handwritten page):

1. (1.81): for each x choose the j that minimizes Σ_k L_kj p(C_k|x). In matrix form: the row
   of posteriors (1×K) times L (K×K) gives the expected cost of each choice j; take the argmin.
   For a data set, P (N×K) @ L, argmin per row.
2. Bayes: p(C_k|x) = p(x|C_k) p(C_k) / p(x).
3. p(x) is the same positive factor for every candidate j, so dividing by it doesn't change
   which j is smallest. Drop it. The priors p(C_k) depend on k and stay inside the sum.
4. Rule: choose the j that minimizes Σ_k L_kj p(x|C_k) p(C_k).

Step 1 alone is already a complete answer (the priors are inside the posteriors). Steps 2–4
give the equivalent form with the priors visible, which the wording "general prior
probabilities" hints at. Its use: if the class balance changes (e.g. the fraud rate in
production differs from training), swap in the new p(C_k) and keep p(x|C_k), without
retraining (generative approach, §1.5.4).

Review note: he first said p(x) can be dropped "because the data are i.i.d."; the actual reason
is step 3 (a common positive factor doesn't change an argmin).

## Exercise 1.24 (\*\*) www (skipped, plan correct)

> Consider a classification problem in which the loss incurred when an input vector from class
> C_k is classified as belonging to class C_j is given by the loss matrix L_kj, and for which
> the loss incurred in selecting the reject option is λ. Find the decision criterion that will
> give the minimum expected loss. Verify that this reduces to the reject criterion discussed in
> Section 1.5.3 when the loss matrix is given by L_kj = 1 − I_kj. What is the relationship
> between λ and the rejection threshold θ?

Context: §1.5.3, the reject option (Fig. 1.26): reject if max_k p(C_k|x) < θ. There θ is
chosen by hand; this exercise derives it from costs.

Plan (his; written down from his spoken explanation on 25.10.2026, no handwritten page):

1. At each x there are K + 1 options: say C_j, expected cost Σ_k L_kj p(C_k|x) (as in 1.22),
   or reject, cost λ whatever the true class. Pick the cheapest:
   reject if min_j Σ_k L_kj p(C_k|x) > λ, otherwise the argmin class.
2. With L = 1 − I the best class costs 1 − max_k p(C_k|x) (Ex. 1.22).
3. Reject if 1 − max_k p(C_k|x) > λ, i.e. max_k p(C_k|x) < 1 − λ. So θ = 1 − λ.

Review notes: first misread λ as the cost of every mistake; it is the cost of the extra action
"reject". He said "priors × L" where posteriors are meant (with the unnormalised
p(x|C_k)p(C_k) of 1.23, the reject cost would have to be λ·p(x)).

## Exercise 1.25 (\*) www

> Consider the generalization of the squared loss function (1.87) for a single target variable
> t to the case of multiple target variables described by the vector **t** given by
> E[L(**t**, **y**(**x**))] = ∫∫ ‖**y**(**x**) − **t**‖² p(**x**, **t**) d**x** d**t** (1.151).
> Using the calculus of variations, show that the function **y**(**x**) for which this
> expected loss is minimized is given by **y**(**x**) = E_**t**[**t**|**x**]. Show that this
> result reduces to (1.89) for the case of a single target variable t.

Context: §1.5.5, loss functions for regression; Appendix D, calculus of variations.

Handwritten original: [`ex-1.25-handwritten.png`](ex-1.25-handwritten.png).

### Solution

Same route as the book's (1.87) → (1.88) → (1.89), with a vector target.

1. ‖**y** − **t**‖² = Σ_i (y_i − t_i)², so its derivative with respect to the vector **y** is
   2(**y** − **t**), component by component. Vector version of (1.88):
   δE/δ**y**(**x**) = 2 ∫ (**y**(**x**) − **t**) p(**x**, **t**) d**t** = 0.
2. **y**(**x**) does not depend on **t**, so it leaves the integral, and ∫ p(**x**, **t**) d**t**
   = p(**x**) (sum rule): **y**(**x**) p(**x**) = ∫ **t** p(**x**, **t**) d**t**.
3. **y**(**x**) = ∫ **t** p(**x**, **t**) d**t** / p(**x**) = ∫ **t** p(**t**|**x**) d**t**
   = E_**t**[**t**|**x**]. ∎
4. With one component, **t** = t and this is (1.89).

### Review notes (professor)

His page has the result (step 3) and "δE/δ**y**(**x**) = 0"; he saw that it is the book's
derivation with a bold **t**. Steps 1–2 (what the derivative equals, and the sum rule that turns
∫ p(**x**, **t**) d**t** into p(**x**)) are filled in here. Each component is a separate scalar
problem, so the vector case adds nothing beyond the derivative of the squared norm.

In practice: training a regression net with MSE makes it predict the conditional mean of the
target, per output, for any number of outputs.

## Exercise 1.26 (\*)

> By expansion of the square in (1.151), derive a result analogous to (1.90) and hence show that
> the function **y**(**x**) that minimizes the expected squared loss for the case of a vector
> **t** of target variables is again given by the conditional expectation of **t**.

Context: §1.5.5, the decomposition (1.90). Same answer as Ex. 1.25, by algebra instead of
calculus of variations, and it also shows what part of the loss no model can remove.

Handwritten original: [`ex-1.26-handwritten-1.png`](ex-1.26-handwritten-1.png) (expansion),
[`ex-1.26-handwritten-2.png`](ex-1.26-handwritten-2.png) (the three terms),
[`ex-1.26-handwritten-3.png`](ex-1.26-handwritten-3.png) (third term = variance, final form).

### Solution

Write E for E[**t**|**x**]. Add and subtract it inside the norm and expand the brackets:

‖**y** − **t**‖² = ‖**y** − E‖² + 2(**y** − E)ᵀ(E − **t**) + ‖E − **t**‖².

Integrate each term against p(**x**, **t**) over **t** (x held fixed):

1. ‖**y**(**x**) − E‖² has no **t**: it leaves the integral and ∫ p(**x**, **t**) d**t** = p(**x**)
   (sum rule). Gives ∫ ‖**y**(**x**) − E‖² p(**x**) d**x**.
2. 2(**y** − E)ᵀ has no **t** and leaves the integral; the rest is
   ∫ (E − **t**) p(**x**, **t**) d**t** = E·p(**x**) − ∫ **t** p(**x**, **t**) d**t**
   = E·p(**x**) − E·p(**x**) = 0, using ∫ **t** p(**x**, **t**) d**t** = p(**x**)·E[**t**|**x**]
   (product rule p(**x**, **t**) = p(**t**|**x**)p(**x**); his line from Ex. 1.25).
3. ∫ ‖E − **t**‖² p(**x**, **t**) d**t** = p(**x**)(E[‖**t**‖²|**x**] − ‖E‖²): the conditional
   variance (summed over components), by (1.40).

So, the vector form of (1.90):

E[L] = ∫ ‖**y**(**x**) − E[**t**|**x**]‖² p(**x**) d**x** + ∫ var[**t**|**x**] p(**x**) d**x**.

Only the first term depends on **y**; it is ≥ 0 and equals 0 exactly when
**y**(**x**) = E[**t**|**x**]. ∎ The second term is the noise in **t**: no model can reduce it.

### Review notes (professor)

His own derivation, with three fixes on the way: the sign of the last term after expanding
(he had −, it is +); he integrated a t-free factor as "∫ … dt = t · …", forgetting that
p(**x**, **t**) depends on **t** (his own question "why am I not considering t inside the
probability?" was the answer); and why ∫ **t** p(**x**, **t**) d**t** = p(**x**)E[**t**|**x**]
(p(**x**, **t**) is not a density of **t**; the product rule turns it into p(**t**|**x**)p(**x**)).
He expanded the third term himself and got E[t²|x] − E[t|x]², i.e. the variance; for the
minimization that term doesn't need simplifying, since it has no **y**.

Book slip, found by him: the printed (1.90) has the second term as ∫ {E[t|x] − t}² p(x) dx,
with t still present after "performing the integral over t". It should be
∫∫ {E[t|x] − t}² p(x, t) dx dt = ∫ var[t|x] p(x) dx, as derived above and as the text right
after (1.90) describes it ("the variance of the distribution of t, averaged over x"). Not in
the official errata (2011).

On page 3 the final line drops p(x) from the second integral: it is ∫ var[t|x] p(x) dx (each
of his three pieces carries the factor p(x)).

What the decomposition teaches (beyond "the minimum is E[t|x]"): the loss of any model =
(how far it is from E[t|x], weighted by p(x), so errors count where data is dense) + (noise
floor ∫ var[t|x] p(x) dx). The floor depends only on the data, not the model: more capacity
can't lower it; only more informative inputs x can (they shrink var[t|x]). The same
add-and-subtract trick gives the bias–variance decomposition in §3.2.

In practice: the irreducible noise term is why validation MSE never reaches zero; it's the
regression analogue of the Bayes error in Ex. 1.21.

# PRML Chapter 1 — Introduction

Source: Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 1, exercises 1.1–1.41.

## Solved

- 1.1 (30.09.2026)
- 1.3 (14.10.2026)
- 1.4 (17.10.2026)
- 1.5 (17.10.2026)
- 1.6 (17.10.2026)
- 1.7 (18.10.2026)
- 1.8 (19.10.2026)
- 1.10 (20.10.2026)
- 1.11 (20.10.2026)

Skipped, plan correct: 1.9 (mode of the Gaussian: dN/dx = 0 gives x = μ).

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

## Exercise 1.12 (\*\*) www (in progress)

> Using the results (1.49) and (1.50), show that
>
> E[xₙxₘ] = μ² + Iₙₘσ²   (1.130)
>
> where xₙ and xₘ denote data points sampled from a Gaussian distribution with mean μ and
> variance σ², and Iₙₘ satisfies Iₙₘ = 1 if n = m and Iₙₘ = 0 otherwise. Hence prove the results
> (1.57) and (1.58).

(1.57): E[μ_ML] = μ. (1.58): E[σ²_ML] = ((N−1)/N)·σ².

Handwritten original, so far: [`ex-1.12-handwritten-1.png`](ex-1.12-handwritten-1.png) (setup),
[`ex-1.12-handwritten-2.png`](ex-1.12-handwritten-2.png) (E[μ_ML]). Below is a transcription.

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

# PRML Chapter 1 — Introduction

Source: Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 1, exercises 1.1–1.41.

## Solved

- 1.1 (30.09.2026)
- 1.3 (14.10.2026)
- 1.4 (17.10.2026)
- 1.5 (17.10.2026)

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

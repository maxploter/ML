# PRML Chapter 1 — Introduction

Source: Bishop, *Pattern Recognition and Machine Learning* (2006), Chapter 1, exercises 1.1–1.41.

## Solved

- 1.1 (30.09.2026)

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

Handwritten original: [`ex-1.1-handwritten.jpg`](ex-1.1-handwritten.jpg). Below is a transcription
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

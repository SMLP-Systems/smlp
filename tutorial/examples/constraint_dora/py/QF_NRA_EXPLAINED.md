# QF_NRA Logic - Quantifier-Free Nonlinear Real Arithmetic

## What is QF_NRA?

**QF_NRA** stands for **Quantifier-Free Nonlinear Real Arithmetic**. It's a logical theory used in SMT (Satisfiability Modulo Theories) solvers to describe and solve problems involving real numbers and nonlinear constraints.

Let's break down each part:

### Q - Quantifier
**Quantifier-Free** means the logic doesn't use quantifiers like:
- ∀ (for all)
- ∃ (there exists)

Instead of: "∀x, x² ≥ 0" (for all x, x squared is non-negative)
We write: "x² ≥ 0" (x squared is non-negative, for some specific x)

### NRA - Nonlinear Real Arithmetic

**Real Arithmetic:** Operations on real numbers (not just integers)
- Variables: x, y, z ∈ ℝ (real numbers like 3.14, -2.5, √2)
- Operations: +, -, ×, ÷

**Nonlinear:** Includes multiplication of variables
- Linear: 2x + 3y ≤ 5 ✓
- Nonlinear: x·y ≤ 5 ✓
- Nonlinear: x² + y² ≤ 1 ✓

## Examples in QF_NRA

### Your Optimization Problem

Your problem is a perfect example of QF_NRA:

```
Minimize: (x1 - 2)² + (x2 - 1)²
Subject to: x1² + x2² ≤ 1
```

This is QF_NRA because:
- ✅ Real numbers (x1, x2 ∈ ℝ)
- ✅ Nonlinear (x1², x2², multiplication)
- ✅ No quantifiers (no ∀ or ∃)

### More Examples

**Example 1: Circle Constraint**
```
x² + y² ≤ 1
```
This describes all points inside or on a unit circle.

**Example 2: Polynomial Constraint**
```
x³ - 2x² + x - 5 ≤ 0
```
A cubic polynomial constraint.

**Example 3: Product Constraint**
```
x·y ≥ 10 ∧ x + y ≤ 20
```
Product of two variables with a linear constraint.

**Example 4: Optimization**
```
x² + y² + z² ≤ 100
x·y·z = 1000
```
Nonlinear constraints in 3D.

## What QF_NRA Can Express

### ✅ Supported Operations

**Basic Arithmetic:**
- Addition: x + y
- Subtraction: x - y
- Multiplication: x · y, x²
- Division: x / y (when y ≠ 0)

**Comparisons:**
- Less than: x < y
- Less or equal: x ≤ y
- Greater than: x > y
- Greater or equal: x ≥ y
- Equal: x = y
- Not equal: x ≠ y

**Boolean Logic:**
- AND: φ₁ ∧ φ₂
- OR: φ₁ ∨ φ₂
- NOT: ¬φ

**Examples:**
```
(x² + y² ≤ 1) ∧ (x ≥ 0) ∧ (y ≥ 0)  // First quadrant of unit circle
(x·y = 12) ∨ (x + y = 7)            // Either product is 12 or sum is 7
```

### ❌ Not Supported (Requires Different Logics)

**Transcendental Functions:**
- sin(x), cos(x), tan(x)
- exp(x), log(x)
- √x (sometimes supported as x^(1/2))

**Integer Constraints:**
- x ∈ ℤ (would need QF_NIA - Quantifier-Free Nonlinear Integer Arithmetic)

**Quantifiers:**
- ∀x, φ(x) (would need NRA without the QF)

## Related Logics

Understanding the landscape of SMT logics:

### Linear vs Nonlinear

| Logic | Name | Variables Multiply? | Example |
|-------|------|---------------------|---------|
| QF_LRA | Linear Real Arithmetic | ❌ No | 2x + 3y ≤ 5 |
| QF_NRA | Nonlinear Real Arithmetic | ✅ Yes | x² + y² ≤ 1 |

### Real vs Integer

| Logic | Name | Number Type | Example |
|-------|------|-------------|---------|
| QF_NRA | Nonlinear Real | ℝ (reals) | x² ≤ 2 |
| QF_NIA | Nonlinear Integer | ℤ (integers) | x² ≤ 2 ∧ x ∈ ℤ |

### Quantifier-Free vs Quantified

| Logic | Name | Has ∀/∃? | Example |
|-------|------|----------|---------|
| QF_NRA | Quantifier-Free NRA | ❌ No | x² + y² ≤ 1 |
| NRA | Full NRA | ✅ Yes | ∀x, x² ≥ 0 |

### Common SMT Logics

```
QF_LRA  - Quantifier-Free Linear Real Arithmetic
QF_NRA  - Quantifier-Free Nonlinear Real Arithmetic  ← Your problem!
QF_LIA  - Quantifier-Free Linear Integer Arithmetic
QF_NIA  - Quantifier-Free Nonlinear Integer Arithmetic
QF_BV   - Quantifier-Free Bit-Vectors
QF_UF   - Quantifier-Free Uninterpreted Functions
```

## Why It Matters for Your Problem

### Your Problem Uses QF_NRA

```python
# Objective: (x1 - 2)² + (x2 - 1)²
# Constraint: x1² + x2² ≤ 1

# In SMT terms:
x1, x2 are Real variables
Constraint: (x1 * x1) + (x2 * x2) <= 1.0  # Nonlinear!
```

This is clearly nonlinear (variables multiplied by themselves), so you need QF_NRA support.

### Why MathSAT Failed

Your MathSAT gave this error:
```
pysmt.exceptions.NonLinearError: ((x2 + -1.0) * (x2 + -1.0))
```

This means your MathSAT installation only supports **QF_LRA** (linear), not **QF_NRA** (nonlinear).

To support QF_NRA, MathSAT needs to be compiled with MPFR library support for handling nonlinear arithmetic.

## How Solvers Handle QF_NRA

### Z3's Approach

Z3 uses multiple techniques:
- **Interval arithmetic**: Bounds on variables
- **Cylindrical algebraic decomposition (CAD)**: Exact solutions
- **Numerical methods**: Approximations when exact is hard

### MathSAT's Approach

MathSAT (with MPFR) uses:
- **DPLL(T)**: Combines SAT solving with theory reasoning
- **MPFR library**: Arbitrary-precision floating-point arithmetic
- **Linear approximations**: Linearize nonlinear constraints

### Difficulty

QF_NRA is **undecidable** in general, meaning:
- ❌ No algorithm can solve all QF_NRA problems
- ⚠️ Solvers may not terminate for some inputs
- ⚠️ Solutions may be approximate, not exact

But for practical problems (like yours), modern solvers work very well!

## Practical Example: Your Problem in SMT-LIB Format

```smt2
; SMT-LIB 2.0 format
(set-logic QF_NRA)  ; Declare we're using QF_NRA

; Declare variables
(declare-fun x1 () Real)
(declare-fun x2 () Real)

; Constraint: x1² + x2² ≤ 1
(assert (<= (+ (* x1 x1) (* x2 x2)) 1.0))

; Objective: minimize (x1-2)² + (x2-1)²
; (SMT solvers don't do optimization directly, but can check feasibility)
(assert (<= (+ (* (- x1 2.0) (- x1 2.0)) 
               (* (- x2 1.0) (- x2 1.0))) 
            1.6))  ; Check if objective can be ≤ 1.6

(check-sat)     ; Is there a solution?
(get-model)     ; If yes, what is it?
```

## Summary

**QF_NRA** = Quantifier-Free Nonlinear Real Arithmetic

**Key Points:**
- ✅ Real numbers (not integers)
- ✅ Variables can multiply: x·y, x²
- ✅ No ∀ or ∃ quantifiers
- ⚠️ Harder than linear (QF_LRA)
- ⚠️ May be approximate
- ✅ Perfect for your optimization problem!

**Why you care:**
- Your problem requires QF_NRA support
- Z3 supports it natively ✓
- MathSAT needs MPFR compilation for it
- Without QF_NRA support, you get NonLinearError

Your optimization problem is a textbook example of QF_NRA!

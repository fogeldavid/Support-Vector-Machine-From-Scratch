# Support Vector Machine From Scratch

> **MGSC 696 — Machine Learning** | McGill MMA
> Soft-margin SVM with kernels, implemented from the dual QP and benchmarked against scikit-learn

---

## Project Overview

This project implements a binary Support Vector Machine from scratch — no SVM library — using the **dual formulation** with a **soft margin** (slack variables) and support for **linear, polynomial, and RBF kernels**. The dual is a convex quadratic program, which we hand to the `cvxopt` QP solver (a general-purpose optimizer, not an SVM package). The implementation is then validated against `sklearn.svm.SVC` on the Breast Cancer Wisconsin dataset (569 samples, 30 features, malignant vs. benign).

**Primal (soft-margin) problem:**

$$\min_{w, b, \xi} \; \tfrac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.} \quad y_i(w^\top \phi(x_i) + b) \geq 1 - \xi_i,\; \xi_i \geq 0$$

**Dual problem (what we actually solve):**

$$\max_\alpha \; \sum_{i=1}^{n}\alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C,\; \sum_i \alpha_i y_i = 0$$

The kernel trick replaces $\phi(x_i)^\top \phi(x_j)$ with $K(x_i, x_j)$, letting the model operate implicitly in high-dimensional feature spaces without ever forming $\phi$.

### Key Results

**3 kernels, from-scratch vs. sklearn (C = 1.0, 426 train / 143 test):**

| Kernel | Source | Test Accuracy | # Support Vectors | Train Time (s) |
|--------|--------|---------------|-------------------|----------------|
| **Linear** | **scratch** | **0.9860** | **34 / 426** | **0.078** |
| Linear | sklearn | 0.9860 | 34 / 426 | 0.003 |
| Polynomial (d=3) | scratch | 0.9650 | 63 / 426 | 0.019 |
| Polynomial (d=3) | sklearn | 0.9720 | 65 / 426 | 0.001 |
| RBF (γ=0.03) | scratch | 0.9790 | 95 / 426 | 0.012 |
| RBF (γ=0.03) | sklearn | 0.9790 | 92 / 426 | 0.001 |

**Why the linear kernel wins?** After standardization, the two classes are very nearly linearly separable in 30 dimensions, so the extra flexibility of RBF and polynomial kernels buys nothing and costs support vectors — the linear model needs only 34 of 426 training points to define its boundary, roughly a third of what RBF requires.

**Prediction agreement with sklearn:**

| Kernel | Agreement | Test points matched |
|--------|-----------|---------------------|
| Linear | 1.0000 | 143 / 143 |
| Polynomial | 0.9930 | 142 / 143 |
| RBF | 1.0000 | 143 / 143 |

Two completely different optimization algorithms — a general interior-point QP solver and libsvm's SMO — converge on the same classifier. The single polynomial disagreement is a numerical-tolerance artifact, not a formulation error.

### Best Model — Detailed Performance (Linear Kernel)

|  | precision | recall | f1-score | support |
|--|-----------|--------|----------|---------|
| malignant (−1) | 0.98 | 0.98 | 0.98 | 53 |
| benign (+1) | 0.99 | 0.99 | 0.99 | 90 |
| **accuracy** | | | **0.99** | **143** |

**Confusion matrix:**

| | pred. malignant | pred. benign |
|--|--|--|
| **actual malignant** | 52 | 1 |
| **actual benign** | 1 | 89 |

Accuracy alone is the wrong lens for a medical dataset — a false negative (a missed malignant tumor) is far costlier than a false positive. The model produces exactly one of each, giving 0.98 recall on the malignant class.

### Effect of C on the Margin

| C | # Support Vectors | Test Accuracy |
|---|-------------------|---------------|
| 0.01 | 322 | 0.6294 |
| 0.1 | 189 | 0.9371 |
| 0.5 | 116 | 0.9720 |
| 1.0 | 95 | 0.9790 |
| **5.0** | **77** | **0.9860** |
| 10.0 | 74 | 0.9790 |
| 100.0 | 67 | 0.9441 |

The regularization sweep traces the bias-variance tradeoff exactly as theory predicts: low $C$ produces a very soft margin where 75% of the training set becomes support vectors and the model underfits badly (62.9% accuracy, barely above the class prior), while $C = 100$ hardens the margin to the point of overfitting. The sweet spot sits around $C = 5$.

### What Drives the Classification? (Linear Weights)

For the linear kernel, the primal weight vector recovers in closed form from the dual solution, $w = \sum_{i \in SV} \alpha_i y_i x_i$. Because features are standardized, the weights are directly comparable:

| Rank | Feature | Weight | Pushes toward |
|------|---------|--------|---------------|
| 1 | worst texture | −1.1950 | malignant |
| 2 | mean compactness | +0.9791 | benign |
| 3 | area error | −0.8927 | malignant |
| 4 | worst concavity | −0.8919 | malignant |
| 5 | worst area | −0.7299 | malignant |
| 6 | worst smoothness | −0.7177 | malignant |
| 7 | radius error | −0.6927 | malignant |
| 8 | mean concavity | −0.6104 | malignant |
| 9 | worst symmetry | −0.6076 | malignant |
| 10 | worst radius | −0.5351 | malignant |

**Key insight:** the "worst" (largest-value) measurements dominate over the means — a tumor is flagged by its most extreme region, not its average appearance. Cosine similarity between our normalized weight vector and sklearn `LinearSVC`'s is **0.824**, confirming both find substantially the same hyperplane orientation (the gap comes from `LinearSVC` optimizing a squared-hinge objective with a regularized bias, a different problem than the one our dual solves).

### Additional Findings
- **Speed:** sklearn is 10–100× faster because SMO is purpose-built for the SVM dual, while cvxopt runs a general interior-point method that forms the full $n \times n$ Gram matrix. The gap widens with dataset size.
- **Interpretability:** the from-scratch model exposes the $\alpha_i$ values, the exact support-vector set, and the margin geometry — all hidden behind sklearn's API.
- **Sparsity:** only 8% of training points (34 of 426) determine the linear decision boundary; the rest could be deleted without changing the model at all.
- **Visualization:** projecting to 2 PCA components shows the decision boundary, margin bands, and circled support vectors — for display only, as the model itself trains on all 30 dimensions.

---

## Getting Started

### Prerequisites
- Python 3.9+
- Jupyter

### Clone and Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd Support-Vector-Machine-From-Scratch

# 2. Install dependencies
pip install numpy matplotlib scikit-learn cvxopt jupyter

# 3. Launch the notebook
jupyter notebook svm_from_scratch.ipynb
```

Run all cells top to bottom — the notebook is self-contained and pulls the Breast Cancer Wisconsin dataset directly from `sklearn.datasets`. Total runtime is under a minute.

---

## Repository Structure

```
Support-Vector-Machine-From-Scratch/
├── svm_from_scratch.ipynb   # Full implementation, experiments, and figures
├── SVM Project.docx         # Written report
├── SVM Project.pdf          # Written report (PDF)
└── README.md
```

---

## Notebook Walkthrough

| # | Section | Purpose |
|---|---------|---------|
| 1 | Load & Preprocess | Breast Cancer Wisconsin, labels remapped to {−1, +1}, standardized, 75/25 stratified split |
| 2 | Kernel Functions | Vectorized linear, polynomial, and RBF Gram matrix computation |
| 3 | `SVMScratch` Class | Dual QP construction and `cvxopt` solve, support vector extraction, bias recovery |
| 4 | Train & Evaluate | All three kernels: accuracy, support vector count, training time |
| 5 | sklearn Comparison | Matched-hyperparameter `SVC` benchmark |
| 6 | Classification Report | Precision, recall, F1, confusion matrix for the best kernel |
| 7 | Effect of C | Regularization sweep over 7 orders of magnitude |
| 8 | Decision Boundary | 2D PCA projection with margin bands and circled support vectors |
| 9 | Feature Importance | Primal weight recovery, top features, cosine similarity vs. `LinearSVC` |
| 10 | Prediction Agreement | Point-by-point comparison of both implementations |

---

## Methodology

### Mapping the SVM Dual to a cvxopt QP

`cvxopt.solvers.qp` minimizes $\tfrac{1}{2}x^\top P x + q^\top x$ subject to $Gx \leq h$ and $Ax = b$. Our dual maps in as:

| cvxopt term | SVM equivalent |
|-------------|----------------|
| $P_{ij}$ | $y_i y_j K(x_i, x_j)$ — the label-weighted Gram matrix |
| $q$ | $-\mathbf{1}$ (the $\sum \alpha_i$ term, negated since cvxopt minimizes) |
| $G, h$ | Box constraint $0 \leq \alpha_i \leq C$, stacked as two inequality blocks |
| $A, b$ | $y^\top \alpha = 0$, the equality constraint from the bias term |

After solving, support vectors are the points with $\alpha_i > 10^{-5}$, and the bias $b$ is averaged over the *margin* support vectors — those with $0 < \alpha_i < C$, which lie exactly on the margin and therefore satisfy $y_i f(x_i) = 1$.

### Kernels Implemented

| Kernel | Definition | Hyperparameters |
|--------|------------|-----------------|
| Linear | $x_i^\top x_j$ | — |
| Polynomial | $(x_i^\top x_j + c)^d$ | degree $d = 3$, $c = 1.0$ |
| RBF (Gaussian) | $\exp(-\gamma \|x_i - x_j\|^2)$ | $\gamma = 0.03 \approx 1/n_\text{features}$ |

The RBF kernel expands $\|x_i - x_j\|^2$ into $\|x_i\|^2 + \|x_j\|^2 - 2x_i^\top x_j$ so the entire Gram matrix computes in one broadcasted pass instead of a Python double loop.

### Experimental Setup
- **Split:** 75/25 stratified, `random_state=42` (426 train / 143 test)
- **Scaling:** `StandardScaler` fit on train only, applied to test — SVMs, and RBF especially, are scale-sensitive
- **Labels:** remapped {0, 1} → {−1, +1}, the convention the dual formulation requires

---

## Dataset

| Dataset | Source |
|---------|--------|
| Breast Cancer Wisconsin (Diagnostic) — 569 samples, 30 features, binary | [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |

---

## Course Context

**MGSC 696 — Machine Learning** at McGill University (MMA program), April 2026.

**Team:** David Fogel · Sebastian Arguedas Soley · Lucas Penney · Aziz Ahmed · Fares Jony

---

## License

MIT

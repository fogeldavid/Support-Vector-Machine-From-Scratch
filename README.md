# Support Vector Machine From Scratch

> **MGSC 696** | McGill MMA | April 30th, 2026
> David Fogel · Sebastian Arguedas Soley · Lucas Penney · Aziz Ahmed · Fares Jony

A binary soft-margin SVM built from scratch on the dual QP formulation, with linear, polynomial, and RBF kernels, validated against `sklearn.svm.SVC` on the Breast Cancer Wisconsin dataset. This page summarises the project — the full write-up is in [SVM Project.pdf](SVM%20Project.pdf), and all code, experiments, and figures are in [svm_from_scratch.ipynb](svm_from_scratch.ipynb).

---

## What the Project Does

Support Vector Machines are supervised models for binary classification. Given labelled training data, the SVM finds the separating boundary that leaves the largest possible gap — the margin — between the two classes, on the principle that a wider gap generalises better to unseen data. Since the margin equals `2 / ‖w‖`, maximising it is the same as minimising `‖w‖²`, which is the core of the objective.

Real data is rarely perfectly separable, so the soft-margin formulation allows points to sit inside the margin or on the wrong side, penalised through a slack variable and a trade-off parameter **C**. A large C punishes every violation and fits the training data tightly; a small C tolerates violations in exchange for a wider, more robust margin.

Solving this directly is awkward, so we solve the dual instead, which introduces one variable `αᵢ` per training point. Most of them come out zero — the points with `αᵢ > 0` are the **support vectors**, and they alone define the boundary. The dual also means the data appears only as dot products, so any **kernel** can be substituted without changing the structure of the problem. Three were implemented: linear (a straight boundary), polynomial (curved, degree 3), and RBF (a similarity score that decays with distance, γ = 0.03).

The dual is a convex quadratic program, which we hand to `cvxopt`, a general-purpose interior-point solver with no knowledge of SVMs. Everything specific to the algorithm — building the kernel matrix, setting up the constraints, extracting support vectors above the 10⁻⁵ threshold, recovering the bias from the points sitting exactly on the margin, and computing predictions — is written from scratch.

The implementation is deliberately binary-only. Extending to more classes requires one-vs-rest or one-vs-one wrappers, which add class-imbalance handling or a quadratic number of models while shifting attention away from the mechanics the project is about. The Breast Cancer Wisconsin data is naturally two-class, so nothing was lost.

---

## Results

The dataset holds 569 biopsy samples with 30 numerical features describing cell nuclei, labelled malignant (212) or benign (357). Features were standardised using training statistics only, and a 75/25 stratified split gave 426 training and 143 test samples.

### From-Scratch vs. sklearn (C = 1.0)

| Kernel | Implementation | Accuracy | # SVs | Train time (s) |
|--------|----------------|----------|-------|----------------|
| Linear | scratch | 0.9860 | 34 | 0.078 |
| Linear | sklearn | 0.9860 | 34 | 0.003 |
| Polynomial | scratch | 0.9650 | 63 | 0.019 |
| Polynomial | sklearn | 0.9720 | 65 | 0.001 |
| RBF | scratch | 0.9790 | 95 | 0.012 |
| RBF | sklearn | 0.9790 | 92 | 0.001 |

Accuracy is near-identical across implementations, confirming both solvers reach the same optimal hyperplane. Point-by-point, the two agree on all 143 test patients for the linear and RBF kernels and on 142 of 143 for the polynomial — a single numerical difference between cvxopt's interior-point method and sklearn's SMO. The real gap is speed: SMO is purpose-built for the SVM dual and solves it through two-variable subproblems, while cvxopt is a general solver that does not exploit the problem's structure.

The linear kernel performs best here, which is itself a finding: after standardisation the two classes are close to linearly separable in 30 dimensions, so the flexible kernels have nothing left to gain and simply need more support vectors to say the same thing.

### Performance on the Best Model (Linear Kernel)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Malignant | 0.98 | 0.98 | 0.98 |
| Benign | 0.99 | 0.99 | 0.99 |

Only 2 of 143 test patients were misclassified: one malignant predicted benign, and one benign predicted malignant. For a medical dataset the two errors are not equivalent — a missed malignant tumour is far costlier than a false alarm — which is why recall on the malignant class matters more than raw accuracy.

### Effect of C

| C | # SVs | Test Accuracy |
|---|-------|---------------|
| 0.01 | 322 | 0.6294 |
| 0.1 | 189 | 0.9371 |
| 0.5 | 116 | 0.9720 |
| 1 | 95 | 0.9790 |
| 5 | 77 | 0.9860 |
| 10 | 74 | 0.9790 |
| 100 | 67 | 0.9441 |

At low C the margin is wide enough to tolerate violations from most of the training set, so three quarters of it becomes support vectors and the model underfits badly. As C rises the margin tightens and fewer points are needed to define it. C = 5 gives the highest test accuracy, though C = 1 is close behind with a slightly wider margin, making it the more conservative choice; past C = 10 accuracy falls away as the model begins to overfit.

### What the Model Learns

Because the linear weight vector can be recovered from the dual solution and the features are standardised, the weights compare directly. The most influential are **worst texture** (−1.195), **mean compactness** (+0.979), **area error** (−0.893), and **worst concavity** (−0.892). Negative weights push toward malignant, and 9 of the top 10 are negative — the model is largely learning what malignant looks like rather than what benign looks like, with mean compactness the strongest benign indicator. That irregular, larger cell nuclei signal malignancy is consistent with the medical literature. Comparing our normalised weight vector against sklearn's `LinearSVC` gives a cosine similarity of 0.824, reflecting broadly the same hyperplane orientation from a completely different optimiser.

---

## Implementation Challenges

Three practical problems came out of the numerical side of the solver. The bias should be recoverable from any support vector sitting exactly on the margin, but floating-point error means few `αᵢ` land precisely in that range, so **b** is averaged over all margin support vectors with a fallback to all support vectors. Solved `αᵢ` are likewise never exactly zero, so a 10⁻⁵ threshold separates genuine support vectors from near-zero noise — too high and valid ones are discarded, too low and noise pollutes the bias estimate. Finally, high-degree polynomial kernels produce very large Gram matrix values that destabilise the QP; fixing degree = 3 and coef0 = 1 kept the matrix well-conditioned.

---

## Conclusion

The from-scratch implementation matched sklearn's accuracy across all three kernels, reaching 99% test accuracy with 2 misclassifications out of 143 patients. The one missed malignant case is a reminder of why recall, not accuracy, is the metric that matters in medical classification.

The broader takeaway is that the SVM's power comes less from the classifier itself than from three ideas working together: the maximum-margin objective, the dual formulation that reduces the problem to a small set of support vectors, and the kernel trick that buys non-linear boundaries at no implementation cost. cvxopt solved the optimisation correctly, and the speed gap against sklearn is a clear illustration of why specialised solvers like SMO are used in production.

---

## Running the Notebook

```bash
git clone <repo-url>
cd Support-Vector-Machine-From-Scratch

pip install numpy matplotlib scikit-learn cvxopt jupyter

jupyter notebook svm_from_scratch.ipynb
```

Python 3.9+ and Jupyter are required. Run the cells top to bottom — the dataset ships with scikit-learn, so nothing needs downloading, and the notebook finishes in under a minute. It covers preprocessing, the three kernel functions, the `SVMScratch` class, training and evaluation, the sklearn comparison, the classification report, the C sweep, a 2D PCA view of the decision boundary with support vectors circled, feature importances, and the prediction-agreement check.

```
Support-Vector-Machine-From-Scratch/
├── svm_from_scratch.ipynb   # Implementation, experiments, and figures
├── SVM Project.docx         # Written report
├── SVM Project.pdf          # Written report (PDF)
└── README.md
```

---

## References

1. M. Andersen, J. Dahl, and L. Vandenberghe, "CVXOPT: A Python package for convex optimization," Version 1.3, 2023. https://cvxopt.org/
2. W. H. Wolberg, W. N. Street, and O. L. Mangasarian, "Breast Cancer Wisconsin (Diagnostic) Data Set," UCI Machine Learning Repository, 1995. https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)
3. Scikit-learn developers, "Support Vector Machines," scikit-learn documentation. https://scikit-learn.org/stable/modules/svm.html
4. Stanford University CS229, "Support Vector Machines Lecture Notes," Andrew Ng. https://cs229.stanford.edu/notes2022fall/main_notes.pdf

# Support Vector Machine From Scratch

> **MGSC 696** | McGill MMA
> Building an SVM classifier from the ground up, then checking it against scikit-learn

---

## What This Project Does

A Support Vector Machine is a classifier that separates two groups by drawing the boundary that leaves the widest possible gap between them. This project builds one from scratch — no SVM library — and uses it to classify malignant tumours from benign ones in the Breast Cancer Wisconsin dataset (569 patients, 30 measurements each).

The only outside help we allow ourselves is a general-purpose optimizer called `cvxopt`, which solves the math problem at the center of an SVM. Everything specific to SVMs — the kernels, the training routine, the prediction step — is written by hand. We then run scikit-learn's `SVC` on the same data to check that our version gets the same answers.

### How It Works, In Plain Terms

Training an SVM means picking a boundary line that (a) separates the two classes and (b) sits as far from both as possible. Written out, the goal is:

$$\min_{w, b, \xi} \; \tfrac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.} \quad y_i(w^\top \phi(x_i) + b) \geq 1 - \xi_i,\; \xi_i \geq 0$$

The first term widens the gap; the second penalizes points that end up on the wrong side. **C** is the dial between the two — crank it up and the model refuses to tolerate mistakes, turn it down and it accepts some in exchange for a cleaner, wider boundary.

That version is hard to solve directly, so we solve its mirror image instead, called the dual:

$$\max_\alpha \; \sum_{i=1}^{n}\alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C,\; \sum_i \alpha_i y_i = 0$$

The dual is worth the detour for two reasons. It hands us one number per training point (the $\alpha_i$), and almost all of them come out as zero — the handful that don't are the **support vectors**, the points sitting closest to the boundary that actually hold it in place. And it lets us swap in a **kernel**, a shortcut function that measures similarity between two points as if we had first bent the data into a much higher-dimensional space, without ever doing that expensive step. That's what lets the same code draw curved boundaries as easily as straight ones.

### What We Found

**Three kernels, ours vs. scikit-learn** (C = 1.0, 426 patients used for training, 143 held back for testing):

| Kernel | Version | Accuracy | Support Vectors | Training Time |
|--------|---------|----------|-----------------|---------------|
| **Straight line (linear)** | **ours** | **98.6%** | **34 of 426** | **0.078 s** |
| Straight line (linear) | scikit-learn | 98.6% | 34 of 426 | 0.003 s |
| Curved (polynomial, d=3) | ours | 96.5% | 63 of 426 | 0.019 s |
| Curved (polynomial, d=3) | scikit-learn | 97.2% | 65 of 426 | 0.001 s |
| Flexible (RBF, γ=0.03) | ours | 97.9% | 95 of 426 | 0.012 s |
| Flexible (RBF, γ=0.03) | scikit-learn | 97.9% | 92 of 426 | 0.001 s |

**The simplest kernel wins.** Once the 30 measurements are put on a common scale, a plain straight-line boundary already separates malignant from benign almost perfectly. The fancier kernels have nothing left to fix, so their extra flexibility just costs complexity — the linear model leans on 34 training patients to define its boundary, while the flexible one needs nearly three times as many.

**Do the two versions agree?**

| Kernel | Agreement | Test patients matched |
|--------|-----------|-----------------------|
| Linear | 100% | 143 of 143 |
| Polynomial | 99.3% | 142 of 143 |
| RBF | 100% | 143 of 143 |

This is the real test of whether the implementation is correct. Our code and scikit-learn's use completely different solving strategies, yet they classify the same patients the same way. The single polynomial disagreement is one borderline patient sitting right on the boundary, where a rounding difference is enough to tip the call.

### How the Best Model Performs

Breaking down the linear model on the 143 held-out patients:

|  | precision | recall | f1-score | patients |
|--|-----------|--------|----------|----------|
| malignant | 0.98 | 0.98 | 0.98 | 53 |
| benign | 0.99 | 0.99 | 0.99 | 90 |
| **overall accuracy** | | | **0.99** | **143** |

|  | called malignant | called benign |
|--|------------------|---------------|
| **actually malignant** | 52 | 1 |
| **actually benign** | 1 | 89 |

For a medical problem, accuracy on its own is misleading — the two kinds of mistake aren't equally bad. Missing a malignant tumor is far worse than flagging a benign one for a second look. The model makes exactly one of each: it catches 52 of the 53 malignant cases.

### Turning the C Dial

| C | Support Vectors | Accuracy |
|---|-----------------|----------|
| 0.01 | 322 | 62.9% |
| 0.1 | 189 | 93.7% |
| 0.5 | 116 | 97.2% |
| 1.0 | 95 | 97.9% |
| **5.0** | **77** | **98.6%** |
| 10.0 | 74 | 97.9% |
| 100.0 | 67 | 94.4% |

Sweeping C across four orders of magnitude shows the tradeoff clearly. At the low end the model is so forgiving that three quarters of the training set ends up as support vectors and it barely does better than guessing the more common class. At the high end it contorts itself to get every training point right and stops generalizing. Somewhere around C = 5 is the sweet spot.

### Which Measurements Matter

For the straight-line version we can read the boundary directly and see how heavily each of the 30 measurements counts. Since everything is on a common scale, the weights compare fairly:

| Rank | Measurement | Weight | Points toward |
|------|-------------|--------|---------------|
| 1 | worst texture | −1.195 | malignant |
| 2 | mean compactness | +0.979 | benign |
| 3 | area error | −0.893 | malignant |
| 4 | worst concavity | −0.892 | malignant |
| 5 | worst area | −0.730 | malignant |
| 6 | worst smoothness | −0.718 | malignant |
| 7 | radius error | −0.693 | malignant |
| 8 | mean concavity | −0.610 | malignant |
| 9 | worst symmetry | −0.608 | malignant |
| 10 | worst radius | −0.535 | malignant |

**The pattern:** "worst" measurements — the most extreme reading found anywhere in the tumor — crowd out the averages. What flags a tumor as malignant is its most abnormal patch, not how it looks on average. That lines up with how a pathologist actually reads a slide.

We also compared our boundary's orientation to scikit-learn's `LinearSVC` and got a cosine similarity of **0.824**, meaning the two point in broadly the same direction. It isn't closer to 1 because `LinearSVC` optimizes a slightly different objective than the one we solve, so a perfect match was never expected.

### A Few Other Takeaways
- **Speed:** scikit-learn is 10–100× faster, and that's expected. Its solver was built specifically for SVMs, while ours is a general-purpose optimizer that builds a full table of every pairwise comparison. The gap would grow on a bigger dataset.
- **Transparency:** writing it ourselves exposes things the library keeps hidden — exactly which patients became support vectors, how much weight each carries, and how wide the final gap is.
- **Most of the data is redundant:** only 8% of the training patients (34 of 426) shape the linear boundary. Delete the other 392 and you'd get the identical model.
- **Seeing it:** squashing the 30 measurements down to 2 dimensions produces a picture of the boundary with the support vectors circled. That's for illustration only — the model itself trains on all 30.

---

## Running It Yourself

You'll need Python 3.9 or newer and Jupyter.

```bash
# 1. Clone the repository
git clone <repo-url>
cd Support-Vector-Machine-From-Scratch

# 2. Install dependencies
pip install numpy matplotlib scikit-learn cvxopt jupyter

# 3. Open the notebook
jupyter notebook svm_from_scratch.ipynb
```

Run the cells top to bottom. Nothing needs downloading — the dataset ships with scikit-learn — and the whole thing finishes in under a minute.

---

## What's In The Repo

```
Support-Vector-Machine-From-Scratch/
├── svm_from_scratch.ipynb   # The implementation, experiments, and figures
├── SVM Project.docx         # Written report
├── SVM Project.pdf          # Written report (PDF)
└── README.md
```

---

## Notebook Walkthrough

| # | Section | What happens |
|---|---------|--------------|
| 1 | Load & prepare the data | Load the dataset, relabel the classes as −1 and +1, put all 30 measurements on a common scale, split 75/25 |
| 2 | Kernel functions | Write the three similarity functions — straight-line, curved, and flexible |
| 3 | The `SVMScratch` class | Set up the optimization problem, solve it, pull out the support vectors and the boundary |
| 4 | Train and evaluate | Run all three kernels and record accuracy, support vector count, and timing |
| 5 | Compare to scikit-learn | Same settings, library implementation, side-by-side results |
| 6 | Detailed scorecard | Precision, recall, and the confusion matrix for the best kernel |
| 7 | The effect of C | Sweep C from 0.01 to 100 and watch the boundary tighten |
| 8 | Picture of the boundary | Squash to 2 dimensions and plot the boundary, the margin, and the support vectors |
| 9 | Which measurements matter | Read the weights off the linear model and compare to scikit-learn's |
| 10 | Do they agree? | Patient-by-patient comparison of both implementations |

---

## Implementation Notes

### Handing the Problem to the Solver

`cvxopt` accepts problems written in one specific standard form — minimize $\tfrac{1}{2}x^\top P x + q^\top x$ subject to $Gx \leq h$ and $Ax = b$ — so most of the work is translating the SVM into that shape:

| Solver expects | We supply |
|----------------|-----------|
| $P$ | The table of pairwise similarities between all training points, signed by their labels |
| $q$ | All −1s (the solver minimizes, our problem maximizes, so signs flip) |
| $G, h$ | The rule that each point's influence stays between 0 and C |
| $A, b$ | The rule that influence balances out evenly across the two classes |

Once it comes back, any point with an influence above $10^{-5}$ counts as a support vector. The boundary's offset is averaged over just the points sitting exactly on the margin — those with influence strictly between 0 and C — because those are the ones whose position we know precisely.

### The Three Kernels

| Kernel | Formula | Settings | What it does |
|--------|---------|----------|--------------|
| Linear | $x_i^\top x_j$ | — | Straight boundary |
| Polynomial | $(x_i^\top x_j + c)^d$ | degree 3, c = 1.0 | Gently curved boundary |
| RBF (Gaussian) | $\exp(-\gamma \|x_i - x_j\|^2)$ | γ = 0.03 | Freely shaped boundary |

The RBF kernel is written so the whole similarity table is computed in one vectorized pass rather than a Python loop over every pair — a rewrite of the distance formula that makes it roughly two orders of magnitude faster.

### Setup Details
- **Split:** 75/25, stratified so both halves keep the same malignant/benign mix, `random_state=42` (426 train, 143 test)
- **Scaling:** fit on the training data only, then applied to the test data — fitting on everything would leak information
- **Labels:** relabeled from 0/1 to −1/+1, which is the convention the math assumes

---

## Data

| Dataset | Source |
|---------|--------|
| Breast Cancer Wisconsin (Diagnostic) — 569 patients, 30 measurements, malignant or benign | [`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |

---

## Course Context

**MGSC 696 — Machine Learning** at McGill University (MMA program), April 2026.

**Team:** David Fogel · Sebastian Arguedas Soley · Lucas Penney · Aziz Ahmed · Fares Jony

---

## License

MIT

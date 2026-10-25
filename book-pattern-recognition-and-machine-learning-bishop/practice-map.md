# Practice map: PRML concepts → notebooks from any course

PRML is the primary source: it gives the concept, the theory and the pen-and-paper exercises.
Coding practice is picked **by concept, not by course**: when a PRML section is done, the
notebooks below for that section are solved next, whichever course they come from.

Courses (all in this repo):

- **MLT**: `course-ml-tartu-2023/` (UT Machine Learning 2023; bonus decisions in its README)
- **ML2**: `course-ml2-tartu-swenlaur/machine-learning-ii/practise-sessions/` (UT ML II, Sven Laur)
- **NN**: `Neural-networks/practices/` (UT Neural Networks)
- **MLV**: `course-ml-voron/MachineLearningSeminars/semN/main.ipynb` (Vorontsov's course, MIPT
  seminars, in Russian; sem1–3 done in 2025, results in `MachineLearningSeminars_results/`)

Matched by notebook title so far. Each row is checked against the notebook's content when its
chapter starts; whether I solved a notebook in an earlier course is not verified.

## Chapter 1 (now)

| PRML | Concept | Practice |
|---|---|---|
| §1.1, §1.3 | overfitting, model selection, cross-validation | ML2 02/06 bias–variance, 02/07–10 cross-validation; MLV sem8 (model selection criteria) |
| §1.2.3 | bootstrap, frequentist error bars | ML2 02/11 bootstrapping methods |
| §1.4 | curse of dimensionality | NN Practice_01 `knn.ipynb` task 1.5; MLT P-3 (curse of dimensionality part) |
| §1.5 | decision theory, reject option, ROC | MLT HW-1 bonus 1 (kNN class probabilities; also needs §2.5.2); MLT P-12 + ML2 02/04 (ROC) |
| §1.5.5 | loss functions for regression (squared, Minkowski) | MLV sem7 (non-standard loss functions) |

## Chapters 2–4 (Nov–Jan)

| PRML | Concept | Practice |
|---|---|---|
| §2.1–2.2 | Bernoulli/binomial, beta prior, Bayesian updating | ML2 03/02 Bayes rule, 03/03 dice vs coin, 03/05 prior and parametrisation |
| §2.3 | Gaussian, covariance | ML2 03/06 + 05/01 normal family, 06 `covariance_estimation`; MLV sem13 (density estimation, maximum likelihood) |
| §2.5.2 | kNN | MLT HW-1 bonus 1; NN Practice_01 `knn.ipynb`; MLV sem4 (metric methods) |
| §3.1–3.2 | linear regression, regularisation, bias–variance | ML2 03/07 regression under additive noise, 03/08 linear models; MLT P-7 (ridge/lasso); MLV sem6 (multivariate regression, SVD); MLV sem2 (SGD, done) |
| §4.1.4 | Fisher's linear discriminant | ML2 05/05–06 LDA |
| §4.3 | logistic regression, softmax | NN Practice_02 `softmax.ipynb` |

## Later chapters

| PRML | Concept | Practice |
|---|---|---|
| Ch. 5 | neural networks, backprop | MLT HW-3 bonus 2 (sigmoid backprop); NN Practice_03–04; MLV sem3 (autograd, done), sem15 |
| Ch. 7 | SVM | MLV sem5 |
| Ch. 9 | K-means, EM | MLT HW-2 bonus 3 (K-means in numpy); ML2 06/03, 07; MLV sem14 |
| Ch. 11 | sampling, Gibbs | ML2 07A/08 |
| Ch. 12 | PCA | ML2 05/02–04; MLT HW-2 bonus 1–2 (t-SNE/UMAP vs PCA) |
| Ch. 13 | Markov chains, HMMs | ML2 04, 08 |
| Ch. 14 | boosting, bagging | MLT HW-5 bonus 3 (AdaBoost from scratch); MLT P-8, P-10; MLV sem11–12 |

Not tied to a chapter: MLT HW-1 bonus 2 (NYC taxi, pandas) is the warm-up for the Kaggle
weekend track; MLT HW-4 CutMix belongs to the deep learning track.

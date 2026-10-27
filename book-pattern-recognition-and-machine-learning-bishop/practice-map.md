# Practice map: PRML concepts → notebooks from any course

PRML is the primary source: it gives the concept, the theory and the pen-and-paper exercises.
Coding practice is picked **by concept, not by course**: when a PRML section is done, the
notebooks below for that section are solved next, whichever course they come from.

**Bold = solve** (the main practice for that section). Plain = extra: only if the concept isn't
landing, or for interest. Homework item numbers from the source are in brackets; `*` marks an
item the source itself flags as optional.

## Status

Audited 26.10.2026 by notebook **content**, not title: every notebook in ML2, MLT, MLV and MSU,
and every HSE PDF, was read down to its homework items and either mapped below or listed under "Not mapped" with a
reason. Whether I solved a notebook in an earlier course is still unverified, except where noted
("done 2023/2025").

Correction from the audit: MLV sem7's "non-standard loss functions" is triplet loss for sentence
embeddings, not Minkowski regression losses. §1.5.5 practice is ML2 03/07 (OLS vs LAD), and sem7
moved to §4.3.3 (Newton–Raphson).

## Courses

| Prefix | Folder | Notes |
|---|---|---|
| ML2 | `course-ml2-tartu-swenlaur/machine-learning-ii/practise-sessions/` | UT ML II (Sven Laur). `02/07` = session 02, notebook 07. 2025 attempts of 01/01 and 06/04 in `../practise-sessions_results_2025/`. Sample solutions for some of 02 and 03 in `sample-solutions/HW2/`: compare only after an attempt |
| MLT | `course-ml-tartu-2023/` | UT Machine Learning 2023. Main homeworks done 2023; bonus decisions in its README |
| MLV | `course-ml-voron/MachineLearningSeminars/semN/main.ipynb` | Vorontsov's course, MIPT seminars (Russian). sem1–3 done 2025 (`MachineLearningSeminars_results/`). `MLV HW1` = `hometask/autumn_semester/task_1/hw1_student.ipynb` |
| MSU | `course-ml-voron/ml-course-msu/` | ВМК МГУ seminars (E. Sokolov), in the same folder. `MSU P2` = `ML17/homeworks-practice/homework-practice-02.ipynb` (P3–P5 likewise); `MSU EM` = `ML16-spring/labs/lab-01-em.ipynb` |
| NN | `Neural-networks/practices/` | UT Neural Networks |
| HSE | `course-ml-hse-bogdanpotanin/Machine-Learning/` | HSE "ML in economics" (B. Potanin). Only the seminar sheets are used: short **by-hand** problems on small tables, no code. `HSE С3.4` = Семинар 3, Задание 4 |

## Chapter 1 (now)

| PRML | Concept | Practice |
|---|---|---|
| §1.1, §1.3 | overfitting, generalisation, model selection, cross-validation | **ML2 01/03 (3.1, 3.2)**, **ML2 02/07 (4.1)**, **ML2 02/10 (8.1\*)**; ML2 01/01 (2025 attempt), 01/02 (2.1), 02/08 (5.1), 02/09 (6.1\*), 02/12 (10.1); reading: ML2 02/tutorials 02–03 (why overfitting occurs); MLV sem8 (model selection criteria, Add-Del) |
| §1.2.3 | frequentist vs Bayesian: confidence vs credible intervals, bootstrap | **ML2 03/01 (1.1\* paradox of confidence intervals)**, **ML2 02/11 (9.1)**; ML2 02/01 (1.1, 1.3), 02/02 (2.1, 2.2), 02/05 (5.1\*) — theory for these: Blom ch. 13–14 |
| §1.2.5, §1.5.5 | Gaussian noise ⇒ squared loss ⇒ mean; Laplace noise ⇒ absolute loss ⇒ median (Ex. 1.27, q = 1) | **ML2 03/07 sections I–III (OLS vs LAD, worked in the notebook)**; HSE С3.4.1 (MSE-optimal F(x) = E[t\|x] for a Poisson target: Ex. 1.25 with numbers); reading first: [Central tendency, solutions to variational problems](https://en.wikipedia.org/wiki/Central_tendency#Solutions_to_variational_problems) (L2 → mean, L1 → median, L0 → mode, L∞ → midrange) |
| §1.4 | curse of dimensionality | NN Practice_01 `knn.ipynb` task 1.5; MLT P-3 (curse of dimensionality part) |
| §1.5.1–1.5.4 | loss matrix, cost-optimal threshold, class priors, ROC | **MLV HW1 Задача 2 (2.1–2.3)**: threshold monotonicity, cost-optimal threshold from c_FP, c_FN (= Ex. 1.22 with numbers), threshold on the logit; ML2 02/04 (4.1\*, 4.2\* ROC envelopes); MLT P-12; MLT HW-6 (done 2023: imbalanced data ↔ §1.5.4); HSE С2.1.4–1.6 (bank: prices of outcomes, optimal threshold; a third action "insure" works like the reject option), HSE С2.2 (profit, gain/lift, AUC) |
| §1.6 | entropy, KL, mutual information | **HSE С3.1** (depth-2 tree by entropy, by hand: information gain H[y] − H[y\|x] = I[x, y], Ex. 1.41 with numbers), after Ex. 1.41. Otherwise Ex. 1.29–1.41 are the practice; in code later: split criteria (§14.4), cross-entropy loss (§4.3.2) |

## Chapter 2

| PRML | Concept | Practice |
|---|---|---|
| §2.1–2.2 | beta/Dirichlet posteriors, conjugacy, sequential updates, Laplace smoothing | **ML2 03/02 (2.1–2.3)**, **ML2 03/03 (3.1)**, **ML2 03/04 (4.1, 4.2)** (Laplace smoothing = Dirichlet posterior mean); ML2 03/05 (5.1), 03/01 (1.2\*), 03/02 (2.4\*); reading: ML2 03/tutorials (coin, dice formulae); MLV sem25 (Thompson sampling: beta posteriors in a bandit) |
| §2.3.1–2.3.3 | multivariate Gaussian, affine maps, conditionals and marginals, linear-Gaussian | **ML2 05/01 (1.1, 1.2)**; ML2 03/06 (6.1, 6.2, 1-D version); reading: ML2 07A/tutorials `multivariate_normal_distribution`, 04/tutorials `normal_distributions_and_belief_propagation` (conditioning) |
| §2.3.4–2.3.5 | ML for the Gaussian; sequential estimation, Robbins–Monro | MLV sem13 (ML section; its homework is Ex. 1.11, done); MLV sem2 (SGD convergence, done 2025) |
| §2.3.6–2.3.9 | Bayesian Gaussian, Student t, periodic | no notebook; PRML exercises only. Mixtures (§2.3.9) → Ch. 9 |
| §2.4 | exponential family | used in code at §4.3.6 (GLM) |
| §2.5.1 | kernel density estimation, bandwidth by leave-one-out | **MLV sem13 (KDE, LOO bandwidth)**; MLV sem4 (Parzen window part) |
| §2.5.2 | k-nearest neighbours | **MLT HW-1 bonus 1** (kNN posteriors K_k/K); NN Practice_01 `knn.ipynb`; MLV sem4 (+ its homework: kNN hyperparameters by CV); HSE С2.1.1–1.3 (feature scaling under Minkowski distance) |

## Chapter 3

| PRML | Concept | Practice |
|---|---|---|
| §3.1.1–3.1.3 | least squares as ML, design matrix, SVD / pseudo-inverse, sequential (LMS) | **MLV sem6 (+ homework: error vs λ, SVD vs direct)**; ML2 03/08 (8.1); MSU `ML15-spring/labs/lab01-linear.ipynb` part 2 (gradient descent by hand); MLV sem2 (SGD, done 2025) |
| §3.1.4 | ridge, lasso, sparsity, feature scaling | **MLV HW1 Задача 1 (1.1–1.4)** (pen-and-paper: closed forms for orthonormal X, soft thresholding, ‖ŵ‖ falls with λ, scaling, intercept not penalised), **MLV HW1 Задача 3 (3.1–3.5)** (Diabetes: scaling, ridge/lasso paths), **ML2 03/07 (7.3)**; ML2 03/07 (7.2, 7.4); MLT P-7; MLT HW-4 ex. 1 (ElasticNet, done 2023); MSU P2 part 1 |
| §3.1.5 | multiple outputs | ML2 05/07 (4.1\*) |
| §3.2 | bias–variance decomposition | **MSU P3 part I (tasks 1–2)**: average model and Monte-Carlo bias/variance, PRML Fig. 3.5–3.6 in code; ML2 02/06 (3.1\*); HSE С3.4.2–4.3 (bias, variance and noise by hand) |
| §3.3 | Bayesian linear regression, predictive variance | MLV sem2 homework (MAP with p(w) = N(0, I), = PRML (3.55)); predictive variance in use: MLV sem26 (§6.4 row) |
| §3.4–3.5 | evidence, Bayesian model comparison | no notebook; PRML exercises only |

## Chapter 4

| PRML | Concept | Practice |
|---|---|---|
| §4.1.1–4.1.3 | discriminant functions, multiclass, least-squares classification | MLV sem1 (binary → multiclass, intercept; done 2025) |
| §4.1.4–4.1.6 | Fisher's linear discriminant, whitening | **ML2 05/05 (5.1 implement LDA)**; ML2 05/06 (6.1–6.4\*) |
| §4.2 | generative classifiers: Gaussian classes ⇒ linear boundary, naive Bayes | **ML2 05/05 section III** (Bayes factors, same notebook as above); MLV sem13 (naive Bayes part); HSE С1.1 (naive Bayes by hand), С1.2.4 (asymptotic bias of naive Bayes) |
| §4.3.1–4.3.4 | logistic regression, cross-entropy, softmax, calibration | **MSU P2 part 2 (task 1: L2 logistic regression by gradient descent, vectorised; tasks 2–3)**, **MSU P2 part 3 task 2** (calibration curves: a threshold on p(x) needs calibrated probabilities, §1.5), **MLV HW1 Задача 4 (4.1–4.5)** (logreg vs kNN vs RBF-SVM vs tree; 4.4 = thresholds again); NN Practice_02 `softmax.ipynb`; ML2 02/08 (5.1) |
| §4.3.3 | Newton–Raphson / IRLS | **MLV sem7** (gradient descent vs Newton–Raphson vs Gauss–Newton) |
| §4.3.6 | canonical link, GLM | MSU P5 (Poisson regression: own loss and gradient, L-BFGS; statsmodels Gaussian / Poisson / NegBin); MLV sem7 (GLM part) |
| §4.4–4.5 | Laplace approximation, BIC, Bayesian logistic regression | no notebook; PRML exercises only |

## Chapter 5

| PRML | Concept | Practice |
|---|---|---|
| §5.1–5.3 | MLP, backpropagation | **MLT HW-3 bonus 2** (sigmoid backprop); NN Practice_03 `two_layer_net`, Practice_04 `FullyConnectedNets`; MSU P4 (computational graph from scratch); MLV sem3 (autograd, done 2025); MLT P-5; HSE С7 (backprop, dropout, SGD vs GD, by hand) |
| §5.5 | weight decay, early stopping, invariance by augmentation, convolutional nets | MLT P-7 (L1/L2 in NNs, augmentation); MLT HW-4 ex. 2–3 (done 2023), HW-4 bonus CutMix (triage); NN Practice_04 `Dropout`, Practice_05; MLV sem15; MSU P4 part 3 (conv, max-pool layers) |
| §5.6–5.7 | mixture density networks, Bayesian neural networks | no notebook |

## Chapters 6–7

| PRML | Concept | Practice |
|---|---|---|
| §6.1–6.2 | kernels, constructing kernels | **MLV sem5 homework** (data where a polynomial kernel beats RBF and linear) |
| §6.3 | Nadaraya–Watson, RBF | MLV sem4 (Nadaraya–Watson, potential functions) |
| §6.4 | Gaussian processes | **MLV sem26** (active learning with a GP: query the point of maximal predictive variance) |
| §7.1 | SVM, soft margin, SVR | **MLV sem5**; MSU P2 part 3 task 1 (LinearSVC, support vectors) |
| §7.2 | RVM | no notebook |

## Chapter 8

| PRML | Concept | Practice |
|---|---|---|
| §8.1–8.2 | Bayesian networks, conditional independence | **HSE С1** (Bayesian network vs naive Bayes on a 15-row table, cross-validated; conditional-independence algebra). No notebook |
| §8.3 | Markov random fields, Hammersley–Clifford, image restoration | **ML2 07A/07 (7.1, 7.3, 7.4)**; ML2 07A/09 (6.1\*); reading: ML2 07A/tutorials `basics_markov_fields` |
| §8.4.1–8.4.4 | sum-product on chains and trees | **ML2 04/01 (1.1, 1.2)** (imputation by belief propagation); reading: ML2 04/tutorials `belief_propagation_in_a_chain`, `_in_a_tree` |
| §8.4.5 | max-sum | ML2 04/01 section III; reading: ML2 04/tutorials `posterior_maximisation_for_chain`, `_for_a_tree` |

## Chapter 9

| PRML | Concept | Practice |
|---|---|---|
| §9.1 | K-means, image segmentation | **ML2 06/03 (3.1–3.3)**, **MLT HW-2 bonus 3** (K-means in numpy); ML2 05/06 section I (colour k-means); MLV sem14 |
| §9.2 | Gaussian mixtures, EM | **ML2 07/03 (3.1–3.3)**, **MSU EM** (Gaussian mixture part); ML2 07/01 (1.1–1.5); MLV sem13 (mixture part) |
| §9.3–9.4 | EM in general, Bernoulli mixtures, lower bound | **ML2 08/01 (1.1)** (Bernoulli mixture = §9.3.3), **MSU EM** (binary sequence with a latent change point); reading: ML2 07/03 section III (lower bound) |

## Chapters 10–12

| PRML | Concept | Practice |
|---|---|---|
| Ch. 10 | variational inference | no PRML-level notebook. VAE versions belong to the deep learning track: MLV sem16, MLV `hometask/task2` task 4 |
| §11.1 | rejection, importance sampling, resampling | ML2 03/03 section IV (rejection sampling); ML2 04/05 section III (weighting and resampling) |
| §11.2 | Metropolis–Hastings | no notebook |
| §11.3 | Gibbs sampling | **ML2 07A/08 (7.1, 7.2)**; ML2 07A/08 (7.3) |
| §12.1 | PCA | **ML2 05/02 (2.1–2.3)**, **ML2 05/04 (4.1)**; ML2 05/03 (2.1), 06/05 (5.1); MLV sem6 (PCA part); MLT P-3; MLT HW-2 bonus 1–2 (t-SNE/UMAP vs PCA); reading: ML2 05/tutorials `direction_with_maximal_variance` |
| §12.2–12.3 | probabilistic PCA, kernel PCA | no notebook |
| §12.4.2 | autoencoders (linear autoencoder = PCA) | MLV sem16 (linear autoencoder part) |

## Chapter 13

| PRML | Concept | Practice |
|---|---|---|
| §13.1 | Markov chains, autoregressive models | **ML2 04/01 (1.3)**, **ML2 04/02 (2.1)**; ML2 04/03 (4.1), 04/04 (4.1\*); MLV sem23 (autoregression) |
| §13.2 | HMM: forward–backward, Viterbi, Baum–Welch | **ML2 04/06 (6.1)**, **ML2 04/07 (7.1)**, **ML2 08/01 (1.2, 1.3)**, **ML2 08/02 (2.1)**; ML2 04/09 (6.1), 08/01 (1.4\*–1.6\*); reading: ML2 04/tutorials `belief_propagation_for_hmm` |
| §13.3 | linear dynamical systems, Kalman filter, particle filter | **ML2 04/08 (8.1)**, **ML2 04/05 (5.1, 5.2)**; ML2 04/08 (8.2\*), 04/05 (5.3) |

## Chapter 14

| PRML | Concept | Practice |
|---|---|---|
| §14.1–14.2 | committees, bagging | MLT P-8; MLT HW-5 ex. 1 (done 2023); MLV sem11 (bagging, random forest); MSU P3 tasks 4–5; HSE С3.2 (bagged kNN and OOB error by hand), С3.3 (ensemble variance when trees are correlated) |
| §14.3 | boosting, exponential loss | **MLT HW-5 bonus 3** (AdaBoost from scratch); MLT P-10; MLV sem11–12 (+ sem12 homework: ComBoost for regression); MSU P3 task 6; HSE С4 (gradient boosting by hand; Newton variant) |
| §14.4 | tree-based models | **MSU P3 task 3** (decision tree from scratch); MLV sem9 (informativeness criteria; homework: Gini) |
| §14.5.1 | mixtures of linear regressions | **ML2 06/04 (4.2, 4.4)** (2025 attempt exists: check it first); ML2 06/04 (4.1, 4.3) |
| §14.5.3 | mixtures of experts | MLV sem11 (Mixture of Experts part) |

## Not mapped, and why

Everything else in these courses, so nothing is dropped silently:

- **Outside PRML (hierarchical clustering, association rules, semi-supervised, anomalies):** ML2
  06/01–02, MLT P-4 (hierarchical part), MLV sem10, MLV sem14 (DBSCAN, self-training parts), MSU
  P6.
- **Deep learning / NLP / recommender track (after PRML):** MLT P-5–P-6, HW-3 (main), HW-4
  CutMix; MLV sem15–22 beyond the parts mapped above, sem7 (triplet-loss part), MLV
  `hometask/task2` (all ten tasks); NN Practice_06–07.
- **Duplicates of bold rows:** MSU `ML15*/ML16*` seminar notebooks and labs (`src/`, `labs/`,
  except the two mapped above): sklearn walkthroughs of kNN, trees, PCA, ensembles. ML2
  `07/01_..._changed.ipynb` is a second copy of 07/01. ML2 `lectures/*/illustration_sources`
  are lecture figure code.
- **Workflow and tooling, no PRML concept:** ML2 00 (pandas), ML2 01/image-generation, ML2 06
  `covariance_estimation` (reference code, no exercises), ML2 02/03 (QQ-plot envelopes; Blom ch.
  13–14 if interested); MSU P1 (missing data); MLT P-1–P-2 (labs, done 2023); MLT HW-1 bonus 2
  (NYC taxi: the Kaggle warm-up).
- **Not doable or empty:** MLT HW-5 bonus 1–2 (closed in-class leaderboard); MLV sem20 (homework
  walkthrough), sem24, sem27 (title only), sem28–30 (LLM, machine translation); MSU contests.
- **Pen-and-paper theory sheets** (MSU `homeworks-theory`, `ML15/ML16/homeworks` PDFs): the PRML
  exercises already cover this ground.
- **HSE beyond the seminar sheets:** С5–С6 and lectures 5–6 are causal inference (double ML,
  ATE, S/T/X-learners, IPW, doubly robust, LATE): not in PRML. Scheduled as its own two-week
  block right after PRML Ch. 4 (late January): lecture 5 → С5, lecture 6 → С6. The homework is a
  semester research report on simulated data (the Kaggle track covers that workflow). Exams
  (some with solutions) repeat the seminar style. Lecture 8 (RNN) is deep learning.
  `2. Решающие деревья.py` is an sklearn walkthrough. Lectures: read one only when a seminar
  problem needs it.

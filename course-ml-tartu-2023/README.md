# University of Tartu — Machine Learning (autumn 2023)

Colab notebooks from the UT Machine Learning course, autumn 2023, exported from Google Drive on
25.10.2026. `practice-sessions/` holds the lab notebooks (`P-N`), `homeworks/` my submitted
homeworks (`HW-N`) with outputs. The main homework exercises are solved; the bonus exercises
were not, apart from a started HW-1 bonus 2.

## Bonus exercises: decision (25.10.2026, revised the same day)

Used as practice next to the main tracks (PRML, then deep learning, Kaggle): theory in code, and
hands-on numpy / pandas / sklearn fluency. Old partial answers may be deleted and redone. Each one is solved
in its own homework notebook, under the bonus heading.

| Bonus | Task | Decision | When |
|---|---|---|---|
| HW-1 B1 | kNN: show the true-class and wrong-class probabilities of misclassified digits | **Solve** | with PRML §2.5.2 (kNN posteriors p(C_k\|x) = K_k/K); reuses §1.5 decision theory |
| HW-1 B2 | NYC taxi Kaggle: beat RMSE 3.04 (late submissions still score) | **Solve** | first Kaggle weekend, warm-up before IEEE-CIS: pandas cleaning, features, RandomForest |
| HW-2 B1 | t-SNE / UMAP on MNIST vs PCA | **Solve** | January, with HW-2 B3: sklearn/umap practice |
| HW-2 B2 | train a model on the t-SNE/UMAP output | **Solve** | right after B1: sklearn pipelines, comparing models |
| HW-2 B3 | K-means from scratch in numpy, loss per step, animation | **Solve** | January, after PRML Ch. 4 (PRML §9.1 is the theory) |
| HW-3 B1 | test my CNN on Stable Diffusion images | Skip | prompting a hosted demo, no framework or theory practice |
| HW-3 B2 | 3-layer net: tanh → sigmoid, redo backprop, explain and fix the result | **Solve** | start of the deep learning track (backprop, vanishing gradients) |
| HW-4 | CutMix from scratch on CIFAR10 | Triage | deep learning track, augmentation; needs a GPU run |
| HW-5 B1, B2 | top of the in-class Kaggle leaderboard | Can't do | the in-class competition closed 19.11.2023 |
| HW-5 B3 | Discrete AdaBoost from scratch (Friedman, Hastie, Tibshirani 2000) | **Solve** | Kaggle gradient-boosting phase, with ISL boosting and PRML §14.3 |
| HW-6 | — | no bonus | |

Solve = do it. Triage = write the plan, do it only if time allows. Skip = not worth the time.

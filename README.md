

<h1 align="center">FDS Challenge 2026 — Pokémon Battle Prediction</h1>

<p align="center">
  <a href="https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2026"><img src="https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle competition"></a>
</p>

<p align="center"><b>Team Fantastic Four: Annika Annika Unm¨ußig, Francesco Anatriello, Valentina Foschi, Ivan Dalla Ragione</b> — solution for the Kaggle competition <a href="https://www.kaggle.com/competitions/fds-pokemon-battles-prediction-2026"><i>FDS Pokémon Battles Prediction 2026</i></a>, Fundamentals of Data Science course.</p>

---

## 📋 Table of contents

- [The challenge](#-the-challenge)
- [The dataset](#-the-dataset)
- [Our submissions](#-our-submissions)
- [Feature engineering, in short](#-feature-engineering-in-short)
- [Repository structure](#-repository-structure)
- [Reproducing the submissions](#-reproducing-the-submissions)

## 🎮 The challenge

Each example is a Pokémon battle in **Gen 2 OU** format between two players, P1 and P2. The task is binary classification: predict `player_won` (1 if P1 wins), observing only the **first 30 turns** of the battle — which in reality lasts 30–80 turns, so the outcome can still flip after the observed window.

Key rules of the competition:

| | |
|---|---|
| 🎯 **Metric** | Accuracy on Kaggle's private leaderboard |
| 🚫 **Constraints** | Classic ML only (no neural networks), no external datasets, no data beyond `train.jsonl` / `test.jsonl` / `type_chart_gen2.csv` |
| 🔍 **Validation** | No access to test labels — all validation is cross-validation on the training set |
| 📤 **Submissions** | One prediction per `battle_id`, max 3 per day, deadline **05/11/2026** |

## 📊 The dataset

| File | Content |
|---|---|
| `train.jsonl` | 10,000 battles, balanced 50/50 on `player_won` |
| `test.jsonl` | 5,000 battles, same structure, no label |
| `type_chart_gen2.csv` | Gen 2 type-effectiveness chart, provided by the competition |


## 🏆 Our submissions

Our three final submissions are three variants of the same feature-engineering pipeline, chosen to be strong but architecturally different — so that a hidden issue in one of them wouldn't compromise all three on the private test set.

| Notebook | Approach | Model | CV (5-fold) | Public LB | Private LB |
|---|---|---|---:|---:|---:|
|submission_6_-_trio___cat_with_new_columns___log.ipynb| Full ensemble | (2 × XGBoost/LightGBM/HistGB + 2 × CatBoost + logistic regression) / 5 | acc 0.7003, AUC 0.7749 | 0.70333 | 0.68542 |
| submission_7_-_cat_boost_only___tuned.ipynb | Single model, different architecture from the ensemble | CatBoost (5 seeds, tuned hyperparameters) | acc 0.6968, AUC 0.7730 | 0.70400 | 0.68657 |
| submission_8_-_terna___logistic__tuned | Safety net, no CatBoost | (3 × XGBoost/LightGBM/HistGB + logistic regression) / 4 | acc 0.6984, AUC 0.7728 | 0.69866 | **0.68771** |


## 🧩 Feature engineering, in short

| Block | What it captures |
|---|---|
| Baseline | team stats, HP & status over time, KOs, team-HP trend, revealed P2 profile, predicted threat, P1 team frequency |
| Material & matchup | Pokémon alive at turn 30, status/species-weighted HP, the two active Pokémon (speed, type edge, boosts), Spikes, recent HP momentum, Leftovers |
| Volatile state | substitute, Leech Seed, Perish Song, confusion, trapping, frozen Pokémon, Reflect/Light Screen |
| Sustain | walls with Leftovers still alive, Pokémon that used Rest/recovery moves |
| Pairwise duels | Gen 2 damage formula — who KOs whom first in every alive-vs-alive matchup |
| Setup dynamics | how stat boosts rose and fell: setup moves, HP paid for setting up, boosts lost on switch |
| Strength ratings | logistic-regression ratings for team strength, species strength, and a "battle token" bag-of-signals rating |

## 🗂️ Repository structure

```
.
├── README.md
├── assets/                                                   # put your own image here, e.g. assets/pk.jpg
├── submission_6_-_trio___cat_with_new_columns___log.ipynb    # full ensemble (public 0.70333 / private 0.68542)
├── submission_7_-_cat_boost_only___tuned.ipynb                # CatBoost only, tuned (public 0.70400 / private 0.68657)
└── submission_8_-_terna___logistic__tuned.ipynb               # trio + logistic, no CatBoost (public 0.69866 / private 0.68771)
```

## ▶️ Reproducing the submissions

Each notebook is self-contained and deterministic (fixed seeds): run top to bottom on Kaggle, with the competition dataset attached, it produces the same `submission.csv`. Approximate runtime on Kaggle CPU: 10 minutes (trio + logistic), 20 minutes (CatBoost only), 25–30 minutes (full ensemble).

**Requirements:** `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, and (for two of the three notebooks) `catboost`.

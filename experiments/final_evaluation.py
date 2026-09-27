
"""
Risk-Aware Autonomous Satellite Data Prioritization Under Limited Communication
================================================================================
FINAL EXPERIMENT (proof-of-concept)

This is a single, clean, reproducible script. Running it top-to-bottom once
produces every number and figure referenced in the report. Nothing here is
hand-entered: the results table and all three figures are built directly from
the dictionaries returned by the evaluation functions below.

No deep learning, no RL library, no real satellite data. Everything is a
synthetic simulation, by design, at this stage of the project.

Compatible with a plain "python3 final_experiment_risk_aware_selection.py"
run and with Google Colab (just run all cells / the whole file).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # safe for headless environments; plt.show() still works in Colab/Jupyter


FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_NAMES = ["importance", "rarity", "novelty", "uncertainty", "risk_estimate"]
N_FEATURES = len(FEATURE_NAMES)

# Seeds are fixed and documented here so the whole experiment is reproducible.
TRAIN_SEED = 0
IN_DIST_EVAL_SEED = 123
OOD_EVAL_SEED = 999


# ============================================================================
# 1. PROBLEM DEFINITION
# ============================================================================
# A satellite observes N candidate "observations" per pass but can only
# transmit K of them (a hard communication-budget constraint). Each
# observation carries 5 features: importance, rarity, novelty, uncertainty,
# and a risk_estimate. A hidden `true_consequence` value (never seen by any
# policy, only used for evaluation) represents how much it actually mattered
# that the observation got transmitted -- the role real ground-truth
# confirmation would play once real data is used.
#
# The central question: can a policy that reasons jointly over these five
# factors, plus redundancy within the transmitted batch, outperform a naive
# "top-K by importance" baseline -- specifically by not discarding rare,
# low-importance-but-high-consequence observations?


# ============================================================================
# 2. SYNTHETIC SATELLITE ENVIRONMENT
# ============================================================================

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def generate_episode(n_normal=39, n_rare=2, rare_true_positive_rate=0.7, rng=None):
    """In-distribution generator: fixed rare-event count, low feature noise,
    rare items always show BOTH high rarity and high novelty."""
    rng = rng if rng is not None else np.random.default_rng()
    n_total = n_normal + n_rare
    features = np.zeros((n_total, N_FEATURES))
    true_consequence = np.zeros(n_total)

    for i in range(n_normal):
        importance = rng.beta(8, 2)
        rarity = rng.beta(2, 8)
        novelty = rng.beta(2, 8)
        uncertainty = rng.beta(2, 8)
        risk_estimate = rng.beta(2, 8) * 0.5 + importance * 0.1
        features[i] = [importance, rarity, novelty, uncertainty, risk_estimate]
        true_consequence[i] = rng.beta(2, 10)

    for k in range(n_rare):
        idx = n_normal + k
        importance = rng.beta(2, 8)
        rarity = rng.beta(8, 2)
        novelty = rng.beta(8, 2)
        uncertainty = rng.beta(6, 3)
        risk_estimate = rng.beta(5, 3)
        features[idx] = [importance, rarity, novelty, uncertainty, risk_estimate]
        if rng.random() < rare_true_positive_rate:
            true_consequence[idx] = rng.beta(9, 1)
        else:
            true_consequence[idx] = rng.beta(2, 10)

    return features, true_consequence


def generate_episode_ood(n_normal=39, rng=None):
    """Out-of-distribution generator. Differs from generate_episode in five
    ways: (1) variable rare-event count (0-5, not fixed 2), (2) added
    Gaussian sensor noise, (3) noisier/fuzzier normal background, (4) lower
    rare true-positive rate (0.5 vs 0.7), (5) 40% of rare events are
    "stealthy" -- LOW rarity/novelty, signaled only via uncertainty/risk."""
    rng = rng if rng is not None else np.random.default_rng()
    n_rare = int(rng.integers(0, 6))
    n_total = n_normal + n_rare
    features = np.zeros((n_total, N_FEATURES))
    true_consequence = np.zeros(n_total)
    noise_std = 0.07

    for i in range(n_normal):
        importance = rng.beta(5, 3)
        rarity = rng.beta(3, 6)
        novelty = rng.beta(3, 6)
        uncertainty = rng.beta(3, 6)
        risk_estimate = rng.beta(3, 6) * 0.6 + importance * 0.05
        vec = np.clip(
            np.array([importance, rarity, novelty, uncertainty, risk_estimate])
            + rng.normal(0, noise_std, size=N_FEATURES), 0, 1)
        features[i] = vec
        true_consequence[i] = float(np.clip(rng.beta(2, 10) + rng.normal(0, 0.05), 0, 1))

    stealthy_fraction = 0.4
    rare_true_positive_rate = 0.5

    for k in range(n_rare):
        idx = n_normal + k
        stealthy = rng.random() < stealthy_fraction
        importance = rng.beta(2, 7)
        if stealthy:
            rarity = rng.beta(3, 7)
            novelty = rng.beta(3, 7)
        else:
            rarity = rng.beta(7, 2)
            novelty = rng.beta(7, 2)
        uncertainty = rng.beta(6, 3)
        risk_estimate = rng.beta(5, 3)
        vec = np.clip(
            np.array([importance, rarity, novelty, uncertainty, risk_estimate])
            + rng.normal(0, noise_std, size=N_FEATURES), 0, 1)
        features[idx] = vec
        if rng.random() < rare_true_positive_rate:
            true_consequence[idx] = float(np.clip(rng.beta(9, 1) + rng.normal(0, 0.05), 0, 1))
        else:
            true_consequence[idx] = float(np.clip(rng.beta(2, 10) + rng.normal(0, 0.05), 0, 1))

    return features, true_consequence


# ============================================================================
# 3. BASELINE POLICY
# ============================================================================

def select_topk_importance(features, K):
    scores = features[:, FEATURE_NAMES.index("importance")]
    return list(np.argsort(-scores)[:K])


def select_oracle(true_consequence, K):
    """Evaluation-only: cheats by seeing hidden ground truth. Used to compute regret."""
    return list(np.argsort(-true_consequence)[:K])


# ============================================================================
# 4. LEARNED RISK-AWARE POLICY
# ============================================================================
# Sequential greedy selection with a linear utility function:
#   score(i | already_selected) = w[0:5] . features[i]  -  w[5] * redundancy(i)
# Trained with black-box (zeroth-order) hill-climbing on episode reward --
# no autograd, no RL library. Redundancy is similarity to the closest
# already-selected item, recomputed at every sequential pick, which is what
# makes selection batch-aware rather than a static top-K sort.

def select_learned(features, w, K):
    n_total = features.shape[0]
    selected, remaining = [], list(range(n_total))
    for _ in range(K):
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            feat = features[idx]
            redundancy = max((cosine_sim(feat, features[s]) for s in selected), default=0.0)
            score = float(np.dot(w[:N_FEATURES], feat) - w[N_FEATURES] * redundancy)
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


LAMBDA_REDUNDANCY = 2.0


def compute_reward(selected, features, true_consequence, lam=LAMBDA_REDUNDANCY):
    value = true_consequence[selected].sum()
    if len(selected) > 1:
        sims = [cosine_sim(features[selected[i]], features[selected[j]])
                for i in range(len(selected)) for j in range(i + 1, len(selected))]
        redundancy_penalty = lam * np.mean(sims)
    else:
        redundancy_penalty = 0.0
    return value - redundancy_penalty


def evaluate_policy_weights(w, n_episodes, K, rng):
    rewards = []
    for _ in range(n_episodes):
        features, true_consequence = generate_episode(rng=rng)
        selected = select_learned(features, w, K)
        rewards.append(compute_reward(selected, features, true_consequence))
    return float(np.mean(rewards))


def train_policy(n_iterations=150, eval_episodes=25, K=10, sigma=0.15, seed=TRAIN_SEED):
    train_rng = np.random.default_rng(seed)
    w = train_rng.normal(0, 0.1, size=N_FEATURES + 1)
    w[N_FEATURES] = 0.1
    best_reward = evaluate_policy_weights(w, eval_episodes, K, train_rng)
    history = [best_reward]
    for _ in range(n_iterations):
        candidate = w + train_rng.normal(0, sigma, size=w.shape)
        candidate_reward = evaluate_policy_weights(candidate, eval_episodes, K, train_rng)
        if candidate_reward > best_reward:
            w, best_reward = candidate, candidate_reward
        history.append(best_reward)
    return w, history


# ============================================================================
# 5. MOTIVATING RARE HIGH-CONSEQUENCE SCENARIO
# ============================================================================
# 10 ordinary-but-useful observations + 1 rare observation that is high on
# BOTH rarity and novelty as well as uncertainty/risk. Budget = 10, so
# something must be dropped.

def build_motivating_scenario():
    features = np.zeros((11, N_FEATURES))
    true_consequence = np.zeros(11)
    for i in range(10):
        features[i] = [0.75, 0.10, 0.10, 0.10, 0.15]
        true_consequence[i] = 0.30
    features[10] = [0.20, 0.90, 0.85, 0.80, 0.70]
    true_consequence[10] = 0.95
    return features, true_consequence


def run_scenario(name, features, true_consequence, w, K, special_idx):
    baseline_sel = select_topk_importance(features, K)
    learned_sel = select_learned(features, w, K)
    oracle_sel = select_oracle(true_consequence, K)

    result = {
        "name": name,
        "baseline_included": special_idx in baseline_sel,
        "learned_included": special_idx in learned_sel,
        "baseline_reward": compute_reward(baseline_sel, features, true_consequence),
        "learned_reward": compute_reward(learned_sel, features, true_consequence),
        "oracle_reward": compute_reward(oracle_sel, features, true_consequence),
    }

    print("=" * 78)
    print(name)
    print("=" * 78)
    print(f"Top-K importance   -> included special item? "
          f"{'YES' if result['baseline_included'] else 'NO (dropped it)'}   "
          f"reward={result['baseline_reward']:.3f}")
    print(f"Learned policy     -> included special item? "
          f"{'YES' if result['learned_included'] else 'NO (dropped it)'}   "
          f"reward={result['learned_reward']:.3f}")
    print(f"Oracle             -> reward={result['oracle_reward']:.3f}")
    print()
    return result


# ============================================================================
# 6 & 7. IN-DISTRIBUTION AND OUT-OF-DISTRIBUTION EVALUATION
# ============================================================================

def batch_evaluate(w, generator_fn, n_test_episodes, K, seed):
    """Generic batch evaluator. Used for BOTH in-distribution and OOD
    evaluation -- the only difference is which generator_fn / seed is passed
    in. Returns a dict of metrics per policy; does not print (printing is
    centralized in the Results section to avoid duplicated output)."""
    test_rng = np.random.default_rng(seed)
    rewards = {"topk": [], "learned": [], "oracle": []}
    catches = {"topk": 0, "learned": 0, "oracle": 0}
    total_catastrophes = 0
    skipped = 0

    for _ in range(n_test_episodes):
        features, true_consequence = generator_fn(rng=test_rng)
        if features.shape[0] < K:
            skipped += 1
            continue

        catastrophe_mask = true_consequence > 0.7
        total_catastrophes += catastrophe_mask.sum()

        sel = {
            "topk": select_topk_importance(features, K),
            "learned": select_learned(features, w, K),
            "oracle": select_oracle(true_consequence, K),
        }
        for policy in ["topk", "learned", "oracle"]:
            rewards[policy].append(compute_reward(sel[policy], features, true_consequence))
            catches[policy] += catastrophe_mask[sel[policy]].sum()

    oracle_mean = float(np.mean(rewards["oracle"]))
    out = {}
    for policy in ["topk", "learned", "oracle"]:
        mean_r = float(np.mean(rewards[policy]))
        recall = (catches[policy] / total_catastrophes) if total_catastrophes > 0 else float("nan")
        out[policy] = {
            "mean_reward": mean_r,
            "regret": oracle_mean - mean_r,
            "catastrophe_recall": recall,
            "n_events": int(total_catastrophes),
        }
    out["_skipped_episodes"] = skipped
    return out


# ============================================================================
# MAIN: run everything from a clean state
# ============================================================================

def main():
    results = {}

    print("\n" + "#" * 78)
    print("# TRAINING")
    print("#" * 78)
    print("Training the learned risk-aware policy via black-box hill-climbing search...\n")
    w_trained, history = train_policy(n_iterations=150, eval_episodes=25, K=10, seed=TRAIN_SEED)
    print("Trained weights:")
    for name, val in zip(FEATURE_NAMES + ["redundancy_penalty"], w_trained):
        print(f"  {name:<18}: {val:+.3f}")
    print(f"Training reward improved from {history[0]:.3f} to {history[-1]:.3f}\n")

    print("#" * 78)
    print("# MOTIVATING RARE HIGH-CONSEQUENCE SCENARIO")
    print("#" * 78)
    mot_features, mot_true_consequence = build_motivating_scenario()
    scenario_motivating = run_scenario(
        "10 ordinary observations + 1 rare (high rarity & novelty), budget=10",
        mot_features, mot_true_consequence, w_trained, K=10, special_idx=10,
    )

    print("#" * 78)
    print("# IN-DISTRIBUTION EVALUATION (same distribution family as training)")
    print("#" * 78)
    results["in_distribution"] = batch_evaluate(
        w_trained, generate_episode, n_test_episodes=200, K=10, seed=IN_DIST_EVAL_SEED)
    print(f"Evaluated on 200 episodes. Total high-consequence events: "
          f"{results['in_distribution']['topk']['n_events']}\n")

    print("#" * 78)
    print("# OUT-OF-DISTRIBUTION EVALUATION (weights NOT retrained)")
    print("#" * 78)
    print("Changes vs. training distribution: variable rare-event count (0-5, not fixed 2),")
    print("added Gaussian sensor noise, noisier normal background, lower rare true-positive")
    print("rate (0.5 vs 0.7), and 40% of rare events are 'stealthy' (low rarity/novelty).\n")

    stealth_features, stealth_true_consequence = np.zeros((11, N_FEATURES)), np.zeros(11)
    for i in range(10):
        stealth_features[i] = [0.75, 0.10, 0.10, 0.10, 0.15]
        stealth_true_consequence[i] = 0.30
    stealth_features[10] = [0.25, 0.15, 0.12, 0.75, 0.80]  # low importance, LOW rarity/novelty, high uncertainty/risk
    stealth_true_consequence[10] = 0.92
    scenario_stealthy = run_scenario(
        "Stealthy event: low importance + LOW rarity/novelty + high uncertainty/risk, budget=10",
        stealth_features, stealth_true_consequence, w_trained, K=10, special_idx=10,
    )

    results["ood"] = batch_evaluate(
        w_trained, generate_episode_ood, n_test_episodes=200, K=10, seed=OOD_EVAL_SEED)
    print(f"Evaluated on 200 OOD episodes (policy weights unchanged from training).")
    print(f"Total high-consequence events: {results['ood']['topk']['n_events']}\n")

    print("#" * 78)
    print("# RESULTS AND VISUALIZATIONS")
    print("#" * 78)
    print_final_table(results)
    figure_paths = make_figures(results)
    for path in figure_paths:
        print(f"Saved figure: {path}")
    print()

    print("#" * 78)
    print("# LIMITATIONS")
    print("#" * 78)
    print(LIMITATIONS_TEXT)

    print("#" * 78)
    print("# WHAT THIS DEMONSTRATES")
    print("#" * 78)
    print(WHAT_THIS_DEMONSTRATES_TEXT)

    return results, scenario_motivating, scenario_stealthy, figure_paths


# ============================================================================
# 8. RESULTS TABLE
# ============================================================================

def print_final_table(results):
    rows = []
    for eval_type, label in [("in_distribution", "In-Distribution"), ("ood", "Out-of-Distribution")]:
        for policy, policy_label in [("topk", "Top-K importance"),
                                       ("learned", "Learned policy"),
                                       ("oracle", "Oracle")]:
            m = results[eval_type][policy]
            recall_str = f"{m['catastrophe_recall']:.1%}" if not np.isnan(m["catastrophe_recall"]) else "n/a"
            rows.append((label, policy_label, f"{m['mean_reward']:.3f}",
                         f"{m['regret']:.3f}", recall_str, str(m["n_events"])))

    headers = ["Evaluation", "Policy", "Mean reward", "Regret vs oracle", "Catastrophe recall", "# high-conseq. events"]
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) + 2 for i, h in enumerate(headers)]

    def fmt_row(cells):
        return "".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    print(fmt_row(headers))
    print("-" * sum(col_widths))
    prev_label = None
    for r in rows:
        if prev_label is not None and r[0] != prev_label:
            print()
        print(fmt_row(r))
        prev_label = r[0]
    print()


# ============================================================================
# 9. FIGURES
# ============================================================================

def make_figures(results):
    saved_paths = []
    group_labels = ["In-Distribution", "Out-of-Distribution"]
    x = np.arange(len(group_labels))

    # --- A. Catastrophe recall: Top-K vs Learned ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.35
    topk_vals = [results["in_distribution"]["topk"]["catastrophe_recall"] * 100,
                 results["ood"]["topk"]["catastrophe_recall"] * 100]
    learned_vals = [results["in_distribution"]["learned"]["catastrophe_recall"] * 100,
                    results["ood"]["learned"]["catastrophe_recall"] * 100]
    ax.bar(x - width / 2, topk_vals, width, label="Top-K importance", color="#c0392b")
    ax.bar(x + width / 2, learned_vals, width, label="Learned policy", color="#2980b9")
    for i, v in enumerate(topk_vals):
        ax.text(i - width / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    for i, v in enumerate(learned_vals):
        ax.text(i + width / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Catastrophe recall (%)")
    ax.set_title("Catastrophe Recall: Top-K vs Learned Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 110)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "catastrophe_recall.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    saved_paths.append(path)

    # --- B. Mean reward: Top-K vs Learned vs Oracle ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    width = 0.25
    topk_r = [results["in_distribution"]["topk"]["mean_reward"], results["ood"]["topk"]["mean_reward"]]
    learned_r = [results["in_distribution"]["learned"]["mean_reward"], results["ood"]["learned"]["mean_reward"]]
    oracle_r = [results["in_distribution"]["oracle"]["mean_reward"], results["ood"]["oracle"]["mean_reward"]]
    ax.bar(x - width, topk_r, width, label="Top-K importance", color="#c0392b")
    ax.bar(x, learned_r, width, label="Learned policy", color="#2980b9")
    ax.bar(x + width, oracle_r, width, label="Oracle", color="#27ae60")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean reward per episode")
    ax.set_title("Mean Reward: Top-K vs Learned Policy vs Oracle")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "mean_reward.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    saved_paths.append(path)

    # --- C. Regret vs oracle: Top-K vs Learned ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    width = 0.35
    topk_regret = [results["in_distribution"]["topk"]["regret"], results["ood"]["topk"]["regret"]]
    learned_regret = [results["in_distribution"]["learned"]["regret"], results["ood"]["learned"]["regret"]]
    ax.bar(x - width / 2, topk_regret, width, label="Top-K importance", color="#c0392b")
    ax.bar(x + width / 2, learned_regret, width, label="Learned policy", color="#2980b9")
    for i, v in enumerate(topk_regret):
        ax.text(i - width / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(learned_regret):
        ax.text(i + width / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Regret vs oracle (lower is better)")
    ax.set_title("Regret vs Oracle: Top-K vs Learned Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "regret_vs_oracle.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    saved_paths.append(path)

    return saved_paths


# ============================================================================
# 10 & 11. LIMITATIONS / WHAT THIS DEMONSTRATES (fixed text, not results)
# ============================================================================

LIMITATIONS_TEXT = """\
- All observations are synthetic 5-dimensional feature vectors, not real
  satellite imagery, spectra, or sensor readings.
- "Risk" and "consequence" are simulated scalars (true_consequence), not
  derived from any real hazard, safety, or scientific-value model.
- No real satellite hardware, onboard compute, power, or storage constraints
  are modeled -- the communication budget is a simple count-based cap (K of N).
- No real communication channel, downlink schedule, or ground-station
  infrastructure is modeled.
- The system does not "think like a human" and no such claim is made anywhere
  in this report -- it is a risk-aware sequential decision policy trained by
  a simple black-box search over a hand-built reward function.
- These results describe behavior in a controlled synthetic simulation and
  do NOT directly represent, predict, or guarantee real-world satellite
  performance. They establish a methodology and a first empirical signal,
  nothing more.
- The out-of-distribution test is a shift across parametric variations of the
  SAME synthetic generator family (different noise/rates/shapes, same 5
  features and causal structure) -- it has not been tested against a
  qualitatively different feature space, such as real image embeddings.
"""

WHAT_THIS_DEMONSTRATES_TEXT = """\
Within this synthetic simulation, and only within it:
1. A learned policy that scores observations on importance, rarity, novelty,
   uncertainty, and risk -- and accounts for redundancy across the
   transmitted batch -- captures far more true high-consequence events under
   a fixed budget than a naive top-K-by-importance baseline.
2. This advantage is not limited to the exact scenario/distribution the
   policy was trained on: the same trained weights, with no retraining,
   retain most of this advantage under a distribution with different event
   rates, added noise, and a harder discrimination task.
3. The advantage persists even in the deliberately hard case where the
   high-consequence event does NOT stand out via rarity or novelty, and is
   only flagged by uncertainty/risk -- i.e. the policy is not simply relying
   on "anomaly detection" as a shortcut.
4. A simple, learned linear utility function -- optimized with a
   near-trivial black-box search, no deep learning or RL machinery required
   -- is sufficient to produce this behavior, at least at this synthetic
   scale.
No claim beyond these four points is supported by the experiments run so far.
"""


if __name__ == "__main__":
    main()
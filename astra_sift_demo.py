"""
ASTRA-SIFT — DEMO OUTPUT
================================================================================
This is a DEMONSTRATION, not a new experiment. It does not change, retrain, or
re-tune anything: it reuses the exact same functions and the exact same
training call (same seed, same hyperparameters) as the final experiment, so
the "ASTRA-SIFT" policy shown here is the identical trained policy already
reported on. Nothing here feeds back into, or alters, the reported results.

Purpose: show, on ONE easy-to-read simulated satellite pass, how ASTRA-SIFT's
selection differs from a naive Top-K-by-importance baseline -- in particular,
whether it preserves a single high-consequence observation that Top-K would
otherwise miss.

Run with: python3 astra_sift_demo.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_NAMES = ["importance", "rarity", "novelty", "uncertainty", "risk_estimate"]
N_FEATURES = len(FEATURE_NAMES)
TRAIN_SEED = 0  # identical to the final experiment -- reproduces the same trained policy

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================================
# UNCHANGED CORE FUNCTIONS
# (copied verbatim from final_experiment_risk_aware_selection.py -- nothing
#  here is modified, retuned, or re-derived)
# ============================================================================

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def generate_episode(n_normal=39, n_rare=2, rare_true_positive_rate=0.7, rng=None):
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
        true_consequence[idx] = rng.beta(9, 1) if rng.random() < rare_true_positive_rate else rng.beta(2, 10)
    return features, true_consequence


def select_topk_importance(features, K):
    scores = features[:, FEATURE_NAMES.index("importance")]
    return list(np.argsort(-scores)[:K])


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
    for _ in range(n_iterations):
        candidate = w + train_rng.normal(0, sigma, size=w.shape)
        candidate_reward = evaluate_policy_weights(candidate, eval_episodes, K, train_rng)
        if candidate_reward > best_reward:
            w, best_reward = candidate, candidate_reward
    return w


# ============================================================================
# DEMO SECTION (new -- everything below this line is presentation only)
# ============================================================================
# One hand-built, fully deterministic simulated satellite pass: 15 candidate
# observations, transmission budget K=10 (so 5 must be dropped). Labels are
# purely descriptive/cosmetic for readability -- they play no role in either
# policy's decision, which only ever sees the 5 numeric feature columns.

def build_demo_pass():
    labels = [
        "Farmland tile - Iowa",
        "Urban block - routine",
        "Coastal monitoring - normal tide",
        "Shipping lane - routine traffic",
        "Forest canopy - healthy",
        "Industrial zone - normal operations",
        "Reservoir level - stable",
        "Highway traffic - typical",
        "Cropland irrigation - normal",
        "Port activity - routine",
        "Cloud cover anomaly - benign",
        "New construction site",
        "Seasonal algae bloom",
        "Minor road closure",
        "Unusual thermal signature - industrial site",  # the critical one
    ]

    # [importance, rarity, novelty, uncertainty, risk_estimate]
    features = np.array([
        [0.82, 0.08, 0.07, 0.06, 0.10],
        [0.78, 0.10, 0.09, 0.08, 0.12],
        [0.75, 0.12, 0.10, 0.10, 0.11],
        [0.80, 0.09, 0.08, 0.07, 0.09],
        [0.73, 0.11, 0.10, 0.09, 0.10],
        [0.77, 0.13, 0.12, 0.11, 0.14],
        [0.70, 0.10, 0.09, 0.08, 0.10],
        [0.74, 0.09, 0.08, 0.07, 0.09],
        [0.79, 0.10, 0.09, 0.08, 0.11],
        [0.81, 0.11, 0.10, 0.09, 0.12],
        [0.35, 0.55, 0.50, 0.45, 0.30],
        [0.40, 0.45, 0.48, 0.35, 0.28],
        [0.38, 0.50, 0.40, 0.30, 0.25],
        [0.42, 0.30, 0.28, 0.20, 0.18],
        [0.22, 0.88, 0.82, 0.78, 0.75],
    ])

    # Hidden ground truth (NOT seen by either policy -- evaluation/highlight only)
    true_consequence = np.array([
        0.15, 0.20, 0.18, 0.12, 0.16, 0.22, 0.14, 0.13, 0.17, 0.19,
        0.25, 0.22, 0.20, 0.15,
        0.93,  # the critical observation
    ])

    return labels, features, true_consequence


def print_observation_table(labels, features):
    print("=" * 100)
    print("1. GENERATED OBSERVATIONS FOR THIS SIMULATED SATELLITE PASS")
    print("=" * 100)
    header = f"{'#':<4}{'Observation':<46}{'Importance':<12}{'Rarity':<10}{'Novelty':<10}{'Uncertainty':<13}{'Risk'}"
    print(header)
    print("-" * len(header))
    for i, (label, feat) in enumerate(zip(labels, features)):
        print(f"{i:<4}{label:<46}{feat[0]:<12.2f}{feat[1]:<10.2f}{feat[2]:<10.2f}{feat[3]:<13.2f}{feat[4]:.2f}")
    print()


def run_demo():
    labels, features, true_consequence = build_demo_pass()
    N, K = len(labels), 10
    critical_idx = int(np.argmax(true_consequence))  # the one hidden high-consequence observation

    print_observation_table(labels, features)

    print(f"Transmission budget for this pass: {K} out of {N} observations can be sent.\n")

    # Use the SAME trained policy as the final experiment (identical training call)
    print("Loading ASTRA-SIFT's trained policy (same training call/seed as the final experiment)...")
    w_trained = train_policy(n_iterations=150, eval_episodes=25, K=10, seed=TRAIN_SEED)
    print("Done.\n")

    topk_selected = select_topk_importance(features, K)
    astra_selected = select_learned(features, w_trained, K)

    print("=" * 100)
    print("2-3. TOP-K IMPORTANCE — WHAT IT WOULD TRANSMIT")
    print("=" * 100)
    for i in sorted(topk_selected):
        marker = "  <-- CRITICAL OBSERVATION" if i == critical_idx else ""
        print(f"  [{i}] {labels[i]}{marker}")
    print()

    print("=" * 100)
    print("4. ASTRA-SIFT — WHAT IT WOULD TRANSMIT")
    print("=" * 100)
    for i in sorted(astra_selected):
        marker = "  <-- CRITICAL OBSERVATION" if i == critical_idx else ""
        print(f"  [{i}] {labels[i]}{marker}")
    print()

    topk_caught = critical_idx in topk_selected
    astra_caught = critical_idx in astra_selected

    print("=" * 100)
    print("5. HIGHLIGHT: THE HIGH-CONSEQUENCE OBSERVATION")
    print("=" * 100)
    print(f"  Observation [{critical_idx}] \"{labels[critical_idx]}\"")
    print(f"  -> Looks unimportant by ordinary standards (importance = {features[critical_idx][0]:.2f}),")
    print(f"     but is highly rare/novel/uncertain/risky (rarity={features[critical_idx][1]:.2f}, "
          f"novelty={features[critical_idx][2]:.2f}, uncertainty={features[critical_idx][3]:.2f}, "
          f"risk={features[critical_idx][4]:.2f}),")
    print(f"     and its true significance (hidden from both policies) is {true_consequence[critical_idx]:.2f} "
          f"-- by far the highest of any observation in this pass.\n")
    print(f"  Top-K importance selected it?      {'YES' if topk_caught else 'NO'}")
    print(f"  ASTRA-SIFT selected it?            {'YES' if astra_caught else 'NO'}")
    print()

    print("=" * 100)
    print("6. SUMMARY")
    print("=" * 100)
    print(f"Top-K selected: {', '.join(str(i) for i in sorted(topk_selected))}")
    print(f"ASTRA-SIFT selected: {', '.join(str(i) for i in sorted(astra_selected))}")
    print(f"Critical observation preserved: {'YES' if astra_caught else 'NO'}")
    if astra_caught and not topk_caught:
        print("\n--> ASTRA-SIFT caught the critical observation that Top-K importance would have missed.")
    elif astra_caught and topk_caught:
        print("\n--> Both approaches happened to catch the critical observation in this particular pass.")
    else:
        print("\n--> ASTRA-SIFT did not catch the critical observation in this particular pass.")

    make_demo_figure(labels, features, true_consequence, topk_selected, astra_selected, critical_idx)

    return {
        "labels": labels, "features": features, "true_consequence": true_consequence,
        "topk_selected": topk_selected, "astra_selected": astra_selected,
        "critical_idx": critical_idx, "topk_caught": topk_caught, "astra_caught": astra_caught,
    }


def make_demo_figure(labels, features, true_consequence, topk_selected, astra_selected, critical_idx):
    """One simple, recruiter-readable chart: importance (x) vs risk_estimate (y),
    color-coded by which policy selected each observation, with the single
    high-consequence observation clearly ringed and labeled."""
    N = len(labels)
    importance = features[:, 0]
    risk = features[:, 4]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(N):
        in_topk = i in topk_selected
        in_astra = i in astra_selected
        if in_topk and in_astra:
            color, cat = "#8e44ad", "Selected by both"
        elif in_astra:
            color, cat = "#2980b9", "ASTRA-SIFT only"
        elif in_topk:
            color, cat = "#c0392b", "Top-K only"
        else:
            color, cat = "#bdc3c7", "Not selected"
        ax.scatter(importance[i], risk[i], s=160, color=color, edgecolor="white",
                   linewidth=0.8, zorder=3, label=cat)

    # Ring and label the critical observation (placed in the empty upper-right
    # region of the chart so it never overlaps the legend or the data clusters)
    ax.scatter(importance[critical_idx], risk[critical_idx], s=500, facecolors="none",
               edgecolors="black", linewidths=2.5, zorder=4)
    ax.annotate("CRITICAL\n(high true consequence,\nlow ordinary importance)",
                xy=(importance[critical_idx], risk[critical_idx]),
                xytext=(0.55, 0.92),
                fontsize=9, ha="left", va="top", color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8),
                arrowprops=dict(arrowstyle="->", color="black"))

    # De-duplicate legend, placed lower-right where there is empty space
    handles, plot_labels = ax.get_legend_handles_labels()
    seen, uniq_handles, uniq_labels = set(), [], []
    for h, l in zip(handles, plot_labels):
        if l not in seen:
            uniq_handles.append(h); uniq_labels.append(l); seen.add(l)
    ax.legend(uniq_handles, uniq_labels, loc="lower left", fontsize=9,
              bbox_to_anchor=(0.0, -0.22), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.2)

    ax.set_xlabel("Importance (naive salience score)")
    ax.set_ylabel("Risk estimate")
    ax.set_title("Demo Pass: What Top-K vs ASTRA-SIFT Transmit\n(budget = 10 of 15 observations)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "demo_pass.png")
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"\nSaved figure: {path}")


if __name__ == "__main__":
    run_demo()

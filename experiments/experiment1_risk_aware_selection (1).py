# %% [markdown]
# # Experiment 1: Risk-Aware Observation Selection Under a Communication Budget
#
# Goal (single sentence): show that a *learned* selection policy which reasons
# over {importance, rarity, novelty, uncertainty, risk} + redundancy can beat
# a naive "top-K by importance" baseline on scenarios where a rare,
# low-importance-but-high-consequence observation would otherwise be dropped.
#
# This script is deliberately small:
#   - No images, no deep learning, no RL library, no dashboard.
#   - "Observations" are 5-dimensional feature vectors, synthetically generated.
#   - The "learned policy" is a linear utility function trained with a simple
#     black-box search (no autograd needed) — this still qualifies as a
#     contextual-bandit-style approach: it learns a context -> utility mapping
#     from reward feedback, it's just optimized with random search instead of
#     gradient descent. That's a legitimate, minimal starting point.
#
# Run with: python experiment1_risk_aware_selection.py

import numpy as np

rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Feature definition
#
# Each observation is a 5-dim vector:
#   [importance, rarity, novelty, uncertainty, risk_estimate]
# All in roughly [0, 1]. A hidden `true_consequence` value (NOT visible to
# any policy, only used for evaluation) represents "how much it actually
# mattered that this got transmitted" — this is our synthetic ground truth,
# playing the role real ground-station confirmation would play later.

FEATURE_NAMES = ["importance", "rarity", "novelty", "uncertainty", "risk_estimate"]
N_FEATURES = len(FEATURE_NAMES)


def cosine_sim(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# %% [markdown]
# ## 2. Synthetic episode generator
#
# Each episode = one satellite pass with N candidate observations.
# Most are "ordinary but useful" (high importance, low rarity/novelty/
# uncertainty, modest true consequence). A small number are "rare" —
# low importance, high rarity/novelty/uncertainty — and *some* (not all,
# to keep it realistic) of these rare ones carry high true_consequence.
# This asymmetry (rare items are only *sometimes* actually important) is
# what makes the problem genuinely about risk, not just novelty detection.

def generate_episode(n_normal=39, n_rare=2, rare_true_positive_rate=0.7, rng=rng):
    n_total = n_normal + n_rare
    features = np.zeros((n_total, N_FEATURES))
    true_consequence = np.zeros(n_total)
    is_rare = np.zeros(n_total, dtype=bool)

    # --- ordinary-but-useful observations ---
    for i in range(n_normal):
        importance = rng.beta(8, 2)          # skewed high
        rarity = rng.beta(2, 8)               # skewed low
        novelty = rng.beta(2, 8)
        uncertainty = rng.beta(2, 8)
        risk_estimate = rng.beta(2, 8) * 0.5 + importance * 0.1  # loosely tracks importance, stays low
        features[i] = [importance, rarity, novelty, uncertainty, risk_estimate]
        # ordinary items have low-ish true consequence, with small noise
        true_consequence[i] = rng.beta(2, 10)

    # --- rare observations ---
    for k in range(n_rare):
        idx = n_normal + k
        importance = rng.beta(2, 8)           # looks unimportant by ordinary standards
        rarity = rng.beta(8, 2)               # skewed high
        novelty = rng.beta(8, 2)
        uncertainty = rng.beta(6, 3)           # model is unsure what this is
        risk_estimate = rng.beta(5, 3)         # onboard risk estimator flags it, imperfectly
        features[idx] = [importance, rarity, novelty, uncertainty, risk_estimate]
        is_rare[idx] = True
        # only a fraction of rare/novel things are TRUE high-consequence events;
        # the rest are false alarms (sensor noise, benign anomalies, etc.)
        if rng.random() < rare_true_positive_rate:
            true_consequence[idx] = rng.beta(9, 1)   # genuinely high consequence
        else:
            true_consequence[idx] = rng.beta(2, 10)  # false alarm, turned out to be nothing

    return features, true_consequence, is_rare


# %% [markdown]
# ## 3. Policies
#
# ### 3a. Baseline: top-K by importance alone
# This is exactly the naive system the whole project argues against.

def select_topk_importance(features, K):
    scores = features[:, FEATURE_NAMES.index("importance")]
    return list(np.argsort(-scores)[:K])


# ### 3b. Oracle (evaluation-only): top-K by true_consequence
# Not a real policy (it cheats by seeing hidden ground truth) — used only
# to compute regret, i.e. "how far from the best-possible batch are we".

def select_oracle(true_consequence, K):
    return list(np.argsort(-true_consequence)[:K])


# ### 3c. Learned policy: greedy sequential selection with a linear utility
# score(i | already_selected) = w[0:5] . features[i]  -  w[5] * redundancy(i)
# where redundancy(i) = similarity to the *closest* already-selected item.
# Picking sequentially (not a one-shot top-K) is what lets redundancy and
# batch diversity actually influence the outcome.

def select_learned(features, w, K):
    n_total = features.shape[0]
    selected = []
    remaining = list(range(n_total))
    for _ in range(K):
        best_idx, best_score = None, -np.inf
        for idx in remaining:
            feat = features[idx]
            if selected:
                redundancy = max(cosine_sim(feat, features[s]) for s in selected)
            else:
                redundancy = 0.0
            score = float(np.dot(w[:N_FEATURES], feat) - w[N_FEATURES] * redundancy)
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


# %% [markdown]
# ## 4. Reward function
#
# R = sum(true_consequence of selected)  -  lambda * average pairwise
#     redundancy within the selected batch
#
# This is the objective the learned policy is trained against. Note the
# policy itself never sees true_consequence directly — only the *reward*
# computed from it during training, exactly like a bandit only observes
# reward for the arms it pulls.

LAMBDA_REDUNDANCY = 2.0


def compute_reward(selected, features, true_consequence, lam=LAMBDA_REDUNDANCY):
    value = true_consequence[selected].sum()
    if len(selected) > 1:
        sims = [
            cosine_sim(features[selected[i]], features[selected[j]])
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
        ]
        redundancy_penalty = lam * np.mean(sims)
    else:
        redundancy_penalty = 0.0
    return value - redundancy_penalty


# %% [markdown]
# ## 5. Training: simple black-box (zeroth-order) policy search
#
# No gradients, no autograd. We just do hill-climbing: propose a small
# random perturbation of the weight vector, keep it if it improves average
# reward over a batch of fresh synthetic episodes, discard it otherwise.
# This is a legitimate (if simple) contextual-bandit-style optimization —
# it learns purely from reward feedback on sampled episodes.

def evaluate_policy_weights(w, n_episodes, K, rng):
    rewards = []
    for _ in range(n_episodes):
        features, true_consequence, _ = generate_episode(rng=rng)
        selected = select_learned(features, w, K)
        rewards.append(compute_reward(selected, features, true_consequence))
    return float(np.mean(rewards))


def train_policy(n_iterations=150, eval_episodes=25, K=10, sigma=0.15, seed=0):
    train_rng = np.random.default_rng(seed)
    w = train_rng.normal(0, 0.1, size=N_FEATURES + 1)
    w[N_FEATURES] = 0.1  # start with a small positive redundancy weight
    best_reward = evaluate_policy_weights(w, eval_episodes, K, train_rng)

    history = [best_reward]
    for it in range(n_iterations):
        candidate = w + train_rng.normal(0, sigma, size=w.shape)
        candidate_reward = evaluate_policy_weights(candidate, eval_episodes, K, train_rng)
        if candidate_reward > best_reward:
            w, best_reward = candidate, candidate_reward
        history.append(best_reward)

    return w, history


# %% [markdown]
# ## 6. The motivating scenario, made concrete and deterministic
#
# Exactly the case from the design doc: 10 "ordinary but useful" observations
# (high importance) plus 1 rare, high-consequence observation that scores low
# on plain importance. Budget = 10, so something MUST be dropped.

def build_motivating_scenario():
    features = np.zeros((11, N_FEATURES))
    true_consequence = np.zeros(11)

    # 10 ordinary, solidly useful observations
    for i in range(10):
        features[i] = [0.75, 0.10, 0.10, 0.10, 0.15]
        true_consequence[i] = 0.30  # useful, but not critical

    # 1 rare, low-importance, high-uncertainty, potentially catastrophic observation
    features[10] = [0.20, 0.90, 0.85, 0.80, 0.70]
    true_consequence[10] = 0.95  # turns out to be the important one

    return features, true_consequence


def run_motivating_scenario(w):
    features, true_consequence = build_motivating_scenario()
    K = 10  # can only send 10 of the 11

    baseline_selected = select_topk_importance(features, K)
    learned_selected = select_learned(features, w, K)
    oracle_selected = select_oracle(true_consequence, K)

    rare_idx = 10
    print("=" * 70)
    print("MOTIVATING SCENARIO: 10 ordinary observations + 1 rare, budget = 10")
    print("=" * 70)
    print(f"Baseline (top-K by importance) selected indices: {sorted(baseline_selected)}")
    print(f"  -> included the rare high-consequence item? "
          f"{'YES' if rare_idx in baseline_selected else 'NO (dropped it)'}")
    print(f"  -> reward: {compute_reward(baseline_selected, features, true_consequence):.3f}")

    print(f"\nLearned policy selected indices: {sorted(learned_selected)}")
    print(f"  -> included the rare high-consequence item? "
          f"{'YES' if rare_idx in learned_selected else 'NO (dropped it)'}")
    print(f"  -> reward: {compute_reward(learned_selected, features, true_consequence):.3f}")

    print(f"\nOracle (cheats, sees true_consequence) selected indices: {sorted(oracle_selected)}")
    print(f"  -> reward: {compute_reward(oracle_selected, features, true_consequence):.3f}")
    print()


# %% [markdown]
# ## 7. Batch evaluation over many random episodes
#
# Reports, averaged over many test episodes:
#   - mean reward for baseline / learned / oracle
#   - regret = oracle_reward - policy_reward  (lower is better)
#   - "catastrophe recall" = fraction of TRUE high-consequence rare items
#     (true_consequence > 0.7) that actually got transmitted

def batch_evaluate(w, n_test_episodes=200, K=10, seed=123):
    test_rng = np.random.default_rng(seed)

    baseline_rewards, learned_rewards, oracle_rewards = [], [], []
    baseline_catches, learned_catches, total_catastrophes = 0, 0, 0

    for _ in range(n_test_episodes):
        features, true_consequence, is_rare = generate_episode(rng=test_rng)

        catastrophe_mask = true_consequence > 0.7
        total_catastrophes += catastrophe_mask.sum()

        baseline_sel = select_topk_importance(features, K)
        learned_sel = select_learned(features, w, K)
        oracle_sel = select_oracle(true_consequence, K)

        baseline_rewards.append(compute_reward(baseline_sel, features, true_consequence))
        learned_rewards.append(compute_reward(learned_sel, features, true_consequence))
        oracle_rewards.append(compute_reward(oracle_sel, features, true_consequence))

        baseline_catches += catastrophe_mask[baseline_sel].sum()
        learned_catches += catastrophe_mask[learned_sel].sum()

    oracle_mean = np.mean(oracle_rewards)
    print("=" * 70)
    print(f"BATCH EVALUATION over {n_test_episodes} random episodes (budget K={K})")
    print("=" * 70)
    print(f"{'Policy':<20}{'Mean reward':<15}{'Regret vs oracle':<20}{'Catastrophe recall'}")
    for name, rewards, catches in [
        ("Top-K importance", baseline_rewards, baseline_catches),
        ("Learned policy", learned_rewards, learned_catches),
        ("Oracle (cheating)", oracle_rewards, None),
    ]:
        mean_r = np.mean(rewards)
        regret = oracle_mean - mean_r
        recall = (catches / total_catastrophes) if (catches is not None and total_catastrophes > 0) else float("nan")
        recall_str = f"{recall:.1%}" if not np.isnan(recall) else "n/a"
        print(f"{name:<20}{mean_r:<15.3f}{regret:<20.3f}{recall_str}")
    print(f"\n(Total true high-consequence events across all episodes: {int(total_catastrophes)})")
    print()


# %% [markdown]
# ## 8. Out-of-distribution generalization test
#
# Everything above trains and evaluates on ONE synthetic distribution. That
# alone can't tell us whether the policy learned something general
# ("weigh uncertainty/risk highly, penalize redundancy") or just overfit to
# the specific shape of the training generator. So here we build a SECOND,
# deliberately different generator, and evaluate the SAME already-trained
# weights on it with zero retraining.
#
# WHAT CHANGES vs. the training distribution (generate_episode):
#   1. Event frequency: training always injects exactly 2 rare events per
#      episode. Here, the count is random per episode (0 to 5) -- some
#      passes have no rare event at all, others have several at once.
#   2. Sensor noise: every feature gets added Gaussian noise (std=0.07,
#      then clipped to [0,1]) before the policy sees it. Training data has
#      no such noise.
#   3. Background "normalcy" is noisier: ordinary observations have higher
#      baseline rarity/novelty/uncertainty (Beta(3,6) instead of Beta(2,8)),
#      so the normal/rare boundary is fuzzier -- harder to separate.
#   4. Harder discrimination: only 50% of rare events are true positives
#      (vs. 70% in training) -- more false alarms to sift through.
#   5. "Stealthy" high-consequence events: 40% of rare events are built to
#      have LOW rarity/novelty (i.e. they don't stand out via those two
#      features at all) but still carry high true_consequence, signaled
#      only through uncertainty/risk_estimate. This directly probes whether
#      the policy leans on rarity/novelty as a crutch or genuinely uses the
#      uncertainty/risk signal.

def generate_episode_ood(n_normal=39, rng=rng):
    n_rare = int(rng.integers(0, 6))  # variable event count: 0-5 (training used a fixed 2)
    n_total = n_normal + n_rare
    features = np.zeros((n_total, N_FEATURES))
    true_consequence = np.zeros(n_total)
    is_rare = np.zeros(n_total, dtype=bool)
    noise_std = 0.07  # sensor/environmental noise absent from training distribution

    for i in range(n_normal):
        importance = rng.beta(5, 3)              # broader/lower than training's Beta(8,2)
        rarity = rng.beta(3, 6)                    # noisier background than training's Beta(2,8)
        novelty = rng.beta(3, 6)
        uncertainty = rng.beta(3, 6)
        risk_estimate = rng.beta(3, 6) * 0.6 + importance * 0.05
        vec = np.clip(
            np.array([importance, rarity, novelty, uncertainty, risk_estimate])
            + rng.normal(0, noise_std, size=N_FEATURES),
            0, 1,
        )
        features[i] = vec
        true_consequence[i] = float(np.clip(rng.beta(2, 10) + rng.normal(0, 0.05), 0, 1))

    stealthy_fraction = 0.4       # fraction of rare events that do NOT show high rarity/novelty
    rare_true_positive_rate = 0.5  # harder than training's 0.7 -> more false alarms

    for k in range(n_rare):
        idx = n_normal + k
        is_rare[idx] = True
        stealthy = rng.random() < stealthy_fraction
        importance = rng.beta(2, 7)
        if stealthy:
            rarity = rng.beta(3, 7)   # embeds close to "normal" -- low rarity
            novelty = rng.beta(3, 7)  # low novelty too
        else:
            rarity = rng.beta(7, 2)
            novelty = rng.beta(7, 2)
        uncertainty = rng.beta(6, 3)
        risk_estimate = rng.beta(5, 3)
        vec = np.clip(
            np.array([importance, rarity, novelty, uncertainty, risk_estimate])
            + rng.normal(0, noise_std, size=N_FEATURES),
            0, 1,
        )
        features[idx] = vec
        if rng.random() < rare_true_positive_rate:
            true_consequence[idx] = float(np.clip(rng.beta(9, 1) + rng.normal(0, 0.05), 0, 1))
        else:
            true_consequence[idx] = float(np.clip(rng.beta(2, 10) + rng.normal(0, 0.05), 0, 1))

    return features, true_consequence, is_rare


def ood_batch_evaluate(w, n_test_episodes=200, K=10, seed=999):
    """Same metrics as batch_evaluate, but sampling from generate_episode_ood
    and using the already-trained weights `w` with NO further training."""
    test_rng = np.random.default_rng(seed)

    baseline_rewards, learned_rewards, oracle_rewards = [], [], []
    baseline_catches, learned_catches, total_catastrophes = 0, 0, 0
    skipped_episodes = 0

    for _ in range(n_test_episodes):
        features, true_consequence, is_rare = generate_episode_ood(rng=test_rng)
        n_total = features.shape[0]
        if n_total < K:
            skipped_episodes += 1
            continue  # can't apply a budget of K if fewer than K candidates exist this pass

        catastrophe_mask = true_consequence > 0.7
        total_catastrophes += catastrophe_mask.sum()

        baseline_sel = select_topk_importance(features, K)
        learned_sel = select_learned(features, w, K)
        oracle_sel = select_oracle(true_consequence, K)

        baseline_rewards.append(compute_reward(baseline_sel, features, true_consequence))
        learned_rewards.append(compute_reward(learned_sel, features, true_consequence))
        oracle_rewards.append(compute_reward(oracle_sel, features, true_consequence))

        baseline_catches += catastrophe_mask[baseline_sel].sum()
        learned_catches += catastrophe_mask[learned_sel].sum()

    oracle_mean = np.mean(oracle_rewards)
    print("=" * 70)
    print(f"OUT-OF-DISTRIBUTION EVALUATION over {n_test_episodes - skipped_episodes} episodes "
          f"(budget K={K}, weights NOT retrained)")
    print("=" * 70)
    print(f"{'Policy':<20}{'Mean reward':<15}{'Regret vs oracle':<20}{'Catastrophe recall'}")
    for name, rewards, catches in [
        ("Top-K importance", baseline_rewards, baseline_catches),
        ("Learned policy", learned_rewards, learned_catches),
        ("Oracle (cheating)", oracle_rewards, None),
    ]:
        mean_r = np.mean(rewards)
        regret = oracle_mean - mean_r
        recall = (catches / total_catastrophes) if (catches is not None and total_catastrophes > 0) else float("nan")
        recall_str = f"{recall:.1%}" if not np.isnan(recall) else "n/a"
        print(f"{name:<20}{mean_r:<15.3f}{regret:<20.3f}{recall_str}")
    print(f"\n(Total true high-consequence events across all OOD episodes: {int(total_catastrophes)})")
    if skipped_episodes:
        print(f"(Skipped {skipped_episodes} episodes with fewer than K candidates.)")
    print()


# %% [markdown]
# ## 8b. Difficult OOD scenario: stealthy high-consequence event
#
# Required case: a high-consequence event with LOW ordinary importance AND
# LOW rarity/novelty (it does NOT stand out as "weird-looking") but HIGH
# uncertainty/risk_estimate. This is harder than the original motivating
# scenario, where the rare item was also high on rarity/novelty -- here we
# strip that crutch away and see if uncertainty/risk alone is enough.

def build_stealthy_scenario():
    features = np.zeros((11, N_FEATURES))
    true_consequence = np.zeros(11)

    for i in range(10):
        features[i] = [0.75, 0.10, 0.10, 0.10, 0.15]
        true_consequence[i] = 0.30

    # low importance, LOW rarity/novelty (blends in), but high uncertainty/risk
    features[10] = [0.25, 0.15, 0.12, 0.75, 0.80]
    true_consequence[10] = 0.92

    return features, true_consequence


def run_stealthy_scenario(w):
    features, true_consequence = build_stealthy_scenario()
    K = 10
    rare_idx = 10

    baseline_selected = select_topk_importance(features, K)
    learned_selected = select_learned(features, w, K)
    oracle_selected = select_oracle(true_consequence, K)

    print("=" * 70)
    print("DIFFICULT OOD SCENARIO: stealthy high-consequence event")
    print("(low importance, LOW rarity/novelty, but high uncertainty/risk)")
    print("=" * 70)
    print(f"Baseline (top-K by importance) selected indices: {sorted(baseline_selected)}")
    print(f"  -> included the stealthy high-consequence item? "
          f"{'YES' if rare_idx in baseline_selected else 'NO (dropped it)'}")
    print(f"  -> reward: {compute_reward(baseline_selected, features, true_consequence):.3f}")

    print(f"\nLearned policy selected indices: {sorted(learned_selected)}")
    print(f"  -> included the stealthy high-consequence item? "
          f"{'YES' if rare_idx in learned_selected else 'NO (dropped it)'}")
    print(f"  -> reward: {compute_reward(learned_selected, features, true_consequence):.3f}")

    print(f"\nOracle selected indices: {sorted(oracle_selected)}")
    print(f"  -> reward: {compute_reward(oracle_selected, features, true_consequence):.3f}")
    print()


# %% [markdown]
# ## 9. Run everything

if __name__ == "__main__":
    print("Training learned policy via black-box (hill-climbing) search...\n")
    w_trained, history = train_policy(n_iterations=150, eval_episodes=25, K=10, seed=0)

    print("Trained weights:")
    for name, val in zip(FEATURE_NAMES + ["redundancy_penalty"], w_trained):
        print(f"  {name:<18}: {val:+.3f}")
    print(f"\nTraining reward improved from {history[0]:.3f} to {history[-1]:.3f}\n")

    run_motivating_scenario(w_trained)
    batch_evaluate(w_trained, n_test_episodes=200, K=10)

    # --- Generalization check: SAME w_trained, NEW distribution, NO retraining ---
    run_stealthy_scenario(w_trained)
    ood_batch_evaluate(w_trained, n_test_episodes=200, K=10)

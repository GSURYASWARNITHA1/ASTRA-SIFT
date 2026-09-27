"""
ASTRA-SIFT — Streamlit Demo App
================================================================================
This app does NOT change the underlying ASTRA-SIFT selection logic, trained
weights, experiments, or reported results. Every core function below
(cosine_sim, generate_episode, select_topk_importance, select_learned,
compute_reward, evaluate_policy_weights, train_policy) is copied verbatim
from final_experiment_risk_aware_selection.py / astra_sift_demo.py. This file
only adds a Streamlit UI layer on top.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

FEATURE_NAMES = ["importance", "rarity", "novelty", "uncertainty", "risk_estimate"]
N_FEATURES = len(FEATURE_NAMES)
TRAIN_SEED = 0  # identical to the final experiment -- reproduces the same trained policy
FIGURES_DIR = os.path.join("results", "figures")


# ============================================================================
# UNCHANGED CORE FUNCTIONS
# (copied verbatim -- nothing here is modified, retuned, or re-derived)
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


@st.cache_resource
def get_trained_policy():
    """Trains once per app process (identical call to the final experiment),
    then reuses the result -- this is NOT retraining on new data, it just
    avoids re-running the same deterministic training every button click."""
    return train_policy(n_iterations=150, eval_episodes=25, K=10, seed=TRAIN_SEED)


# ============================================================================
# DEMO PASS (identical to astra_sift_demo.py -- presentation data only)
# ============================================================================

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
    true_consequence = np.array([
        0.15, 0.20, 0.18, 0.12, 0.16, 0.22, 0.14, 0.13, 0.17, 0.19,
        0.25, 0.22, 0.20, 0.15,
        0.93,  # the critical observation
    ])
    return labels, features, true_consequence


def make_comparison_figure(labels, features, topk_selected, astra_selected, critical_idx):
    N = len(labels)
    importance = features[:, 0]
    risk = features[:, 4]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(N):
        in_topk, in_astra = i in topk_selected, i in astra_selected
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

    ax.scatter(importance[critical_idx], risk[critical_idx], s=500, facecolors="none",
               edgecolors="black", linewidths=2.5, zorder=4)
    ax.annotate("CRITICAL\n(high true consequence,\nlow ordinary importance)",
                xy=(importance[critical_idx], risk[critical_idx]),
                xytext=(0.55, 0.92), fontsize=9, ha="left", va="top", color="black",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8),
                arrowprops=dict(arrowstyle="->", color="black"))

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
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="ASTRA-SIFT", layout="wide")

st.title("ASTRA-SIFT")
st.markdown("**Risk-aware satellite information selection under a limited communication budget.**")

st.markdown(
    "This is a synthetic, proof-of-concept demonstration. It does not use real "
    "satellite imagery or real satellite hardware, and the numbers shown are "
    "either the actual output of a fixed demonstration pass, or the actual "
    "reported results from the full 200-episode evaluation experiments -- "
    "nothing here is invented or re-tuned for this app."
)

run_clicked = st.button("Run Satellite Pass Demo")

if run_clicked:
    labels, features, true_consequence = build_demo_pass()
    N, K = len(labels), 10
    critical_idx = int(np.argmax(true_consequence))

    st.header("1. Generated Observations")
    st.caption(f"One simulated satellite pass: {N} candidate observations, transmission budget = {K}.")
    df = pd.DataFrame(features, columns=[n.replace("_", " ").title() for n in FEATURE_NAMES])
    df.insert(0, "Observation", labels)
    st.dataframe(df, width='stretch')

    with st.spinner("Loading ASTRA-SIFT's trained policy..."):
        w_trained = get_trained_policy()

    topk_selected = select_topk_importance(features, K)
    astra_selected = select_learned(features, w_trained, K)
    topk_caught = critical_idx in topk_selected
    astra_caught = critical_idx in astra_selected

    st.header("2. What Each Method Selected")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top-K Importance")
        for i in sorted(topk_selected):
            tag = " 🔴 CRITICAL" if i == critical_idx else ""
            st.write(f"[{i}] {labels[i]}{tag}")
    with col2:
        st.subheader("ASTRA-SIFT")
        for i in sorted(astra_selected):
            tag = " 🔴 CRITICAL" if i == critical_idx else ""
            st.write(f"[{i}] {labels[i]}{tag}")

    st.header("3. The Critical Observation")
    st.markdown(
        f"**[{critical_idx}] \"{labels[critical_idx]}\"** looks unimportant by ordinary "
        f"standards (importance = {features[critical_idx][0]:.2f}), but is highly rare, "
        f"novel, uncertain, and risky (rarity={features[critical_idx][1]:.2f}, "
        f"novelty={features[critical_idx][2]:.2f}, uncertainty={features[critical_idx][3]:.2f}, "
        f"risk={features[critical_idx][4]:.2f}). Its hidden true significance -- not seen "
        f"by either method -- is {true_consequence[critical_idx]:.2f}, the highest of any "
        f"observation in this pass."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.error("Top-K importance selected it? NO") if not topk_caught else st.success("Top-K importance selected it? YES")
    with c2:
        st.success("ASTRA-SIFT selected it? YES") if astra_caught else st.error("ASTRA-SIFT selected it? NO")

    st.header("4. Comparison Chart")
    fig = make_comparison_figure(labels, features, topk_selected, astra_selected, critical_idx)
    st.pyplot(fig)

    st.header("5. Final Evaluation Results (from the full 200-episode experiments)")
    st.caption(
        "These three figures are the actual saved outputs of the full in-distribution / "
        "out-of-distribution evaluation reported separately -- they are not regenerated "
        "or recomputed by this app."
    )
    figure_files = [
        ("catastrophe_recall.png", "Catastrophe recall: Top-K vs ASTRA-SIFT"),
        ("mean_reward.png", "Mean reward: Top-K vs ASTRA-SIFT vs Oracle"),
        ("regret_vs_oracle.png", "Regret vs oracle: Top-K vs ASTRA-SIFT"),
    ]
    any_found = False
    for fname, caption in figure_files:
        path = os.path.join(FIGURES_DIR, fname)
        if os.path.exists(path):
            st.image(path, caption=caption, width='stretch')
            any_found = True
        else:
            st.info(f"'{fname}' not found in {FIGURES_DIR}/ -- run the final experiment "
                    f"script first to generate it.")
    if not any_found:
        st.warning(
            "No final evaluation figures were found. Place mean_reward.png, "
            "catastrophe_recall.png, and regret_vs_oracle.png in results/figures/ "
            "(generated by the final experiment script) to display them here."
        )

    st.header("6. Conclusion")
    if astra_caught and not topk_caught:
        st.markdown(
            "**In this illustrative pass, ASTRA-SIFT transmitted the one observation that "
            "actually mattered, while a naive top-K-by-importance system would have missed "
            "it entirely** -- by favoring several ordinary, low-risk observations that "
            "individually looked more \"important\", but collectively carried far less real "
            "consequence. This is a single fixed demonstration, not a new performance claim: "
            "the actual measured recall/regret numbers come from the 200-episode evaluation "
            "figures above."
        )
    elif astra_caught and topk_caught:
        st.markdown(
            "**Both methods happened to transmit the critical observation in this particular "
            "pass.** ASTRA-SIFT is designed to catch this kind of low-importance, high-risk "
            "observation even when a naive top-K system would not -- see the evaluation "
            "figures above for the measured difference across 200 episodes."
        )
    else:
        st.markdown(
            "**Neither method transmitted the critical observation in this particular pass.** "
            "This is a single fixed demonstration; the measured difference in catastrophe "
            "recall between the two approaches is reported in the evaluation figures above."
        )
else:
    st.info("Click **Run Satellite Pass Demo** above to generate one simulated satellite pass "
            "and compare Top-K importance against ASTRA-SIFT.")

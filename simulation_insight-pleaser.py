"""
Toy simulations for Insights 1–3 (non-MELD version):

1. Signal quality (variance + entropy) for honest vs pleaser
2. Confidence trap: higher confidence, lower accuracy
3. Transfer / generalization: training on honest vs pleasers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11

# ------------------------------------------------------------
# Emotion space (same as MELD mapping)
# ------------------------------------------------------------

EMOTIONS = {
    "happy":      {"val": 0.8},
    "excited":    {"val": 0.7},
    "content":    {"val": 0.6},
    "neutral":    {"val": 0.0},
    "bored":      {"val": -0.3},
    "annoyed":    {"val": -0.4},
    "sad":        {"val": -0.7},
    "frustrated": {"val": -0.6},
}

EMOTION_NAMES = list(EMOTIONS.keys())
N = len(EMOTION_NAMES)

def valence(e):
    return EMOTIONS[e]["val"]

POSITIVE = [e for e in EMOTION_NAMES if valence(e) > 0.3]
NEGATIVE = [e for e in EMOTION_NAMES if valence(e) < -0.2]


# ============================================================
# 1) Insight 1 — Signal quality
# ============================================================

def sample_random_emotion():
    """Uniform random emotion (toy environment)."""
    return np.random.choice(EMOTION_NAMES)


def simulate_signal_quality(mask_prob=0.8, n_steps=500, n_runs=100):
    honest_vals, pleaser_vals = [], []
    honest_emos, pleaser_emos = [], []

    for _ in range(n_runs):
        for _ in range(n_steps):
            e = sample_random_emotion()
            honest_emos.append(e)
            honest_vals.append(valence(e))

            # pleaser: mask negative emotions with prob mask_prob
            if valence(e) < 0 and np.random.rand() < mask_prob:
                e_mask = np.random.choice(POSITIVE)
            else:
                e_mask = e

            pleaser_emos.append(e_mask)
            pleaser_vals.append(valence(e_mask))

    # valence variance
    var_h = np.var(honest_vals)
    var_p = np.var(pleaser_vals)

    # categorical entropy over emotion labels
    counts_h = np.array([honest_emos.count(e) for e in EMOTION_NAMES], dtype=float)
    counts_p = np.array([pleaser_emos.count(e) for e in EMOTION_NAMES], dtype=float)

    probs_h = counts_h / counts_h.sum()
    probs_p = counts_p / counts_p.sum()

    ent_h = entropy(probs_h, base=2)  # bits
    ent_p = entropy(probs_p, base=2)

    return var_h, var_p, ent_h, ent_p


# ============================================================
# 2) Insight 2 — Confidence trap
# ============================================================

class Learner:
    """Simple transition-matrix learner."""

    def __init__(self):
        self.C = np.zeros((N, N))

    def observe(self, prev, curr):
        i = EMOTION_NAMES.index(prev)
        j = EMOTION_NAMES.index(curr)
        self.C[i, j] += 1

    def model(self):
        M = self.C + 0.1           # pseudocounts
        return M / M.sum(axis=1, keepdims=True)

    def confidence(self):
        M = self.model()
        return M.max(axis=1).mean()

    def mse_to_true(self, TRUE):
        M = self.model()
        return np.mean((M - TRUE) ** 2)


def random_transition_matrix():
    T = np.random.rand(N, N)
    return T / T.sum(axis=1, keepdims=True)


def simulate_confidence_trap(mask_prob=0.8, n_obs=250, n_runs=60):
    TRUE = random_transition_matrix()

    conf_h_list, conf_p_list = [], []
    acc_h_list, acc_p_list = [], []

    for _ in range(n_runs):
        L_honest = Learner()
        L_pleaser = Learner()

        prev_h = np.random.choice(EMOTION_NAMES)
        prev_p_true = np.random.choice(EMOTION_NAMES)

        for _ in range(n_obs):
            # honest speaker: sees & expresses TRUE state
            e_h = np.random.choice(EMOTION_NAMES,
                                   p=TRUE[EMOTION_NAMES.index(prev_h)])
            L_honest.observe(prev_h, e_h)
            prev_h = e_h

            # pleaser speaker: internal true state + masked expression
            e_p_true = np.random.choice(EMOTION_NAMES,
                                        p=TRUE[EMOTION_NAMES.index(prev_p_true)])
            if valence(e_p_true) < 0 and np.random.rand() < mask_prob:
                e_p_exp = np.random.choice(POSITIVE)
            else:
                e_p_exp = e_p_true

            L_pleaser.observe(prev_p_true, e_p_exp)
            prev_p_true = e_p_exp

        conf_h_list.append(L_honest.confidence())
        conf_p_list.append(L_pleaser.confidence())

        acc_h_list.append(1 - L_honest.mse_to_true(TRUE))
        acc_p_list.append(1 - L_pleaser.mse_to_true(TRUE))

    return (np.mean(conf_h_list), np.mean(conf_p_list),
            np.mean(acc_h_list), np.mean(acc_p_list))


# ============================================================
# 3) Insight 3 — Transfer / generalization
# ============================================================

def simulate_transfer(mask_prob=0.8, n_train=200, n_test=120, n_runs=60):
    results = {"H→H": [], "P→H": [], "H→P": [], "P→P": []}

    for _ in range(n_runs):
        TRUE = random_transition_matrix()

        # train on honest data
        L_honest = Learner()
        prev = np.random.choice(EMOTION_NAMES)
        for _ in range(n_train):
            e = np.random.choice(EMOTION_NAMES,
                                 p=TRUE[EMOTION_NAMES.index(prev)])
            L_honest.observe(prev, e)
            prev = e
        M_h = L_honest.model()

        # train on masked (pleaser) data
        L_pleaser = Learner()
        prev = np.random.choice(EMOTION_NAMES)
        for _ in range(n_train):
            e = np.random.choice(EMOTION_NAMES,
                                 p=TRUE[EMOTION_NAMES.index(prev)])
            if valence(e) < 0 and np.random.rand() < mask_prob:
                e_mask = np.random.choice(POSITIVE)
            else:
                e_mask = e
            L_pleaser.observe(prev, e_mask)
            prev = e_mask
        M_p = L_pleaser.model()

        def test(model, pleaser_test=False):
            prev = np.random.choice(EMOTION_NAMES)
            correct = 0
            for _ in range(n_test):
                e_true = np.random.choice(
                    EMOTION_NAMES,
                    p=TRUE[EMOTION_NAMES.index(prev)]
                )
                if pleaser_test and valence(e_true) < 0 and np.random.rand() < mask_prob:
                    e_obs = np.random.choice(POSITIVE)
                else:
                    e_obs = e_true

                pred = EMOTION_NAMES[np.argmax(model[EMOTION_NAMES.index(prev)])]
                correct += (pred == e_obs)
                prev = e_obs
            return correct / n_test

        results["H→H"].append(test(M_h, pleaser_test=False))
        results["P→H"].append(test(M_p, pleaser_test=False))
        results["H→P"].append(test(M_h, pleaser_test=True))
        results["P→P"].append(test(M_p, pleaser_test=True))

    return {k: np.mean(v) for k, v in results.items()}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Running toy simulations for Insights 1–3...\n")

    # ----- Insight 1: signal quality -----
    var_h, var_p, ent_h, ent_p = simulate_signal_quality()
    print("Insight 1 — variance (honest, pleaser):", var_h, var_p)
    print("Insight 1 — entropy  (honest, pleaser):", ent_h, ent_p)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax_var, ax_ent = axes

    ax_var.bar(["Honest", "Pleaser"], [var_h, var_p])
    ax_var.set_title("Valence variance")
    ax_var.set_ylabel("Variance")

    ax_ent.bar(["Honest", "Pleaser"], [ent_h, ent_p])
    ax_ent.set_title("Emotion entropy")
    ax_ent.set_ylabel("Entropy (bits)")

    fig.suptitle("Insight 1: Pleasers have smoother, lower-entropy signals")
    fig.tight_layout()
    plt.show()

    # ----- Insight 2: confidence trap -----
    c_h, c_p, a_h, a_p = simulate_confidence_trap()
    print("\nInsight 2 — mean confidence (honest, pleaser):", c_h, c_p)
    print("Insight 2 — mean accuracy   (honest, pleaser):", a_h, a_p)
        # --- Plot Insight 2: confidence trap (optional) ---
    labels = ["Honest", "Pleaser"]

    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 4))
    ax_c, ax_a = axes2

    ax_c.bar(labels, [c_h, c_p])
    ax_c.set_title("Mean confidence")
    ax_c.set_ylabel("Confidence")

    ax_a.bar(labels, [a_h, a_p])
    ax_a.set_title("Mean accuracy vs TRUE")
    ax_a.set_ylabel("Accuracy")

    fig2.suptitle("Insight 2: Pleasers are more confident but less accurate")
    fig2.tight_layout()
    plt.show()


    # ----- Insight 3: transfer / generalization -----
    transfer_results = simulate_transfer()
    print("\nInsight 3 — transfer accuracies:")
    for k, v in transfer_results.items():
        print(f"  {k}: {v:.3f}")
    # --- Plot Insight 3: transfer performance (optional) ---
    keys = ["H→H", "P→H", "H→P", "P→P"]
    vals = [transfer_results[k] for k in keys]

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(keys, vals)
    ax3.set_ylim(0, max(vals) + 0.05)
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Insight 3: Transfer between honest / pleaser worlds")
    plt.tight_layout()
    plt.show()

"""
Simulation Insights 1–3:
1. Signal Quality (variance + entropy)
2. Confidence Trap
3. Transfer / Generalization

This script implements the abstract, non-MELD version of the model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import entropy

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# ------------------------------------------------------------
# Emotion setup (toy / abstract model)
# ------------------------------------------------------------

EMOTIONS = {
    'happy':      {'val': 0.8},
    'excited':    {'val': 0.7},
    'content':    {'val': 0.6},
    'neutral':    {'val': 0.0},
    'bored':      {'val': -0.3},
    'annoyed':    {'val': -0.4},
    'sad':        {'val': -0.7},
    'frustrated': {'val': -0.6},
}

EMOTION_NAMES = list(EMOTIONS.keys())
N = len(EMOTION_NAMES)
POSITIVE = [e for e in EMOTION_NAMES if EMOTIONS[e]['val'] > 0.3]
NEGATIVE = [e for e in EMOTION_NAMES if EMOTIONS[e]['val'] < -0.2]

def get_valence(e):
    return EMOTIONS[e]['val']


# ------------------------------------------------------------
# 1. Insight 1: Signal Quality
# ------------------------------------------------------------

def sample_random_emotion():
    """Uniform random events → random emotions for abstract simulation."""
    return np.random.choice(EMOTION_NAMES)

def simulate_signal_quality(mask_prob=0.8, n_steps=500, n_runs=100):

    honest_vals = []
    pleaser_vals = []

    for _ in range(n_runs):
        h_vals = []
        p_vals = []

        for _ in range(n_steps):
            e = sample_random_emotion()
            h_vals.append(get_valence(e))

            # pleaser masking rule
            if get_valence(e) < 0 and np.random.rand() < mask_prob:
                e2 = np.random.choice(POSITIVE)
            else:
                e2 = e
            p_vals.append(get_valence(e2))

        honest_vals.extend(h_vals)
        pleaser_vals.extend(p_vals)

    # Variance
    var_h = np.var(honest_vals)
    var_p = np.var(pleaser_vals)

    # Entropy
    def entropy_over_categories(vals):
        cats = [EMOTION_NAMES[np.argmin([abs(get_valence(e)-v) for e in EMOTION_NAMES])]
                for v in vals]
        counts = np.array([cats.count(e) for e in EMOTION_NAMES])
        return entropy(counts / counts.sum())

    ent_h = entropy_over_categories(honest_vals)
    ent_p = entropy_over_categories(pleaser_vals)

    return var_h, var_p, ent_h, ent_p


# ------------------------------------------------------------
# 2. Insight 2: Confidence Trap
# ------------------------------------------------------------

class Learner:
    def __init__(self):
        self.C = np.zeros((N, N))

    def observe(self, a, b):
        i = EMOTION_NAMES.index(a)
        j = EMOTION_NAMES.index(b)
        self.C[i, j] += 1

    def model(self):
        M = self.C + 0.1
        return M / M.sum(axis=1, keepdims=True)

    def confidence(self):
        M = self.model()
        max_probs = M.max(axis=1)
        return max_probs.mean()

    def mse_to_true(self, TRUE):
        M = self.model()
        return np.mean((M - TRUE)**2)


def random_transition_matrix():
    """Generate toy TRUE_TRANSITIONS."""
    T = np.random.rand(N, N)
    T = T / T.sum(axis=1, keepdims=True)
    return T


def simulate_confidence_trap(mask_prob=0.8, n_obs=250, n_runs=60):
    TRUE = random_transition_matrix()
    conf_h, conf_p = [], []
    acc_h, acc_p = [], []

    for _ in range(n_runs):
        Lh = Learner()
        Lp = Learner()

        prev_h = np.random.choice(EMOTION_NAMES)
        prev_p = np.random.choice(EMOTION_NAMES)

        for _ in range(n_obs):
            # honest
            e = np.random.choice(EMOTION_NAMES, p=TRUE[EMOTION_NAMES.index(prev_h)])
            Lh.observe(prev_h, e)
            prev_h = e

            # pleaser
            e2 = np.random.choice(EMOTION_NAMES, p=TRUE[EMOTION_NAMES.index(prev_p)])
            if get_valence(e2) < 0 and np.random.rand() < mask_prob:
                e2_exp = np.random.choice(POSITIVE)
            else:
                e2_exp = e2
            Lp.observe(prev_p, e2_exp)
            prev_p = e2_exp

        conf_h.append(Lh.confidence())
        conf_p.append(Lp.confidence())

        acc_h.append(1 - Lh.mse_to_true(TRUE))
        acc_p.append(1 - Lp.mse_to_true(TRUE))

    return np.mean(conf_h), np.mean(conf_p), np.mean(acc_h), np.mean(acc_p)


# ------------------------------------------------------------
# 3. Insight 3: Transfer Learning
# ------------------------------------------------------------

def simulate_transfer(mask_prob=0.8, n_train=200, n_test=120, n_runs=60):
    results = {"H→H": [], "P→H": [], "H→P": [], "P→P": []}

    for _ in range(n_runs):
        TRUE = random_transition_matrix()

        # train honest
        Lh = Learner()
        prev = np.random.choice(EMOTION_NAMES)
        for _ in range(n_train):
            e = np.random.choice(EMOTION_NAMES, p=TRUE[EMOTION_NAMES.index(prev)])
            Lh.observe(prev, e)
            prev = e
        Mh = Lh.model()

        # train pleaser
        Lp = Learner()
        prev = np.random.choice(EMOTION_NAMES)
        for _ in range(n_train):
            e = np.random.choice(EMOTION_NAMES, p=TRUE[EMOTION_NAMES.index(prev)])
            if get_valence(e) < 0 and np.random.rand() < mask_prob:
                e2 = np.random.choice(POSITIVE)
            else:
                e2 = e
            Lp.observe(prev, e2)
            prev = e2
        Mp = Lp.model()

        # test on honest
        def test(model, pleaser_test=False):
            prev = np.random.choice(EMOTION_NAMES)
            correct = 0
            for _ in range(n_test):
                e = np.random.choice(EMOTION_NAMES, p=TRUE[EMOTION_NAMES.index(prev)])
                if pleaser_test and get_valence(e) < 0 and np.random.rand() < mask_prob:
                    e_actual = np.random.choice(POSITIVE)
                else:
                    e_actual = e
                pred = EMOTION_NAMES[np.argmax(model[EMOTION_NAMES.index(prev)])]
                correct += (pred == e_actual)
                prev = e_actual
            return correct / n_test

        results["H→H"].append(test(Mh, pleaser_test=False))
        results["P→H"].append(test(Mp, pleaser_test=False))
        results["H→P"].append(test(Mh, pleaser_test=True))
        results["P→P"].append(test(Mp, pleaser_test=True))

    # plot
    print({
        k: np.mean(v) for k, v in results.items()
    })


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Running Insight 1–3 simulations...")

    # Insight 1 Example
    var_h, var_p, ent_h, ent_p = simulate_signal_quality()
    print("Insight 1 — Variance honest/pleaser:", var_h, var_p)
    print("Insight 1 — Entropy honest/pleaser:", ent_h, ent_p)

        # --- Plot Insight 1: signal quality (honest vs pleaser) ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax_var, ax_ent = axes

    # Variance plot
    ax_var.bar(['Honest', 'Pleaser'], [var_h, var_p])
    ax_var.set_title("Valence variance")
    ax_var.set_ylabel("Variance")

    # Entropy plot
    ax_ent.bar(['Honest', 'Pleaser'], [ent_h, ent_p])
    ax_ent.set_title("Emotion entropy")
    ax_ent.set_ylabel("Entropy (a.u.)")

    fig.suptitle("Insight 1: Pleasers have smoother, lower-entropy signals")
    fig.tight_layout()
    plt.show()

    # Insight 2 Example
    c_h, c_p, a_h, a_p = simulate_confidence_trap()
    print("Insight 2 — Confidences:", c_h, c_p)
    print("Insight 2 — Accuracies:", a_h, a_p)

    # Insight 3 Example
    simulate_transfer()

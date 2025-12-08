import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- basic plotting style ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# where the CSVs from R live (change to "." if they’re in the root)
DATA_DIR = "."

# ------------------------------------------------------------
# 1. Emotion space
# ------------------------------------------------------------

EMOTIONS = [
    "happy", "excited", "content", "neutral",
    "bored", "annoyed", "sad", "frustrated",
]
E_INDEX = {e: i for i, e in enumerate(EMOTIONS)}
N_EMO = len(EMOTIONS)

VALENCE = {
    "happy": 0.8, "excited": 0.7, "content": 0.6,
    "neutral": 0.0,
    "bored": -0.3, "annoyed": -0.4,
    "sad": -0.7, "frustrated": -0.6,
}

POSITIVE = [e for e, v in VALENCE.items() if v > 0.3]
NEGATIVE = [e for e, v in VALENCE.items() if v < -0.2]


def mask_emotion(true_emotion, base_mask_prob, env_scale=1.0):
    """
    If the true emotion is negative, flip to a random positive
    with probability base_mask_prob * env_scale (capped at 1).
    Otherwise pass the emotion through unchanged.
    """
    eff_prob = min(1.0, base_mask_prob * env_scale)
    if VALENCE[true_emotion] < 0 and np.random.rand() < eff_prob:
        return np.random.choice(POSITIVE)
    else:
        return true_emotion


def load_global_transitions(path=DATA_DIR):
    df = pd.read_csv(f"{path}/P_global.csv")
    df = df.set_index("from")
    P = df[EMOTIONS].values
    P = P / P.sum(axis=1, keepdims=True)
    return P


def load_speaker_matrices(path=DATA_DIR):
    df = pd.read_csv(f"{path}/speaker_transition_long.csv")
    mats = {}
    for sp, g in df.groupby("Speaker"):
        M = np.zeros((N_EMO, N_EMO))
        for _, row in g.iterrows():
            i = E_INDEX[row["from"]]
            j = E_INDEX[row["to"]]
            M[i, j] = row["prob"]
        M = M / M.sum(axis=1, keepdims=True)
        mats[sp] = M
    return mats


def load_mask_stats(path=DATA_DIR):
    return pd.read_csv(f"{path}/speaker_masking_stats.csv")


# ------------------------------------------------------------
# 2. Learner class
# ------------------------------------------------------------

class Learner:
    def __init__(self):
        self.C = np.zeros((N_EMO, N_EMO))

    def observe(self, prev, curr):
        i = E_INDEX[prev]
        j = E_INDEX[curr]
        self.C[i, j] += 1

    def model(self):
        M = self.C + 0.1
        return M / M.sum(axis=1, keepdims=True)

    def confidence(self):
        M = self.model()
        max_probs = M.max(axis=1)
        return max_probs.mean()


def cosine_similarity(A, B):
    v1 = A.flatten()
    v2 = B.flatten()
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if den == 0 else num / den


# ------------------------------------------------------------
# 3. Dyadic simulation
# ------------------------------------------------------------

def simulate_dyad(TRUE_A, TRUE_B,
                  base_mask_A, base_mask_B,
                  env_scale=1.0,
                  n_steps=240):
    """
    One dyadic run:
      - A has true dynamics TRUE_A, baseline masking base_mask_A
      - B has true dynamics TRUE_B, baseline masking base_mask_B
      - env_scale scales both mask probs (situational pressure)
      - Each builds a transition model over the other's expressed states.
    """
    LA = Learner()  # A's model of B
    LB = Learner()  # B's model of A

    true_A = "neutral"
    true_B = "neutral"
    expr_prev_A = "neutral"
    expr_prev_B = "neutral"

    for _ in range(n_steps):
        # internal transitions
        idx_A = E_INDEX[true_A]
        idx_B = E_INDEX[true_B]
        true_A = np.random.choice(EMOTIONS, p=TRUE_A[idx_A])
        true_B = np.random.choice(EMOTIONS, p=TRUE_B[idx_B])

        # masking when expressing
        expr_A = mask_emotion(true_A, base_mask_A, env_scale)
        expr_B = mask_emotion(true_B, base_mask_B, env_scale)

        # each learns from the other's expressed states
        LA.observe(expr_prev_B, expr_B)
        LB.observe(expr_prev_A, expr_A)

        expr_prev_A = expr_A
        expr_prev_B = expr_B

    MA = LA.model()
    MB = LB.model()

    # Accuracy: A tries to approximate B's true dynamics, and vice versa
    accA = 1 - np.mean((MA - TRUE_B) ** 2)
    accB = 1 - np.mean((MB - TRUE_A) ** 2)
    avg_acc = 0.5 * (accA + accB)

    confA = LA.confidence()
    confB = LB.confidence()
    avg_conf = 0.5 * (confA + confB)

    sim = cosine_similarity(MA, MB)
    subjective_alignment = sim * avg_conf

    return avg_acc, avg_conf, sim, subjective_alignment


def sweep_env_scales(TRUE_A, TRUE_B,
                     base_mask_A, base_mask_B,
                     scales=np.linspace(0.0, 1.5, 6),
                     n_runs=80, n_steps=240):
    accs, confs, sims, aligns = [], [], [], []

    for s in scales:
        acc_list, conf_list, sim_list, align_list = [], [], [], []
        for _ in range(n_runs):
            avg_acc, avg_conf, sim, subj = simulate_dyad(
                TRUE_A, TRUE_B,
                base_mask_A, base_mask_B,
                env_scale=s,
                n_steps=n_steps,
            )
            acc_list.append(avg_acc)
            conf_list.append(avg_conf)
            sim_list.append(sim)
            align_list.append(subj)

        accs.append(np.mean(acc_list))
        confs.append(np.mean(conf_list))
        sims.append(np.mean(sim_list))
        aligns.append(np.mean(align_list))

    return (np.array(scales),
            np.array(accs),
            np.array(confs),
            np.array(sims),
            np.array(aligns))


# ------------------------------------------------------------
# 4. Main: Monica & Chandler
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)

    # Load data exported from R
    P_global = load_global_transitions()
    speaker_mats = load_speaker_matrices()
    mask_df = load_mask_stats()

    # Dyad: Monica (higher masking) & Chandler (lower masking)
    sp_A = "Phoebe"
    sp_B = "Ross"

    TRUE_A = speaker_mats[sp_A]
    TRUE_B = speaker_mats[sp_B]

    base_mask_A = mask_df.loc[mask_df["Speaker"] == sp_A, "masking_rate"].iloc[0]
    base_mask_B = mask_df.loc[mask_df["Speaker"] == sp_B, "masking_rate"].iloc[0]

    print(f"{sp_A} base masking rate: {base_mask_A:.3f}")
    print(f"{sp_B} base masking rate: {base_mask_B:.3f}")

    scales = np.linspace(0.0, 1.5, 6)
    scales, accs, confs, sims, aligns = sweep_env_scales(
        TRUE_A, TRUE_B,
        base_mask_A, base_mask_B,
        scales=scales,
        n_runs=80,
        n_steps=240,
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax1, ax2, ax3 = axes

    ax1.plot(scales, accs, marker="o")
    ax1.set_xlabel("Environmental masking scale")
    ax1.set_ylabel("Accuracy vs true dynamics")
    ax1.set_title(f"Accuracy: {sp_A} & {sp_B}")

    ax2.plot(scales, confs, marker="o", label="Confidence")
    ax2.plot(scales, sims, marker="o", label="Model similarity")
    ax2.set_xlabel("Environmental masking scale")
    ax2.set_ylabel("Value")
    ax2.set_title("Confidence & similarity")
    ax2.legend()

    ax3.plot(scales, aligns, marker="o")
    ax3.set_xlabel("Environmental masking scale")
    ax3.set_ylabel("Subjective alignment")
    ax3.set_title("Illusion of mutual understanding")

    fig.suptitle(f"Insight 4 with real speakers: {sp_A} & {sp_B}", y=1.03)
    fig.tight_layout()
    plt.show()

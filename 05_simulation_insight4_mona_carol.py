import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# ------------------------------------------------------------
# 1. Emotion space (must match R mapping)
# ------------------------------------------------------------

EMOTIONS = [
    'happy', 'excited', 'content', 'neutral',
    'bored', 'annoyed', 'sad', 'frustrated'
]
E_INDEX = {e: i for i, e in enumerate(EMOTIONS)}
N_EMO = len(EMOTIONS)

# simple valence map for masking rule
VALENCE = {
    'happy': 0.8, 'excited': 0.7, 'content': 0.6,
    'neutral': 0.0,
    'bored': -0.3, 'annoyed': -0.4,
    'sad': -0.7, 'frustrated': -0.6,
}
POSITIVE = [e for e, v in VALENCE.items() if v > 0.3]
NEGATIVE = [e for e, v in VALENCE.items() if v < -0.2]


def mask_emotion(true_emotion, base_mask_prob, env_scale=1.0):
    """
    Mask negative emotions with probability base_mask_prob * env_scale,
    capped at 1. Positive emotions pass through unchanged.
    """
    eff_prob = min(1.0, base_mask_prob * env_scale)
    if VALENCE[true_emotion] < 0 and np.random.rand() < eff_prob:
        return np.random.choice(POSITIVE)
    else:
        return true_emotion


# ------------------------------------------------------------
# 2. Load MELD-based parameters exported from R
# ------------------------------------------------------------

def load_global_transitions(path="."):
    """Load global transition matrix (not required for Insight 4, but kept for completeness)."""
    df = pd.read_csv(f"{path}/P_global.csv")
    df = df.set_index("from")
    P = df[EMOTIONS].values
    P = P / P.sum(axis=1, keepdims=True)
    return P


def load_speaker_matrices(path="."):
    """
    Load long-format speaker matrices and return dict: speaker -> 8x8 matrix.
    Requires a CSV with columns: Speaker, from, to, prob.
    """
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


def load_mask_stats(path="."):
    """Load per-speaker masking rates from MELD."""
    return pd.read_csv(f"{path}/speaker_masking_stats.csv")


# ------------------------------------------------------------
# 3. Learner class
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
# 4. Dyadic simulation using real speakers
# ------------------------------------------------------------

def simulate_dyad(TRUE_A, TRUE_B,
                  base_mask_A, base_mask_B,
                  env_scale=1.0,
                  n_steps=240):
    """
    One dyadic run:
      - A has true dynamics TRUE_A, baseline masking base_mask_A
      - B has true dynamics TRUE_B, baseline masking base_mask_B
      - env_scale scales both mask probs (situation pressure to please)
      - Each learns a transition model over the other's expressed states.
    """
    LA = Learner()  # A's model of B
    LB = Learner()  # B's model of A

    true_A = 'neutral'
    true_B = 'neutral'
    expr_prev_A = 'neutral'
    expr_prev_B = 'neutral'

    for _ in range(n_steps):
        # internal transitions
        idx_A = E_INDEX[true_A]
        idx_B = E_INDEX[true_B]
        true_A = np.random.choice(EMOTIONS, p=TRUE_A[idx_A])
        true_B = np.random.choice(EMOTIONS, p=TRUE_B[idx_B])

        # masking when expressing
        expr_A = mask_emotion(true_A, base_mask_A, env_scale)
        expr_B = mask_emotion(true_B, base_mask_B, env_scale)

        # each learns from other's expressed states
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
                n_steps=n_steps
            )
            acc_list.append(avg_acc)
            conf_list.append(avg_conf)
            sim_list.append(sim)
            align_list.append(subj)

        accs.append(np.mean(acc_list))
        confs.append(np.mean(conf_list))
        sims.append(np.mean(sim_list))
        aligns.append(np.mean(align_list))

    return np.array(scales), np.array(accs), np.array(confs), np.array(sims), np.array(aligns)


# ------------------------------------------------------------
# 5. Main: run Insight 4 with real Mona & Carol
# ------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(0)

    # Load data exported from R
    # (P_global is loaded but not directly used here)
    P_global = load_global_transitions(".")
    speaker_mats = load_speaker_matrices(".")
    mask_df = load_mask_stats(".")

    # Dyad: Mona (high-ish masking) & Carol (moderate masking)
    sp_A = "Mona"
    sp_B = "Carol"

    TRUE_A = speaker_mats[sp_A]
    TRUE_B = speaker_mats[sp_B]

    # Get their baseline masking rates from MELD
    base_mask_A = mask_df.loc[mask_df["Speaker"] == sp_A, "masking_rate"].iloc[0]
    base_mask_B = mask_df.loc[mask_df["Speaker"] == sp_B, "masking_rate"].iloc[0]

    print(f"{sp_A} base masking rate: {base_mask_A:.3f}")
    print(f"{sp_B} base masking rate: {base_mask_B:.3f}")

    # Sweep environmental pressure to please
    scales = np.linspace(0.0, 1.5, 6)
    scales, accs, confs, sims, aligns = sweep_env_scales(
        TRUE_A, TRUE_B,
        base_mask_A, base_mask_B,
        scales=scales,
        n_runs=80,
        n_steps=240
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax1, ax2, ax3 = axes

    ax1.plot(scales, accs, marker='o')
    ax1.set_xlabel("Environmental masking scale")
    ax1.set_ylabel("Accuracy vs true dynamics")
    ax1.set_title(f"Accuracy: {sp_A} & {sp_B}")

    ax2.plot(scales, confs, marker='o', label="Confidence")
    ax2.plot(scales, sims, marker='o', label="Model similarity")
    ax2.set_xlabel("Environmental masking scale")
    ax2.set_ylabel("Value")
    ax2.set_title("Confidence & similarity")
    ax2.legend()

    ax3.plot(scales, aligns, marker='o')
    ax3.set_xlabel("Environmental masking scale")
    ax3.set_ylabel("Subjective alignment")
    ax3.set_title("Illusion of mutual understanding")

    fig.suptitle(f"Insight 4 with real speakers: {sp_A} & {sp_B}", y=1.03)
    fig.tight_layout()
    plt.show()

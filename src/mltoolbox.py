import numpy as np


# --- start: confusion matrix ---

def make_class_probs(K, kind="balanced", imbalance=None):
    """
    Returns pi (class distribution) length K.
    kind:
      - "balanced"
      - "imbalanced" (provide imbalance list/array of length K)
    """
    if kind == "balanced":
        pi = np.ones(K) / K
    elif kind == "imbalanced":
        if imbalance is None:
            raise ValueError("For kind='imbalanced', pass imbalance=[...] of length K.")
        pi = np.array(imbalance, dtype=float)
        pi = pi / pi.sum()
    else:
        raise ValueError("Unknown kind.")
    return pi

def make_transition_matrix(K, diag_acc=0.75, structure=None):
    """
    Returns P (KxK) where P[t, p] = P(y_pred=p | y_true=t).

    diag_acc: baseline probability to be correct (diagonal).
    structure: optional dict to impose extra confusion patterns, e.g.
      structure = {
        (2, 1): 0.20  # for true class 2, allocate 20% mass to predicting class 1
      }
    Remaining off-diagonal mass is spread uniformly among other classes.
    """
    if not (0 <= diag_acc <= 1):
        raise ValueError("diag_acc must be in [0,1].")

    P = np.zeros((K, K), dtype=float)

    for t in range(K):
        # start with diagonal mass
        P[t, t] = diag_acc

        # remaining mass for wrong predictions
        remaining = 1.0 - diag_acc

        # allocate structured confusions for this true class t
        if structure is not None:
            # take all specified (t, p) for this t
            structured_targets = [(a, p, w) for (a, p), w in structure.items() if a == t]
            structured_mass = sum(w for _, _, w in structured_targets)

            if structured_mass > remaining + 1e-12:
                raise ValueError(
                    f"Structured mass {structured_mass} exceeds available off-diagonal mass {remaining} for class {t}."
                )

            # assign structured masses
            for _, p, w in structured_targets:
                if p == t:
                    raise ValueError("Structured confusion cannot target the same class (diagonal).")
                P[t, p] += w

            remaining -= structured_mass

        # distribute leftover off-diagonal mass uniformly across the remaining wrong classes
        wrong_classes = [p for p in range(K) if p != t and P[t, p] == 0.0]
        if len(wrong_classes) > 0:
            P[t, wrong_classes] += remaining / len(wrong_classes)
        else:
            # edge case: all off-diagonal mass was assigned by structure
            # remaining should be ~0
            pass

    # numeric sanity: row-normalize (tiny floating errors)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def simulate_labels(N, K, pi, P, seed=0):
    """
    Simulate y_true ~ Categorical(pi)
    then y_pred | y_true=t ~ Categorical(P[t])
    """
    rng = np.random.default_rng(seed)

    pi = np.array(pi, dtype=float)
    if not np.isclose(pi.sum(), 1.0):
        raise ValueError("pi must sum to 1.")

    if P.shape != (K, K):
        raise ValueError("P must be shape (K,K).")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Each row of P must sum to 1.")

    y_true = rng.choice(K, size=N, p=pi)
    y_pred = np.array([rng.choice(K, p=P[t]) for t in y_true], dtype=int)
    return y_true, y_pred


def expected_accuracy(pi, P):
    """E[1(y_pred=y_true)] = sum_k pi[k] * P[k,k]"""
    pi = np.array(pi, dtype=float)
    return float(np.sum(pi * np.diag(P)))

def empirical_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def get_confusion_matrix(y_true, y_pred):
    K = len(np.unique(y_true))
    cm = np.zeros((K,K), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t,p] +=1

    return cm

def plot_confusion_matrix(cm):
    import seaborn as sns
    import matplotlib.pyplot as plt

    class_names = [f"Class {i}" for i in range(cm.shape[0])]

    plt.figure(figsize=(6, 5))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white"
    )

    # Move x-axis to top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Labels
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Actual Class", fontsize=12)

    plt.title("Confusion Matrix", pad=30)
    plt.tight_layout()
    plt.show()


def get_TP_FP_FN_TN(cm):
    import pandas as pd

    cm = np.asarray(cm)
    K = cm.shape[0]
    N = cm.sum()
    out = []

    for k in range(K):
        TP = cm[k, k]
        FN = cm[k, :].sum() - TP
        FP = cm[:, k].sum() - TP
        TN = N - (TP + FP + FN)
        support = cm[k, :].sum()

        out.append({
            "class": k,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "support (actual count)": support
        })

    return pd.DataFrame(out)



# ---- end: confusion matrix ---
"""RCOD Section 5 experiment: can RCOD swarm metrics predict shock onsets?

Pre-registered in DESIGN.md BEFORE Run 1 ran; Amendments 1 (precursor fix)
and 2 (hierarchical 3-level swarms) written before their runs.
"""
import json
import sys
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

from swarm import swarm_metrics, hierarchical_swarm, temporal_hierarchy

T = 24000
N_SHOCKS = 8
HORIZON = 30
LAGS = [0, 5, 10, 20]
MACRO_W = 252
SEEDS = list(range(10))


def gen_series(seed):
    # Run 2+ (see DESIGN.md Amendment 1): 150-step precursor ramp before each break.
    rng = np.random.default_rng(seed)
    phi = 0.9
    PRE = 150
    x = np.zeros(T)
    eps = rng.standard_normal(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + eps[t]
    onsets, kinds = [], []
    t = 2500 + int(rng.integers(0, 800))
    i = 0
    while len(onsets) < N_SHOCKS and t + PRE + 700 < T - 1500:
        kind = "vol" if i % 2 == 0 else "mean"
        if kind == "vol":
            x[t : t + PRE] *= np.linspace(1.0, 3.0, PRE)
            x[t + PRE : t + PRE + 600] *= 6.0
        else:
            x[t : t + PRE] += np.linspace(0.0, 2.0, PRE)
            x[t + PRE : t + PRE + 600] += 4.0
        onsets.append(t + PRE)
        kinds.append(kind)
        t += PRE + 600 + 1500 + int(rng.integers(0, 1500))
        i += 1
    return x, np.array(onsets), kinds


def lagged_stack(series_list, idx):
    cols = []
    for s in series_list:
        for l in LAGS:
            cols.append(s[idx - l])
    return np.column_stack(cols)


def build_features(x, metrics, warmup):
    med, mac, dis = metrics["median_mu"], metrics["macro_mu"], metrics["dissonance"]
    idx = np.arange(warmup, T - HORIZON)
    F_rcod = lagged_stack([med, mac, dis], idx)
    F_dis = lagged_stack([dis], idx)
    w = 60
    mean = np.full(T, np.nan); std = np.full(T, np.nan)
    ac1 = np.full(T, np.nan); maxd = np.full(T, np.nan)
    # fill from warmup-max(LAGS): lagged features reach back before warmup
    for t in range(warmup - max(LAGS), T - HORIZON):
        s = x[t - w + 1 : t + 1]
        mean[t] = s.mean()
        std[t] = s.std()
        ac1[t] = np.corrcoef(s[:-1], s[1:])[0, 1] if s.std() > 1e-12 else 0.0
        maxd[t] = np.abs(np.diff(s)).max()
    F_raw = lagged_stack([mean, std, ac1, maxd], idx)
    return idx, F_rcod, F_dis, F_raw


def labels_primary(idx, onsets):
    return np.array([np.any((onsets > t) & (onsets <= t + HORIZON)) for t in idx]).astype(int)


def evaluate_arm(F, y, idx, onsets, seed):
    n = len(idx)
    i1, i2 = int(0.6 * n), int(0.8 * n)
    sc = StandardScaler().fit(F[:i1])
    Fs = sc.transform(F)
    clf = LogisticRegression(max_iter=5000, random_state=seed).fit(Fs[:i1], y[:i1])
    p_test = clf.predict_proba(Fs[i2:])[:, 1]
    p_val = clf.predict_proba(Fs[i1:i2])[:, 1]
    best_t, best_f1 = 0.5, -1
    for t_ in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y[i1:i2], (p_val >= t_).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t_
    alarms = idx[i2:][p_test >= best_t]
    test_onsets = onsets[(onsets >= idx[i2]) & (onsets < idx[-1])]
    leads, missed = [], 0
    for o in test_onsets:
        cand = alarms[(alarms >= o - 200) & (alarms <= o)]
        if len(cand):
            leads.append(float(o - cand.min()))
        else:
            missed += 1
    if len(test_onsets):
        fa = np.sum(~np.any(np.abs(alarms[:, None] - test_onsets[None, :]) <= 200, axis=1))
    else:
        fa = len(alarms)
    return {
        "pr_auc": float(average_precision_score(y[i2:], p_test)),
        "roc_auc": float(roc_auc_score(y[i2:], p_test)),
        "median_lead": float(np.median(leads)) if leads else None,
        "missed": missed,
        "fa_per_1000": float(fa / max(1, (n - i2)) * 1000),
        "thresh": float(best_t),
    }


def run_seed_hier(seed):
    """Run 3: single-level arms A/C plus hierarchical arm D, all on the same
    (deeper-warmup) index grid so the comparison is apples-to-apples."""
    x, onsets, _ = gen_series(seed)
    hier = hierarchical_swarm(x, levels=3, macro_w=MACRO_W, seed=0)
    L3, names = hier[3]
    assert L3.shape[1] == 27, L3.shape
    # Warmup NaNs compound across levels; measure the true warmup instead of
    # guessing it, and assert no NaNs survive past it.
    bad = np.isnan(L3).any(axis=1)
    first_ok = int(np.argmax(~bad))
    assert not bad[first_ok:].any(), "NaNs beyond warmup region"
    warmup = max(3 * MACRO_W + max(LAGS), first_ok + max(LAGS))
    metrics = swarm_metrics(x, macro_w=MACRO_W, seed=0)
    idx, F_rcod, F_dis, F_raw = build_features(x, metrics, warmup)
    y = labels_primary(idx, onsets)
    F_hier = lagged_stack([L3[:, j] for j in range(L3.shape[1])], idx)
    assert not np.isnan(F_hier).any(), "NaN in hierarchical features"
    return {
        "seed": seed,
        "n_test_pos": int(y[int(0.8 * len(idx)) :].sum()),
        "rcod": evaluate_arm(F_rcod, y, idx, onsets, seed),
        "raw": evaluate_arm(F_raw, y, idx, onsets, seed),
        "hier": evaluate_arm(F_hier, y, idx, onsets, seed),
    }


def main_hier():
    results = [run_seed_hier(s) for s in SEEDS]
    with open("results_hier.json", "w") as f:
        json.dump(results, f, indent=1)
    pr = {k: np.array([r[k]["pr_auc"] for r in results]) for k in ("rcod", "raw", "hier")}
    roc = {k: np.array([r[k]["roc_auc"] for r in results]) for k in ("rcod", "raw", "hier")}
    print("=== Run 3: hierarchical (3-level) swarms vs single-level vs raw ===")
    print("--- PR-AUC (primary; pre-registered) ---")
    for k in ("rcod", "raw", "hier"):
        print(f"  {k:5s}: mean={pr[k].mean():.3f} sd={pr[k].std():.3f} "
              f"values={[round(v, 3) for v in pr[k]]}")
    _, p_hr = wilcoxon(pr["hier"], pr["raw"], alternative="greater")
    _, p_ha = wilcoxon(pr["hier"], pr["rcod"], alternative="greater")
    print(f"Wilcoxon hier>raw: p={p_hr:.4f} | hier>rcod-single: p={p_ha:.4f}")
    print("--- ROC-AUC (for comparability with Jacob's historical AUC memory) ---")
    for k in ("rcod", "raw", "hier"):
        print(f"  {k:5s}: mean={roc[k].mean():.3f} sd={roc[k].std():.3f}")
    print("--- operational (hier arm, F1-tuned threshold) ---")
    h = results[0]["hier"]
    leads = [r["hier"]["median_lead"] for r in results if r["hier"]["median_lead"] is not None]
    fa = np.mean([r["hier"]["fa_per_1000"] for r in results])
    miss = np.mean([r["hier"]["missed"] for r in results])
    print(f"  median_lead={np.median(leads) if leads else None} fa/1000={fa:.2f} missed/seed={miss:.1f}")
    print("=== pre-registered verdict (Amendment 2) ===")
    if p_hr >= 0.05:
        print("KILL: hier PR-AUC NOT significantly > raw -> NEGATIVE for hierarchical variant")
    else:
        print("PASS: hier PR-AUC significantly > raw -> POSITIVE, earns real-data follow-up")


def run_seed_temporal(seed, block_sizes=(84, 21, 6)):
    """Run 4/5: Jacob's '3 levels deep' = nested temporal partitions.
    Arm E (temporal hierarchy) vs A (single-level trio) vs C (raw), all on
    the same index grid. block_sizes=(84,21,6) is Run 4 (his illustrative
    numbers); (128,32,8) is Run 5 (dyadic ladder, Amendment 4)."""
    x, onsets, _ = gen_series(seed)
    th = temporal_hierarchy(x, block_sizes=block_sizes)
    warmup = 5 * block_sizes[0]  # 4 coarsest blocks completed + mu defined
    idx = np.arange(warmup, T - HORIZON)
    feats, latest = [], []
    for lv in (1, 2, 3):
        mu, W = th[lv]["mu"], th[lv]["W"]
        b = (idx + 1) // W - 1  # latest completed block at each t
        assert (b >= 4).all() and not np.isnan(mu[b]).any()
        feats.extend([mu[b - k] for k in (3, 2, 1, 0)])
        latest.append(mu[b])
    latest = np.column_stack(latest)
    feats.append(np.median(latest, axis=1))
    feats.append(latest.std(axis=1))
    F_temp = np.column_stack(feats)
    assert not np.isnan(F_temp).any()
    metrics = swarm_metrics(x, macro_w=MACRO_W, seed=0)
    _, F_rcod, F_dis, F_raw = build_features(x, metrics, warmup)
    y = labels_primary(idx, onsets)
    return {
        "seed": seed,
        "n_test_pos": int(y[int(0.8 * len(idx)) :].sum()),
        "rcod": evaluate_arm(F_rcod, y, idx, onsets, seed),
        "raw": evaluate_arm(F_raw, y, idx, onsets, seed),
        "temp": evaluate_arm(F_temp, y, idx, onsets, seed),
    }


def main_temporal():
    results = [run_seed_temporal(s) for s in SEEDS]
    with open("results_temporal.json", "w") as f:
        json.dump(results, f, indent=1)
    pr = {k: np.array([r[k]["pr_auc"] for r in results]) for k in ("rcod", "raw", "temp")}
    roc = {k: np.array([r[k]["roc_auc"] for r in results]) for k in ("rcod", "raw", "temp")}
    print("=== Run 4: nested temporal partitions (Jacob's '3 levels deep') ===")
    for k in ("rcod", "raw", "temp"):
        valid = ~np.isnan(roc[k])
        print(f"  {k:5s} PR-AUC  mean={pr[k].mean():.4f} sd={pr[k].std():.4f} "
              f"values={np.round(pr[k], 3).tolist()}")
        print(f"       ROC-AUC mean={roc[k][valid].mean():.3f} (n={valid.sum()}/10 valid)")
    _, p_tr = wilcoxon(pr["temp"], pr["raw"], alternative="greater")
    _, p_ta = wilcoxon(pr["temp"], pr["rcod"], alternative="greater")
    print(f"Wilcoxon temp>raw: p={p_tr:.4f} | temp>rcod-single: p={p_ta:.4f}")
    leads = [r["temp"]["median_lead"] for r in results if r["temp"]["median_lead"] is not None]
    fa = np.mean([r["temp"]["fa_per_1000"] for r in results])
    miss = np.mean([r["temp"]["missed"] for r in results])
    print(f"temp operational: median_lead={np.median(leads) if leads else None} "
          f"fa/1000={fa:.2f} missed/seed={miss:.1f}")
    print("=== pre-registered verdict (Amendment 3) ===")
    if p_tr >= 0.05:
        print("KILL: temp PR-AUC NOT significantly > raw -> NEGATIVE for temporal-hierarchy variant")
    else:
        print("PASS: temp PR-AUC significantly > raw -> POSITIVE, earns real-data follow-up")


def main_dyadic():
    results = []
    for s in SEEDS:
        r = run_seed_temporal(s, block_sizes=(128, 32, 8))
        r["dyad"] = r.pop("temp")
        results.append(r)
    with open("results_dyadic.json", "w") as f:
        json.dump(results, f, indent=1)
    pr = {k: np.array([r[k]["pr_auc"] for r in results]) for k in ("rcod", "raw", "dyad")}
    roc = {k: np.array([r[k]["roc_auc"] for r in results]) for k in ("rcod", "raw", "dyad")}
    print("=== Run 5: dyadic ladder (128/32/8) — precise structure, last redesign ===")
    for k in ("rcod", "raw", "dyad"):
        valid = ~np.isnan(roc[k])
        print(f"  {k:5s} PR-AUC  mean={pr[k].mean():.4f} sd={pr[k].std():.4f} "
              f"values={np.round(pr[k], 3).tolist()}")
        print(f"       ROC-AUC mean={roc[k][valid].mean():.3f} (n={valid.sum()}/10 valid)")
    _, p_tr = wilcoxon(pr["dyad"], pr["raw"], alternative="greater")
    _, p_ta = wilcoxon(pr["dyad"], pr["rcod"], alternative="greater")
    print(f"Wilcoxon dyad>raw: p={p_tr:.4f} | dyad>rcod-single: p={p_ta:.4f}")
    leads = [r["dyad"]["median_lead"] for r in results if r["dyad"]["median_lead"] is not None]
    fa = np.mean([r["dyad"]["fa_per_1000"] for r in results])
    miss = np.mean([r["dyad"]["missed"] for r in results])
    print(f"dyad operational: median_lead={np.median(leads) if leads else None} "
          f"fa/1000={fa:.2f} missed/seed={miss:.1f}")
    print("=== pre-registered verdict (Amendment 4, one-shot) ===")
    if p_tr >= 0.05:
        print("KILL: dyad PR-AUC NOT significantly > raw -> NEGATIVE; section-5 thread CLOSED")
    else:
        print("PASS: dyad PR-AUC significantly > raw -> POSITIVE, earns real-data follow-up")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hier":
        main_hier()
    elif len(sys.argv) > 1 and sys.argv[1] == "temporal":
        main_temporal()
    elif len(sys.argv) > 1 and sys.argv[1] == "dyadic":
        main_dyadic()
    else:
        print("pass 'hier', 'temporal' or 'dyadic' to run a comparison")

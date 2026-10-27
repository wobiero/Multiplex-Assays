"""
Multiplex Panel Value-of-Diagnostic-Information Explorer

Streamlit companion app for the manuscript
"Decision Value and Optimal Composition of Multiplex Diagnostic Panels:
A Value-of-Diagnostic-Information Framework".

Run with:  streamlit run multiplex_voi_app.py
"""

import itertools
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Full-width elements: newer Streamlit uses width="stretch"; older versions use
# use_container_width=True (and treat `width` as a pixel count).
def _st_version():
    parts = []
    for p in st.__version__.split(".")[:2]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


STRETCH = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}

# --------------------------------------------------------------------------------------
# Chart styling
# --------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"


def style(fig, x_title, y_title, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#b5b4ae")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=True,
                     zerolinecolor="rgba(128,128,128,0.5)")
    return fig


def vline(fig, x, label, color=GREY, level=0):
    fig.add_vline(x=x, line_dash="dot", line_color=color, line_width=1,
                  annotation_text=label, annotation_position="top right",
                  annotation=dict(yshift=-16 * level, font_size=11))


def money(v, digits=2):
    return f"−${abs(v):,.{digits}f}" if v < 0 else f"${v:,.{digits}f}"


# --------------------------------------------------------------------------------------
# Core model: single binary decision (Eqs. 1–8)
# --------------------------------------------------------------------------------------
def expected_utility(p, U):
    """Best action and its expected utility at disease probability p.
    U = dict(t1, t0, n1, n0): utilities of Treat/No-treat under D+ / D−."""
    eu_t = p * U["t1"] + (1 - p) * U["t0"]
    eu_n = p * U["n1"] + (1 - p) * U["n0"]
    return (eu_t, "treat") if eu_t >= eu_n else (eu_n, "no-treat")


def threshold(U):
    B = U["t1"] - U["n1"]
    H = U["n0"] - U["t0"]
    return B, H, H / (H + B)


def single_test(p0, Se, Sp, U):
    """Gross EVDI (Eq. 7) and the quantities behind it."""
    p_pos = p0 * Se + (1 - p0) * (1 - Sp)
    p_neg = 1 - p_pos
    post_pos = p0 * Se / p_pos if p_pos > 0 else 0.0
    post_neg = p0 * (1 - Se) / p_neg if p_neg > 0 else 0.0
    eu0, a0 = expected_utility(p0, U)
    eu_pos, a_pos = expected_utility(post_pos, U)
    eu_neg, a_neg = expected_utility(post_neg, U)
    eu_test = p_pos * eu_pos + p_neg * eu_neg
    return dict(p_pos=p_pos, post_pos=post_pos, post_neg=post_neg, eu0=eu0, a0=a0,
                a_pos=a_pos, a_neg=a_neg, evdi=eu_test - eu0)


def evdi(p0, Se, Sp, U):
    return single_test(p0, Se, Sp, U)["evdi"]


def testing_thresholds(Se, Sp, pt):
    """Pauker–Kassirer testing and test–treatment thresholds (Annex Eqs. A4–A5)."""
    lo = pt * (1 - Sp) / (Se * (1 - pt) + pt * (1 - Sp))
    hi = pt * Sp / ((1 - Se) * (1 - pt) + pt * Sp)
    return lo, hi


def solve_on_grid(f, lo=0.0, hi=1.0, n=20001):
    """First root of f on [lo, hi], or None."""
    xs = np.linspace(lo, hi, n)
    ys = np.array([f(x) for x in xs])
    s = np.sign(ys)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    a, b = xs[i], xs[i + 1]
    for _ in range(60):
        m = (a + b) / 2
        if np.sign(f(m)) == np.sign(f(a)):
            a = m
        else:
            b = m
    return (a + b) / 2


# --------------------------------------------------------------------------------------
# Panel model: one decision state informed by K jointly observed binary targets
# --------------------------------------------------------------------------------------
def panel_values(names, cells, probs, prevs, U):
    """Gross value of every subset S: EU(S) − EU(∅), where EU(S) conditions the decision
    on the targets in S (partitioning the joint cells by those targets' values)."""
    K = len(names)
    out = {}
    for r in range(K + 1):
        for S in itertools.combinations(range(K), r):
            groups = {}
            for cell, pr, pv in zip(cells, probs, prevs):
                key = tuple(cell[i] for i in S)
                g = groups.setdefault(key, [0.0, 0.0])
                g[0] += pr
                g[1] += pr * pv
            eu = sum(m * expected_utility(w / m, U)[0] for m, w in groups.values() if m > 0)
            out[S] = eu
    base = out[()]
    return {S: v - base for S, v in out.items()}


def panel_analysis(names, gross, c, F):
    K = len(names)

    def nmb(S):
        return gross[S] - c * len(S) - (F if S else 0.0)

    rows = []
    for S in gross:
        rows.append(dict(Panel="{" + ", ".join(names[i] for i in S) + "}" if S else "∅ (no panel)",
                         Targets=len(S), Gross=gross[S], Cost=c * len(S) + (F if S else 0.0),
                         NMB=nmb(S)))
    table = pd.DataFrame(rows).sort_values("NMB", ascending=False).reset_index(drop=True)
    best = max(gross, key=nmb)

    # greedy forward selection
    S, path = (), []
    while True:
        cands = [tuple(sorted(S + (k,))) for k in range(K) if k not in S]
        if not cands:
            break
        nxt = max(cands, key=nmb)
        gain = nmb(nxt) - nmb(S)
        if gain <= 1e-12:
            break
        added = [k for k in nxt if k not in S][0]
        path.append((names[added], gain))
        S = nxt
    greedy = S

    # Shapley values of gross value
    shap = []
    for i in range(K):
        tot = 0.0
        others = [k for k in range(K) if k != i]
        for r in range(K):
            for T in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(K - r - 1) / math.factorial(K)
                tot += w * (gross[tuple(sorted(T + (i,)))] - gross[T])
        shap.append(tot)

    # pairwise interactions I(i,j | ∅)
    inter = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in range(K):
        for j in range(K):
            if i == j:
                inter.iloc[i, j] = np.nan
            else:
                inter.iloc[i, j] = gross[tuple(sorted((i, j)))] - gross[(i,)] - gross[(j,)]
    return table, best, nmb(best), greedy, nmb(greedy), path, shap, inter


PRESETS = {
    "Paper Case A – substitutes (§10.2–10.3)": dict(
        names=["X", "Y"],
        probs={(1, 1): 0.01, (1, 0): 0.29, (0, 1): 0.16, (0, 0): 0.54},
        prevs={(1, 1): 0.60, (1, 0): 0.10, (0, 1): 0.05, (0, 0): 0.03},
        U=dict(t1=300.0, t0=-40.0, n1=-150.0, n0=0.0)),
    "Paper Case B – complements (§10.2–10.3)": dict(
        names=["X", "Y"],
        probs={(1, 1): 0.01, (1, 0): 0.29, (0, 1): 0.16, (0, 0): 0.54},
        prevs={(1, 1): 0.70, (1, 0): 0.03, (0, 1): 0.04, (0, 0): 0.02},
        U=dict(t1=300.0, t0=-40.0, n1=-150.0, n0=0.0)),
    "Three targets – independent markers (illustrative)": dict(
        names=["A", "B", "C"],
        probs=None,  # built from independent marginals below
        marg=[0.30, 0.20, 0.10],
        prevs_fn=lambda a, b, c: min(0.95, 0.02 + 0.10 * a + 0.06 * b + 0.15 * c + 0.40 * a * b * c),
        U=dict(t1=300.0, t0=-40.0, n1=-150.0, n0=0.0)),
}
for _k in (2, 3, 4):
    PRESETS[f"Blank template – {_k} targets (edit the table)"] = dict(
        names=["T1", "T2", "T3", "T4"][:_k], probs=None, marg=[0.20] * _k,
        prevs_fn=lambda *cell: 0.05, U=dict(t1=300.0, t0=-40.0, n1=-150.0, n0=0.0))


def preset_frame(key):
    pr = PRESETS[key]
    names = pr["names"]
    K = len(names)
    cells = list(itertools.product([1, 0], repeat=K))
    rows = []
    for cell in cells:
        if pr.get("probs"):
            p, v = pr["probs"][cell], pr["prevs"][cell]
        else:
            p = float(np.prod([m if x else 1 - m for m, x in zip(pr["marg"], cell)]))
            v = pr["prevs_fn"](*cell)
        row = {n: ("+" if x else "−") for n, x in zip(names, cell)}
        row["P(cell)"] = round(p, 6)
        row["P(D+ | cell)"] = round(v, 6)
        rows.append(row)
    return pd.DataFrame(rows), pr["U"]


# --------------------------------------------------------------------------------------
# Panel size (Proposition 1, §10.4)
# --------------------------------------------------------------------------------------
def panel_size_curves(g1, r, n_max, c, a, phi, L):
    n = np.arange(0, n_max + 1)
    g = g1 * r ** np.arange(n_max)
    G = np.concatenate([[0.0], np.cumsum(g)])
    per_result = G - c * n - phi * L * a * n
    per_patient = G - c * n - phi * L * (1 - (1 - a) ** n)
    return n, G, per_result, per_patient


# --------------------------------------------------------------------------------------
# Reporting policy (§10.5)
# --------------------------------------------------------------------------------------
def reporting_policies(q, p0, Se, Sp, B, H, pos_rest, phi_full, phi_comment, L, impl, c, miss, leak):
    U = dict(t1=B, t0=-H, n1=0.0, n0=0.0)
    sub = single_test(p0, Se, Sp, U)
    G = q * sub["evdi"]
    reflex_full = (1 - q) * pos_rest * phi_full * L
    reflex_comment = (1 - q) * pos_rest * phi_comment * L
    sel_perfect = G - impl
    sel_imperfect = (1 - miss) * G - leak * reflex_full - impl
    rows = [
        ("Full release", reflex_full, 0.0, G - reflex_full),
        (f"Interpretive comment (φ: {phi_full:g} → {phi_comment:g})", reflex_comment, 0.0, G - reflex_comment),
        ("Selective release, perfect subgroup identification", 0.0, impl, sel_perfect),
        (f"Selective release, {miss:.0%} missed / {leak:.0%} leaked", leak * reflex_full, impl, sel_imperfect),
        ("Suppress W", 0.0, 0.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["Reporting policy", "Expected reflex cost", "Implementation cost",
                                     "Net value, fixed panel (assay sunk)"])
    df["ΔNMB of adding W to a configurable panel"] = [v - c for v in df.iloc[:4, 3]] + [0.0]
    return df, sub, G


# --------------------------------------------------------------------------------------
# Public-health value (Annex A6)
# --------------------------------------------------------------------------------------
def ph_value(f, pi, Cd, lam, D0):
    f = np.asarray(f, dtype=float)
    mu = lam * f
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = np.where(mu > 0, (1 - np.exp(-mu * D0)) / mu, D0)
    return pi * Cd * (D0 - delay), delay


# ======================================================================================
# UI
# ======================================================================================
st.set_page_config(page_title="Multiplex Panel VOI Explorer", page_icon="🧪", layout="wide")
st.title("Multiplex Panel Value-of-Information Explorer")
st.caption("Companion to *Decision Value and Optimal Composition of Multiplex Diagnostic Panels*. "
           "All monetary values are net monetary benefit (NMB) per patient unless stated. "
           "Defaults reproduce the paper's illustrative examples.")

tabs = st.tabs(["1 · Single-target EVDI", "2 · Panel composition", "3 · Panel size",
                "4 · Reporting policy", "5 · Public-health value", "About"])

# --------------------------------------------------------------------------------------
# Tab 1: single target
# --------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Single target: treatment threshold, EVDI, and net value")
    st.markdown("Default inputs reproduce the de-escalation example in §10.1: continue vs. stop "
                "empiric therapy at hour 24.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Test and prevalence**")
        p0 = st.slider("Pretest probability p₀", 0.0, 1.0, 0.30, 0.005, key="t1p0")
        Se = st.slider("Sensitivity Se", 0.50, 1.0, 0.92, 0.005, key="t1se")
        Sp = st.slider("Specificity Sp", 0.50, 1.0, 0.96, 0.005, key="t1sp")
        cost = st.number_input("Assay / pathway cost C_T ($)", 0.0, 10000.0, 15.0, 1.0, key="t1c")
    with c2:
        st.markdown("**Utilities, action 1 (treat / continue)**")
        t1 = st.number_input("U(treat, D+)", value=500.0, step=10.0, key="t1u1")
        t0 = st.number_input("U(treat, D−)", value=-50.0, step=10.0, key="t1u2")
    with c3:
        st.markdown("**Utilities, action 2 (no treat / stop)**")
        n1 = st.number_input("U(no treat, D+)", value=-800.0, step=10.0, key="t1u3")
        n0 = st.number_input("U(no treat, D−)", value=0.0, step=10.0, key="t1u4")
    U1 = dict(t1=t1, t0=t0, n1=n1, n0=n0)
    B, H, pt = threshold(U1)
    if B <= 0 or H <= 0:
        st.error("The model needs B = U(treat,D+) − U(no treat,D+) > 0 and "
                 "H = U(no treat,D−) − U(treat,D−) > 0.")
    else:
        res = single_test(p0, Se, Sp, U1)
        lo, hi = testing_thresholds(Se, Sp, pt)
        m = st.columns(5)
        m[0].metric("Treatment threshold pₜ", f"{pt:.4f}", help="Eq. (2): H / (H + B)")
        m[1].metric(f"Post-test p₊ → {res['a_pos']}", f"{res['post_pos']:.4f}")
        m[2].metric(f"Post-test p₋ → {res['a_neg']}", f"{res['post_neg']:.4f}")
        m[3].metric("Gross EVDI", money(res["evdi"]), help="Eq. (7)")
        m[4].metric("Net value", money(res["evdi"] - cost), help="Eq. (8): gross EVDI − C_T")
        st.markdown(
            f"B = {B:,.0f}, H = {H:,.0f}. Without testing the preferred action is **{res['a0']}**. "
            f"Testing has value only for p₀ between the Pauker–Kassirer testing threshold "
            f"**{lo:.4f}** and test–treatment threshold **{hi:.4f}**.")

        ps = np.linspace(0.0005, 0.9995, 800)
        ev_p = [evdi(p, Se, Sp, U1) for p in ps]
        fig = go.Figure()
        fig.add_scatter(x=ps, y=ev_p, name="Gross EVDI", line=dict(color=BLUE, width=2))
        fig.add_scatter(x=ps, y=np.array(ev_p) - cost, name="Net value", line=dict(color=ORANGE, width=2))
        for lvl, (x, lab) in enumerate([(lo, "testing thr."), (pt, "pₜ"), (hi, "test–treat thr."), (p0, "p₀")]):
            vline(fig, x, lab, AQUA if lab == "p₀" else GREY, level=lvl)
        st.plotly_chart(style(fig, "Pretest probability p₀", "$ per patient"), **STRETCH)

        cc = st.columns(2)
        ses = np.linspace(0.5, 1.0, 400)
        ev_se = [evdi(p0, s, Sp, U1) for s in ses]
        f2 = go.Figure()
        f2.add_scatter(x=ses, y=ev_se, name="Gross EVDI", line=dict(color=BLUE, width=2))
        f2.add_scatter(x=ses, y=np.array(ev_se) - cost, name="Net value", line=dict(color=ORANGE, width=2))
        vline(f2, Se, "Se", AQUA)
        cc[0].plotly_chart(style(f2, "Sensitivity (p₀, Sp fixed)", "$ per patient", 320), **STRETCH)
        sps = np.linspace(0.5, 1.0, 400)
        ev_sp = [evdi(p0, Se, s, U1) for s in sps]
        f3 = go.Figure()
        f3.add_scatter(x=sps, y=ev_sp, name="Gross EVDI", line=dict(color=BLUE, width=2))
        f3.add_scatter(x=sps, y=np.array(ev_sp) - cost, name="Net value", line=dict(color=ORANGE, width=2))
        vline(f3, Sp, "Sp", AQUA)
        cc[1].plotly_chart(style(f3, "Specificity (p₀, Se fixed)", "$ per patient", 320), **STRETCH)

        be_se = solve_on_grid(lambda s: evdi(p0, s, Sp, U1) - cost, 0.5, 1.0)
        be_sp = solve_on_grid(lambda s: evdi(p0, Se, s, U1) - cost, 0.5, 1.0)
        notes = []
        if be_se is not None:
            notes.append(f"breakeven sensitivity ≈ **{be_se:.4f}** (holding Sp = {Sp:.3f})")
        if be_sp is not None:
            notes.append(f"breakeven specificity ≈ **{be_sp:.4f}** (holding Se = {Se:.3f})")
        if notes:
            st.markdown("At this cost, " + "; ".join(notes) + ".")
        st.caption("Gross EVDI is piecewise affine in each of p₀, Se and Sp when the others are fixed "
                   "(Proposition 2), and exactly zero outside the testing range.")

# --------------------------------------------------------------------------------------
# Tab 2: panel composition with dependence
# --------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Panel composition: marginal value, interactions, and greedy failure")
    st.markdown(
        "One treatment decision (state D) is informed by up to four jointly observed binary targets. "
        "Edit the joint distribution of the targets and the probability of D in each cell. The app "
        "evaluates **every subset**, finds the NMB-maximizing panel (Eq. 11), and compares it with "
        "greedy forward selection. As in the paper, targets are treated as directly observed signals.")
    pc = st.columns([2, 1])
    with pc[1]:
        preset = st.selectbox("Start from", list(PRESETS.keys()), key="t2preset")
        c_marg = st.number_input("Marginal cost per added target c ($)", 0.0, 1000.0, 0.50, 0.10, key="t2c")
        F_fixed = st.number_input("Fixed run cost if any target is run F ($)", 0.0, 1000.0, 0.0, 1.0,
                                  key="t2F", help="0 when targets are added to a panel already run for other indications.")
        st.markdown("**Utilities for the decision on D**")
        df0, Up = preset_frame(preset)
        u_t1 = st.number_input("U(treat, D+)", value=Up["t1"], key=f"t2u1{preset}")
        u_t0 = st.number_input("U(treat, D−)", value=Up["t0"], key=f"t2u2{preset}")
        u_n1 = st.number_input("U(no treat, D+)", value=Up["n1"], key=f"t2u3{preset}")
        u_n0 = st.number_input("U(no treat, D−)", value=Up["n0"], key=f"t2u4{preset}")
    with pc[0]:
        edited = st.data_editor(df0, key=f"t2ed{preset}", **STRETCH, hide_index=True,
                                disabled=[c for c in df0.columns if c not in ("P(cell)", "P(D+ | cell)")])
    U2 = dict(t1=u_t1, t0=u_t0, n1=u_n1, n0=u_n0)
    names = [c for c in edited.columns if c not in ("P(cell)", "P(D+ | cell)")]
    probs = edited["P(cell)"].astype(float).clip(lower=0).to_numpy()
    prevs = edited["P(D+ | cell)"].astype(float).clip(0, 1).to_numpy()
    tot = probs.sum()
    if tot <= 0:
        st.error("Cell probabilities must sum to a positive number.")
    else:
        if abs(tot - 1) > 1e-6:
            st.warning(f"Cell probabilities sum to {tot:.4f}; they have been rescaled to 1.")
            probs = probs / tot
        cells = [tuple(1 if edited.iloc[k][n] == "+" else 0 for n in names) for k in range(len(edited))]
        _, _, pt2 = threshold(U2)
        gross = panel_values(names, cells, probs, prevs, U2)
        table, best, best_v, greedy, greedy_v, path, shap, inter = panel_analysis(names, gross, c_marg, F_fixed)
        marg_prev = float(np.dot(probs, prevs))
        lab = lambda S: "{" + ", ".join(names[i] for i in S) + "}" if S else "∅"
        m = st.columns(4)
        m[0].metric(f"Marginal P(D+)  (pₜ = {pt2:.4f})", f"{marg_prev:.4f}")
        m[1].metric(f"Optimal panel  (NMB {money(best_v)})", lab(best))
        m[2].metric(f"Greedy selection  (NMB {money(greedy_v)})", lab(greedy))
        m[3].metric("Loss from greedy", money(best_v - greedy_v))
        if best_v - greedy_v > 1e-9:
            st.error(f"Greedy forward selection stops at {lab(greedy)} and forgoes {money(best_v - greedy_v)} "
                     "per patient: the targets interact (complements), so no single addition pays on its own.")
        cols = st.columns([3, 2])
        with cols[0]:
            st.markdown("**All subsets** (gross value over ∅, cost, and net NMB)")
            st.dataframe(table.style.format({"Gross": "{:.2f}", "Cost": "{:.2f}", "NMB": "{:.2f}"}),
                         **STRETCH, hide_index=True)
            st.download_button("Download subsets (CSV)", table.to_csv(index=False).encode(),
                               "panel_subsets.csv", "text/csv")
        with cols[1]:
            st.markdown("**Pairwise interaction of gross value, I(i, j | ∅)** (Eq. 20)")
            st.dataframe(inter.style.format("{:.2f}", na_rep="—"), **STRETCH)
            st.caption("Negative: substitutes. Positive: complements.")
            st.markdown("**Shapley attribution of gross value**")
            st.dataframe(pd.DataFrame({"Target": names, "Shapley value": shap})
                         .style.format({"Shapley value": "{:.3f}"}), **STRETCH, hide_index=True)
            st.caption("Shapley values attribute value; they do not identify the best subset.")
        fig = go.Figure()
        order = table.sort_values("Targets")
        fig.add_bar(x=order["Panel"], y=order["NMB"],
                    marker_color=[ORANGE if p == lab(best) or (p.startswith("∅") and not best) else BLUE
                                  for p in order["Panel"]],
                    hovertemplate="%{x}: %{y:.2f}<extra></extra>")
        st.plotly_chart(style(fig, "Panel", "NMB per patient ($)", 320).update_layout(hovermode="closest"),
                        **STRETCH)

# --------------------------------------------------------------------------------------
# Tab 3: panel size
# --------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Panel size: accumulation of non-actionable results")
    st.markdown("Candidate targets are ordered by declining gross value g_k = g₁·r^(k−1). Each costs c at "
                "the margin and returns a non-actionable positive with probability a, which triggers an "
                "unintended action with probability φ at excess cost L. Defaults reproduce §10.4 and Figure 2.")
    cc = st.columns(4)
    g1 = cc[0].number_input("Gross value of first target g₁ ($)", 0.0, 1000.0, 6.0, 0.5, key="t3g1")
    r = cc[0].slider("Decay ratio r", 0.0, 1.0, 0.70, 0.01, key="t3r")
    nmax = cc[1].slider("Candidate targets", 5, 60, 30, 1, key="t3n")
    c3 = cc[1].number_input("Marginal cost c ($)", 0.0, 100.0, 0.50, 0.05, key="t3c")
    a3 = cc[2].number_input("P(non-actionable positive) per target, a", 0.0, 1.0, 0.02, 0.005, format="%.3f", key="t3a")
    phi3 = cc[2].slider("φ (reflex action)", 0.0, 1.0, 0.40, 0.05, key="t3phi")
    L3 = cc[3].number_input("Excess loss L ($)", 0.0, 10000.0, 60.0, 5.0, key="t3L")
    n, G, pr_, pp_ = panel_size_curves(g1, r, nmax, c3, a3, phi3, L3)
    i1, i2 = int(np.argmax(pr_)), int(np.argmax(pp_))
    m = st.columns(4)
    m[0].metric(f"Optimal size, per-result loss  (NMB {money(pr_[i1])})", f"{i1}")
    m[1].metric(f"Optimal size, per-patient loss  (NMB {money(pp_[i2])})", f"{i2}")
    m[2].metric(f"Gross value at n = {nmax}", money(G[-1]))
    m[3].metric("P(≥1 non-actionable) at n*", f"{1 - (1 - a3) ** i1:.1%}")
    cols = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=n, y=G, name="Gross value", line=dict(color=GREY, dash="dash", width=1.5))
    fig.add_scatter(x=n, y=pr_, name="Net, per-result loss (Eq. 15)", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=n, y=pp_, name="Net, per-patient loss (Eq. 16)", line=dict(color=ORANGE, width=2))
    fig.add_scatter(x=[i1], y=[pr_[i1]], mode="markers", marker=dict(size=10, color=BLUE, line=dict(color="white", width=2)),
                    showlegend=False, hovertemplate="optimum n=%{x}: %{y:.2f}<extra></extra>")
    cols[0].plotly_chart(style(fig, "Number of targets on panel, n", "NMB per patient ($)"), **STRETCH)
    sps = st.multiselect("Per-target specificity curves", [0.99, 0.995, 0.999, 0.9999], [0.99, 0.995, 0.999], key="t3sp")
    f2 = go.Figure()
    for sp, col in zip(sorted(sps), [BLUE, ORANGE, AQUA, GREY]):
        f2.add_scatter(x=n, y=100 * (1 - sp ** n), name=f"Sp = {sp}", line=dict(color=col, width=2))
    cols[1].plotly_chart(style(f2, "Number of targets on panel, n", "P(≥1 false positive), %"), **STRETCH)
    st.caption("Proposition 1: E[N_S] = Σ a_j under any dependence; with independent errors, "
               "P(≥1 false positive) = 1 − Π Sp_j for a patient negative for every target.")

# --------------------------------------------------------------------------------------
# Tab 4: reporting policy
# --------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Reporting policy as the decision variable")
    st.markdown("Target W is actionable only in a subgroup. Defaults reproduce §10.5 and Table 2.")
    cc = st.columns(4)
    q = cc[0].slider("Subgroup share q", 0.0, 1.0, 0.15, 0.01, key="t4q")
    p04 = cc[0].slider("Subgroup p₀", 0.0, 1.0, 0.08, 0.005, key="t4p0")
    Se4 = cc[1].slider("Se", 0.5, 1.0, 0.95, 0.005, key="t4se")
    Sp4 = cc[1].slider("Sp", 0.5, 1.0, 0.98, 0.005, key="t4sp")
    B4 = cc[2].number_input("B: benefit of treating D+ ($)", 0.0, 1e5, 400.0, 10.0, key="t4B")
    H4 = cc[2].number_input("H: harm of treating D− ($)", 0.01, 1e5, 20.0, 1.0, key="t4H")
    pos4 = cc[3].slider("P(W+) outside subgroup", 0.0, 1.0, 0.10, 0.01, key="t4pos")
    L4 = cc[3].number_input("Excess loss L ($)", 0.0, 1e5, 60.0, 5.0, key="t4L")
    cc2 = st.columns(4)
    phif = cc2[0].slider("φ, full release", 0.0, 1.0, 0.50, 0.05, key="t4phif")
    phic = cc2[1].slider("φ, with interpretive comment", 0.0, 1.0, 0.20, 0.05, key="t4phic")
    impl = cc2[2].number_input("Selective-release cost per test ($)", 0.0, 100.0, 0.30, 0.05, key="t4impl")
    c4 = cc2[3].number_input("Marginal cost of W, configurable panel ($)", 0.0, 100.0, 0.50, 0.05, key="t4c")
    cc3 = st.columns(4)
    miss = cc3[0].slider("Subgroup patients missed", 0.0, 1.0, 0.20, 0.05, key="t4miss")
    leak = cc3[1].slider("Other patients released", 0.0, 1.0, 0.20, 0.05, key="t4leak")
    df4, sub4, G4 = reporting_policies(q, p04, Se4, Sp4, B4, H4, pos4, phif, phic, L4, impl, c4, miss, leak)
    m = st.columns(3)
    m[0].metric("Subgroup gross EVDI", money(sub4["evdi"]))
    m[1].metric("Gross value per tested patient", money(G4))
    m[2].metric("Expected reflex cost, full release", money(df4.iloc[0]["Expected reflex cost"]))
    ib = df4["Net value, fixed panel (assay sunk)"].idxmax()
    st.success(f"Best policy for a fixed panel: **{df4.iloc[ib]['Reporting policy']}** "
               f"({money(df4.iloc[ib]['Net value, fixed panel (assay sunk)'])} per tested patient).")
    st.dataframe(df4.style.format({c: "{:.2f}" for c in df4.columns[1:]}), **STRETCH, hide_index=True)
    worst = df4.iloc[0]["ΔNMB of adding W to a configurable panel"]
    bestc = df4["ΔNMB of adding W to a configurable panel"].max()
    if worst < 0 < bestc:
        st.info(f"Composition reverses with reporting: adding W loses {money(abs(worst))} under full release "
                f"but gains {money(bestc)} under the best reporting policy.")
    fig = go.Figure()
    fig.add_bar(y=df4["Reporting policy"], x=df4["Net value, fixed panel (assay sunk)"], orientation="h",
                marker_color=BLUE, hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    st.plotly_chart(style(fig, "Net value per tested patient, fixed panel ($)", "", 300)
                    .update_layout(hovermode="closest"), **STRETCH)

# --------------------------------------------------------------------------------------
# Tab 5: public-health value
# --------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Public-health value and sampled inclusion of a surveillance target")
    st.markdown("Target E changes no clinical decision but speeds outbreak detection. It is run in a fraction f "
                "of panels, with results routed only to public-health authorities. Defaults reproduce Annex A6.")
    cc = st.columns(4)
    N5 = cc[0].number_input("Panels per year N", 100, 10_000_000, 100_000, 1000, key="t5N")
    c5 = cc[0].number_input("Cost per panel including E ($)", 0.0, 100.0, 0.50, 0.05, key="t5c")
    sp5 = cc[1].number_input("Specificity of E", 0.9, 1.0, 0.999, 0.0005, format="%.4f", key="t5sp")
    conf5 = cc[1].number_input("Confirmatory test per false positive ($)", 0.0, 1e5, 100.0, 10.0, key="t5conf")
    pi5 = cc[2].slider("Outbreak probability per year π", 0.0, 1.0, 0.20, 0.01, key="t5pi")
    lam5 = cc[2].number_input("Infected patients tested per day λ", 0.01, 1000.0, 2.0, 0.5, key="t5lam")
    D05 = cc[3].number_input("Days to recognition without E, D₀", 1.0, 365.0, 21.0, 1.0, key="t5D0")
    Cd5 = cc[3].number_input("Outbreak cost averted per day earlier C_d ($)", 0.0, 1e8, 40_000.0, 1000.0, key="t5Cd")
    ceff = c5 + (1 - sp5) * conf5
    fs = np.linspace(0.001, 1.0, 2000)
    V, delay = ph_value(fs, pi5, Cd5, lam5, D05)
    net = V - ceff * fs * N5
    k = int(np.argmax(net))
    f_closed = math.sqrt(pi5 * Cd5 / (lam5 * ceff * N5)) if ceff > 0 else float("nan")
    V1, _ = ph_value([1.0], pi5, Cd5, lam5, D05)
    m = st.columns(4)
    m[0].metric("Effective cost per panel c′", money(ceff))
    m[1].metric("Optimal inclusion fraction f*", f"{fs[k]:.3f}",
                help=f"Closed form (Eq. A7): {f_closed:.3f}" + (" (above 1, so universal inclusion)" if f_closed > 1 else ""))
    m[2].metric(f"Net value at f*  ({money(net[k] / N5)} per panel)", money(net[k], 0) + "/yr")
    m[3].metric("Net value, universal", money(float(V1[0] - ceff * N5), 0) + "/yr")
    fig = go.Figure()
    fig.add_scatter(x=fs, y=V, name="Value of earlier detection", line=dict(color=GREY, dash="dash", width=1.5))
    fig.add_scatter(x=fs, y=net, name="Net value", line=dict(color=BLUE, width=2))
    vline(fig, fs[k], "f*", ORANGE)
    st.plotly_chart(style(fig, "Fraction of panels including E, f", "$ per year"), **STRETCH)
    rows = []
    for f in sorted({1.0, 0.5, round(float(fs[k]), 3), 0.1, 0.05, 0.02}, reverse=True):
        v, d = ph_value([f], pi5, Cd5, lam5, D05)
        rows.append(dict(f=f, **{"Expected days to detection": float(d[0]), "Value ($/yr)": float(v[0]),
                                 "Inclusion cost ($/yr)": ceff * f * N5,
                                 "Net ($/yr)": float(v[0]) - ceff * f * N5,
                                 "Net per panel ($)": (float(v[0]) - ceff * f * N5) / N5}))
    st.dataframe(pd.DataFrame(rows).style.format({"f": "{:.3f}", "Expected days to detection": "{:.2f}",
                                                  "Value ($/yr)": "{:,.0f}", "Inclusion cost ($/yr)": "{:,.0f}",
                                                  "Net ($/yr)": "{:,.0f}", "Net per panel ($)": "{:.2f}"}),
                 **STRETCH, hide_index=True)
    st.caption("V_ph(f) = π·C_d·[D₀ − (1 − e^(−λfD₀))/(λf)] (Eq. A6). f* = √(π·C_d / (λ·c′·N)) when e^(−λfD₀) is negligible (Eq. A7). "
               "Illustrative inputs; real applications need pathogen-specific estimates and uncertainty analysis.")

# --------------------------------------------------------------------------------------
# About
# --------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("What the app computes")
    st.markdown("**Treatment threshold and EVDI** (Eqs. 1–8)")
    st.latex(r"p_t=\frac{H}{H+B},\qquad \mathrm{EVDI}^{gross}=\sum_{r}P(r)\max_a E[U(a,D)\mid p_r]-\max_a E[U(a,D)\mid p_0]")
    st.markdown("**Panel objective and marginal value** (Eqs. 9–11, 20)")
    st.latex(r"S^*=\arg\max_{S\subseteq\mathcal T}NMB(S),\quad \Delta NMB(i\mid S)=NMB(S\cup i)-NMB(S),\quad I(i,j\mid S)=\Delta NMB(i\mid S\cup j)-\Delta NMB(i\mid S)")
    st.markdown("**Non-actionable results** (Proposition 1, Eqs. 15–16)")
    st.latex(r"E[L_{reflex}]=\sum_{j\in S}a_j\phi_jL_j\ \ \text{(per result)},\qquad E[L_{reflex}]=\phi L\,P(N_S\ge1)\ \ \text{(per patient)}")
    st.markdown("**Public-health value** (Eq. 18, Annex A6)")
    st.latex(r"NMB_{soc}(S,\rho_c,\rho_{ph})=NMB(S,\rho_c)+V_{ph}(S,\rho_{ph})")
    st.markdown(
        "**Scope.** The app implements the paper's binary-decision illustrations. Panel targets are treated as "
        "directly observed signals informing one decision; assay error for those targets, multiple decisions, "
        "and parameter uncertainty (PSA, EVPPI, EVSI) are not modeled. Results are illustrative and conditional "
        "on the inputs.")

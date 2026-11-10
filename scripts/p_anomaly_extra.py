"""Z-Score page extras: data illustration on a return series + choosing the threshold k / window."""
import math
from dio import *

W = 20


VOLS = (0.005, 0.02, 0.007, 0.024, 0.006)          # 5 volatility regimes (calm / stressed alternating)


def make_series(n=260, seed=21, n_anom=10):
    """Daily returns with volatility regimes and injected labelled anomalies (4–6 local σ)."""
    r = seeded(seed)
    rets, labels = [], []
    anom_days = set(r.sample(range(W + 5, n), n_anom))
    seg = n / len(VOLS)
    for t in range(n):
        vol = VOLS[min(int(t / seg), len(VOLS) - 1)]
        x = r.gauss(0, vol)
        lab = t in anom_days
        if lab:
            x = (1 if r.random() < 0.5 else -1) * vol * r.uniform(4.0, 6.0)
        rets.append(x)
        labels.append(lab)
    closes = [100.0]
    for x in rets[1:]:
        closes.append(closes[-1] * (1 + x))
    return rets, labels, closes


def rolling_z(rets, w=W):
    """z_t uses the PREVIOUS w returns only (no look-ahead)."""
    out = []
    for t in range(len(rets)):
        if t < w:
            out.append((None, None, None)); continue
        win = rets[t - w:t]
        mu = sum(win) / w
        sd = math.sqrt(sum((v - mu) ** 2 for v in win) / (w - 1))
        out.append((mu, sd, (rets[t] - mu) / sd))
    return out


def prf(z, labels, k):
    tp = fp = fn = 0
    for (mu, sd, zz), lab in zip(z, labels):
        if zz is None: continue
        flag = abs(zz) > k
        tp += flag and lab
        fp += flag and not lab
        fn += (not flag) and lab
    p = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    f = 2 * p * rc / (p + rc) if p + rc else 0
    return p, rc, f, tp, fp, fn


def build(pg: Page, cat):
    rets, labels, closes = make_series()
    z = rolling_z(rets)
    n = len(rets)

    # ---------------------------------------------------------------- ⑦ data illustration
    x0, y0 = 400, pg.max_y() + 60
    pg.panel(x0, y0, 2000, 700, 7, "Data Illustration: rolling Z-score on daily returns",
             f"Simulated {n} trading days · window w = {W} · 5 volatility regimes (calm ↔ stressed) · {sum(labels)} injected anomalies")
    # table around the first flagged anomaly
    first = next(t for t in range(W, n) if abs(z[t][2]) > 3)
    rows = [["t", "Close", "Return \\(r_t\\)", "\\(\\mu_{t}\\) (prev 20)", "\\(\\sigma_{t}\\) (prev 20)",
             "\\(z_t\\)", "Flag |z|&gt;3"]]
    hl = ()
    for i, t in enumerate(range(first - 5, first + 4)):
        mu, sd, zz = z[t]
        flag = abs(zz) > 3
        if flag: hl = hl + (i + 1,)
        rows.append([t, f"{closes[t]:.2f}", f"{rets[t] * 100:+.2f}%", f"{mu * 100:+.3f}%", f"{sd * 100:.3f}%",
                     f"<b>{zz:+.2f}</b>" if flag else f"{zz:+.2f}",
                     '<font color="#FF0000"><b>ANOMALY</b></font>' if flag else "–"])
    pg.table(x0 + 20, y0 + 70, 620, rows, size=11, highlight_rows=hl)
    pg.math("z_t = \\frac{r_t - \\mu_t}{\\sigma_t},\\quad \\mu_t = \\frac{1}{w}\\sum_{i=t-w}^{t-1} r_i", x0 + 20,
            y0 + 330, 620, 50, 16)
    pg.note("only past data → no look-ahead bias", x0 + 380, y0 + 385, x0 + 470, y0 + 362, w=230, color=RED)
    pg.bullets(x0 + 20, y0 + 420, 620, [
        "Prices are non-stationary → work on <b>returns</b> \\(r_t = \\frac{P_t}{P_{t-1}} - 1\\)",
        "A <b>rolling</b> window adapts μ and σ to the current volatility regime",
        "The current value is excluded from its own window, so a big spike cannot hide itself",
        f"Row t = {first}: the return is {abs(z[first][2]):.1f} rolling σ away from the mean → flagged",
        "<b>Masking</b>: after a spike enters the window, σ stays inflated for w days (band bulges)"],
        head="How the algorithm reads the table:")
    pf_first = (first, rets[first] * 100)
    # plot with rolling bands
    lo = min(min(rets), min(z[t][0] - 3 * z[t][1] for t in range(W, n))) * 100 * 1.05
    hi = max(max(rets), max(z[t][0] + 3 * z[t][1] for t in range(W, n))) * 100 * 1.05
    p = pg.axes(x0 + 720, y0 + 90, 1220, 480, (0, n), (lo, hi), "t", "r_t\\ (\\%)", label_size=12,
                origin=(0, lo), xticks=(0, 50, 100, 150, 200, 250), yticks=(-10, -5, 0, 5, 10))
    p.hline(0, color=GREY)
    p.poly([(t, rets[t] * 100) for t in range(n)], color=BLUE, width=1)
    up = [(t, (z[t][0] + 3 * z[t][1]) * 100) for t in range(W, n)]
    dn = [(t, (z[t][0] - 3 * z[t][1]) * 100) for t in range(W, n)]
    p.poly(up, color=ORANGE, width=1.5, dashed=True)
    p.poly(dn, color=ORANGE, width=1.5, dashed=True)
    tp = fp = 0
    for t in range(W, n):
        if abs(z[t][2]) > 3:
            p.scatter([(t, rets[t] * 100)], color=RED, r=4.5)
            if not labels[t]: fp += 1
        if labels[t]:
            px, py = p.P(t, rets[t] * 100)
            pg.ellipse(px - 7, py - 7, 14, 14, stroke=GREEN, sw=1.5)
    seg = n / len(VOLS)
    for i, v in enumerate(VOLS):
        if i: p.vline(i * seg, lo, hi, color=GREY)
        p.text(f"{'calm' if v < 0.01 else 'stressed'} σ={v * 100:.1f}%", (i + 0.5) * seg, hi, dy=8, w=140, size=10,
               color="#555555")
    bx, by = p.P(seg * 0.55, up[int(seg * 0.55) - W][1])
    pg.note("μ ± 3σ rolling band (adapts to volatility)", bx + 30, by - 60, bx, by, w=250, color=ORANGE)
    pg.html('<font color="#FF0000">●</font> flagged |z| &gt; 3 &nbsp;&nbsp; <font color="#009900">○</font> injected '
            '(true) anomaly', x0 + 720, y0 + 610, 600, 20, 11)
    pg.ask("When does a return become an anomaly?",
           f"when it leaves the rolling ±3σ band: day {first} is {abs(z[first][2]):.1f}σ away; "
           "the band widens in stressed regimes",
           x0 + 20, y0 + 565, *p.P(*pf_first), w=620)
    pp, rc, f1, tpn, fpn, fnn = prf(z, labels, 3)
    pg.html(f"At k = 3: TP = {tpn}, FP = {fpn}, FN = {fnn} → precision = {pp:.2f}, recall = {rc:.2f}, "
            f'<span style="background:{HL}">F1 = {f1:.2f}</span>', x0 + 720, y0 + 640, 1200, 22, 12)

    # ---------------------------------------------------------------- ⑧ choosing k and the window
    y0 = y0 + 740
    pg.panel(x0, y0, 2000, 640, 8, "Finding the optimal hyperparameters: threshold k and window w",
             "Use labelled history (or injected anomalies) and pick the value that maximizes F1 on validation data")
    ks = [1.5 + 0.1 * i for i in range(36)]
    res = [prf(z, labels, k) for k in ks]
    best = max(range(len(ks)), key=lambda i: (res[i][2], -abs(ks[i] - 3)))
    p = pg.axes(x0 + 80, y0 + 100, 640, 330, (1.5, 5.0), (0, 1.05), "k", "", label_size=13, origin=(1.5, 0),
                xticks=(1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5), yticks=(0.25, 0.5, 0.75, 1), tick_fmt="{:g}")
    for idx, col, lab in [(0, BLUE, "Precision"), (1, ORANGE, "Recall"), (2, RED, "F1")]:
        p.poly([(k, rr[idx]) for k, rr in zip(ks, res)], color=col, width=2.5 if idx == 2 else 1.5)
        p.text(lab, 5.0, res[-1][idx], dx=40, size=11, color=col, bold=True)
    kb = ks[best]
    p.vline(kb, 0, 1.0, color=GREEN, width=1.5)
    p.scatter([(kb, res[best][2])], color=GREEN, r=5)
    pg.note(f"best k = {kb:.1f} → F1 = {res[best][2]:.2f}", x0 + 300, y0 + 75, *p.P(kb, res[best][2]), w=170,
            color=GREEN)
    pg.ask("Which threshold k should I use?",
           f"k = {kb:.1f}: max F1 = {res[best][2]:.2f}; lower k → false alarms, higher k → misses",
           x0 + 470, y0 + 103, *p.P(kb + 0.03, res[best][2]), w=240)
    pg.text("small k → many false alarms (low precision)", x0 + 90, y0 + 445, 300, 18, 10, color=BLUE)
    pg.text("large k → misses anomalies (low recall)", x0 + 450, y0 + 445, 280, 18, 10, color=ORANGE)
    pg.math(r"P=\frac{TP}{TP+FP},\ R=\frac{TP}{TP+FN},\ F_1=\frac{2PR}{P+R}", x0 + 80, y0 + 480, 620, 44, 15)
    pg.bullets(x0 + 20, y0 + 535, 720, [
        "No labels? Choose k from the tolerated alarm rate: |z|&gt;2 ≈ 4.6 %, |z|&gt;3 ≈ 0.27 % of normal days (if returns were normal)",
        "Fat tails (kurtosis &gt; 3) → normal-based rates are too optimistic → validate on history"], size=11)
    # window trade-off
    wx = x0 + 800
    pg.text("Grid search over (w, k): best F1 for each window", wx, y0 + 70, 540, 22, 13, bold=True)
    rows = [["w", "best k", "Precision", "Recall", "F1", "Behaviour"]]
    beh = {5: "σ very noisy", 10: "reacts fast", 20: "balanced", 60: "slow to adapt", 120: "almost static"}
    WS = (5, 10, 20, 60, 120)
    fbs = []
    for w in WS:
        zw = rolling_z(rets, w)
        cand = [(prf(zw, labels, k), k) for k in ks]
        (pr, rcw, fw, *_), kk = max(cand, key=lambda c: (c[0][2], -abs(c[1] - 3)))
        fbs.append(fw)
        rows.append([w, f"{kk:.1f}", f"{pr:.2f}", f"{rcw:.2f}", f"{fw:.2f}", beh[w]])
    # static (global) z
    mu_g = sum(rets) / n
    sd_g = math.sqrt(sum((v - mu_g) ** 2 for v in rets) / (n - 1))
    zg = [(None, None, None)] * W + [(mu_g, sd_g, (v - mu_g) / sd_g) for v in rets[W:]]
    (pg_, rg_, fg_, *_), kg = max([(prf(zg, labels, k), k) for k in ks], key=lambda c: c[0][2])
    rows.append(["static", f"{kg:.1f}", f"{pg_:.2f}", f"{rg_:.2f}", f"{fg_:.2f}", "one global μ, σ"])
    hl = (1 + max(range(len(WS)), key=lambda i: fbs[i]),)
    pg.table(wx, y0 + 100, 540, rows, size=12, highlight_rows=hl)
    pg.html("Static z uses one σ for all regimes → misses anomalies in calm periods and "
            "over-flags normal days in stressed periods.", wx, y0 + 272, 540, 40, 11)
    pg.bullets(wx, y0 + 320, 520, [
        "Grid-search (k, w) on a <b>validation period</b>, then confirm on a later test period (walk-forward)",
        "Keep the window strictly in the past; refit nothing on the test period",
        "w trades <b>adaptivity</b> (small w) against <b>stability of σ</b> (large w)"], head="Procedure:", size=11)
    # contamination box
    cx = x0 + 1380
    pg.rect(cx, y0 + 70, 590, 200,
            "<b>Contamination (expected anomaly rate)</b><br><br>"
            f"Here: {sum(labels)} / {n} = {sum(labels) / n * 100:.1f} % of days.<br>"
            "Many libraries (Isolation Forest, LOF, One-Class SVM) take <i>contamination</i> instead of k: "
            "they flag the top c % most extreme scores.<br><br>"
            "Equivalent for Z-score: choose k = the (1 − c) quantile of |z| on training data.<br><br>"
            "Set c from domain knowledge (e.g. known fraud rate) — a wrong c directly sets precision / recall.",
            fill="#F5F5F5", stroke="#999999", size=12, align="left", valign="top", raw=True)
    qs = sorted(abs(t[2]) for t in z if t[2] is not None)
    kc = qs[int(len(qs) * (1 - sum(labels) / n))]
    pg.html(f'Quantile rule with c = {sum(labels) / n * 100:.1f} % → '
            f'<span style="background:{HL}">k ≈ {kc:.2f}</span> (compare best-F1 k = {kb:.1f})',
            cx, y0 + 285, 590, 24, 12)
    pg.bullets(cx, y0 + 330, 590, [
        "Masking: several anomalies close together inflate σ → use median / MAD (panel ⑤)",
        "Swamping: very low k marks normal neighbours as anomalies",
        "Multivariate data → Mahalanobis distance or Isolation Forest"], head="Pitfalls:", size=11)

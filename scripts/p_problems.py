"""Common Data Problems page (base: Curse of Dimensionality) - model-impact view of data problems."""
import math
from dio import *

LINK = lambda pid: f"data:page/id,{pid}"


# ---------------------------------------------------------------------- helpers
def solve(A, b):
    """Gaussian elimination for small dense systems."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[c][c] != 0:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def ols(X, y):
    """X rows without intercept -> [b0, b1, ...]."""
    Z = [[1.0] + list(r) for r in X]
    p = len(Z[0])
    A = [[sum(z[i] * z[j] for z in Z) for j in range(p)] for i in range(p)]
    b = [sum(z[i] * yy for z, yy in zip(Z, y)) for i in range(p)]
    return solve(A, b)


def logreg(X, y, iters=1500, lr=0.15):
    w = [0.0, 0.0, 0.0]
    n = len(X)
    for _ in range(iters):
        g = [0.0, 0.0, 0.0]
        for (a, b), t in zip(X, y):
            z = w[0] + w[1] * a + w[2] * b
            p = 1 / (1 + math.exp(-max(-30, min(30, z))))
            e = p - t
            g[0] += e; g[1] += e * a; g[2] += e * b
        w = [wi - lr * gi / n for wi, gi in zip(w, g)]
    return w


def clip_line(w, xr, yr):
    """Segment of w0 + w1 x + w2 y = 0 inside the box."""
    pts = []
    w0, w1, w2 = w
    for xv in xr:
        if abs(w2) > 1e-12:
            yv = -(w0 + w1 * xv) / w2
            if yr[0] <= yv <= yr[1]: pts.append((xv, yv))
    for yv in yr:
        if abs(w1) > 1e-12:
            xv = -(w0 + w2 * yv) / w1
            if xr[0] <= xv <= xr[1]: pts.append((xv, yv))
    pts = sorted(set((round(a, 6), round(b, 6)) for a, b in pts))
    return [pts[0], pts[-1]] if len(pts) >= 2 else None


def skewness(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    return sum((x - m) ** 3 for x in v) / len(v) / s ** 3


def build(pg: Page, cat):
    panel_curse(pg)
    panel_imbalance(pg)
    panel_collinear(pg)
    panel_values(pg)
    panel_drift(pg)
    panel_master(pg)


# ---------------------------------------------------------------------- ⑤ leakage, drift, backtest biases
def panel_drift(pg):
    x0, y0 = 400, 1780
    pg.panel(x0, y0, 1100, 720, 5, "Leakage, Non-stationarity & Drift",
             "The future or the production world differs from the training table")
    r = seeded(31)
    # leaky column
    n = 500
    ret = [r.gauss(0, 1) for _ in range(n)]
    up = [1 if v > 0 else 0 for v in ret]
    honest = [0.15 * v + r.gauss(0, 1) for v in ret]
    leak = [v + r.gauss(0, 0.05) for v in ret]
    acc_h = sum((1 if f > 0 else 0) == u for f, u in zip(honest, up)) / n
    acc_l = sum((1 if f > 0 else 0) == u for f, u in zip(leak, up)) / n
    pg.text("Leaky column: ret_next is known only AFTER the bar closes", x0 + 30, y0 + 66, 500, 18, 12, bold=True)
    LK = lambda s: f'<font color="#FF0000"><b>{s}</b></font>'
    pg.table(x0 + 30, y0 + 90, 300, [["t", "rsi", "ema_gap", LK("ret_next"), "up"]] +
             [[i + 1, f"{50 + 10 * honest[i]:.0f}", f"{honest[i]:+.2f}", LK(f"{ret[i]:+.2f}"), up[i]]
              for i in range(5)], size=10, h=120)
    q = pg.axes(x0 + 370, y0 + 100, 140, 110, (0, 2), (0, 1), "", "", arrow=False, yticks=(0.5, 1))
    q.bars([(0.5, acc_l), (1.5, acc_h)], 0.6, fill=LIGHTRED, stroke=RED, labels=True)
    pg.text("with leak", x0 + 360, y0 + 214, 80, 14, 9, align="center")
    pg.text("honest", x0 + 430, y0 + 214, 80, 14, 9, align="center")
    pg.ask("Why is my validation accuracy almost perfect?",
           f"A column leaks the answer: {acc_l:.0%} with ret_next vs {acc_h:.0%} honestly; live, the column does not "
           "exist → audit timestamps", x0 + 30, y0 + 240, *q.P(0.5, acc_l), w=300)
    # covariate drift
    tr = [r.gauss(0, 1) for _ in range(2000)]
    pr = [r.gauss(0.6, 1.3) for _ in range(2000)]
    allv = sorted(set(tr + pr))
    ks, i, j, st, sp = 0, 0, 0, sorted(tr), sorted(pr)
    for v in allv[::7]:
        while i < len(st) and st[i] <= v: i += 1
        while j < len(sp) and sp[j] <= v: j += 1
        ks = max(ks, abs(i / len(st) - j / len(sp)))
    edges = [st[int(len(st) * k / 10)] for k in range(1, 10)]
    def frac(v):
        c = [0] * 10
        for x in v: c[sum(1 for e in edges if x > e)] += 1
        return [max(1e-4, k / len(v)) for k in c]
    ft, fp = frac(tr), frac(pr)
    psi = sum((b - a) * math.log(b / a) for a, b in zip(ft, fp))
    dx = x0 + 580
    pg.text("Covariate drift: same feature, train vs production", dx, y0 + 66, 500, 18, 12, bold=True)
    ht, bw = hist(tr, -4, 5, 27)
    hp, _ = hist(pr, -4, 5, 27)
    top = max(c for _, c in ht + hp) * 1.15
    c = pg.axes(dx + 30, y0 + 100, 440, 150, (-4, 5), (0, top), "x", "", label_size=11, arrow=False,
                xticks=(-4, -2, 0, 2, 4))
    c.bars(ht, bw, fill=LIGHTBLUE, stroke=BLUE, opacity=60)
    c.bars(hp, bw, fill=LIGHTRED, stroke=RED, opacity=50)
    pg.html('<font color="#3333FF">■ train</font>  <font color="#FF0000">■ production</font>', dx + 360, y0 + 96,
            140, 18, 10)
    pg.html(f"KS statistic = <b>{ks:.2f}</b> · PSI = <b>{psi:.2f}</b>  (PSI &lt; 0.1 stable · 0.1–0.25 watch · "
            f"&gt; 0.25 major shift)", dx + 10, y0 + 270, 500, 18, 11)
    pg.math(r"PSI=\sum_{b}(p_b-q_b)\ln\frac{p_b}{q_b}", dx + 10, y0 + 292, 260, 40, 13)
    pg.ask("Is production data still like my training data?",
           f"No: PSI {psi:.2f} &gt; 0.25 and KS {ks:.2f} — the feature shifted; retrain or recalibrate",
           dx + 280, y0 + 300, *c.P(1.8, top * 0.5), w=230)
    # concept drift
    pg.text("Concept drift: same inputs, the relationship changes", x0 + 30, y0 + 400, 500, 18, 12, bold=True)
    A = [(xv, 0.8 * xv + r.gauss(0, 0.5)) for xv in [r.uniform(-2, 2) for _ in range(30)]]
    B = [(xv, -0.5 * xv + r.gauss(0, 0.5)) for xv in [r.uniform(-2, 2) for _ in range(30)]]
    bA, bB = ols([(a,) for a, _ in A], [b for _, b in A]), ols([(a,) for a, _ in B], [b for _, b in B])
    d = pg.axes(x0 + 60, y0 + 440, 300, 180, (-2.2, 2.2), (-2.5, 2.5), "x", "y", label_size=11, origin=(0, 0))
    d.scatter(A, color=BLUE, r=3); d.scatter(B, color=RED, r=3)
    d.curve(lambda v: bA[0] + bA[1] * v, -2.2, 2.2, n=2, curved=False, color=BLUE, width=2)
    d.curve(lambda v: bB[0] + bB[1] * v, -2.2, 2.2, n=2, curved=False, color=RED, width=2)
    pg.html(f'<font color="#3333FF">● 2023: slope {bA[1]:+.2f}</font><br><font color="#FF0000">● 2024: slope '
            f'{bB[1]:+.2f}</font>', x0 + 380, y0 + 450, 150, 36, 11)
    pg.ask("Can a model trained on 2023 trade 2024?",
           f"Not here: the slope flipped from {bA[1]:+.2f} to {bB[1]:+.2f}; the x distribution looks the same, so only "
           "monitoring the error reveals it", x0 + 370, y0 + 510, *d.P(1.5, bB[0] + bB[1] * 1.5), w=200)
    # survivorship bias
    sx = x0 + 600
    pg.text("Backtest biases: survivorship & look-ahead", sx, y0 + 400, 480, 18, 12, bold=True)
    s = pg.axes(sx + 20, y0 + 440, 300, 160, (0, 100), (0, 3.0), "t", "price", label_size=10, arrow=False)
    for seed in range(100, 200):          # find a universe with 2-4 delisted stocks
        rr, paths = seeded(seed), []
        for k in range(8):
            pth, v = [], 1.0
            for t in range(101):
                pth.append((t, min(v, 2.95)))
                v *= math.exp(rr.gauss(0.0, 0.05))
                if v < 0.5: break
            paths.append(pth)
        if 2 <= sum(1 for p_ in paths if p_[-1][0] < 100) <= 4: break
    finals_all, finals_surv = [], []
    for pth in paths:
        dead = pth[-1][0] < 100
        s.poly(pth[::2] + [pth[-1]], color=RED if dead else BLUE, width=1.2)
        if dead:
            s.text("✘", pth[-1][0], pth[-1][1], size=12, color=RED, w=20)
            finals_all.append(0.0)
        else:
            finals_all.append(pth[-1][1]); finals_surv.append(pth[-1][1])
    ra = sum(finals_all) / len(finals_all) - 1
    rs = sum(finals_surv) / len(finals_surv) - 1
    pg.html(f"avg return of <b>survivors only</b>: <b>{rs:+.0%}</b><br>avg incl. <font color='#FF0000'>delisted</font>: "
            f"<b>{ra:+.0%}</b>", sx + 340, y0 + 450, 170, 52, 11)
    pg.ask("Why does my backtest beat reality?",
           f"Delisted stocks are missing: survivors show {rs:+.0%} but the real universe gave {ra:+.0%}",
           sx + 340, y0 + 520, *s.P(60, 1.4), w=170)
    pg.rect(sx + 20, y0 + 625, 480, 48,
            "<b>Look-ahead:</b> signal computed from Close(t) but the trade fills at Open(t) — or a feature "
            "uses a revised / centred value. Fix: point-in-time data, <font face='Courier New'>shift(1)</font>, "
            "trade at t+1.", fill="#FFF5F5", stroke=RED, dashed=True, size=10, align="left", valign="top", raw=True)
    pg.text("Non-stationarity (mean / variance drift over time) → tests & fixes on the ADF page", x0 + 30, y0 + 690,
            540, 18, 10, italic=True, color="#3333FF", link=LINK("tstests"))


# ---------------------------------------------------------------------- ⑥ master table
def panel_master(pg):
    x0, y0 = 1540, 1780
    pg.panel(x0, y0, 860, 720, 6, "Problem → Symptom → Detection → Remedy")
    rows = [["Problem", "Symptom", "Detect with", "Remedy"],
            ["Curse of dimensionality", "kNN / K-Means degrade as columns grow", "n / p ratio, distance ratio",
             "feature selection, PCA, L1"],
            ["Small data", "scores swing across folds", "CV std, learning curve", "simpler model, regularize, more rows"],
            ["Multicollinearity", "coefficients flip sign", "VIF &gt; 10, correlation matrix", "drop / combine, Ridge, PCA"],
            ["Class imbalance", "high accuracy, low recall", "class counts, PR curve",
             "class weights, SMOTE (train fold), threshold"],
            ["Label noise", "train error won't drop", "inspect errors, label agreement", "relabel, robust loss"],
            ["Leaky target", "validation too good to be true", "importance spike, timestamp audit",
             "remove column, time-aware pipeline"],
            ["Missing data", "NaNs, biased subsets", "missingness by column / group", "kNN / model impute, is_missing flag"],
            ["Outliers", "slope / mean jumps", "Z-score, IQR, Cook's distance", "check, cap, robust models"],
            ["Skewed features", "mean ≫ median", "skewness, histogram", "log / Box-Cox"],
            ["Non-stationarity", "statistics shift over time", "ADF / KPSS, rolling stats", "differencing, returns, retrain"],
            ["Data / concept drift", "live metric decays", "PSI, KS, monitoring", "retrain, sliding window, alerts"],
            ["Look-ahead / survivorship", "backtest ≫ live results", "timestamp audit, delisted data",
             "point-in-time data, shift(1)"]]
    th = len(rows) * 26
    pg.table(x0 + 20, y0 + 50, 820, rows, size=12, h=th, highlight_rows=(1,))
    pg.ask("What explains 'great backtest, bad live trading'?",
           "Usually leakage, look-ahead or survivorship bias (rows 6 and 12) — audit timestamps and the universe first",
           x0 + 440, y0 + 70 + th, x0 + 110, y0 + 50 + int(len(rows) * 21.2) - 8, w=400)
    pg.bullets(x0 + 20, y0 + 80 + th, 400, [
        "Profile every column: type, missing %, unique values, skew",
        "Count rows per class and rows per feature (n / p)",
        "Check timestamps: is every feature known at prediction time?",
        "Split by time / group before any fitting or resampling",
        "Compare train vs recent data (PSI, KS) before deploying"], head="Pre-modelling checklist:", size=11)


def hist(vals, lo, hi, bins):
    w = (hi - lo) / bins
    c = [0] * bins
    for v in vals:
        k = int((v - lo) / w)
        if 0 <= k < bins: c[k] += 1
    return [(lo + (k + 0.5) * w, c[k]) for k in range(bins)], w


# ---------------------------------------------------------------------- ③ multicollinearity & small data
def panel_collinear(pg):
    x0, y0 = 400, 1040
    pg.panel(x0, y0, 940, 700, 3, "Multicollinearity & Small Data",
             "Correlated columns and few rows both make estimates unstable")
    r = seeded(5)
    n = 40
    x1 = [r.gauss(0, 1) for _ in range(n)]
    x2 = [a + r.gauss(0, 0.1) for a in x1]
    x3 = [r.gauss(0, 1) for _ in range(n)]
    y = [3 * a + c + r.gauss(0, 1) for a, c in zip(x1, x3)]
    both, alone = [], []
    for _ in range(40):
        idx = [r.randrange(n) for _ in range(n)]
        b = ols([(x1[i], x2[i], x3[i]) for i in idx], [y[i] for i in idx])
        both.append((b[1], b[2]))
        alone.append(ols([(x1[i], x3[i]) for i in idx], [y[i] for i in idx])[1])
    lo = min(min(a for a, _ in both), min(b for _, b in both)) - 0.5
    hi = max(max(a for a, _ in both), max(b for _, b in both)) + 0.5
    p = pg.axes(x0 + 70, y0 + 100, 320, 260, (lo, hi), (lo, hi), "\\beta_1", "\\beta_2", label_size=12,
                origin=(lo, lo))
    p.curve(lambda v: 3 - v, lo, hi, n=2, curved=False, dashed=True, color=GREY)
    p.scatter(both, color=BLUE, r=3.5)
    p.scatter([(a, 0) for a in alone] if lo < 0 < hi else [], color=GREEN, r=3)
    p.text("β₁ + β₂ ≈ 3", hi - 1.2, 3 - (hi - 1.2), dx=40, size=10, color=GREY, w=80)
    sd = lambda v: math.sqrt(sum((a - sum(v) / len(v)) ** 2 for a in v) / len(v))
    sd_both, sd_alone = sd([a for a, _ in both]), sd(alone)
    pg.html(f'<font color="#3333FF">● y ~ x₁ + x₂ + x₃</font>: sd(β₁) = <b>{sd_both:.2f}</b><br>'
            f'<font color="#009900">● y ~ x₁ + x₃</font> (x₂ dropped): sd(β₁) = <b>{sd_alone:.2f}</b>',
            x0 + 60, y0 + 380, 360, 36, 11)
    pg.ask("Can I trust each coefficient on its own?",
           f"Not with x₁ and x₂ both in: β₁ jumps around (sd {sd_both:.2f}) while only the sum ≈ 3 is stable; "
           f"dropping x₂ gives sd {sd_alone:.2f}", x0 + 470, y0 + 300, *p.P(*both[len(both) // 2]), w=300)
    # VIF
    cols = {"x₁": x1, "x₂": x2, "x₃": x3}
    vif = {}
    for name, v in cols.items():
        others = [c for k, c in cols.items() if k != name]
        b = ols(list(zip(*others)), v)
        pred = [b[0] + sum(bi * oi for bi, oi in zip(b[1:], row)) for row in zip(*others)]
        m = sum(v) / n
        r2 = 1 - sum((a - q) ** 2 for a, q in zip(v, pred)) / sum((a - m) ** 2 for a in v)
        vif[name] = (r2, 1 / (1 - r2))
    vx = x0 + 470
    pg.text("Variance Inflation Factor", vx, y0 + 80, 300, 20, 12, bold=True)
    pg.math(r"VIF_j = \frac{1}{1 - R_j^2}", vx, y0 + 105, 200, 44, 16)
    pg.text("R²ⱼ = fit of feature j from the other features", vx + 200, y0 + 115, 240, 30, 10, color="#555555")
    rows = [["feature", "R²ⱼ", "VIF", "verdict"]]
    for name, (r2, vv) in vif.items():
        rows.append([name, f"{r2:.3f}", f"{vv:,.1f}", "✘ redundant" if vv > 10 else "✔ ok"])
    pg.table(vx, y0 + 160, 300, rows, size=11, h=95, highlight_rows=(1, 2))
    pg.text("Rule of thumb: VIF > 5 worry · > 10 act (drop, combine, PCA, Ridge)", vx, y0 + 262, 440, 18, 10,
            italic=True)
    pg.text("(green = x₁-only model's β₁, plotted at β₂ = 0)", x0 + 60, y0 + 418, 360, 16, 9, italic=True,
            color="#555555")
    # small data: spurious correlation
    sx, sy = x0 + 40, y0 + 470
    pg.text("Small data: correlation between two INDEPENDENT columns (300 repeats)", sx, sy - 10, 560, 20, 12,
            bold=True)
    for k, nn in enumerate((10, 100)):
        rs = []
        for _ in range(300):
            a = [r.gauss(0, 1) for _ in range(nn)]; b = [r.gauss(0, 1) for _ in range(nn)]
            ma, mb = sum(a) / nn, sum(b) / nn
            cov = sum((u - ma) * (w - mb) for u, w in zip(a, b))
            rs.append(cov / math.sqrt(sum((u - ma) ** 2 for u in a) * sum((w - mb) ** 2 for w in b)))
        data, bw = hist(rs, -1, 1, 20)
        q = pg.axes(sx + 20 + k * 290, sy + 30, 240, 110, (-1, 1), (0, max(c for _, c in data) * 1.15), "r", "",
                    label_size=11, xticks=(-1, -0.5, 0, 0.5, 1), arrow=False)
        q.bars(data, bw, fill=LIGHTBLUE if k else LIGHTRED, stroke=BLUE if k else RED)
        big = sum(1 for v in rs if abs(v) > 0.5) / 300
        pg.html(f"<b>n = {nn}</b>: |r| &gt; 0.5 in <b>{big:.0%}</b> of samples", sx + 20 + k * 290, sy + 160, 260,
                18, 11)
        if k == 0: big10 = big
    pg.ask("Is a strong correlation in 10 rows real?",
           f"Often not: pure noise gives |r| &gt; 0.5 in {big10:.0%} of 10-row samples; with 100 rows almost never",
           x0 + 620, y0 + 470, sx + 100, sy + 90, w=290)
    pg.bullets(x0 + 620, y0 + 570, 300, ["Fewer, stronger features; regularization", "Cross-validate, report intervals",
                                         "Simple models first; collect more rows"], head="Small-data fixes:", size=10)


# ---------------------------------------------------------------------- ④ missing data, outliers, skew
def panel_values(pg):
    x0, y0 = 1380, 1040
    pg.panel(x0, y0, 1020, 700, 4, "Missing Data, Outliers & Skew", "Bad values distort statistics and fits")
    NaN = '<font color="#FF0000"><b>NaN</b></font>'
    specs = [("MCAR — completely at random", [["25", "30k"], ["41", NaN], ["33", "52k"], ["58", "71k"], ["29", NaN]],
              "Sensor glitch / random loss → drop rows or simple impute"),
             ("MAR — depends on other columns", [["22", NaN], ["24", NaN], ["45", "60k"], ["51", "75k"], ["23", NaN]],
              "Young people skip income → impute using age (kNN / model)"),
             ("MNAR — depends on the value itself", [["40", "35k"], ["47", NaN], ["39", "42k"], ["52", NaN],
                                                      ["44", "38k"]],
              "High earners hide income → add is_missing flag, model it")]
    for k, (title, rows, fix) in enumerate(specs):
        tx = x0 + 25 + k * 330
        pg.text(title, tx, y0 + 70, 310, 18, 11, bold=True)
        pg.table(tx + 70, y0 + 92, 160, [["age", "income"]] + rows, size=10, h=120)
        pg.text(fix, tx, y0 + 216, 310, 30, 10, color="#555555", align="center")
    pg.ask("Can I just drop rows with missing values?",
           "Only if MCAR. Under MAR / MNAR dropping biases the data (here: young or rich people vanish)",
           x0 + 680, y0 + 255, x0 + 520, y0 + 205, w=320)
    # outlier pulling the regression line
    r = seeded(9)
    pts = [(xv, 1 + 0.8 * xv + r.gauss(0, 0.6)) for xv in [0.5 + 0.65 * i for i in range(14)]]
    out = (9.5, -3.0)
    b_clean = ols([(a,) for a, _ in pts], [b for _, b in pts])
    b_dirty = ols([(a,) for a, _ in pts + [out]], [b for _, b in pts + [out]])
    p = pg.axes(x0 + 70, y0 + 350, 360, 200, (0, 10), (-4, 10), "x", "y", label_size=11, origin=(0, -4))
    p.scatter(pts, r=3.5)
    p.scatter([out], color=RED, r=5)
    p.curve(lambda v: b_clean[0] + b_clean[1] * v, 0, 10, n=2, curved=False, color=GREEN, width=2)
    p.curve(lambda v: b_dirty[0] + b_dirty[1] * v, 0, 10, n=2, curved=False, color=RED, width=2, dashed=True)
    pg.html(f'<font color="#009900">— without outlier: slope {b_clean[1]:.2f}</font><br>'
            f'<font color="#FF0000">- - with 1 outlier: slope {b_dirty[1]:.2f}</font>', x0 + 140, y0 + 296, 300, 36, 11)
    pg.ask("How much can ONE bad row change the model?",
           f"A lot: one typo moves the slope from {b_clean[1]:.2f} to {b_dirty[1]:.2f} — check it, cap it, or use robust "
           "(Huber) regression", x0 + 210, y0 + 575, *p.P(*out), w=280)
    # skewed feature
    vol = [math.exp(r.gauss(11, 0.9)) / 1e5 for _ in range(400)]
    mean_v, med_v = sum(vol) / len(vol), sorted(vol)[200]
    sk, sk_log = skewness(vol), skewness([math.log(v) for v in vol])
    data, bw = hist(vol, 0, 6, 24)
    q = pg.axes(x0 + 540, y0 + 330, 300, 170, (0, 6), (0, max(c for _, c in data) * 1.2), "volume (×100k)", "",
                label_size=10, arrow=False, xticks=(0, 2, 4, 6))
    q.bars(data, bw, fill=LIGHTBLUE, stroke=BLUE)
    q.vline(med_v, color=GREEN, width=1.5)
    q.vline(mean_v, color=RED, width=1.5)
    q.text(f"median {med_v:.2f}", med_v, data[0][1] * 1.1 if False else max(c for _, c in data) * 1.12, dx=-36,
           size=9, color=GREEN, w=80)
    q.text(f"mean {mean_v:.2f}", mean_v, max(c for _, c in data) * 0.95, dx=40, size=9, color=RED, w=80)
    pg.html(f"skewness = <b>{sk:.2f}</b> → after log: <b>{sk_log:.2f}</b>", x0 + 560, y0 + 525, 300, 18, 11)
    pg.ask("Is the mean a good 'typical' value here?",
           f"No: the long tail pulls the mean ({mean_v:.2f}) above the median ({med_v:.2f}); log-transform or use the median",
           x0 + 690, y0 + 560, *q.P(mean_v, max(c for _, c in data) * 0.6), w=300)
    pg.html('Fixes in detail → <a href="data:page/id,featureeng">Feature Engineering</a> (imputation, scaling, '
            'log / Box-Cox) · <a href="data:page/id,anomaly">Anomaly Detection</a> (finding outliers)',
            x0 + 30, y0 + 660, 960, 20, 11)


# ---------------------------------------------------------------------- ① curse of dimensionality
def panel_curse(pg):
    x0, y0 = 400, 240
    pg.panel(x0, y0, 940, 760, 1, "★ Curse of Dimensionality: what it does to a model",
             "Every extra column adds volume; with fixed rows the data becomes sparse and distances lose meaning")
    r = seeded(11)
    # kNN accuracy vs number of irrelevant columns
    def sample(n):
        rows = []
        for _ in range(n):
            c = 1 if r.random() < 0.5 else 0
            m = 0.8 if c else -0.8
            rows.append(([r.gauss(m, 1) for _ in range(2)], [r.gauss(0, 1) for _ in range(200)], c))
        return rows
    train, test = sample(80), sample(120)
    dvals = [0, 5, 10, 20, 50, 100, 150, 200]
    dist = [[sum((a - b) ** 2 for a, b in zip(t[0], s[0])) for s in train] for t in test]
    accs, prev = [], 0
    for d in dvals:
        for i, t in enumerate(test):
            row = dist[i]
            for j, s in enumerate(train):
                row[j] += sum((a - b) ** 2 for a, b in zip(t[1][prev:d], s[1][prev:d]))
        prev = d
        ok = 0
        for i, t in enumerate(test):
            nn = sorted(range(len(train)), key=lambda j: dist[i][j])[:5]
            vote = sum(train[j][2] for j in nn)
            ok += (1 if vote >= 3 else 0) == t[2]
        accs.append(ok / len(test))
    p = pg.axes(x0 + 70, y0 + 100, 400, 220, (0, 200), (0.4, 1.0), "d", "acc", label_size=12,
                xticks=(0, 50, 100, 150, 200), yticks=(0.5, 0.75, 1.0), origin=(0, 0.4))
    p.hline(0.5, color=RED)
    p.text("coin flip", 185, 0.5, dy=-10, size=10, color=RED, w=70)
    p.poly(list(zip(dvals, accs)), color=BLUE, width=2)
    p.scatter(list(zip(dvals, accs)), r=4)
    for d, a in zip(dvals, accs):
        if d in (0, 20, 50, 100, 200):
            p.text(f"{a:.2f}", d, a, dx=14 if d == 0 else 0, dy=-14, size=10, w=40)
    pg.text("5-NN · 2 informative features + d random columns · 80 training rows", x0 + 40, y0 + 335, 460, 18, 10,
            italic=True, color="#555555", align="center")
    acc100 = accs[dvals.index(100)]
    pg.ask("Do extra columns help a distance-based model?",
           f"No: accuracy falls from {accs[0]:.2f} to {acc100:.2f} with 100 random columns — drop or compress them",
           x0 + 260, y0 + 380, *p.P(100, acc100), w=230)

    # same 20 rows, 2 vs 50 columns
    rx = x0 + 530
    pg.text("Same 20 rows: distance from row 1 to the others", rx, y0 + 80, 390, 20, 12, bold=True)
    U = [[r.random() for _ in range(50)] for _ in range(20)]
    ratios = []
    for k, ncol in enumerate((2, 50)):
        ds = sorted(math.sqrt(sum((U[0][c] - U[i][c]) ** 2 for c in range(ncol))) for i in range(1, 20))
        ratios.append(ds[0] / ds[-1])
        q = pg.axes(rx + 20, y0 + 120 + k * 135, 360, 90, (0, 20), (0, ds[-1] * 1.1), "", "", arrow=False)
        q.bars([(i + 1, v) for i, v in enumerate(ds)], 0.7, fill=LIGHTBLUE if k == 0 else LIGHTRED,
               stroke=BLUE if k == 0 else RED)
        pg.html(f"<b>{ncol} columns</b>: nearest / farthest = <b>{ds[0] / ds[-1]:.2f}</b>", rx + 20,
                y0 + 214 + k * 135, 360, 18, 11)
    pg.ask("Is the 'nearest' row still special?",
           f"In 2-D yes (ratio {ratios[0]:.2f}); in 50-D every row is almost equally far (ratio {ratios[1]:.2f})",
           rx + 140, y0 + 420, rx + 260, y0 + 330, w=240)
    hdr = ["row", "c1", "c2", "c3", "…", "c50"]
    rows = [hdr] + [[i + 1, f"{U[i][0]:.2f}", f"{U[i][1]:.2f}", f"{U[i][2]:.2f}", "…", f"{U[i][49]:.2f}"]
                    for i in range(3)] + [["…", "…", "…", "…", "…", "…"]]
    pg.table(rx + 20, y0 + 500, 300, rows, size=10, h=110)
    pg.text("2 columns = only c1, c2 · 50 columns = c1 … c50 (uniform random)", rx + 10, y0 + 615, 380, 30, 10,
            italic=True, color="#555555")

    # density table
    pg.text("Rows needed to keep 10 points per axis", x0 + 30, y0 + 470, 300, 20, 12, bold=True)
    pg.table(x0 + 30, y0 + 495, 220, [["d (columns)", "rows ≈ 10^d"], ["1", "10"], ["2", "100"], ["3", "1,000"],
                                      ["5", "100,000"], ["10", "10,000,000,000"]], size=11, h=150, highlight_rows=(5,))
    pg.bullets(x0 + 280, y0 + 470, 250, [
        '<a href="data:page/id,featureeng">Feature selection</a> (drop weak columns)',
        '<a href="data:page/id,dimred">PCA / embeddings</a> (compress)',
        "Regularization (L1 zeroes columns)",
        "More rows · tree models (less distance-based)"], head="Remedies:", size=11)
    pg.text("Geometry of high dimensions (volume, distance concentration) → PCA page ⑦", x0 + 30, y0 + 700, 880, 18,
            10, italic=True, color="#3333FF", link=LINK("dimred"))


# ---------------------------------------------------------------------- ② class imbalance
def panel_imbalance(pg):
    x0, y0 = 1380, 240
    pg.panel(x0, y0, 1020, 760, 2, "Class Imbalance: the rare class is the one that matters",
             "Fraud, crashes, defaults, big moves — usually 1–5 % of rows")
    r = seeded(23)
    # class bar + accuracy paradox
    pg.rect(x0 + 30, y0 + 80, 294, 26, "class 0: 980 (98 %)", fill=LIGHTBLUE, stroke=BLUE, size=10)
    pg.rect(x0 + 324, y0 + 80, 6, 26, fill=RED, stroke=RED)
    pg.text("class 1: 20 (2 %)", x0 + 236, y0 + 108, 120, 16, 10, color=RED, align="right")
    pg.text("'Always predict 0' model:", x0 + 30, y0 + 130, 300, 18, 12, bold=True)
    pg.table(x0 + 30, y0 + 152, 300, [["", "pred 0", "pred 1"], ["true 0", "980 (TN)", "0 (FP)"],
                                      ["true 1", '<font color="#FF0000">20 (FN)</font>', "0 (TP)"]], size=11, h=70)
    pg.html(f'Accuracy = <b>98 %</b> · <span style="background:{HL}">Recall = 0 %</span> · F1 = 0',
            x0 + 30, y0 + 228, 300, 18, 11)
    pg.ask("Is 98 % accuracy good here?",
           "No: the model never finds a single rare case (recall 0) — judge with recall, F1 and PR-AUC",
           x0 + 30, y0 + 256, x0 + 180, y0 + 236, w=300)

    # SMOTE + boundary before / after (computed logistic regression)
    maj = [(r.gauss(0, 1.0), r.gauss(0, 1.0)) for _ in range(150)]
    mino = [(r.gauss(1.3, 0.7), r.gauss(1.3, 0.7)) for _ in range(12)]
    synth, segs = [], []
    for i, a in enumerate(mino):
        nbrs = sorted((m for m in mino if m is not a), key=lambda b: (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)[:3]
        while len(synth) < (i + 1) * 12:
            b = nbrs[r.randrange(3)]
            lam = r.random()
            synth.append((a[0] + lam * (b[0] - a[0]), a[1] + lam * (b[1] - a[1])))
            if len(segs) < 10 and r.random() < 0.15: segs.append((a, b))
    w_before = logreg(maj + mino, [0] * len(maj) + [1] * len(mino))
    w_after = logreg(maj + mino + synth, [0] * len(maj) + [1] * (len(mino) + len(synth)))
    rec = lambda w: sum(1 for a, b in mino if w[0] + w[1] * a + w[2] * b > 0) / len(mino)
    rb, ra = rec(w_before), rec(w_after)
    xr, yr = (-3, 4), (-3, 4)
    px, py = x0 + 380, y0 + 90
    p = pg.axes(px, py, 300, 250, xr, yr, "x_1", "x_2", label_size=11)
    p.scatter([m for m in maj if xr[0] < m[0] < xr[1] and yr[0] < m[1] < yr[1]], color=BLUE, r=2.2)
    for a, b in segs:
        p.poly([a, b], color=ORANGE, width=0.75)
    p.scatter(synth[::3], color=ORANGE, r=2.2)
    p.scatter(mino, color=RED, r=4)
    for w, col, dash, lab in [(w_before, GREY, True, "before"), (w_after, GREEN, False, "after SMOTE")]:
        seg = clip_line(w, xr, yr)
        if seg: p.poly(seg, color=col, dashed=dash, width=2)
    pg.html('<font color="#3333FF">● majority</font> <font color="#FF0000">● minority</font> '
            '<font color="#FF8000">● synthetic</font><br><font color="#808080">- - boundary before</font> '
            '<font color="#009900">— after SMOTE</font>', px, py + 268, 320, 32, 10)
    pg.math(r"x_{new} = x_i + \lambda\,(x_{nn} - x_i),\ \ \lambda \sim U(0,1)", x0 + 700, y0 + 95, 300, 40, 13)
    pg.text("SMOTE: new minority points on the segment between a minority point and one of its k nearest "
            "minority neighbours (orange lines)", x0 + 705, y0 + 140, 290, 46, 10)
    fa = lambda w: sum(1 for a, b in maj if w[0] + w[1] * a + w[2] * b > 0)
    pg.ask("Does balancing move the decision boundary?",
           f"Yes: it shifts toward the majority; minority recall {rb:.0%} → {ra:.0%}, "
           f"false alarms {fa(w_before)} → {fa(w_after)} of 150",
           x0 + 705, y0 + 200, *p.P(0.6, 0.6), w=290)

    # resampling strategies (counts)
    bx, by = x0 + 30, y0 + 430
    pg.text("Resampling the TRAINING set", bx, by - 6, 420, 20, 12, bold=True)
    groups = [("original", 980, 20, ""), ("under-sample", 20, 20, "drop majority"),
              ("over-sample", 980, 980, "duplicate minority"), ("SMOTE", 980, 980, "synthetic minority")]
    for k, (name, a, b, note) in enumerate(groups):
        gx = bx + 10 + k * 105
        for j, (v, col, st) in enumerate([(a, LIGHTBLUE, BLUE), (b, LIGHTRED if k < 3 else "#FFE6CC",
                                                                  RED if k < 3 else ORANGE)]):
            h = max(2, v / 980 * 100)
            pg.rect(gx + j * 34, by + 125 - h, 30, h, fill=col, stroke=st)
            pg.text(str(v), gx + j * 34 - 6, by + 125 - h - 15, 42, 14, 9, align="center")
        pg.text(name, gx - 10, by + 128, 90, 16, 10, bold=True, align="center")
        pg.text(note, gx - 10, by + 143, 90, 14, 9, color="#555555", align="center")
    pg.line([(bx, by + 125), (bx + 430, by + 125)])

    # class weights
    wx = x0 + 470
    pg.text("Class weights (cost-sensitive loss)", wx, by - 6, 280, 20, 12, bold=True)
    pg.math(r"L=-\frac{1}{n}\sum_i w_{y_i}\big[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)\big]", wx - 10, by + 20,
            290, 40, 12)
    pg.math(r"w_c=\frac{n}{K\,n_c}", wx, by + 66, 120, 40, 14)
    pg.html(f'w₁ = 1000/(2·20) = <b>25</b><br>w₀ = 1000/(2·980) = <b>0.51</b>', wx + 120, by + 68, 160, 40, 11)
    pg.text("One missed rare case costs as much as 49 false alarms → no resampling needed", wx, by + 116, 270, 30,
            10, italic=True, color="#555555")

    # threshold moving: PR curve
    pos = [1 / (1 + math.exp(-r.gauss(-0.5, 1.2))) for _ in range(100)]
    neg = [1 / (1 + math.exp(-r.gauss(-4.0, 1.2))) for _ in range(4900)]
    curve, best = [], None
    for k in range(1, 99):
        t = k / 100
        tp = sum(1 for s in pos if s >= t); fp = sum(1 for s in neg if s >= t)
        prec = tp / (tp + fp) if tp + fp else 1.0
        recl = tp / 100
        f1 = 2 * prec * recl / (prec + recl) if prec + recl else 0
        curve.append((recl, prec, t, f1))
        if best is None or f1 > best[3]: best = (recl, prec, t, f1)
    at05 = min(curve, key=lambda c: abs(c[2] - 0.5))
    tx = x0 + 770
    pg.text("Threshold moving (PR curve, 2 % positives)", tx - 10, by - 6, 240, 20, 11, bold=True)
    q = pg.axes(tx + 20, by + 24, 190, 110, (0, 1), (0, 1), "recall", "", label_size=10, xticks=(0.5, 1),
                yticks=(0.5, 1))
    pg.text("precision", tx - 30, by + 60, 50, 14, 9, color="#555555", align="right")
    q.poly([(c[0], c[1]) for c in curve], color=BLUE, width=1.8)
    q.scatter([(at05[0], at05[1])], color=GREY, r=4)
    q.scatter([(best[0], best[1])], color=RED, r=4.5)
    q.text("t = 0.5", at05[0], at05[1], dx=-26, dy=10, size=9, color=GREY, w=50)
    q.text(f"best F1, t = {best[2]:.2f}", best[0], best[1], dx=40, dy=-10, size=9, color=RED, w=100)
    pg.ask("Is 0.5 the right cut-off for a rare class?",
           f"No: at 0.5 recall is {at05[0]:.0%}; the best-F1 cut-off t = {best[2]:.2f} gives recall {best[0]:.0%}, "
           f"precision {best[1]:.0%}",
           x0 + 470, by + 150, *q.P(best[0], best[1]), w=290)

    # when to use + warning
    pg.table(x0 + 20, y0 + 650, 640, [
        ["Remedy", "Use when", "Watch out"],
        ["Class weights", "first choice; most models support it", "probabilities need recalibration"],
        ["Under-sampling", "huge majority, fast training", "throws data away"],
        ["Over-sampling / SMOTE", "few minority rows, kNN / linear models", "overfits copies; SMOTE blurs boundaries"],
        ["Threshold moving", "always, at decision time", "choose t on validation data"]], size=10, h=95)
    pg.rect(x0 + 680, y0 + 650, 320, 95,
            '<b><font color="#FF0000">⚠ Resample only inside the training fold.</font></b><br>Never resample before '
            'the split or the validation set gets synthetic copies → optimistic scores. Evaluate with recall, '
            'F1, PR-AUC on the real class ratio.', fill="#FFF5F5", stroke=RED, dashed=True, size=10, align="left",
            valign="top", raw=True)

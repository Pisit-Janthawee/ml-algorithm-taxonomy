"""Extra panels for the Logistic Regression page: data illustration + hyperparameter / threshold search."""
import math
from dio import *
from l_common import *


def sig(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def standardize(cols):
    out, stats = [], []
    for c in cols:
        m = sum(c) / len(c)
        s = math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) or 1
        out.append([(v - m) / s for v in c]); stats.append((m, s))
    return out, stats


def fit_logreg(X, y, lam=0.0, lr=0.3, iters=400):
    """X rows include intercept 1 in col 0; L2 penalty lam (= 1/C) on non-intercept weights."""
    p, n = len(X[0]), len(X)
    lr = min(lr, 1.0 / (1.0 + lam / n))
    w = [0.0] * p
    for _ in range(iters):
        g = [0.0] * p
        for r, t in zip(X, y):
            e = sig(sum(a * b for a, b in zip(r, w))) - t
            for j in range(p): g[j] += e * r[j]
        for j in range(p):
            w[j] -= lr * (g[j] / n + (lam * w[j] / n if j else 0))
    return w


def logloss(X, y, w):
    s = 0
    for r, t in zip(X, y):
        p = min(max(sig(sum(a * b for a, b in zip(r, w))), 1e-9), 1 - 1e-9)
        s -= t * math.log(p) + (1 - t) * math.log(1 - p)
    return s / len(y)


def build(pg: Page, cat):
    y0 = pg.max_y() + 60
    data_illustration(pg, 400, y0)
    tuning(pg, 1440, y0)


# ---------------------------------------------------------------------------------------- ⑥
def data_illustration(pg, x0, y0):
    pg.panel(x0, y0, 1000, 760, 6, "Data Illustration: rows → probabilities → classes",
             "Predict tomorrow's direction (Up = 1 / Down = 0) from EMA gap and RSI on 10 trading days")
    raw = [(1.20, 55, 1), (-0.80, 62, 0), (0.40, 38, 1), (-1.50, 45, 0), (1.80, 71, 1),
           (-0.30, 30, 1), (0.60, 68, 0), (-1.10, 58, 0), (0.90, 49, 1), (-0.20, 52, 0)]
    rows = [[d + 1, e, s, u] for d, (e, s, u) in enumerate(raw)]
    (e_s, r_s), stats = standardize([[row[1] for row in rows], [row[2] for row in rows]])
    X = [[1, a, b] for a, b in zip(e_s, r_s)]
    Y = [row[3] for row in rows]
    w = fit_logreg(X, Y, lam=0.2, lr=0.5, iters=800)
    zs = [sum(a * b for a, b in zip(rr, w)) for rr in X]
    ps = [sig(z) for z in zs]

    pg.text("Step 1 · Raw data", x0 + 20, y0 + 62, 300, 20, 12, bold=True)
    t = [["Day", "EMA gap %", "RSI", '<span style="background:#FFFF00">Up?</span>']]
    t += [[d, f"{e:+.2f}", s, f"<b>{u}</b>"] for d, e, s, u in rows]
    pg.table(x0 + 20, y0 + 86, 250, t, size=11, h=290)
    arrow_label(pg, x0 + 278, y0 + 230, x0 + 318, y0 + 230, "")
    pg.text("Step 2 · Standardize → linear score → sigmoid → threshold 0.5", x0 + 330, y0 + 62, 650, 20, 12,
            bold=True)
    o = [["Day", "\\(x_1'\\)", "\\(x_2'\\)", "\\(z = w^Tx\\)", "\\(\\hat{p}=\\sigma(z)\\)", "\\(\\hat{y}\\)", "actual",
          ""]]
    correct = 0
    for (d, _, _, u), a, b, z, p in zip(rows, e_s, r_s, zs, ps):
        yh = 1 if p >= 0.5 else 0
        ok = yh == u
        correct += ok
        o.append([d, f"{a:+.2f}", f"{b:+.2f}", f"{z:+.2f}", f"<b>{p:.2f}</b>", yh, u,
                  '<font color="#009900">✔</font>' if ok else '<font color="#FF0000">✘</font>'])
    pg.table(x0 + 330, y0 + 86, 420, o, size=11, h=290)
    pg.html(f'<b>Fitted weights</b> (standardized features):<br>'
            f'\\(w_0={w[0]:+.2f},\\ w_1={w[1]:+.2f},\\ w_2={w[2]:+.2f}\\)<br><br>'
            f'<span style="background:{HL}">Accuracy = {correct}/10</span><br><br>'
            f'<font color="#555555">EMA gap ↑ → P(Up) ↑ (w₁ &gt; 0)<br>RSI ↑ → P(Up) ↓ (w₂ &lt; 0, overbought)</font>',
            x0 + 765, y0 + 90, 220, 200, 11)
    # sigmoid with rows placed on it
    p = pg.axes(x0 + 80, y0 + 430, 520, 220, (-4, 4), (0, 1), "z", "\\hat{p}", label_size=12, xticks=(-4, -2, 0, 2, 4),
                yticks=(0.5, 1))
    p.curve(sig, -4, 4, n=60, width=2)
    p.hline(0.5, color=ORANGE)
    p.text("threshold 0.5", 4, 0.5, dx=40, size=10, color=ORANGE, w=80)
    for (d, _, _, u), z in zip(rows, zs):
        zz = max(-4, min(4, z))
        p.scatter([(zz, sig(zz))], color=BLUE if u else RED, r=5)
        p.text(str(d), zz, sig(zz), dy=-12, size=9)
    pg.html('<font color="#3333FF">● actual Up</font> &nbsp; <font color="#FF0000">● actual Down</font>', x0 + 90,
            y0 + 410, 220, 18, 11)
    pg.bullets(x0 + 640, y0 + 440, 340, [
        "Every row becomes one point on the sigmoid",
        "Right of z = 0 → predicted Up, left → Down",
        "Points of the wrong colour on a side are the errors",
        "Distance from 0.5 = confidence; use p̂ for position sizing, not only the class"], head="Reading the plot:")
    kc = max(range(len(zs)), key=lambda k: abs(zs[k]))
    zc = max(-4, min(4, zs[kc]))
    pg.ask("How sure is the model about each day?",
           f"distance of p̂ from 0.5: day {kc + 1} (p̂ = {ps[kc]:.2f}) is the most confident; points near the "
           "0.5 line are coin-flips → bet smaller",
           x0 + 640, y0 + 575, *p.P(zc, sig(zc)), w=340)
    pg.bullets(x0 + 20, y0 + 690, 960, [
        "Standardize features first (weights become comparable, gradient descent converges faster); keep the train-set mean/σ for new days"],
        size=11)


def tuning(pg, x0, y0):
    pg.panel(x0, y0, 960, 760, 7, "Finding the optimal hyperparameter & threshold",
             "Regularization strength C = 1/λ by validation, then the decision threshold on validation data")
    r = seeded(31)
    n, p_noise = 240, 12
    rows, ys = [], []
    for _ in range(n):
        f = [r.gauss(0, 1) for _ in range(2 + p_noise)]
        z = -0.9 + 1.5 * f[0] - 1.0 * f[1]
        ys.append(1 if r.random() < sig(z) else 0)
        rows.append([1] + f)
    Xt, yt, Xv, yv = rows[:40], ys[:40], rows[40:], ys[40:]
    logC = [-2.5 + 0.5 * i for i in range(12)]
    tr, va, ws = [], [], []
    for lc in logC:
        w = fit_logreg(Xt, yt, lam=1 / 10 ** lc, lr=0.5, iters=1500)
        ws.append(w)
        tr.append(logloss(Xt, yt, w)); va.append(logloss(Xv, yv, w))
    ib = va.index(min(va))
    pg.text("(a) Validation curve for C (40 train rows, 2 useful + 12 noise features)", x0 + 20, y0 + 62, 460, 20, 12,
            bold=True)
    top = max(max(tr), max(va)) * 1.1
    p = pg.axes(x0 + 70, y0 + 125, 360, 170, (-2.7, 3.2), (0, top), "\\log_{10}C", "log loss", label_size=11,
                xticks=(-2, -1, 0, 1, 2, 3))
    p.poly(list(zip(logC, tr)), color=BLUE, width=2)
    p.poly(list(zip(logC, va)), color=RED, width=2)
    p.scatter(list(zip(logC, tr)), color=BLUE, r=3)
    p.scatter(list(zip(logC, va)), color=RED, r=3)
    p.vline(logC[ib], 0, top, color=GREEN, width=1.5)
    p.text(f"best C = 10^{logC[ib]:g}", logC[ib], top, dy=-8, size=11, color=GREEN, bold=True, w=130)
    p.text("strong penalty<br>underfit", -2.0, top * 0.15, size=10, w=90, h=30, raw=True)
    p.text("weak penalty<br>overfit", 2.5, top * 0.4, size=10, w=90, h=30, raw=True)
    pg.html('<font color="#3333FF"><b>— train</b></font> &nbsp; <font color="#FF0000"><b>— validation</b></font>',
            x0 + 300, y0 + 86, 160, 18, 10)
    # coefficient shrinkage
    pg.text("Coefficients shrink as C decreases (L2)", x0 + 20, y0 + 330, 440, 18, 12, bold=True)
    q = pg.axes(x0 + 70, y0 + 380, 360, 110, (-2.7, 3.2), (-2.5, 2.5), "\\log_{10}C", "w_j", label_size=11,
                origin=(-2.7, 0))
    for j in range(1, 15):
        col = [GREEN, RED][j - 1] if j <= 2 else "#AAAAAA"
        q.poly([(lc, max(-2.5, min(2.5, w[j]))) for lc, w in zip(logC, ws)], color=col, width=2 if j <= 2 else 1)
    pg.html('<font color="#009900">— w₁ (useful)</font> <font color="#FF0000">— w₂ (useful)</font> '
            '<font color="#888888">— 12 noise</font>', x0 + 80, y0 + 500, 380, 18, 10)

    # threshold selection on validation set with best C
    w = ws[ib]
    pv = [sig(sum(a * b for a, b in zip(rr, w))) for rr in Xv]
    ths = [0.05 * i for i in range(1, 20)]
    P, R, F = [], [], []
    for t in ths:
        tp = sum(1 for p_, y_ in zip(pv, yv) if p_ >= t and y_ == 1)
        fp = sum(1 for p_, y_ in zip(pv, yv) if p_ >= t and y_ == 0)
        fn = sum(1 for p_, y_ in zip(pv, yv) if p_ < t and y_ == 1)
        pr = tp / (tp + fp) if tp + fp else 1.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        P.append(pr); R.append(rc); F.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    it = F.index(max(F))
    pos = sum(yv) / len(yv)
    pg.text(f"(b) Decision threshold on validation set ({pos:.0%} positives)", x0 + 500, y0 + 62, 440, 20, 12,
            bold=True)
    s = pg.axes(x0 + 550, y0 + 125, 360, 170, (0, 1), (0, 1.05), "threshold", "", label_size=11,
                xticks=(0, 0.25, 0.5, 0.75, 1), yticks=(0.5, 1))
    s.poly(list(zip(ths, P)), color=BLUE, width=2)
    s.poly(list(zip(ths, R)), color=RED, width=2)
    s.poly(list(zip(ths, F)), color=GREEN, width=2.5)
    s.vline(ths[it], 0, 1.05, color=BLACK, width=1)
    s.vline(0.5, 0, 1.05, color=ORANGE)
    s.scatter([(ths[it], F[it])], color=GREEN, r=5)
    s.text(f"max F1 = {F[it]:.2f}<br>at t = {ths[it]:.2f}", ths[it], F[it], dy=-40, dx=-60, size=10, color=GREEN,
           raw=True, w=100, h=30)
    s.text("default 0.5", 0.5, 1.05, dy=-8, size=9, color=ORANGE, w=70)
    pg.html('<font color="#3333FF"><b>— precision</b></font> <font color="#FF0000"><b>— recall</b></font> '
            '<font color="#009900"><b>— F1</b></font>', x0 + 700, y0 + 86, 220, 18, 10)
    pg.math("F_1=\\frac{2\\cdot P\\cdot R}{P+R}", x0 + 560, y0 + 320, 200, 44, 14)
    pg.math("t^*=\\arg\\min_t\\ \\left(c_{FP}\\cdot FP(t)+c_{FN}\\cdot FN(t)\\right)", x0 + 520, y0 + 370, 420, 40, 13)
    pg.text("business-cost threshold (e.g. a missed crash costs more than a false alarm)", x0 + 520, y0 + 410, 420,
            18, 10, align="center", color="#555555")
    pg.table(x0 + 560, y0 + 440, 360, [["Threshold", "Precision", "Recall", "F1"]] +
             [[f"{ths[k]:.2f}", f"{P[k]:.2f}", f"{R[k]:.2f}", f"{F[k]:.2f}"] for k in (3, 9, it, 15)],
             size=10, h=100, highlight_rows=(3,))
    pg.ask("How strong should the regularization be?",
           f"C = 10^{logC[ib]:g}: lowest validation log loss; a weaker penalty lets the 12 noise features overfit, "
           "a stronger one underfits",
           x0 + 20, y0 + 528, *p.P(logC[ib], va[ib]), w=500)
    pg.ask("Which probability cut-off should trigger a trade?",
           f"t = {ths[it]:.2f} maximises F1 ({F[it]:.2f}); with real costs choose the t that minimises "
           "c_FP·FP + c_FN·FN",
           x0 + 560, y0 + 548, x0 + 900, y0 + 512, w=380)
    pg.bullets(x0 + 20, y0 + 640, 920, [
        "Tune <b>C</b> (and penalty L1 / L2) with K-fold CV on a log grid; L1 drives noise-feature weights exactly to 0",
        "Tune the <b>threshold</b> after C, on validation data: max F1, a target recall / precision, or minimum expected cost",
        "<b>Class imbalance</b>: accuracy is misleading → use class_weight='balanced', PR-AUC / F1, stratified folds, or resampling (SMOTE)",
        "Time series: tune with walk-forward splits; the base rate of 'Up' days drifts, so re-check the threshold periodically"],
        head="How to choose:", size=11)

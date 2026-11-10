"""Extra panels for the Simple/Multiple Linear Regression page: data illustration + hyperparameter search."""
import math
from dio import *
from l_common import *


def build(pg: Page, cat):
    y0 = pg.max_y() + 60
    data_illustration(pg, 400, y0)
    tuning(pg, 1440, y0)


# ---------------------------------------------------------------------------------------- ⑥
def data_illustration(pg, x0, y0):
    pg.panel(x0, y0, 1000, 760, 6, "Data Illustration: from a table to predictions",
             "Multiple linear regression on 8 trading days (features → next-day return)")
    r = seeded(11)
    rows = []
    for d in range(8):
        ema = round(r.uniform(-2, 2), 2)
        rsi = round(r.uniform(25, 75))
        vol = round(r.uniform(0.5, 3.0), 2)
        y = round(0.10 + 0.45 * ema - 0.02 * (rsi - 50) + 0.10 * vol + r.gauss(0, 0.25), 2)
        rows.append((d + 1, ema, rsi, vol, y))
    X = [[1, e, s, v] for _, e, s, v, _ in rows]
    Y = [row[4] for row in rows]
    beta = fit_ls(X, Y)
    yh = predict(X, beta)
    ybar = sum(Y) / len(Y)
    r2 = 1 - sum((a - b) ** 2 for a, b in zip(Y, yh)) / sum((a - ybar) ** 2 for a in Y)

    # 1. raw table
    pg.text("Step 1 · Raw tabular data (n = 8, p = 3)", x0 + 20, y0 + 62, 400, 20, 12, bold=True)
    t = [["Day", "EMA gap %<br>\\(x_1\\)", "RSI<br>\\(x_2\\)", "Volatility %<br>\\(x_3\\)",
          '<span style="background:#FFFF00">Next return %<br>\\(y\\)</span>']]
    t += [[d, f"{e:+.2f}", s, f"{v:.2f}", f"<b>{yv:+.2f}</b>"] for d, e, s, v, yv in rows]
    pg.table(x0 + 20, y0 + 86, 400, t, size=11, h=250)
    # 2. design matrix
    arrow_label(pg, x0 + 430, y0 + 200, x0 + 480, y0 + 200, "", BLACK)
    pg.text("Step 2 · Design matrix X (add intercept column of 1s)", x0 + 490, y0 + 62, 480, 20, 12, bold=True)
    m = "\\\\".join(f"1 & {e:+.2f} & {s} & {v:.2f}" for _, e, s, v, _ in rows[:3])
    yv = "\\\\".join(f"{row[4]:+.2f}" for row in rows[:3])
    pg.math(f"X=\\begin{{bmatrix}}{m}\\\\ \\vdots & \\vdots & \\vdots & \\vdots\\end{{bmatrix}}_{{8\\times4}}"
            f"\\quad y=\\begin{{bmatrix}}{yv}\\\\ \\vdots\\end{{bmatrix}}", x0 + 490, y0 + 90, 490, 110, 14)
    pg.note("column of 1s → intercept β₀", x0 + 500, y0 + 205, x0 + 560, y0 + 160, w=170, color=RED)
    pg.text("Step 3 · Fit: Normal Equation (or Gradient Descent)", x0 + 490, y0 + 235, 480, 20, 12, bold=True)
    pg.math("\\hat{\\beta}=(X^{T}X)^{-1}X^{T}y", x0 + 490, y0 + 258, 250, 40, 16)
    pg.html(f'<span style="background:{HL}">β̂ = [{beta[0]:+.3f}, {beta[1]:+.3f}, {beta[2]:+.4f}, {beta[3]:+.3f}]</span>'
            f'<br><font color="#555555" style="font-size:10px">true generator: [0.10*, 0.45, −0.02, 0.10] '
            f'(*intercept absorbs −0.02·50 = +1.0 → 1.10)</font>', x0 + 745, y0 + 255, 245, 60, 11, valign="middle")
    # 3. output table
    pg.line([(x0 + 220, y0 + 345), (x0 + 220, y0 + 375)], arrow="block", width=2)
    pg.text("Step 4 · Output: prediction ŷ = Xβ̂ and residual e = y − ŷ", x0 + 20, y0 + 382, 460, 20, 12, bold=True)
    o = [["Day", "\\(y\\)", "\\(\\hat{y}\\)", "\\(e = y-\\hat{y}\\)", "\\(e^2\\)"]]
    for (d, *_), a, b in zip(rows, Y, yh):
        o.append([d, f"{a:+.2f}", f"{b:+.2f}", f'<font color="{RED if abs(a - b) > 0.25 else "#000000"}">'
                                                f"{a - b:+.2f}</font>", f"{(a - b) ** 2:.3f}"])
    o.append(["", "", "", "MSE", f"<b>{mse(Y, yh):.3f}</b>"])
    pg.table(x0 + 20, y0 + 406, 400, o, size=11, h=270, highlight_rows=(9,))
    # actual vs predicted
    lo, hi = min(Y + yh) - 0.3, max(Y + yh) + 0.3
    p = pg.axes(x0 + 540, y0 + 420, 250, 220, (lo, hi), (lo, hi), "y", "\\hat{y}", label_size=12)
    p.poly([(lo, lo), (hi, hi)], dashed=True, color=GREY)
    for a, b in zip(Y, yh):
        p.vline(a, a, b, color=RED, dashed=False)
    p.scatter(list(zip(Y, yh)), r=4)
    pg.text("Actual vs predicted (dashed = perfect fit)", x0 + 500, y0 + 655, 330, 18, 11, align="center")
    pg.html(f'<b>R² = {r2:.3f}</b>', x0 + 830, y0 + 520, 150, 24, 14)
    pg.text("red segment = residual", x0 + 830, y0 + 548, 150, 18, 10, color=RED)
    pg.ask("Does the fitted model reproduce the table?",
           f"R² = {r2:.3f}: the 3 features explain {r2 * 100:.0f} % of the variation in next-day return",
           x0 + 825, y0 + 570, x0 + 870, y0 + 545, w=165)
    pg.bullets(x0 + 20, y0 + 690, 960, [
        "Each <b>row</b> = one observation (time step), each <b>column</b> = one feature; the model learns one weight per column",
        "In practice: scale features, split train / validation by time (no shuffling for time series) and evaluate on unseen days"],
        size=11)


# ---------------------------------------------------------------------------------------- ⑦
def tuning(pg, x0, y0):
    pg.panel(x0, y0, 960, 760, 7, "Finding the optimal hyperparameter",
             "Model complexity (polynomial degree) and regularization strength α by validation / K-fold CV")
    r = seeded(23)
    xs = [r.uniform(-1, 1) for _ in range(40)]
    ys = [math.sin(2.5 * x) + r.gauss(0, 0.3) for x in xs]
    xt, yt, xv, yv = xs[:20], ys[:20], xs[20:], ys[20:]
    degs = list(range(1, 11))
    tr, va, fits = [], [], {}
    for d in degs:
        b = fit_ls(polyfeat(xt, d), yt, alpha=1e-9)
        fits[d] = b
        tr.append(mse(yt, predict(polyfeat(xt, d), b)))
        va.append(mse(yv, predict(polyfeat(xv, d), b)))
    best = degs[va.index(min(va))]
    cap = 0.6
    pg.text("(a) Validation curve: error vs polynomial degree", x0 + 20, y0 + 62, 440, 20, 12, bold=True)
    p = pg.axes(x0 + 70, y0 + 125, 360, 160, (0.5, 10.5), (0, cap), "degree", "MSE", label_size=11,
                xticks=degs, yticks=(0.2, 0.4))
    p.poly([(d, min(v, cap)) for d, v in zip(degs, tr)], color=BLUE, width=2)
    p.poly([(d, min(v, cap)) for d, v in zip(degs, va)], color=RED, width=2)
    p.scatter([(d, min(v, cap)) for d, v in zip(degs, tr)], color=BLUE, r=3)
    p.scatter([(d, min(v, cap)) for d, v in zip(degs, va)], color=RED, r=3)
    p.vline(best, 0, cap, color=GREEN, width=1.5)
    p.text(f"best degree = {best}", best, cap, dy=-8, size=11, color=GREEN, bold=True, w=130)
    p.text("underfit<br>(high bias)", 1.6, cap * 0.8, size=10, w=80, h=30, raw=True)
    p.text("overfit<br>(high variance)", 9.3, cap * 0.8, size=10, w=90, h=30, raw=True)
    pg.html('<font color="#3333FF"><b>— train</b></font> &nbsp; <font color="#FF0000"><b>— validation</b></font>',
            x0 + 300, y0 + 86, 160, 18, 10)
    for k, d in enumerate([1, best, 10]):
        q = pg.axes(x0 + 40 + k * 150, y0 + 330, 120, 90, (-1, 1), (-2, 2), "", "", arrow=False)
        q.scatter(list(zip(xt, yt)), r=2)
        q.curve(lambda v, b=fits[d]: sum(c * v ** i for i, c in enumerate(b)), -1, 1, n=60, color=RED, width=1.5)
        lab = ["degree 1: underfit", f"degree {best}: just right", "degree 10: overfit"][k]
        pg.text(lab, x0 + 30 + k * 150, y0 + 425, 140, 16, 10, bold=True, align="center",
                color=GREEN if k == 1 else BLACK)

    pg.ask("Which polynomial degree generalises best?",
           f"degree {best}: lowest validation error ({va[best - 1]:.2f}); higher degrees keep lowering train error "
           "but fit the noise",
           x0 + 30, y0 + 450, *p.P(best, min(va[best - 1], cap)), w=440)

    # ridge alpha by K-fold CV
    pg.text("(b) Ridge α by 5-fold CV (degree-10 features)", x0 + 500, y0 + 62, 440, 20, 12, bold=True)
    logs = [-6 + 0.5 * i for i in range(17)]
    folds = kfold(40, 5)
    means, ses = [], []
    for la in logs:
        errs = []
        for f in folds:
            tri = [i for i in range(40) if i not in f]
            b = fit_ls(polyfeat([xs[i] for i in tri], 10), [ys[i] for i in tri], alpha=10 ** la)
            errs.append(mse([ys[i] for i in f], predict(polyfeat([xs[i] for i in f], 10), b)))
        m = sum(errs) / 5
        sd = math.sqrt(sum((e - m) ** 2 for e in errs) / 4)
        means.append(m); ses.append(sd / math.sqrt(5))
    imin = means.index(min(means))
    thr = means[imin] + ses[imin]
    i1 = max(i for i in range(len(logs)) if means[i] <= thr)
    top = min(max(m + s for m, s in zip(means, ses)) * 1.1, 1.0)
    q = pg.axes(x0 + 550, y0 + 125, 370, 160, (-6.3, 2.3), (0, top), "\\log_{10}\\alpha", "CV MSE", label_size=11,
                xticks=(-6, -4, -2, 0, 2))
    for la, m, s in zip(logs, means, ses):
        q.poly([(la, min(m - s, top)), (la, min(m + s, top))], color=GREY)
    q.poly([(la, min(m, top)) for la, m in zip(logs, means)], color=BLUE, width=2)
    q.scatter([(la, min(m, top)) for la, m in zip(logs, means)], r=3)
    q.hline(thr, color=ORANGE)
    q.scatter([(logs[imin], means[imin])], color=RED, r=5)
    q.scatter([(logs[i1], means[i1])], color=GREEN, r=5)
    q.text(f"min CV error<br>α = 1e{logs[imin]:g}", logs[imin], means[imin], dy=30, size=10, color=RED, raw=True,
           w=90, h=30)
    q.text(f"1-SE rule<br>α = 1e{logs[i1]:g}", logs[i1], means[i1], dy=-30, size=10, color=GREEN, raw=True, w=90,
           h=30)
    q.text("min + 1 SE", 2.3, thr, dx=-30, dy=-9, size=9, color=ORANGE, w=70)
    pg.text("← overfit (α small)           underfit (α large) →", x0 + 550, y0 + 300, 370, 16, 10, align="center",
            color="#555555")
    # k-fold diagram
    pg.text("K-fold cross-validation (K = 5)", x0 + 520, y0 + 330, 400, 20, 12, bold=True)
    for i in range(5):
        pg.text(f"Fold {i + 1}", x0 + 520, y0 + 356 + i * 24, 50, 20, 10)
        for j in range(5):
            val = i == j
            pg.rect(x0 + 575 + j * 58, y0 + 358 + i * 24, 56, 18, "valid" if val else "train",
                    fill="#FFE6CC" if val else LIGHTBLUE, stroke=ORANGE if val else BLUE, size=9)
        pg.text(f"→ score{i + 1}", x0 + 870, y0 + 356 + i * 24, 70, 20, 10)
    pg.math("CV(\\alpha)=\\frac{1}{K}\\sum_{k=1}^{K}MSE_k(\\alpha)", x0 + 560, y0 + 480, 330, 44, 14)
    pg.bullets(x0 + 20, y0 + 560, 920, [
        f"Validation curve: pick the complexity where <b>validation</b> error is lowest (degree {best}); training error always keeps falling",
        "Grid of α on a <b>log scale</b>; choose the α with min CV error, or the <b>1-SE rule</b> (simplest model within one standard error of the min)",
        "Lasso: same procedure — larger α → more coefficients exactly 0 (feature selection); Elastic Net: grid over α × l1_ratio",
        "Time series: use <b>walk-forward / TimeSeriesSplit</b> instead of shuffled K-fold; refit on all training data with the chosen value",
        "Final check on an untouched <b>test set</b> — never tune on it"], head="How to choose:", size=11)

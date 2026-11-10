"""Overfitting & Underfitting page (base: Bias–Variance Trade-off): how the data table causes it, and fixes."""
import math
from dio import *

RED_BG = "#FFCCCC"
GRN = "#009900"


# ---------------------------------------------------------------- tiny numeric helpers (pure Python)
def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[c][c] != 0:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def polyfit(xs, ys, deg, lam=0.0):
    X = [[x ** d for d in range(deg + 1)] for x in xs]
    p = deg + 1
    A = [[sum(r[i] * r[j] for r in X) + (lam if (i == j and i > 0) else 0) for j in range(p)] for i in range(p)]
    b = [sum(r[i] * y for r, y in zip(X, ys)) for i in range(p)]
    return solve(A, b)


def peval(c, x):
    return sum(ci * x ** i for i, ci in enumerate(c))


def mse(a, b):
    return sum((u - v) ** 2 for u, v in zip(a, b)) / len(a)


def red(s):
    return f'<span style="background:{RED_BG}">{s}</span>'


def mini(pg, x, y, w, title, rows, why, fix, h_table=None):
    """Mini table + red 'why' + green 'fix'."""
    pg.text(title, x, y, w, 20, 12, bold=True)
    ht = pg.table(x, y + 24, w, rows, size=10, h=h_table)
    pg.html(f'<font color="#FF0000"><b>✘ </b>{why}</font>', x, y + 30 + ht, w, 34, 10)
    pg.html(f'<font color="#009900"><b>✔ Fix: </b></font>{fix}', x, y + 64 + ht, w, 30, 10)


# ---------------------------------------------------------------- data for panel 1
def dataset():
    r = seeded(11)
    xs = [round(0.5 + i * 0.85, 2) for i in range(12)]
    ys = [round(5 + 3 * math.sin(x / 1.5) + r.gauss(0, 0.5), 2) for x in xs]
    test = {2, 5, 8, 10}
    return xs, ys, test


def build(pg: Page, cat):
    xs, ys, test = dataset()
    tr = [i for i in range(12) if i not in test]
    txs, tys = [xs[i] for i in tr], [ys[i] for i in tr]
    lin = polyfit(txs, tys, 1)
    cub = polyfit(txs, tys, 3)

    def nn(x):
        j = min(range(len(txs)), key=lambda k: abs(txs[k] - x))
        return tys[j]

    models = [("Underfit: straight line", lambda x: peval(lin, x), BLUE),
              ("Good: cubic", lambda x: peval(cub, x), GRN),
              ("Overfit: 1-NN (memorise)", nn, RED)]
    preds = [[f(x) for x in xs] for _, f, _ in models]
    errs = []
    for p_ in preds:
        errs.append((mse([p_[i] for i in tr], tys), mse([p_[i] for i in test], [ys[i] for i in test])))

    # ------------------------------------------------------------ ① one table, three models
    x0, y0 = 400, 240
    pg.panel(x0, y0, 2000, 680, 1, "One data table, three models: underfit · good · overfit",
             "12 houses: size x (100 m²) → price y (M); 8 rows train the model, 4 rows test it")
    rows = [["#", "size x", "price y", "split", "ŷ underfit", "ŷ good", "ŷ overfit"]]
    for i in range(12):
        is_t = i in test
        sp = '<font color="#FF0000"><b>test</b></font>' if is_t else "train"
        cells = [f"{p_[i]:.2f}" for p_ in preds]
        rows.append([i + 1, f"{xs[i]:.2f}", f"{ys[i]:.2f}", sp] + cells)
    rows.append(["", "", "", "<b>Train MSE</b>"] + [f"<b>{e[0]:.2f}</b>" for e in errs])
    rows.append(["", "", "", "<b>Test MSE</b>"] + [f"<b>{e[1]:.2f}</b>" for e in errs])
    ht = pg.table(x0 + 20, y0 + 70, 640, rows, size=11, h=15 * 21 + 4, highlight_rows=(14,))
    pg.html('<b>Read it:</b> underfit → <font color="#3333FF">both errors high</font> · good → both low, small gap · '
            'overfit → <font color="#FF0000">train MSE = 0 but test MSE high</font> (memorised the 8 rows)',
            x0 + 20, y0 + 80 + ht, 640, 40, 11)
    for k, ((name, f, col), (etr, ete)) in enumerate(zip(models, errs)):
        px = x0 + 720 + k * 430
        pg.text(name, px - 10, y0 + 70, 380, 22, 13, bold=True, color=col, align="center")
        p = pg.axes(px + 20, y0 + 120, 330, 230, (0, 10.5), (0, 9.5), "x", "y", label_size=12)
        if k == 2:   # step function
            order = sorted(range(len(txs)), key=lambda j: txs[j])
            pts, prev = [], 0.0
            for a, j in enumerate(order):
                right = (txs[j] + txs[order[a + 1]]) / 2 if a + 1 < len(order) else 10.5
                pts += [(prev, tys[j]), (right, tys[j])]
                prev = right
            p.poly(pts, color=col, width=2)
        else:
            p.curve(f, 0, 10.5, n=60, color=col, width=2, curved=(k == 1))
        p.scatter([(xs[i], ys[i]) for i in tr], color=BLUE, r=4)
        for i in test:
            p.vline(xs[i], min(ys[i], f(xs[i])), max(ys[i], f(xs[i])), color=RED, dashed=True)
            pg.ellipse(*[c - 5 for c in p.P(xs[i], ys[i])], 10, 10, stroke=RED, sw=2)
        pg.html(f"train MSE = <b>{etr:.2f}</b> &nbsp; test MSE = <b>{ete:.2f}</b>", px, y0 + 375, 360, 22, 12,
                align="center")
        diag = ["High bias: misses the curve", "Balanced: captures the shape", "High variance: fits every bump"][k]
        pg.text(diag, px, y0 + 398, 360, 20, 11, italic=True, align="center", color=col)
    pg.html('<span style="color:#3333FF">●</span> train row &nbsp; <span style="color:#FF0000">○</span> test row '
            '(red dashed = test error)', x0 + 720, y0 + 430, 600, 20, 11)
    pg.math(r"E\big[(y-\hat{f}(x))^2\big] = \underbrace{\text{Bias}^2}_{\text{underfit}} + "
            r"\underbrace{\text{Variance}}_{\text{overfit}} + \underbrace{\sigma^2}_{\text{noise}}",
            x0 + 760, y0 + 470, 760, 70, 20)
    pg.bullets(x0 + 720, y0 + 560, 1260, [
        "<b>Underfitting</b> (high bias): the model is too simple for the pattern in the table → wrong on train and test alike",
        "<b>Overfitting</b> (high variance): the model learns row-specific noise → perfect on train, poor on new rows",
        "The test rows are the only honest judge — never tune on them (see Model Evaluation page)"], size=11)

    # ------------------------------------------------------------ ② how the table causes overfitting
    x0, y0 = 400, 960
    pg.panel(x0, y0, 1200, 680, 2, "How the TABLE causes overfitting",
             "Six data problems that let a model memorise instead of learn")
    cw, chh = 370, 310
    r = seeded(5)
    # (a) p >> n
    hdr = [f"x{j}" for j in range(1, 11)] + ["y"]
    rows_a = [hdr] + [[r.randint(1, 9) for _ in range(10)] + [r.choice([0, 1])] for _ in range(6)]
    mini(pg, x0 + 20, y0 + 70, cw, "(a) More columns than rows (p ≫ n)", rows_a,
         "10 features, 6 rows → a model can fit even random labels exactly",
         "fewer features, L1/L2 regularization, collect more rows")
    # (b) ID column
    rows_b = [["ID", "RSI", "EMA gap", "up?"]] + [[red(1000 + i), v, g, u] for i, (v, g, u) in
                                                  enumerate([(55, 0.4, 1), (31, -0.8, 0), (62, 0.9, 1), (48, -0.1, 0),
                                                             (70, 1.2, 1)])]
    mini(pg, x0 + 20 + (cw + 20), y0 + 70, cw, "(b) ID / row-number column", rows_b,
         "unique per row → a tree splits on ID and memorises; new IDs mean nothing",
         "drop IDs, timestamps-as-numbers, file/row indexes")
    # (c) leaky column
    rows_c = [["date", "RSI", red("ret_next"), "y = sign"]] + [
        [d, v, red(f"{rr:+.1f}%"), 1 if rr > 0 else 0] for d, v, rr in
        [("Mon", 55, 0.8), ("Tue", 61, -0.4), ("Wed", 47, 1.1), ("Thu", 52, -1.3), ("Fri", 66, 0.2)]]
    mini(pg, x0 + 20 + 2 * (cw + 20), y0 + 70, cw, "(c) Leaky column (future / target copy)", rows_c,
         "ret_next is known only AFTER the label time → 100 % validation accuracy, fails live",
         "keep only information available at time t; shift features by 1")
    # (d) noisy labels / outlier
    rows_d = [["size", "rooms", "price"]] + [[s, rm, p] for s, rm, p in [(1.2, 2, 3.1), (1.5, 3, 3.6), (2.0, 3, 4.2)]] + \
             [[red(2.1), red(3), red("41.0")], [2.6, 4, 5.3]]
    mini(pg, x0 + 20, y0 + 70 + chh, cw, "(d) Noisy labels & outliers", rows_d,
         "a typo (41.0 instead of 4.1) — a flexible model bends to fit this one row",
         "clean / cap outliers, robust loss (MAE, Huber), min_samples_leaf")
    # (e) duplicates across split
    rows_e = [["RSI", "EMA gap", "up?", "split"]] + [[55, 0.4, 1, "train"], [31, -0.8, 0, "train"],
                                                      [red(62), red(0.9), red(1), "train"], [48, -0.1, 0, "train"],
                                                      [red(62), red(0.9), red(1), '<font color="#FF0000">test</font>']]
    mini(pg, x0 + 20 + (cw + 20), y0 + 70 + chh, cw, "(e) Duplicate rows in train AND test", rows_e,
         "the test row was already seen → test score is inflated, real error is higher",
         "deduplicate before splitting; group / time-based splits")
    # (f) rare categories
    rows_f = [["sector", "count", "up-rate"]] + [["Tech", 120, "0.54"], ["Bank", 95, "0.49"], [red("Mining"), red(1), red("1.00")],
                                                 [red("Shipping"), red(1), red("0.00")], [red("Gaming"), red(2), red("1.00")]]
    mini(pg, x0 + 20 + 2 * (cw + 20), y0 + 70 + chh, cw, "(f) Rare category levels", rows_f,
         "levels with 1–2 rows get 'perfect' rates → pure memorisation",
         "group rare levels into 'Other', smoothed target encoding")

    panel3(pg)
    panel4(pg)
    panel5(pg, [xs[i] for i in tr], [ys[i] for i in tr])
    panel6(pg)


# ---------------------------------------------------------------- ③ underfitting causes
def panel3(pg):
    x0, y0 = 1640, 960
    pg.panel(x0, y0, 760, 680, 3, "How the TABLE causes underfitting",
             "The signal is missing or hidden from the model")
    cw, chh = 350, 310
    rows_a = [["date", "RSI", "next ret"]] + [[d, v, f"{rr:+.1f}%"] for d, v, rr in
                                              [("Mon", 52, 2.1), ("Tue", 51, -1.8), ("Wed", 53, 1.9), ("Thu", 50, -2.2),
                                               ("Fri", 52, 2.0)]] + [[red("news?"), red("—"), red("missing")]]
    mini(pg, x0 + 20, y0 + 70, cw, "(a) Key feature not in the table", rows_a,
         "returns are driven by news; RSI barely moves → any model predicts ~average",
         "add the driving feature (news sentiment, volume, macro data)")
    rows_b = [["timestamp", "sector", "y"]] + [[red(ts), red(c), y] for ts, c, y in
                                               [(1715839200, 1, 0.4), (1715842800, 2, 1.1), (1715846400, 3, 0.2),
                                                (1715850000, 1, 0.9)]]
    mini(pg, x0 + 20 + cw + 20, y0 + 70, cw, "(b) Wrong representation", rows_b,
         "raw Unix time hides the hour pattern; sector coded 1,2,3 fakes an order",
         "hour / weekday (sin, cos), one-hot sectors (see Feature Engineering)")
    rows_c = [["x₁", "x₂", "y", "x₁·x₂"]] + [[a, b, int(a * b > 0), a * b] for a, b in
                                            [(1, 1), (1, -1), (-1, 1), (-1, -1)]]
    mini(pg, x0 + 20, y0 + 70 + chh, cw, "(c) Interaction not represented (XOR)", rows_c,
         "a linear model on x₁, x₂ alone gets 50 % — the signal is in their product",
         "add x₁·x₂ (last column) or use trees / kernels / MLP")
    rows_d = [["RSI raw", "binned", "next ret"]] + [[v, red(b), r_] for v, b, r_ in
                                                    [(22, "low", "+1.8%"), (45, "low", "−0.2%"), (55, "high", "+0.1%"),
                                                     (81, "high", "−1.9%")]]
    mini(pg, x0 + 20 + cw + 20, y0 + 70 + chh, cw, "(d) Over-aggressive binning / cleaning", rows_d,
         "2 bins merge oversold (22) with neutral (45) → the extremes' signal is lost",
         "keep the raw value or finer bins; don't delete 'outliers' that carry signal")


# ---------------------------------------------------------------- 1-D regression tree (for ④)
def fit_tree(xs, ys, depth):
    n = len(xs)
    mean = sum(ys) / n
    if depth == 0 or n < 2:
        return mean
    pairs = sorted(zip(xs, ys))
    sx = [p[0] for p in pairs]
    sy = [p[1] for p in pairs]
    tot, tot2 = sum(sy), sum(v * v for v in sy)
    best, ls, ls2 = None, 0.0, 0.0
    for i in range(1, n):
        ls += sy[i - 1]
        ls2 += sy[i - 1] ** 2
        if sx[i] == sx[i - 1]:
            continue
        rs, rs2 = tot - ls, tot2 - ls2
        sse = (ls2 - ls * ls / i) + (rs2 - rs * rs / (n - i))
        if best is None or sse < best[0]:
            best = (sse, i)
    if best is None:
        return mean
    i = best[1]
    thr = (sx[i - 1] + sx[i]) / 2
    return (thr, fit_tree(sx[:i], sy[:i], depth - 1), fit_tree(sx[i:], sy[i:], depth - 1))


def tpred(t, x):
    while isinstance(t, tuple):
        t = t[1] if x < t[0] else t[2]
    return t


def sim(n, r):
    xs = [r.uniform(0, 10) for _ in range(n)]
    return xs, [math.sin(x / 1.5) * 3 + r.gauss(0, 0.8) for x in xs]


# ---------------------------------------------------------------- ④ diagnosis
def panel4(pg):
    x0, y0 = 400, 1700
    pg.panel(x0, y0, 980, 680, 4, "Diagnosis: complexity curve & learning curves",
             "Regression tree on simulated data (train n = 80, validation n = 400)")
    r = seeded(21)
    xtr, ytr = sim(80, r)
    xva, yva = sim(400, r)
    depths = list(range(1, 13))
    tr_e, va_e = [], []
    for d in depths:
        t = fit_tree(xtr, ytr, d)
        tr_e.append(mse([tpred(t, x) for x in xtr], ytr))
        va_e.append(mse([tpred(t, x) for x in xva], yva))
    best = depths[min(range(len(depths)), key=lambda k: va_e[k])]
    top = max(va_e + tr_e) * 1.1
    p = pg.axes(x0 + 70, y0 + 90, 480, 300, (0.5, 12.5), (0, top), "max\\_depth", "MSE", label_size=12,
                origin=(0.5, 0), xticks=depths, yticks=tuple(v for v in (0.5, 1, 1.5, 2, 2.5, 3) if v < top))
    # zones
    for a, b, col, lab in [(0.5, best - 0.5, LIGHTBLUE, "underfit"), (best - 0.5, best + 0.5, "#D5E8D4", "sweet spot"),
                           (best + 0.5, 12.5, LIGHTRED, "overfit")]:
        pa, pb = p.P(a, top), p.P(b, 0)
        pg.rect(pa[0], pa[1], pb[0] - pa[0], pb[1] - pa[1], fill=col, stroke="none", opacity=45)
        pg.text(lab, pa[0], pa[1] + 4, pb[0] - pa[0], 16, 10, bold=True, align="center")
    p.poly(list(zip(depths, tr_e)), color=BLUE, width=2)
    p.poly(list(zip(depths, va_e)), color=RED, width=2)
    p.scatter(list(zip(depths, tr_e)), color=BLUE, r=3)
    p.scatter(list(zip(depths, va_e)), color=RED, r=3)
    p.hline(0.64, color=GREY)
    pg.html('<font color="#3333FF">— train MSE</font> &nbsp; <font color="#FF0000">— validation MSE</font> '
            '&nbsp; <font color="#808080">- - noise σ² = 0.64</font>', x0 + 80, y0 + 415, 460, 20, 11)
    pg.html(f"best depth = <b>{best}</b>: validation MSE {va_e[best - 1]:.2f}; depth 12: train "
            f"{tr_e[-1]:.2f} vs validation {va_e[-1]:.2f}", x0 + 80, y0 + 437, 470, 34, 11)
    # learning curves
    ns = [10, 20, 40, 80, 160, 320]
    for k, (d, lab, col) in enumerate([(1, "depth 1 (underfit)", BLUE), (12, "depth 12 (overfit)", RED)]):
        trc, vac = [], []
        for n in ns:
            rr = seeded(100 + n)
            xa, ya = sim(n, rr)
            t = fit_tree(xa, ya, d)
            trc.append(mse([tpred(t, x) for x in xa], ya))
            vac.append(mse([tpred(t, x) for x in xva], yva))
        q = pg.axes(x0 + 640, y0 + 90 + k * 200, 290, 130, (0, 340), (0, 5), "n", "", label_size=11,
                    xticks=(10, 80, 160, 320), yticks=(2, 4))
        q.poly([(n, min(v, 5)) for n, v in zip(ns, trc)], color=BLUE, width=1.5)
        q.poly([(n, min(v, 5)) for n, v in zip(ns, vac)], color=RED, width=1.5)
        q.hline(0.64, color=GREY)
        pg.text(f"Learning curve (MSE vs rows n): {lab}", x0 + 600, y0 + 58 + k * 200, 360, 18, 11, bold=True,
                color=col)
    pg.text("underfit: curves meet HIGH → more rows won't help", x0 + 620, y0 + 240, 330, 16, 10, italic=True)
    pg.text("overfit: big gap that shrinks with more rows", x0 + 620, y0 + 440, 330, 16, 10, italic=True)
    pg.table(x0 + 20, y0 + 490, 940, [
        ["Train error", "Validation error", "Gap", "Diagnosis", "First thing to try"],
        ["high", "high", "small", '<font color="#3333FF"><b>Underfitting</b></font>', "richer features / more complex model"],
        ["low", "high", "large", '<font color="#FF0000"><b>Overfitting</b></font>', "simplify, regularize, more rows, drop leaky / ID columns"],
        ["low", "low", "small", '<font color="#009900"><b>Good fit</b></font>', "confirm on the untouched test set"],
        ["≈ 0", "≈ 0 (suspicious)", "none", "<b>Leakage</b>", "look for future / target-copy columns"]], size=11, h=130)
    pg.text("Train ≫ validation error (the reverse) usually means a bug or different distributions", x0 + 20,
            y0 + 630, 940, 18, 10, italic=True, color="#555555")


# ---------------------------------------------------------------- ⑤ remedies drawn
def ridge_path(xs, ys, deg=6, lams=None):
    cols = [[x ** d for x in xs] for d in range(1, deg + 1)]
    Z = []
    for c in cols:
        m = sum(c) / len(c)
        s = math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) or 1
        Z.append([(v - m) / s for v in c])
    ym = sum(ys) / len(ys)
    yc = [v - ym for v in ys]
    n = len(xs)
    out = []
    for lam in lams:
        A = [[sum(Z[i][k] * Z[j][k] for k in range(n)) + (lam if i == j else 0) for j in range(deg)] for i in range(deg)]
        b = [sum(Z[i][k] * yc[k] for k in range(n)) for i in range(deg)]
        out.append(solve(A, b))
    return out


def panel5(pg, txs, tys):
    x0, y0 = 1420, 1700
    pg.panel(x0, y0, 980, 700, 5, "Remedies, drawn", "Shrink, prune, stop early, drop out, add data")
    # ridge path
    logs = [k / 4 for k in range(-16, 13)]          # log10 λ from -4 to 3
    path = ridge_path(txs, tys, 6, [10 ** v for v in logs])
    lim = 8
    p = pg.axes(x0 + 60, y0 + 100, 380, 210, (-4, 3), (-lim, lim), "\\log_{10}\\lambda", "", label_size=11,
                origin=(-4, 0), xticks=(-4, -2, 0, 2))
    cols = [BLUE, RED, GREEN, ORANGE, "#9673A6", "#555555"]
    for j in range(6):
        p.poly([(lv, max(-lim, min(lim, c[j]))) for lv, c in zip(logs, path)], color=cols[j], width=1.5)
    pg.text("Ridge (L2) path: coefficients βⱼ of a degree-6 polynomial on the 8 train rows of ①", x0 + 30, y0 + 62, 440, 18, 11, bold=True)
    pg.text("small λ: wild coefficients (overfit) → large λ: all → 0 (underfit); pick λ by CV",
            x0 + 40, y0 + 330, 430, 30, 10, italic=True)
    pg.math(r"J = \text{MSE} + \lambda\sum_j \beta_j^2 \;\;(L2) \qquad \text{MSE} + \lambda\sum_j|\beta_j| \;\;(L1)",
            x0 + 20, y0 + 362, 460, 36, 13)
    # L1 vs L2 geometry
    for k, (lab, l1) in enumerate([("L2: circle → small weights", False), ("L1: diamond → some β = 0", True)]):
        cx, cy = x0 + 600 + k * 200, y0 + 200
        pg.line([(cx - 80, cy), (cx + 85, cy)], arrow="classic", width=0.75)
        pg.line([(cx, cy + 80), (cx, cy - 90)], arrow="classic", width=0.75)
        pg.math(r"\beta_1", cx + 80, cy - 4, 30, 20, 11)
        pg.math(r"\beta_2", cx - 10, cy - 110, 30, 20, 11)
        R = 45
        if l1:
            pg.line([(cx, cy - R), (cx + R, cy), (cx, cy + R), (cx - R, cy), (cx, cy - R)], color=BLUE, width=2)
        else:
            pg.ellipse(cx - R, cy - R, 2 * R, 2 * R, stroke=BLUE, sw=2)
        ex, ey = cx + 55, cy - 95
        for s in (22, 44, 66):
            pg.ellipse(ex - s * 1.4, ey - s * 0.8, 2.8 * s, 1.6 * s, stroke=RED, sw=0.75, dashed=True)
        pg.dot(ex, ey, r=3, color=RED)
        tx, ty = (cx, cy - R) if l1 else (cx + R * 0.45, cy - R * 0.89)
        pg.dot(tx, ty, r=5, color=GREEN)
        pg.text(lab, cx - 90, cy + 95, 190, 18, 10, bold=True, align="center")
    pg.text("loss contours (red) meet the penalty region (blue) at the solution (green)", x0 + 510, y0 + 320, 440, 18,
            10, italic=True, align="center")
    # pruning
    bx, by = x0 + 30, y0 + 420
    pg.text("Pruning a tree", bx, by, 300, 18, 12, bold=True)

    def node(x, y, lab, leaf=False, cut=False):
        return pg.rect(x, y, 56, 24, lab, fill=("#D5E8D4" if leaf else "#FFFFFF"), size=9,
                       stroke=RED if cut else BLACK, dashed=cut)
    a = node(bx + 95, by + 28, "x<4.6")
    b_, c_ = node(bx + 30, by + 74, "x<2.1"), node(bx + 160, by + 74, "x<7.9")
    leaves = [node(bx + 5 + i * 52, by + 124, l, True, cut=True) for i, l in enumerate(["7.5", "8.1", "3.0", "2.2", "4.6"])]
    for t in (b_, c_): pg.connect(a, t, orth=False, arrow=None)
    for i, t in enumerate(leaves): pg.connect(b_ if i < 2 else c_, t, orth=False, arrow=None, color=RED, dashed=True)
    pg.text("depth 3, leaves of 1 row → cut (red) → depth 2", bx, by + 158, 300, 30, 10, italic=True)
    # early stopping + dropout
    ex0 = x0 + 340
    pg.text("Early stopping", ex0, by, 260, 18, 12, bold=True)
    q = pg.axes(ex0 + 20, by + 30, 230, 110, (0, 60), (0, 1.2), "epoch", "", label_size=10)
    q.curve(lambda e: 1.1 * math.exp(-e / 10) + 0.05, 0, 60, color=BLUE, width=1.5)
    q.curve(lambda e: 1.1 * math.exp(-e / 10) + 0.25 + 0.0002 * (e - 18) ** 2 * (e > 18), 0, 60, color=RED,
            width=1.5)
    q.vline(18, 0, 1.2, color=GREEN, width=1.5)
    q.text("stop", 18, 1.1, dx=18, size=10, color=GREEN)
    dx0 = x0 + 650
    pg.text("Dropout (p = 0.5)", dx0, by, 280, 18, 12, bold=True)
    layers = [3, 5, 5, 1]
    pos = []
    for li, nn_ in enumerate(layers):
        col = []
        for k in range(nn_):
            cx_, cy_ = dx0 + 20 + li * 80, by + 50 + k * 26 + (5 - nn_) * 13
            col.append((cx_, cy_))
        pos.append(col)
    dropped = {(1, 1), (1, 3), (2, 0), (2, 2), (2, 4)}
    for li in range(3):
        for a_i, (ax_, ay_) in enumerate(pos[li]):
            for b_i, (bx_, by_) in enumerate(pos[li + 1]):
                if (li, a_i) in dropped or (li + 1, b_i) in dropped: continue
                pg.line([(ax_, ay_), (bx_, by_)], color="#AAAAAA", width=0.5)
    for li, col in enumerate(pos):
        for k, (cx_, cy_) in enumerate(col):
            off = (li, k) in dropped
            pg.ellipse(cx_ - 8, cy_ - 8, 16, 16, "✕" if off else "", fill="#EEEEEE" if off else LIGHTBLUE,
                       stroke=GREY if off else BLUE, size=9, color=RED)
    pg.text("random neurons switched off each step → no single path can memorise", dx0, by + 172, 300, 30, 10,
            italic=True)
    pg.bullets(x0 + 20, y0 + 645, 940, [
        "<b>More rows</b> lowers variance (overfit gap shrinks, see ④); <b>augmentation</b> (noise, bootstraps, synthetic samples) "
        "helps when real rows are scarce"], size=11)


# ---------------------------------------------------------------- ⑥ remedy decision table
def panel6(pg):
    x0, y0 = 400, 2420
    pg.panel(x0, y0, 2000, 470, 6, "Remedy decision table: problem → fix → what to check in the data table")
    U = '<font color="#3333FF"><b>Underfit</b></font>'
    O = '<font color="#FF0000"><b>Overfit</b></font>'
    rows = [["Problem", "Fix", "Tabular check (look at the data table)", "Typical knob"],
            [U, "Add richer / missing features", "Is the real driver a column at all? correlation / MI of each column with y", "new data source"],
            [U, "Better representation", "Timestamps → hour / weekday; nominal → one-hot; skewed → log", "encoders, transforms"],
            [U, "More complex model / interactions", "Do residuals still show a pattern vs any column?", "degree, max_depth ↑, hidden units ↑"],
            [U, "Less regularization, train longer", "Train error still high and flat", "λ ↓, epochs ↑, C ↑"],
            [O, "More rows / augmentation", "rows ÷ columns ratio (aim ≫ 10); learning curve still dropping?", "collect data, bootstrap"],
            [O, "Drop ID / leaky / duplicate columns & rows", "Unique-value count per column = rows? features computed after time t? duplicate rows across split?", "drop, shift(1), dedupe"],
            [O, "Fewer / cleaner features", "Many weak, correlated columns; rare levels with 1–2 rows", "feature selection, group rare levels"],
            [O, "Regularization", "Large coefficients with alternating signs", "λ (L1/L2), weight decay"],
            [O, "Prune / limit trees", "Leaves with 1–2 rows", "max_depth, min_samples_leaf, ccp_alpha"],
            [O, "Early stopping, dropout", "Validation loss rising while train loss falls", "patience, dropout p"],
            [O, "Honest validation & ensembling", "Is the split by time / group? Is the test set untouched?", "CV, purged CV, bagging"]]
    pg.table(x0 + 20, y0 + 50, 1960, rows, size=11, h=12 * 21 + 6)
    pg.html(f'<span style="background:{HL}"><b>Rule of thumb:</b></span> first make the model able to fit the training rows '
            '(fix underfitting), then make it generalise (fix overfitting) — and always judge on rows it has never seen.',
            x0 + 20, y0 + 320, 1960, 24, 12)
    pg.bullets(x0 + 20, y0 + 352, 1960, [
        "Overfitting is mostly a <b>data-table</b> problem (too few rows, too many / leaky columns, noisy labels) before it is a model problem",
        "Underfitting is mostly a <b>feature</b> problem: the information is not in the columns, or not in a form the model can use",
        "Related pages: Model Evaluation (validation, learning curves), Feature Engineering (representation), Explainability (suspicious top feature = leakage)"],
        size=11)

    # ------------------------------------------------------------ interpretation callouts (all panels)
    pg.ask("Is the model memorising rows instead of learning the pattern?",
           "overfit: train MSE 0.00 but test 2.63; the cubic (0.05 / 0.51) has the smallest gap, so it generalises",
           1800, 700, 2149, 627, w=540)
    pg.ask("Why is the test score suspiciously good?",
           "the test row (62, 0.9) duplicates a training row, so it was already seen; dedupe before splitting",
           430, 1562, 1012, 1473, w=1000)
    pg.ask("Why can't a linear model learn this table?",
           "the signal lives in the product x₁·x₂ (each alone = 50 %); adding that column makes it linear",
           1660, 1560, 1856, 1381, w=720)
    pg.ask("Will collecting more rows fix the overfitting?",
           "partly: the depth-12 gap shrinks as n grows (variance); depth-1 curves stay high, so data won't fix underfit",
           1175, 2190, 1184, 2081, w=195)
    pg.ask("Which features does L1 switch off?",
           "the solution lands on a corner (β₁ = 0), so L1 drops weak features; L2 only shrinks them",
           2025, 2047, 2218, 1853, w=360)
    pg.ask("What should I check in my table when validation looks too good?",
           "unique values per column (IDs), features computed after time t, and duplicate rows across splits",
           440, 2490, 933, 2605, w=440)

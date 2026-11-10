"""Feature Engineering page (base: Standardization / Z-score)."""
import math
from dio import *


def counts(values, lo, hi, bins):
    w = (hi - lo) / bins
    c = [0] * bins
    for v in values:
        k = int((v - lo) / w)
        if 0 <= k < bins: c[k] += 1
    return [(lo + (k + 0.5) * w, n) for k, n in enumerate(c)], w


def mean(v): return sum(v) / len(v)


def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def skew(v):
    m, s = mean(v), std(v)
    return sum(((x - m) / s) ** 3 for x in v) / len(v)


def quantile(v, q):
    s = sorted(v)
    pos = q * (len(s) - 1)
    i = int(pos)
    return s[i] if i + 1 >= len(s) else s[i] + (s[i + 1] - s[i]) * (pos - i)


def build(pg: Page, cat):
    r = seeded(22)

    # ------------------------------------------------------------------ ① standardization
    x0, y0 = 400, 240
    pg.panel(x0, y0, 980, 700, 1, "★ Standardization (Z-score)", "Put every feature on the same scale: mean 0, std 1")
    price = [r.gauss(150, 20) for _ in range(200)]
    rsi = [min(99, max(1, r.gauss(50, 15))) for _ in range(200)]
    mp, sp, mr, sr = mean(price), std(price), mean(rsi), std(rsi)
    zp = [(v - mp) / sp for v in price]
    zr = [(v - mr) / sr for v in rsi]
    pg.math(r"z = \frac{x - \mu}{\sigma}", x0 + 30, y0 + 55, 200, 50, 20)
    pg.note("μ, σ computed on the TRAINING set", x0 + 230, y0 + 62, x0 + 175, y0 + 90, w=200, color=RED)
    # before
    pg.text("Before: different units & ranges", x0 + 30, y0 + 120, 420, 20, 12, bold=True)
    d, w = counts(price, 80, 220, 20)
    p = pg.axes(x0 + 50, y0 + 150, 380, 110, (80, 220), (0, max(n for _, n in d) * 1.2), "price", "", label_size=11,
                xticks=(100, 150, 200))
    p.bars(d, w, fill=LIGHTBLUE, stroke=BLUE, opacity=80)
    pg.text(f"μ = {mp:.1f},  σ = {sp:.1f}", x0 + 260, y0 + 150, 170, 16, 10, color=BLUE)
    d, w = counts(rsi, 0, 100, 20)
    p = pg.axes(x0 + 50, y0 + 300, 380, 110, (0, 100), (0, max(n for _, n in d) * 1.2), "RSI", "", label_size=11,
                xticks=(0, 50, 100))
    p.bars(d, w, fill="#FFE6CC", stroke="#D79B00", opacity=80)
    pg.text(f"μ = {mr:.1f},  σ = {sr:.1f}", x0 + 260, y0 + 300, 170, 16, 10, color="#D79B00")
    # after (common axis)
    pg.text("After: both on the same z-axis", x0 + 520, y0 + 120, 420, 20, 12, bold=True)
    dp, w = counts(zp, -3.5, 3.5, 21)
    dr, _ = counts(zr, -3.5, 3.5, 21)
    top = max(max(n for _, n in dp), max(n for _, n in dr)) * 1.2
    p = pg.axes(x0 + 540, y0 + 150, 400, 260, (-3.5, 3.5), (0, top), "z", "", label_size=12,
                xticks=(-3, -2, -1, 0, 1, 2, 3), origin=(-3.5, 0))
    p.bars([(c - w / 4, n) for c, n in dp], w / 2, fill=LIGHTBLUE, stroke=BLUE, opacity=80)
    p.bars([(c + w / 4, n) for c, n in dr], w / 2, fill="#FFE6CC", stroke="#D79B00", opacity=80)
    p.vline(0, 0, top * 0.95, color=RED)
    pg.html('<font color="#3333FF">■ price</font>&nbsp;&nbsp;<font color="#D79B00">■ RSI</font>',
            x0 + 800, y0 + 150, 140, 18, 11)
    pg.line([(x0 + 445, y0 + 280), (x0 + 520, y0 + 280)], arrow="classic", width=2)
    # worked table
    rows = [["row", "price", "RSI", r"\(z_{price}\)", r"\(z_{RSI}\)"]]
    for i in range(4):
        rows.append([i + 1, f"{price[i]:.1f}", f"{rsi[i]:.1f}", f"{zp[i]:+.2f}", f"{zr[i]:+.2f}"])
    pg.text("Worked rows", x0 + 30, y0 + 432, 300, 18, 12, bold=True)
    pg.table(x0 + 30, y0 + 455, 400, rows, size=11, h=120)
    pg.html(f"e.g. z = ({price[0]:.1f} − {mp:.1f}) / {sp:.1f} = <b>{zp[0]:+.2f}</b>  → "
            f"{abs(zp[0]):.2f} σ {'above' if zp[0] > 0 else 'below'} the mean", x0 + 30, y0 + 580, 420, 20, 11)
    # pipeline diagram
    px, py = x0 + 500, y0 + 440
    pg.text("Fit on train only (no leakage)", px, py - 8, 440, 18, 12, bold=True)
    tr = pg.rect(px, py + 20, 120, 40, "Train data", fill=LIGHTBLUE, stroke=BLUE, size=11)
    fit = pg.rect(px + 160, py + 20, 130, 40, "scaler.fit()<br>μ, σ", fill=HL, size=11, bold=True, raw=True)
    tt = pg.rect(px + 330, py + 20, 110, 40, "transform", size=11)
    va = pg.rect(px, py + 90, 120, 40, "Val / Test data", fill="#F5F5F5", stroke=GREY, size=11)
    tv = pg.rect(px + 330, py + 90, 110, 40, "transform", size=11)
    pg.connect(tr, fit, orth=False)
    pg.connect(fit, tt, orth=False)
    pg.connect(va, tv, orth=False)
    pg.line([(px + 225, py + 60), (px + 225, py + 75), (px + 385, py + 75), (px + 385, py + 90)], color=RED,
            dashed=True, arrow="classic")
    pg.text("reuse train μ, σ", px + 232, py + 58, 110, 16, 10, color=RED)
    pg.bullets(x0 + 500, y0 + 590, 460, [
        "Distance models (kNN, K-Means, SVM) & PCA: large-unit features dominate otherwise",
        "Gradient descent: round cost contours → faster convergence",
        "Regularization (Ridge / Lasso) penalises all coefficients fairly",
        "Trees don't need scaling (splits are order-based)"], head="Why it matters:", size=10, h=90)

    # ------------------------------------------------------------------ ② scaling comparison
    x0 = 1420
    pg.panel(x0, y0, 980, 700, 2, "Scaling methods compared (skewed data + one outlier)",
             "Same 30 values, three scalers, common axis")
    data = sorted([round(r.lognormvariate(2.0, 0.35), 1) for _ in range(29)]) + [45.0]
    m_, s_ = mean(data), std(data)
    mn, mx = min(data), max(data)
    med, q1, q3 = quantile(data, 0.5), quantile(data, 0.25), quantile(data, 0.75)
    iqr = q3 - q1
    scal = [("Raw x", data, BLACK),
            ("Standardization  (x − μ)/σ", [(v - m_) / s_ for v in data], BLUE),
            ("Min-Max  (x − min)/(max − min)", [(v - mn) / (mx - mn) for v in data], "#D79B00"),
            ("Robust  (x − median)/IQR", [(v - med) / iqr for v in data], GREEN)]
    p = Plot(pg, x0 + 270, y0 + 80, 660, 300, (-2, 12), (0, 5))
    for k, (lab, vals, col) in enumerate(scal):
        yy = 4.3 - k * 1.1
        if k == 0:
            pg.text(lab + "  (own scale, shown ÷ 4)", x0 + 30, p.P(0, yy)[1] - 10, 240, 20, 11, bold=True, color=col)
            vals = [v / 4 for v in vals]
        else:
            pg.text(lab, x0 + 30, p.P(0, yy)[1] - 10, 240, 20, 11, bold=True, color=col)
        p.hline(yy, -2, 12, color="#CCCCCC", dashed=False)
        for v in vals[:-1]:
            pg.dot(*p.P(v, yy + r.uniform(-0.15, 0.15)), r=3.2, color=col)
        pg.dot(*p.P(vals[-1], yy), r=5, color=RED)
    pg.line([p.P(-2, -0.2), p.P(12, -0.2)], arrow="classic")
    for t in range(-2, 13, 2):
        px_, py_ = p.P(t, -0.2)
        pg.line([(px_, py_), (px_, py_ + 4)])
        pg.text(str(t), px_ - 15, py_ + 5, 30, 14, 9, align="center", color="#555555")
    ox_, oy_ = p.P(45 / 4, 4.3)
    pg.note("outlier (45)", ox_ - 120, oy_ - 34, ox_ - 4, oy_ - 4, w=90, color=RED)
    mx_, my_ = p.P(0.25, 2.1)
    pg.note("Min-Max: outlier squashes all other values into ~[0, 0.2]", mx_ + 80, my_ + 14, mx_ + 4, my_ + 4,
            w=330, color="#D79B00")
    pg.text(f"μ = {m_:.1f}, σ = {s_:.1f} · min = {mn}, max = {mx} · median = {med:.1f}, IQR = {iqr:.1f}",
            x0 + 30, y0 + 425, 920, 18, 11, align="center", color="#555555")
    pg.table(x0 + 30, y0 + 455, 920, [
        ["Scaler", "Formula", "Range", "Outlier effect", "Use when"],
        ["Standardization", r"\(\frac{x-\mu}{\sigma}\)", "unbounded, mean 0", "pulls μ, inflates σ",
         "roughly normal features; linear models, PCA, SVM"],
        ["Min-Max", r"\(\frac{x-x_{min}}{x_{max}-x_{min}}\)", "[0, 1]", "very sensitive",
         "bounded inputs (images, NN), no big outliers"],
        ["Robust", r"\(\frac{x-\tilde{x}}{IQR}\)", "unbounded, median 0", "resistant",
         "heavy tails / outliers (returns, volume)"],
        ["Max-Abs", r"\(\frac{x}{|x|_{max}}\)", "[−1, 1]", "sensitive", "sparse data (keeps zeros)"]], size=11, h=170)
    pg.text("None of them change the SHAPE of the distribution → for skew use a transform (③)", x0 + 30, y0 + 640,
            920, 20, 11, italic=True, color=RED, align="center")

    # ------------------------------------------------------------------ ③ distribution transforms
    x0, y0 = 400, 980
    pg.panel(x0, y0, 980, 680, 3, "Fixing skew: log, Box-Cox, Yeo-Johnson",
             "Right-skewed trading volume (millions) → closer to normal")
    vol = [r.lognormvariate(0.0, 0.8) for _ in range(300)]
    n = len(vol)

    def bc(x, lam):
        return math.log(x) if abs(lam) < 1e-9 else (x ** lam - 1) / lam

    def yj(x, lam):  # x >= 0 branch
        return math.log(x + 1) if abs(lam) < 1e-9 else ((x + 1) ** lam - 1) / lam

    def best(fn, jac):
        bestl, bestv = 0, -1e18
        for k in range(-40, 41):
            lam = k / 40
            y = [fn(v, lam) for v in vol]
            m = mean(y)
            var = sum((t - m) ** 2 for t in y) / n
            ll = -n / 2 * math.log(var) + (lam - 1) * sum(jac(v) for v in vol)
            if ll > bestv: bestl, bestv = lam, ll
        return bestl

    lbc = best(bc, math.log)
    lyj = best(yj, lambda v: math.log(v + 1))
    variants = [("Raw volume", vol, BLACK), ("log(x)", [math.log(v) for v in vol], BLUE),
                (f"Box-Cox  λ = {lbc:.2f}", [bc(v, lbc) for v in vol], GREEN),
                (f"Yeo-Johnson  λ = {lyj:.2f}", [yj(v, lyj) for v in vol], "#9673A6")]
    for k, (lab, vals, col) in enumerate(variants):
        cx, cy = x0 + 50 + (k % 2) * 460, y0 + 90 + (k // 2) * 220
        lo, hi = min(vals), max(vals)
        d, w = counts(vals, lo, hi + 1e-9, 22)
        p = pg.axes(cx, cy + 20, 380, 140, (lo, hi), (0, max(c for _, c in d) * 1.2), "", "", arrow=False)
        p.bars(d, w, fill="#FFFFFF" if k == 0 else LIGHTBLUE, stroke=col, opacity=90)
        g1 = skew(vals)
        pg.html(f"<b>{lab}</b> &nbsp; skewness g₁ = <span style='background:{HL if abs(g1) < 0.3 else '#FFFFFF'}'>"
                f"{g1:+.2f}</span>", cx, cy - 4, 380, 20, 12)
    pg.math(r"\text{Box-Cox: } y=\begin{cases}\frac{x^{\lambda}-1}{\lambda} & \lambda\neq0\\ \ln x & \lambda=0\end{cases}"
            r"\quad (x\gt 0)", x0 + 30, y0 + 525, 460, 56, 14)
    pg.math(r"\text{Yeo-Johnson: same idea on } (x+1)\text{, also works for } x \le 0", x0 + 500, y0 + 525, 460, 56,
            13)
    pg.bullets(x0 + 30, y0 + 585, 920, [
        "λ is chosen by maximum likelihood (here on a grid −1 … 1); λ = 0 ⇒ log, λ = 1 ⇒ no change, λ = 0.5 ⇒ ≈ √x",
        "Check with a Q-Q plot / skewness (see Distributions page); trees don't need it, linear models & z-score do",
        "Returns are already ~symmetric → use log-returns instead of prices; fit λ on training data only"], size=11,
        head="Notes:")

    # ------------------------------------------------------------------ ④ encoding
    x0 = 1420
    pg.panel(x0, y0, 980, 680, 4, "Encoding categorical features",
             "Models need numbers — the right encoding depends on the data type (NOIR page)")
    raw = [["Tech", "Low", 1], ["Energy", "High", 0], ["Bank", "Mid", 1],
           ["Tech", "Mid", 1], ["Bank", "High", 0], ["Energy", "Low", 0]]
    pg.text("Raw data", x0 + 30, y0 + 70, 200, 18, 12, bold=True)
    pg.table(x0 + 30, y0 + 92, 240, [["sector<br><i>nominal</i>", "rating<br><i>ordinal</i>", "up (y)"]] +
             [[a, b, c] for a, b, c in raw], size=11, h=190)
    # one-hot
    pg.line([(x0 + 280, y0 + 180), (x0 + 320, y0 + 180)], arrow="classic", width=1.5)
    pg.text("One-Hot (sector)", x0 + 330, y0 + 70, 260, 18, 12, bold=True, color=BLUE)
    oh = [['<s><font color="#FF0000">is_Bank</font></s>', "is_Energy", "is_Tech"]] + \
         [[f'<font color="#999999">{int(a == "Bank")}</font>', int(a == "Energy"), int(a == "Tech")] for a, _, _ in raw]
    pg.table(x0 + 330, y0 + 92, 250, oh, size=11, h=190)
    pg.text("drop_first: remove one column (Bank = both 0) → avoids the dummy-variable trap "
            "(perfect multicollinearity in linear models)", x0 + 330, y0 + 288, 270, 46, 10, color=RED)
    # ordinal
    pg.text("Ordinal (rating)", x0 + 620, y0 + 70, 200, 18, 12, bold=True, color=GREEN)
    pg.table(x0 + 620, y0 + 92, 140, [["rating", "code"], ["Low", 1], ["Mid", 2], ["High", 3]], size=11, h=110)
    pg.text("keeps the order Low < Mid < High; never use it for nominal data (implies Tech > Bank)", x0 + 620,
            y0 + 208, 330, 34, 10, italic=True)
    # target encoding
    pg.text("Target (mean) encoding (sector) with smoothing", x0 + 30, y0 + 345, 500, 18, 12, bold=True,
            color="#9673A6")
    pg.math(r"enc(c) = \frac{n_c\,\bar{y}_c + m\,\bar{y}}{n_c + m}", x0 + 30, y0 + 370, 300, 50, 16)
    gm = sum(t for _, _, t in raw) / len(raw)
    rows = [["sector", r"\(n_c\)", r"\(\bar{y}_c\)", "enc (m = 2)"]]
    for s in ("Tech", "Energy", "Bank"):
        ys = [t for a, _, t in raw if a == s]
        rows.append([s, len(ys), f"{mean(ys):.2f}", f"<b>{(len(ys) * mean(ys) + 2 * gm) / (len(ys) + 2):.2f}</b>"])
    pg.table(x0 + 340, y0 + 360, 300, rows, size=11, h=110)
    pg.text(f"global mean ȳ = {gm:.2f}; smoothing m pulls rare categories toward it", x0 + 30, y0 + 425, 300, 34,
            10, italic=True)
    pg.html('<b><font color="#FF0000">⚠ Leakage:</font></b> compute it <b>out-of-fold</b> (K-fold on train) — '
            'using the row\'s own target leaks the label', x0 + 660, y0 + 360, 300, 60, 11)
    pg.table(x0 + 30, y0 + 490, 920, [
        ["Encoding", "Data type", "Pros", "Cons"],
        ["One-Hot", "nominal, low cardinality", "no false order", "many columns, sparse"],
        ["Ordinal", "ordinal", "1 column, keeps order", "assumes equal spacing"],
        ["Target / Mean", "nominal, high cardinality", "1 column, strong signal", "leakage risk → out-of-fold"],
        ["Hashing / Binary", "very high cardinality (tickers, IDs)", "fixed size", "collisions, not interpretable"]],
        size=11, h=150)

    # ------------------------------------------------------------------ ⑤ discretization & feature creation
    x0, y0 = 400, 1700
    pg.panel(x0, y0, 980, 720, 5, "Discretization & creating new features")
    vals = sorted(round(r.lognormvariate(1.5, 0.5), 2) for _ in range(40))
    lo, hi = vals[0], vals[-1]
    ew = [lo + (hi - lo) * k / 4 for k in range(5)]
    ef = [vals[0]] + [quantile(vals, k / 4) for k in (1, 2, 3)] + [vals[-1]]
    pg.text("Binning the same 40 values into 4 bins", x0 + 30, y0 + 50, 500, 18, 12, bold=True)
    for k, (lab, edges, col) in enumerate([("Equal-width (same interval size)", ew, BLUE),
                                           ("Equal-frequency (quantiles, same count)", ef, GREEN)]):
        p = Plot(pg, x0 + 60, y0 + 100 + k * 90, 860, 40, (0, hi * 1.05), (0, 1))
        pg.text(lab, x0 + 40, y0 + 76 + k * 90, 400, 18, 11, bold=True, color=col)
        pg.line([p.P(0, 0.3), p.P(hi * 1.05, 0.3)], color="#999999")
        for v in vals:
            pg.dot(*p.P(v, 0.3 + r.uniform(-0.12, 0.12)), r=3, color=col)
        for e in edges[1:-1]:
            p.vline(e, -0.1, 1.0, color=RED, dashed=True, width=1.5)
        cnt = [sum(1 for v in vals if edges[b] <= v <= edges[b + 1] and (b == 3 or v < edges[b + 1]))
               for b in range(4)]
        for b in range(4):
            p.text(f"n={cnt[b]}", (edges[b] + edges[b + 1]) / 2, 0.95, size=10, w=50, color=RED)
    pg.text("equal-width: outliers leave empty bins · equal-frequency: balanced bins, uneven widths; "
            "fit the edges on training data", x0 + 40, y0 + 270, 900, 18, 10, italic=True)
    # interaction XOR
    pg.text("Interaction term: x₁·x₂ makes a non-linear pattern linear", x0 + 30, y0 + 310, 500, 18, 12, bold=True)
    pts = [(r.uniform(-1, 1), r.uniform(-1, 1)) for _ in range(70)]
    p = pg.axes(x0 + 60, y0 + 350, 220, 180, (-1, 1), (-1, 1), "x_1", "x_2", label_size=11, origin=(0, 0))
    for a, b in pts:
        pg.dot(*p.P(a, b), r=3.2, color=BLUE if a * b > 0 else RED)
    pg.text("no straight line separates the classes", x0 + 50, y0 + 545, 240, 18, 10, italic=True, align="center")
    pg.line([(x0 + 300, y0 + 440), (x0 + 350, y0 + 440)], arrow="classic", width=1.5)
    q = Plot(pg, x0 + 370, y0 + 420, 220, 40, (-1, 1), (0, 1))
    pg.line([q.P(-1, 0.5), q.P(1, 0.5)], arrow="classic")
    for a, b in pts:
        pg.dot(*q.P(a * b, 0.5 + r.uniform(-0.25, 0.25)), r=3.2, color=BLUE if a * b > 0 else RED)
    q.vline(0, 0, 1, color=GREEN, dashed=False, width=2)
    pg.math(r"z = x_1 x_2", x0 + 420, y0 + 470, 120, 30, 14)
    pg.text("threshold z = 0 separates perfectly", x0 + 370, y0 + 500, 220, 18, 10, italic=True, align="center",
            color=GREEN)
    # cyclical encoding
    cx, cy, R = x0 + 790, y0 + 450, 95
    pg.text("Cyclical time: hour → (sin, cos)", x0 + 650, y0 + 310, 320, 18, 12, bold=True)
    pg.ellipse(cx - R, cy - R, 2 * R, 2 * R, stroke="#999999")
    pg.line([(cx - R - 15, cy), (cx + R + 15, cy)], color="#CCCCCC")
    pg.line([(cx, cy + R + 15), (cx, cy - R - 15)], color="#CCCCCC")
    for h in range(24):
        ang = 2 * math.pi * h / 24
        x, y = cx + R * math.sin(ang), cy - R * math.cos(ang)
        col = RED if h in (23, 0) else BLUE
        pg.dot(x, y, r=4 if h in (23, 0) else 3, color=col)
        if h % 3 == 0 or h == 23:
            pg.text(str(h), x + 9 * math.sin(ang) - 10, y - 9 * math.cos(ang) - 8, 20, 16, 9, align="center",
                    color=col)
    pg.text("23 h and 0 h are neighbours (raw 23 vs 0 looks far)", cx - 150, cy + R + 22, 300, 18, 10, color=RED,
            align="center")
    pg.math(r"x_{sin}=\sin\frac{2\pi h}{24},\ \ x_{cos}=\cos\frac{2\pi h}{24}", x0 + 640, y0 + 600, 320, 40, 13)
    pg.bullets(x0 + 30, y0 + 590, 600, [
        "Polynomial terms x², x³ let linear models fit curves (see Regression page)",
        "Ratios & differences: price / EMA, high − low (range), volume / avg volume",
        "Same idea for weekday (7), month (12), day-of-year (365)"], head="More feature ideas:", size=11)

    # ------------------------------------------------------------------ ⑥ time-series / trading features
    x0 = 1420
    pg.panel(x0, y0, 980, 720, 6, "Trading features: indicators, lags & rolling windows",
             "Simulated 120-bar price · every feature uses only past data")
    pr, pv = [], 100.0
    for _ in range(120):
        pv *= math.exp(r.gauss(0.0006, 0.012))
        pr.append(pv)

    def ema(v, n_):
        a, out = 2 / (n_ + 1), [v[0]]
        for x in v[1:]: out.append(a * x + (1 - a) * out[-1])
        return out

    e10, e20 = ema(pr, 10), ema(pr, 20)
    bb_m, bb_u, bb_l = [], [], []
    for i in range(19, 120):
        w_ = pr[i - 19:i + 1]
        m, s = mean(w_), std(w_)
        bb_m.append((i, m)); bb_u.append((i, m + 2 * s)); bb_l.append((i, m - 2 * s))
    gains, losses, rsi_ = [], [], []
    for i in range(1, 120):
        d_ = pr[i] - pr[i - 1]
        gains.append(max(d_, 0)); losses.append(max(-d_, 0))
    ag, al = mean(gains[:14]), mean(losses[:14])
    for i in range(14, 119):
        ag = (ag * 13 + gains[i]) / 14
        al = (al * 13 + losses[i]) / 14
        rsi_.append((i + 1, 100 - 100 / (1 + ag / al) if al > 0 else 100))
    lo_, hi_ = min(min(pr), min(b for _, b in bb_l)), max(max(pr), max(b for _, b in bb_u))
    p = pg.axes(x0 + 70, y0 + 80, 860, 210, (0, 119), (lo_ - 1, hi_ + 1), "t", "price", label_size=11)
    p.poly(bb_u, color=GREY, dashed=True)
    p.poly(bb_l, color=GREY, dashed=True)
    p.poly(list(enumerate(pr)), color=BLACK, width=1.5)
    p.poly(list(enumerate(e10)), color=BLUE, width=1.5)
    p.poly(list(enumerate(e20)), color=ORANGE, width=1.5)
    pg.html('<b>— price</b> &nbsp;<font color="#3333FF">— EMA(10)</font> &nbsp;<font color="#FF8000">— EMA(20)</font>'
            ' &nbsp;<font color="#808080">- - Bollinger(20, 2σ)</font>', x0 + 400, y0 + 62, 540, 18, 10)
    q = pg.axes(x0 + 70, y0 + 320, 860, 80, (0, 119), (0, 100), "t", "RSI", label_size=11, yticks=(30, 70))
    q.hline(70, color=RED); q.hline(30, color=GREEN)
    q.poly(rsi_, color="#9673A6", width=1.2)
    pg.math(r"EMA_t=\alpha p_t+(1-\alpha)EMA_{t-1},\ \alpha=\tfrac{2}{n+1}", x0 + 30, y0 + 420, 330, 36, 12)
    pg.math(r"RSI=100-\frac{100}{1+\overline{gain}/\overline{loss}}", x0 + 360, y0 + 420, 260, 36, 12)
    pg.math(r"BB=SMA_{20}\pm2\sigma_{20}", x0 + 640, y0 + 420, 300, 36, 12)
    rets = [None] + [pr[i] / pr[i - 1] - 1 for i in range(1, 120)]
    rows = [["t", "close", "ret", "ret_lag1", "roll_mean3 (shift 1)"]]
    for t in range(6):
        lag = rets[t - 1] if t >= 1 else None
        rm = mean([rets[k] for k in range(t - 3, t)]) if t >= 4 else None
        f = lambda v: "NaN" if v is None else f"{v * 100:+.2f}%"
        rows.append([t, f"{pr[t]:.2f}", f(rets[t]), f(lag), f(rm)])
    pg.table(x0 + 30, y0 + 470, 470, rows, size=10, h=150)
    pg.text("first rows are NaN → drop them; never back-fill from the future", x0 + 30, y0 + 625, 470, 18, 10,
            italic=True, color=RED)
    pg.bullets(x0 + 530, y0 + 465, 430, [
        "Features at time t use data up to t (shift(1) for anything built from the target)",
        "Rolling windows must be trailing, never centered",
        "Scale / impute with statistics from the training window only",
        "Missing values: forward-fill prices; median for tabular; add an 'is_missing' flag",
        "Re-check after every new feature: train score ≫ test score ⇒ suspect leakage"],
        head="Leakage checklist:", size=10, h=150)

    # ------------------------------------------------------------ interpretation callouts
    pg.ask("How unusual is this row compared with the others?",
           "row 3: price z = +3.53 → 3.5 σ above the mean, a likely outlier",
           430, 862, 657, 772, w=420)
    pg.ask("Which scaler should I use when the data has outliers?",
           "Robust (median / IQR): the outlier doesn't inflate its spread; Min-Max squashes the rest into ~[0, 0.2]",
           1440, 822, 1653, 760, w=255)
    pg.ask("Did the transform make the feature roughly symmetric?",
           "yes: skewness +2.33 → −0.02 (log, Box-Cox) and +0.07 (Yeo-Johnson); |g₁| < 0.5 ≈ symmetric",
           1000, 1565, 1145, 1455, w=365)
    pg.ask("Can the model use the order Low < Mid < High?",
           "yes — ordinal codes 1, 2, 3 keep the order; nominal sectors need one-hot so no fake order is implied",
           2050, 1225, 2110, 1125, w=330)
    pg.ask("Why add the product x₁·x₂ as a new column?",
           "on z = x₁x₂ one threshold (z = 0) separates red from blue, so a linear model can solve it",
           840, 2290, 879, 2138, w=200)
    pg.ask("Can this feature be computed at prediction time without the future?",
           "yes: ret_lag1 and roll_mean3 are shifted one bar, so row t only uses data up to t−1 (first rows NaN)",
           1830, 2262, 1755, 2199, w=540)

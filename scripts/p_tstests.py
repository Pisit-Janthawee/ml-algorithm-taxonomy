"""Time Series Hypothesis Tests & Stationarity page (base: Augmented Dickey-Fuller), incl. cointegration & ECM."""
import math
from dio import *


# ---------------------------------------------------------------- numeric helpers
def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def inv(A):
    n = len(A)
    cols = [solve(A, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def ols(X, y):
    k = len(X[0])
    XtX = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    Xty = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    b = solve(XtX, Xty)
    res = [v - sum(bi * xi for bi, xi in zip(b, r)) for r, v in zip(X, y)]
    s2 = sum(e * e for e in res) / (len(y) - k)
    iv = inv(XtX)
    se = [math.sqrt(max(s2 * iv[i][i], 1e-12)) for i in range(k)]
    sst = sum((v - sum(y) / len(y)) ** 2 for v in y)
    r2 = 1 - sum(e * e for e in res) / sst if sst else 0
    return b, se, res, r2


def adf(y, lags=1, trend=False):
    dy = [y[t] - y[t - 1] for t in range(1, len(y))]
    X, Y = [], []
    for t in range(lags + 1, len(y)):
        row = [1.0] + ([float(t)] if trend else []) + [y[t - 1]] + [dy[t - 1 - i] for i in range(1, lags + 1)]
        X.append(row)
        Y.append(dy[t - 1])
    b, se, _, _ = ols(X, Y)
    g = 2 if trend else 1
    return b[g], b[g] / se[g]


def ar1(n, phi, r, s=1.0, c=0.0):
    y = [0.0]
    for _ in range(n - 1):
        y.append(c + phi * y[-1] + r.gauss(0, s))
    return y


def rolling(y, w):
    m, sd = [], []
    for i in range(len(y)):
        seg = y[max(0, i - w + 1):i + 1]
        mu = sum(seg) / len(seg)
        m.append(mu)
        sd.append(math.sqrt(sum((v - mu) ** 2 for v in seg) / len(seg)))
    return m, sd


def acf(y, k):
    n, mu = len(y), sum(y) / len(y)
    d = sum((v - mu) ** 2 for v in y)
    return sum((y[t] - mu) * (y[t - k] - mu) for t in range(k, n)) / d


def series(pg, x, y, w, h, data, color=BLUE, yr=None, step=2, width=1, xlabel="t", ylabel="", label_size=11,
           arrow=True):
    lo, hi = (min(data), max(data)) if yr is None else yr
    pad = (hi - lo) * 0.08 or 1
    p = pg.axes(x, y, w, h, (0, len(data) - 1), (lo - pad, hi + pad), xlabel, ylabel, label_size=label_size,
                origin=(0, lo - pad), arrow=arrow)
    p.poly([(i, data[i]) for i in range(0, len(data), step)], color=color, width=width)
    return p


def build(pg: Page, cat):
    r = seeded(19)
    N = 200

    # ------------------------------------------------------------ ① gallery
    x0, y0 = 400, 240
    pg.panel(x0, y0, 1100, 640, 1, "Stationary vs Non-stationary (simulated, n = 200)",
             "Red = rolling mean (window 20), dashed = rolling mean ± rolling std")
    wn = ar1(N, 0.3, r)
    trend = [0.05 * t + r.gauss(0, 1) for t in range(N)]
    seas = [3 * math.sin(2 * math.pi * t / 25) + r.gauss(0, 0.7) for t in range(N)]
    rw = ar1(N, 1.0, r)
    brk = [(0 if t < 110 else 5) + r.gauss(0, 1) for t in range(N)]
    hv = [r.gauss(0, 0.3 + 2.2 * t / N) for t in range(N)]
    gal = [("AR(1), φ = 0.3", wn, "✔ Stationary", GREEN),
           ("Trend", trend, "✘ mean changes", RED),
           ("Seasonality (period 25)", seas, "✘ mean depends on t (calendar)", RED),
           ("Random walk (unit root)", rw, "✘ variance grows ∝ t", RED),
           ("Structural break at t = 110", brk, "✘ mean jumps (regime change)", RED),
           ("Changing variance", hv, "✘ variance changes (heteroscedastic)", RED)]
    for k, (name, d, tag, col) in enumerate(gal):
        cx = x0 + 50 + (k % 3) * 350
        cy = y0 + 90 + (k // 3) * 270
        pg.text(name, cx - 10, cy - 30, 320, 20, 12, bold=True)
        p = series(pg, cx, cy, 290, 150, d)
        m, sd = rolling(d, 20)
        p.poly([(i, m[i]) for i in range(19, N, 3)], color=RED, width=1.5)
        p.poly([(i, m[i] + sd[i]) for i in range(19, N, 3)], color=ORANGE, dashed=True)
        p.poly([(i, m[i] - sd[i]) for i in range(19, N, 3)], color=ORANGE, dashed=True)
        pg.text(tag, cx - 10, cy + 172, 320, 20, 11, bold=True, color=col)
    pg.text("Most ML / statistical models assume the data-generating process does not change over time → test first, "
            "transform if needed.", x0 + 20, y0 + 600, 1060, 20, 11, italic=True, color="#555555")

    # ------------------------------------------------------------ ② definitions + unit root intuition
    x0 = 1540
    pg.panel(x0, y0, 860, 640, 2, "Weak Stationarity & the Unit Root", "What exactly must stay constant?")
    pg.math("E[y_t] = \\mu", x0 + 30, y0 + 70, 220, 34, 16)
    pg.math("Var(y_t) = \\sigma^2", x0 + 260, y0 + 70, 240, 34, 16)
    pg.math("Cov(y_t, y_{t-k}) = \\gamma_k", x0 + 520, y0 + 70, 300, 34, 16)
    pg.text("constant mean", x0 + 30, y0 + 104, 220, 16, 10, align="center", color="#555555")
    pg.text("constant variance", x0 + 260, y0 + 104, 240, 16, 10, align="center", color="#555555")
    pg.text("depends only on lag k, not on t", x0 + 520, y0 + 104, 300, 16, 10, align="center", color="#555555")
    pg.html("<b>Weak (covariance) stationarity</b> = the 3 conditions above.<br>"
            "<b>Strict stationarity</b> = the whole joint distribution is shift-invariant (stronger, rarely tested).",
            x0 + 30, y0 + 130, 800, 40, 12)
    pg.text("Shock response: y₀ = 1, then yₜ = φ·yₜ₋₁ (no new noise)", x0 + 30, y0 + 185, 800, 20, 13, bold=True)
    p = pg.axes(x0 + 80, y0 + 240, 500, 215, (0, 30), (0, 2.0), "t", "y_t", label_size=12,
                xticks=(0, 10, 20, 30), yticks=(0.5, 1, 1.5, 2))
    for phi, col, lab in [(0.5, GREEN, "φ = 0.5  → shock fades (stationary)"),
                          (1.0, ORANGE, "φ = 1  → shock persists forever (unit root)"),
                          (1.02, RED, "φ = 1.02 → shock explodes")]:
        p.poly([(t, phi ** t) for t in range(31)], color=col, width=2)
    for k, (col, lab) in enumerate([(GREEN, "φ = 0.5 → shock fades (stationary)"),
                                    (ORANGE, "φ = 1 → persists forever (unit root)"),
                                    (RED, f"φ = 1.02 → grows ({1.02 ** 30:.2f} at t = 30)")]):
        pg.text(lab, x0 + 600, y0 + 250 + k * 34, 240, 30, 11, bold=True, color=col)
    pg.math("y_t = y_{t-1} + \\varepsilon_t \\;\\Rightarrow\\; y_t = y_0 + \\sum_{i=1}^{t}\\varepsilon_i,\\quad "
            "Var(y_t) = t\\,\\sigma^2", x0 + 30, y0 + 480, 800, 44, 16)
    pg.bullets(x0 + 20, y0 + 530, 820, [
        "<b>Unit root</b>: the root of \\(1-\\phi z = 0\\) is \\(z = 1\\) → variance grows with t → <b>non-stationary</b>",
        "Prices are usually unit-root (random walk); <b>returns</b> \\(\\Delta \\log p_t\\) are usually stationary",
        "Regressing one unit-root series on another gives <b>spurious</b> results (see ⑤)"], head="Key idea:")

    # ------------------------------------------------------------ ③ ADF test
    x0, y0 = 400, 920
    pg.panel(x0, y0, 1100, 680, 3, "★ Augmented Dickey-Fuller (ADF) Test",
             "Is there a unit root? Regress the change on the previous level")
    pg.math("\\Delta y_t = \\alpha + \\beta t + \\gamma\\, y_{t-1} + \\sum_{i=1}^{p}\\delta_i\\,\\Delta y_{t-i} + "
            "\\varepsilon_t", x0 + 120, y0 + 75, 640, 56, 22)
    pg.note("constant (drift)", x0 + 60, y0 + 140, x0 + 260, y0 + 112, w=110)
    pg.note("trend (optional)", x0 + 200, y0 + 160, x0 + 320, y0 + 115, w=110)
    pg.note("γ = φ − 1 : the coefficient being tested", x0 + 330, y0 + 165, x0 + 400, y0 + 115, w=230, color=RED)
    pg.note("lagged differences 'augment' DF → remove autocorrelation", x0 + 560, y0 + 140, x0 + 600, y0 + 115,
            w=260)
    pg.html("\\(H_0: \\gamma = 0\\) → <b>unit root</b> (non-stationary)<br>"
            "\\(H_1: \\gamma \\lt 0\\) → <b>stationary</b> (left-tailed test)", x0 + 800, y0 + 70, 290, 50, 12)
    pg.math("\\tau = \\frac{\\hat{\\gamma}}{SE(\\hat{\\gamma})}", x0 + 850, y0 + 125, 200, 50, 18)
    # simulated Dickey-Fuller distribution
    rs = seeded(7)
    taus = []
    for _ in range(1500):
        y = [0.0]
        for _ in range(100):
            y.append(y[-1] + rs.gauss(0, 1))
        taus.append(adf(y, lags=0)[1])
    lo_, hi_, bins = -5, 3, 32
    bw = (hi_ - lo_) / bins
    cnt = [0] * bins
    for t_ in taus:
        k = int((t_ - lo_) / bw)
        if 0 <= k < bins: cnt[k] += 1
    dens = [(lo_ + (k + 0.5) * bw, c / (len(taus) * bw)) for k, c in enumerate(cnt)]
    pg.text("Distribution of τ under H₀ (1500 simulated random walks) is NOT normal", x0 + 30, y0 + 200, 560, 20,
            12, bold=True)
    p = pg.axes(x0 + 60, y0 + 240, 500, 220, (lo_, hi_), (0, 0.55), "\\tau", "", label_size=12, origin=(lo_, 0),
                xticks=(-4, -2.86, -2, 0, 2))
    p.bars(dens, bw, fill=LIGHTBLUE, stroke=BLUE, opacity=80)
    p.curve(normal_pdf, lo_, hi_, n=80, color=BLACK, width=1.5, dashed=True)
    p.vline(-2.86, 0, 0.5, color=RED, dashed=False, width=1.5)
    p.text("Dickey-Fuller (simulated)", -3.0, 0.53, w=150, size=10, color=BLUE, bold=True)
    p.text("N(0,1)", 1.1, 0.42, w=60, size=10)
    p.text("reject H₀ (5 %)", -4.2, 0.16, w=100, size=10, color=RED)
    pg.table(x0 + 60, y0 + 495, 500, [
        ["Critical values (constant, large n)", "1 %", "5 %", "10 %"],
        ["Dickey-Fuller τ", "−3.43", "−2.86", "−2.57"],
        ["Normal z (left tail)", "−2.33", "−1.645", "−1.28"]], size=11, h=70)
    pg.text("Using normal critical values would reject H₀ far too often", x0 + 60, y0 + 570, 500, 18, 10,
            italic=True, color=RED, align="center")
    # applied to simulated series
    rw2 = ar1(300, 1.0, seeded(3))
    st2 = ar1(300, 0.5, seeded(4))
    g_rw, t_rw = adf(rw2, lags=1)
    g_st, t_st = adf(st2, lags=1)
    q1 = series(pg, x0 + 640, y0 + 225, 200, 100, rw2, step=3)
    q2 = series(pg, x0 + 640, y0 + 375, 200, 100, st2, step=3)
    pg.text("Random walk", x0 + 640, y0 + 205, 200, 16, 11, bold=True)
    pg.text("AR(1), φ = 0.5", x0 + 640, y0 + 355, 200, 16, 11, bold=True)
    pg.html(f"γ̂ = {g_rw:.3f}<br>τ = <b>{t_rw:.2f}</b> &gt; −2.86<br>"
            f'<font color="#FF0000"><b>fail to reject H₀</b></font><br>→ unit root', x0 + 880, y0 + 225, 200, 80, 11)
    pg.html(f"γ̂ = {g_st:.3f}<br>τ = <b>{t_st:.2f}</b> &lt; −2.86<br>"
            f'<span style="background:{HL}"><b>reject H₀</b></span><br>→ stationary', x0 + 880, y0 + 375, 200, 80, 11)
    pg.text("(my own OLS on n = 300, constant + 1 lag)", x0 + 640, y0 + 485, 440, 16, 10, italic=True,
            color="#555555")
    pg.bullets(x0 + 620, y0 + 510, 460, [
        "Choose lag p by AIC / BIC (residuals must be white noise)",
        "Include trend term only if the series visibly trends (critical value then −3.41 at 5 %)",
        "Low power: near-unit-root (φ = 0.97) often not rejected → cross-check with KPSS ④"], head="Practice:",
        size=11)

    # ------------------------------------------------------------ ④ ADF vs KPSS + make it stationary
    x0 = 1540
    pg.panel(x0, y0, 860, 680, 4, "ADF vs KPSS  ·  Making a Series Stationary")
    pg.table(x0 + 20, y0 + 50, 520, [
        ["ADF (H₀: unit root)", "KPSS (H₀: stationary)", "Conclusion"],
        ["reject", "do not reject", f'<span style="background:{HL}">Stationary</span>'],
        ["do not reject", "reject", "Unit root → difference it"],
        ["reject", "reject", "Difference-stationary / check breaks"],
        ["do not reject", "do not reject", "Inconclusive (too little data)"]], size=11, h=125)
    pg.bullets(x0 + 560, y0 + 50, 280, [
        r"<b>KPSS</b>: \(y_t = \xi t + r_t + \varepsilon_t\), LM test that the random-walk variance is 0",
        "<b>Phillips-Perron</b>: like ADF but corrects autocorrelation non-parametrically",
        "<b>Zivot-Andrews</b>: unit-root test allowing one unknown structural break"], size=10, h=140)
    pg.text("Fix → re-test (ADF τ, constant + 1 lag)", x0 + 20, y0 + 195, 820, 20, 13, bold=True)
    rr = seeded(11)
    base_rw = ar1(200, 1.0, rr)
    diff = [base_rw[t] - base_rw[t - 1] for t in range(1, 200)]
    tr = [0.06 * t + rr.gauss(0, 1) for t in range(200)]
    b_, _, res_tr, _ = ols([[1.0, float(t)] for t in range(200)], tr)
    ss = [3 * math.sin(2 * math.pi * t / 25) + 0.03 * t + rr.gauss(0, 0.6) for t in range(200)]
    sd_ = [ss[t] - ss[t - 25] for t in range(25, 200)]
    expo = [math.exp(0.012 * t + 0.03 * sum(rr.gauss(0, 1) for _ in range(1))) * (1 + 0.04 * rr.gauss(0, 1))
            for t in range(200)]
    logd = [math.log(expo[t]) - math.log(expo[t - 1]) for t in range(1, 200)]
    fixes = [("Random walk", base_rw, "1st difference  Δyₜ = yₜ − yₜ₋₁", diff),
             ("Trend", tr, "Detrend: residual of y on t", res_tr),
             ("Seasonal (s = 25)", ss, "Seasonal difference yₜ − yₜ₋₂₅", sd_),
             ("Exponential growth", expo, "Log, then difference (log-return)", logd)]
    for k, (nm, before, how, after) in enumerate(fixes):
        yy = y0 + 230 + k * 92
        series(pg, x0 + 30, yy + 10, 200, 60, before, step=3, xlabel="", arrow=False)
        pg.text(nm, x0 + 30, yy - 6, 200, 16, 10, bold=True)
        pg.line([(x0 + 240, yy + 58), (x0 + 330, yy + 58)], arrow="classic")
        pg.text(how, x0 + 238, yy + 6, 96, 48, 9, color="#555555", valign="bottom")
        series(pg, x0 + 340, yy + 10, 185, 60, after, step=2, color=GREEN, xlabel="", arrow=False)
        tb = adf(before, 1)[1]
        ta = adf(after, 1)[1]
        pg.html(f"τ: {tb:.2f} → <b>{ta:.2f}</b><br>"
                + (f'<span style="background:{HL}">stationary</span>' if ta < -2.86 else "still unit root"),
                x0 + 540, yy + 18, 140, 40, 10)
    # fractional differencing trade-off
    q = pg.axes(x0 + 700, y0 + 250, 130, 120, (0, 1), (0, 1), "d", "", label_size=11, xticks=(0, 0.5, 1))
    q.curve(lambda d: 1 - d ** 0.6, 0, 1, color=BLUE, width=2)
    q.curve(lambda d: min(1, 1.6 * d), 0, 1, color=RED, width=2, curved=False)
    pg.html('<b>Fractional differencing</b><br><font color="#3333FF">— memory kept</font><br>'
            '<font color="#FF0000">— stationarity</font><br>pick the smallest d that passes ADF (often 0.3–0.5)',
            x0 + 690, y0 + 390, 160, 90, 10)
    pg.text("Over-differencing adds noise and destroys predictive memory — difference only as much as needed.",
            x0 + 20, y0 + 610, 820, 30, 11, italic=True, color=RED)

    # ------------------------------------------------------------ ⑤ cointegration
    x0, y0 = 400, 1640
    pg.panel(x0, y0, 1100, 720, 5, "Cointegration (Engle-Granger) & Pairs Trading",
             "Two non-stationary series that share one stochastic trend: their spread is stationary")
    rc = seeded(23)
    n = 250
    xs = [50.0]
    for _ in range(n - 1):
        xs.append(xs[-1] + rc.gauss(0, 1))
    u = ar1(n, 0.7, rc, s=1.0)
    ys = [10 + 1.5 * a + e for a, e in zip(xs, u)]
    (a_hat, b_hat), _, res, r2c = ols([[1.0, a] for a in xs], ys)
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    p = pg.axes(x0 + 60, y0 + 90, 480, 200, (0, n - 1), (lo - 5, hi + 5), "t", "price", label_size=11,
                origin=(0, lo - 5))
    p.poly([(i, xs[i]) for i in range(0, n, 2)], color=BLUE, width=1.2)
    p.poly([(i, ys[i]) for i in range(0, n, 2)], color=ORANGE, width=1.2)
    pg.html('<font color="#3333FF"><b>— x<sub>t</sub></b> (random walk)</font><br>'
            '<font color="#FF8000"><b>— y<sub>t</sub> = 10 + 1.5 x<sub>t</sub> + u<sub>t</sub></b></font>',
            x0 + 70, y0 + 66, 300, 34, 10)
    # spread
    mu_u = sum(res) / n
    sd_u = math.sqrt(sum((e - mu_u) ** 2 for e in res) / n)
    q = pg.axes(x0 + 60, y0 + 345, 480, 170, (0, n - 1), (-4 * sd_u, 4 * sd_u), "t", "", label_size=11,
                origin=(0, -4 * sd_u))
    q.hline(0, color=BLACK, dashed=True)
    q.hline(2 * sd_u, color=RED)
    q.hline(-2 * sd_u, color=GREEN)
    q.poly([(i, res[i]) for i in range(0, n, 2)], color=BLUE, width=1)
    for i in range(1, n):
        if res[i - 1] < 2 * sd_u <= res[i]:
            q.scatter([(i, res[i])], color=RED, r=4)
        if res[i - 1] > -2 * sd_u >= res[i]:
            q.scatter([(i, res[i])], color=GREEN, r=4)
    q.text("+2σ: short y / long x", n - 1, 2 * sd_u, dx=-70, dy=-10, w=140, size=9, color=RED)
    q.text("−2σ: long y / short x", n - 1, -2 * sd_u, dx=-70, dy=12, w=140, size=9, color=GREEN)
    q.text("exit at mean", 10, 0, dx=40, dy=-9, w=80, size=9)
    pg.text(f"Spread  û = y − {a_hat:.2f} − {b_hat:.3f}·x   (mean-reverting)", x0 + 60, y0 + 322, 480, 18, 11,
            bold=True)
    t_x = adf(xs, 1)[1]
    t_y = adf(ys, 1)[1]
    t_u = adf(res, 1)[1]
    # Engle-Granger
    ex = x0 + 590
    pg.text("Engle-Granger 2-step", ex, y0 + 70, 480, 22, 14, bold=True)
    pg.math("1)\\ \\ y_t = a + \\beta x_t + u_t \\quad (OLS)", ex, y0 + 98, 480, 36, 15, align="left")
    pg.math("2)\\ \\ ADF\\ on\\ \\hat{u}_t:\\ H_0: \\text{no cointegration (unit root)}", ex, y0 + 138, 480, 36, 14,
            align="left")
    pg.table(ex, y0 + 185, 480, [
        ["Series", "ADF τ", "Result"],
        ["x (price A)", f"{t_x:.2f}", "unit root"],
        ["y (price B)", f"{t_y:.2f}", "unit root"],
        [f"spread û  (β̂ = {b_hat:.3f}, true 1.5)", f"<b>{t_u:.2f}</b>",
         f'<span style="background:{HL}">stationary → cointegrated</span>']], size=11, h=100)
    pg.text("Residual-based critical values are stricter (2 series, 5 %: ≈ −3.34, not −2.86) because β is estimated.",
            ex, y0 + 292, 480, 30, 10, italic=True, color=RED)
    # spurious regression
    rsp = seeded(10)
    w1, w2 = [0.0], [0.0]
    for _ in range(n - 1):
        w1.append(w1[-1] + rsp.gauss(0, 1))
        w2.append(w2[-1] + rsp.gauss(0, 1))
    (asp, bsp), sesp, ressp, r2sp = ols([[1.0, a] for a in w1], w2)
    t_sp = adf(ressp, 1)[1]
    pg.text("Contrast: spurious regression", ex, y0 + 335, 480, 22, 14, bold=True)
    sp = pg.axes(ex + 20, y0 + 375, 200, 130, (min(w1) - 1, max(w1) + 1), (min(w2) - 1, max(w2) + 1), "w_1", "w_2",
                 label_size=11)
    sp.scatter([(w1[i], w2[i]) for i in range(0, n, 3)], r=2)
    sp.curve(lambda v: asp + bsp * v, n=2, curved=False, color=RED, width=1.5)
    pg.html(f"Two <b>independent</b> random walks:<br>β̂ = {bsp:.2f}, t = {bsp / sesp[1]:.1f}, R² = {r2sp:.2f}<br>"
            f"but ADF on residuals τ = {t_sp:.2f} → <font color='#FF0000'><b>not cointegrated</b></font><br>"
            "→ the 'relationship' is meaningless", ex + 240, y0 + 380, 240, 100, 11)
    pg.bullets(x0 + 20, y0 + 545, 1060, [
        "Both series must be I(1) (unit root in levels, stationary after one difference) before testing cointegration",
        "<b>Johansen test</b>: many series at once, finds the number of cointegrating vectors (trace / max-eigenvalue statistics)",
        "Pairs trading: trade the spread when it is ±2σ from its mean; exit when it reverts; re-estimate β on a rolling window",
        "Cointegration ≠ correlation: returns can be correlated without cointegration (and vice versa)"],
        head="Notes:")

    # ------------------------------------------------------------ ⑥ error correction model
    x0 = 1540
    pg.panel(x0, y0, 860, 720, 6, "Error Correction Model (ECM)", "Short-run dynamics + pull back to the long-run equilibrium")
    pg.math("\\Delta y_t = \\alpha + \\gamma\\,(y_{t-1} - \\beta x_{t-1}) + \\delta\\,\\Delta x_t + \\varepsilon_t",
            x0 + 60, y0 + 70, 700, 50, 22)
    pg.note("γ < 0 : speed of adjustment", x0 + 140, y0 + 135, x0 + 285, y0 + 108, w=180, color=RED)
    pg.note("error-correction term = yesterday's disequilibrium (long run)", x0 + 330, y0 + 140, x0 + 420,
            y0 + 112, w=250, color=BLUE)
    pg.note("short-run effect of x", x0 + 640, y0 + 135, x0 + 640, y0 + 110, w=140, color=GREEN)
    # simulate ECM response to a shock
    gam = -0.25
    dev = [0.0] * 5 + [4.0]
    for _ in range(30):
        dev.append(dev[-1] * (1 + gam))
    p = pg.axes(x0 + 70, y0 + 210, 470, 200, (0, len(dev) - 1), (-0.5, 4.5), "t", "y - \\beta x", label_size=11,
                origin=(0, -0.5), yticks=(0, 2, 4))
    p.hline(0, color=BLACK, dashed=True)
    p.poly([(i, dev[i]) for i in range(len(dev))], color=BLUE, width=2)
    p.scatter([(i, dev[i]) for i in range(5, len(dev), 3)], r=3)
    hl = math.log(0.5) / math.log(1 + gam)
    p.hline(2.0, 0, 5 + hl, color=RED)
    p.vline(5 + hl, -0.5, 2.0, color=RED)
    p.text("shock: disequilibrium = 4", 5, 4, dx=95, w=160, size=10)
    p.text(f"half of it gone after {hl:.1f} periods", 5 + hl, 2.0, dx=105, dy=-10, w=200, size=10, color=RED)
    p.text("long-run equilibrium y = βx", len(dev) - 1, 0, dx=-90, dy=-12, w=180, size=10)
    pg.math("\\text{half-life} = \\frac{\\ln 0.5}{\\ln(1+\\gamma)} = \\frac{\\ln 0.5}{\\ln 0.75} = "
            f"{hl:.2f}", x0 + 560, y0 + 230, 280, 50, 15)
    pg.text("γ = −0.25 → 25 % of the gap closes each period", x0 + 560, y0 + 290, 280, 34, 11, align="center",
            color=RED)
    pg.table(x0 + 560, y0 + 335, 280, [
        ["γ", "meaning"],
        ["−1 < γ < 0", "converges (cointegrated)"],
        ["γ = 0", "no correction (not cointegrated)"],
        ["γ > 0", "diverges (unstable)"]], size=10, h=90)
    pg.bullets(x0 + 20, y0 + 450, 820, [
        "<b>Granger representation theorem</b>: if x and y are cointegrated, an ECM exists (and vice versa)",
        "Estimate in 2 steps: (1) long-run β from the cointegrating regression, (2) OLS of Δy on û<sub>t−1</sub>, Δx<sub>t</sub> and lags",
        "All terms are stationary (differences + stationary spread) → standard t / F inference is valid",
        "Multivariate version: <b>VECM</b> (Vector ECM) after a Johansen test",
        "Trading: |γ| tells how fast a spread reverts; half-life sets the holding period / look-back window"],
        head="Notes:")

    # ------------------------------------------------------------ ⑦ residual & causality tests
    x0, y0 = 400, 2400
    pg.panel(x0, y0, 2000, 470, 7, "Residual & Causality Tests", "After fitting a model: is anything left in the residuals?")
    # Ljung-Box
    rl = seeded(31)
    wn2 = [rl.gauss(0, 1) for _ in range(300)]
    ar2 = ar1(300, 0.6, rl)
    pg.text("Ljung-Box", x0 + 30, y0 + 60, 440, 20, 14, bold=True)
    pg.math("Q = n(n+2)\\sum_{k=1}^{h}\\frac{\\hat{\\rho}_k^2}{n-k} \\sim \\chi^2_{h}", x0 + 30, y0 + 85, 440, 44, 15)
    for j, (dat, nm) in enumerate([(wn2, "white-noise residuals"), (ar2, "autocorrelated residuals")]):
        cx = x0 + 30 + j * 225
        rho = [acf(dat, k) for k in range(1, 11)]
        Q = 300 * 302 * sum(rh * rh / (300 - k) for k, rh in enumerate(rho, 1))
        a = pg.axes(cx + 10, y0 + 150, 180, 110, (0.3, 10.7), (-0.4, 0.8), "k", "", label_size=10,
                    origin=(0.3, 0), arrow=False)
        a.bars([(k, rh) for k, rh in enumerate(rho, 1)], 0.35, fill=BLUE if j == 0 else RED, stroke="none")
        band = 1.96 / math.sqrt(300)
        a.hline(band, color=GREY)
        a.hline(-band, color=GREY)
        ok = Q < 18.31
        pg.html(f"{nm}<br>Q(10) = <b>{Q:.1f}</b> {'&lt;' if ok else '&gt;'} 18.31 → "
                + ("no autocorrelation ✔" if ok else '<font color="#FF0000">autocorrelation ✘</font>'),
                cx, y0 + 275, 215, 50, 10)
    pg.text("H₀: ρ₁ = … = ρₕ = 0 (residuals are white noise)", x0 + 30, y0 + 335, 440, 18, 10, italic=True)
    # Durbin-Watson
    dx_ = x0 + 520
    pg.text("Durbin-Watson", dx_, y0 + 60, 420, 20, 14, bold=True)
    pg.math("DW = \\frac{\\sum_{t=2}^{n}(e_t-e_{t-1})^2}{\\sum_{t=1}^{n}e_t^2} \\approx 2(1-\\hat{\\rho}_1)", dx_,
            y0 + 85, 420, 50, 15)
    pg.line([(dx_ + 20, y0 + 200), (dx_ + 400, y0 + 200)], width=2)
    for v, lab, col in [(0, "0<br>positive<br>autocorr.", RED), (2, "2<br>no<br>autocorr.", GREEN),
                        (4, "4<br>negative<br>autocorr.", RED)]:
        px = dx_ + 20 + v / 4 * 380
        pg.line([(px, y0 + 192), (px, y0 + 208)], width=2)
        pg.html(f'<div style="text-align:center"><font color="{col}">{lab}</font></div>', px - 45, y0 + 212, 90,
                50, 10, align="center")
    pg.rect(dx_ + 20 + 1.5 / 4 * 380, y0 + 180, 0.25 * 380, 14, fill="#CCFFCC", stroke="none", opacity=80)
    pg.text("≈ 1.5 – 2.5 acceptable", dx_ + 150, y0 + 160, 140, 16, 10, align="center", color=GREEN)
    pg.text("Only lag-1; use Breusch-Godfrey for higher lags", dx_ + 20, y0 + 280, 400, 18, 10, italic=True)
    # ARCH-LM
    ax_ = x0 + 990
    pg.text("ARCH-LM (Engle)", ax_, y0 + 60, 440, 20, 14, bold=True)
    rg = seeded(5)
    ret, s2 = [], 0.2
    for _ in range(250):
        e = rg.gauss(0, math.sqrt(s2))
        ret.append(e)
        s2 = 0.05 + 0.25 * e * e + 0.7 * s2
    a = pg.axes(ax_ + 20, y0 + 95, 400, 80, (0, 249), (-max(map(abs, ret)), max(map(abs, ret))), "", "r_t",
                label_size=10, origin=(0, 0), arrow=False)
    a.poly([(i, ret[i]) for i in range(250)], color=BLUE, width=0.8)
    b = pg.axes(ax_ + 20, y0 + 195, 400, 70, (0, 249), (0, max(v * v for v in ret)), "t", "r_t^2", label_size=10,
                arrow=False)
    b.poly([(i, ret[i] ** 2) for i in range(250)], color=RED, width=0.8)
    pg.math("\\hat{e}_t^2 = a_0 + \\sum_{i=1}^{q} a_i \\hat{e}_{t-i}^2;\\quad LM = nR^2 \\sim \\chi^2_q", ax_,
            y0 + 285, 440, 40, 13)
    pg.text("Volatility clusters (big moves follow big moves) → H₀ 'no ARCH' rejected → use GARCH", ax_, y0 + 330,
            440, 30, 10, italic=True)
    # Granger causality
    gx = x0 + 1470
    pg.text("Granger Causality", gx, y0 + 60, 500, 20, 14, bold=True)
    for k in range(4):
        pg.ellipse(gx + 20 + k * 95, y0 + 100, 56, 30, f"x(t−{4 - k})" if k < 3 else "x(t)", size=10,
                   stroke=BLUE)
        pg.ellipse(gx + 20 + k * 95, y0 + 180, 56, 30, f"y(t−{4 - k})" if k < 3 else "y(t)", size=10,
                   stroke=ORANGE, sw=2 if k == 3 else 1)
    for k in range(3):
        pg.line([(gx + 48 + k * 95, y0 + 130), (gx + 48 + 3 * 95 - 5, y0 + 180)], color=BLUE, arrow="classic",
                dashed=True, width=0.8)
        pg.line([(gx + 76 + k * 95, y0 + 195), (gx + 20 + 3 * 95, y0 + 195)] if k == 2 else
                [(gx + 48 + k * 95, y0 + 210), (gx + 48 + 3 * 95, y0 + 210)], color=ORANGE, arrow="classic",
                width=0.8)
    pg.html("Restricted: \\(y_t = \\sum a_i y_{t-i} + e_t\\)<br>Unrestricted: \\(y_t = \\sum a_i y_{t-i} + "
            "\\sum b_i x_{t-i} + e_t\\)", gx, y0 + 240, 500, 44, 12)
    pg.math("F = \\frac{(RSS_R - RSS_U)/p}{RSS_U/(n-2p-1)}", gx + 40, y0 + 290, 360, 44, 14)
    pg.text("H₀: all bᵢ = 0 (x does not Granger-cause y). 'Predictive', not true causation; both series must be "
            "stationary.", gx, y0 + 340, 500, 30, 10, italic=True)
    asks(pg)
    pg.ask("Does knowing x's past improve the forecast of y?",
           "if the F-test rejects H₀ (all bᵢ = 0), lagged x adds predictive information → x Granger-causes y",
           gx, y0 + 382, gx + 20 + 3 * 95 + 28, y0 + 212, w=500)


def asks(pg):
    """Interpretation callouts for panels ①–⑥ (absolute positions)."""
    x0, y0 = 400, 240          # ① gallery
    pg.ask("Can I model this series as it is?",
           "only if mean & spread stay constant (AR(1) ✓); the five ✗ series need a transform first",
           x0 + 640, y0 + 566, x0 + 980, y0 + 470, w=430)
    x0, y0 = 1540, 240         # ② unit root
    pg.ask("Do shocks to this series ever fade away?",
           "φ = 0.5 yes (stationary); φ = 1 never (unit root); φ = 1.02 they grow (explosive)",
           x0 + 580, y0 + 355, x0 + 557, y0 + 357, w=240)
    x0, y0 = 400, 920          # ③ ADF
    pg.ask("Is this price series stationary?",
           "τ = −2.22 > −2.86 → cannot reject a unit root: difference it first",
           x0 + 968, y0 + 255, x0 + 890, y0 + 262, w=128)
    x0, y0 = 1540, 920         # ④ fixes
    pg.ask("How do I make a non-stationary series usable?",
           "apply the matching fix (difference / detrend / seasonal diff / log) until ADF says stationary",
           x0 + 525, y0 + 578, x0 + 545, y0 + 560, w=320)
    x0, y0 = 400, 1640         # ⑤ cointegration
    pg.ask("When do I enter and exit a pairs trade?",
           "spread above +2σ → short y / long x; below −2σ → reverse; exit at the mean (the spread reverts)",
           x0 + 610, y0 + 530, x0 + 477, y0 + 390, w=460)
    x0, y0 = 1540, 1640        # ⑥ ECM
    pg.ask("How fast does the pair return to equilibrium?",
           "γ = −0.25 → 25 % of the gap closes each period; half-life ≈ 2.4 periods = holding time",
           x0 + 560, y0 + 560, x0 + 800, y0 + 258, w=280)

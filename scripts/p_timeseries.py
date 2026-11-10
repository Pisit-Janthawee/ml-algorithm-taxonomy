"""AR / MA / ARMA page: stationarity, white noise & AR(1), ACF/PACF, model formulas, fitting, forecast."""
import math
from dio import *

N = 200


def _sim():
    r = seeded(21)
    e = [r.gauss(0, 1) for _ in range(N + 1)]
    ar = [0.0]
    for t in range(1, N):
        ar.append(0.8 * ar[-1] + e[t])
    ma = [e[t] + 0.7 * e[t - 1] for t in range(1, N + 1)]
    rw = [0.0]
    for t in range(1, N):
        rw.append(rw[-1] + e[t] + 0.08)
    return e[:N], ar, ma, rw


def _acf(x, k):
    m = sum(x) / len(x)
    d = sum((v - m) ** 2 for v in x)
    return sum((x[t] - m) * (x[t - k] - m) for t in range(k, len(x))) / d


def _pacf(x, K):
    """Durbin-Levinson."""
    r = [1.0] + [_acf(x, k) for k in range(1, K + 1)]
    phi, out = [], []
    for k in range(1, K + 1):
        if k == 1:
            new = [r[1]]
        else:
            num = r[k] - sum(phi[j] * r[k - 1 - j] for j in range(k - 1))
            den = 1 - sum(phi[j] * r[j + 1] for j in range(k - 1))
            pkk = num / den
            new = [phi[j] - pkk * phi[k - 2 - j] for j in range(k - 1)] + [pkk]
        phi = new
        out.append(phi[-1])
    return out


def _series(pg, x, y, w, h, data, yr, color=BLUE, label=None, mean=None, band=None, t0=0, xlabel="t"):
    p = pg.axes(x, y, w, h, (t0, t0 + len(data)), yr, xlabel, "", label_size=11,
                origin=(t0, yr[0]))
    p.poly([(t0 + i, max(yr[0], min(yr[1], v))) for i, v in enumerate(data)], color=color, width=1)
    if mean is not None:
        p.hline(mean, color=RED, dashed=True)
    if band is not None:
        p.hline(mean + band, color=GREY); p.hline(mean - band, color=GREY)
    return p


def _bars(pg, x, y, w, h, vals, title, n=N, color=BLUE):
    K = len(vals)
    p = pg.axes(x, y, w, h, (0, K + 1), (-1, 1), "\\text{lag}", "", label_size=10, origin=(0, 0),
                xticks=(1, 5, 10, 15), yticks=(-1, 1))
    ci = 1.96 / math.sqrt(n)
    p.hline(ci, color=RED); p.hline(-ci, color=RED)
    for k, v in enumerate(vals, 1):
        a, b = p.P(k, 0), p.P(k, v)
        pg.line([a, b], color=color if abs(v) > ci else GREY, width=3)
        pg.dot(*b, r=3, color=color if abs(v) > ci else GREY)
    pg.text(title, x, y - 24, w, 18, 12, bold=True, align="center")
    return p


def build(pg: Page, cat):
    e, ar, ma, rw = _sim()

    # ---------------------------------------------------------------- ① stationarity
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 640, 1, "Stationarity", "Background: constant mean, variance and autocorrelation over time")
    pg.text("✔ Stationary (AR(1), φ = 0.8)", x0 + 30, y0 + 70, 300, 18, 12, bold=True, color=GREEN)
    _series(pg, x0 + 60, y0 + 100, 520, 110, ar[:120], (-5, 5), mean=0, band=2 * 1.67)
    pg.note("constant mean", x0 + 480, y0 + 72, x0 + 520, y0 + 155, w=90, color=RED)
    pg.note("±2σ band", x0 + 40, y0 + 225, x0 + 120, y0 + 120, w=70, color=GREY)
    pg.text("✘ Non-stationary (random walk with drift)", x0 + 30, y0 + 250, 400, 18, 12, bold=True, color=RED)
    _series(pg, x0 + 60, y0 + 280, 520, 110, rw[:120], (-6, 22), color=ORANGE)
    pg.note("mean drifts upward → trend", x0 + 380, y0 + 270, x0 + 520, y0 + 300, w=170, color=ORANGE)
    pg.text("Fix: differencing  Δyₜ = yₜ − yₜ₋₁  → stationary", x0 + 30, y0 + 425, 500, 18, 12, bold=True)
    d = [rw[t] - rw[t - 1] for t in range(1, 120)]
    _series(pg, x0 + 60, y0 + 455, 520, 80, d, (-4, 4), color=GREEN, mean=0, label="\\Delta y_t")
    pg.bullets(x0 + 20, y0 + 560, 330, [
        "AR / MA / ARMA require a stationary series;  test with <b>ADF</b> (p &lt; 0.05 → stationary)",
        "Differencing d times = the 'I' in ARIMA(p, <b>d</b>, q);  prices → returns is the same idea"],
        head="Notes:", h=80)
    pg.ask("How do I make a trending price usable for ARMA?",
           "difference it: Δy keeps a constant mean (red line) → stationary; for prices use returns",
           x0 + 360, y0 + 540, x0 + 540, y0 + 500, w=270)

    # ---------------------------------------------------------------- ② white noise & AR(1)
    x0 = 1080
    pg.panel(x0, y0, 620, 640, 2, "White Noise → AR(1)", "Background: random shocks + memory of the past")
    pg.text("White noise εₜ ~ N(0, 1): no memory", x0 + 30, y0 + 70, 400, 18, 12, bold=True)
    _series(pg, x0 + 60, y0 + 100, 500, 90, e[:100], (-3.5, 3.5), color=GREY, mean=0, label="\\varepsilon_t")
    pg.text("AR(1): today = φ × yesterday + shock", x0 + 30, y0 + 215, 400, 18, 12, bold=True)
    _series(pg, x0 + 60, y0 + 245, 500, 90, ar[:100], (-5, 5), mean=0)
    pg.math("y_t = c + \\varphi\\,y_{t-1} + \\varepsilon_t", x0 + 30, y0 + 355, 300, 40, 18)
    pg.note("φ = 0.8: strong memory → smooth waves", x0 + 330, y0 + 360, x0 + 205, y0 + 375, w=230, color=BLUE)
    # lag plot
    p = pg.axes(x0 + 80, y0 + 420, 150, 150, (-5, 5), (-5, 5), "y_{t-1}", "y_t", label_size=11, origin=(-5, -5))
    p.scatter([(ar[t - 1], ar[t]) for t in range(1, 150) if abs(ar[t]) < 4.9 and abs(ar[t - 1]) < 4.9], r=2)
    p.curve(lambda v: 0.8 * v, -5, 5, n=2, curved=False, color=RED, width=1.5)
    pg.text("Lag plot: slope ≈ φ", x0 + 70, y0 + 585, 180, 18, 11, align="center", color=RED)
    pg.bullets(x0 + 280, y0 + 420, 320, [
        "\\(|\\varphi| \\lt 1\\) → stationary (shocks fade)",
        "\\(\\varphi = 1\\) → random walk (non-stationary)",
        "\\(\\varphi \\lt 0\\) → zig-zag series",
        "Mean of AR(1): \\(\\mu = \\frac{c}{1-\\varphi}\\)",
        "Variance: \\(\\frac{\\sigma^2}{1-\\varphi^2}\\)"], head="Rules of φ:", h=130)
    pg.ask("How strongly does today depend on yesterday?",
           "lag-plot slope ≈ φ = 0.8 → 80 % of yesterday's deviation carries over",
           x0 + 270, y0 + 555, *p.P(3, 2.4), w=330)

    # ---------------------------------------------------------------- ③ ACF / PACF
    x0 = 1740
    pg.panel(x0, y0, 660, 640, 3, "Autocorrelation: ACF & PACF", "Background: choose p and q from the correlogram")
    K = 15
    acf = [_acf(ar, k) for k in range(1, K + 1)]
    pacf = _pacf(ar, K)
    _bars(pg, x0 + 60, y0 + 110, 250, 150, acf, "ACF of AR(1): tails off (≈ 0.8ᵏ)")
    _bars(pg, x0 + 370, y0 + 110, 250, 150, pacf, "PACF of AR(1): cuts off after lag 1", color=ORANGE)
    pg.note("95% band ±1.96/√n", x0 + 450, y0 + 285, x0 + 600, y0 + 175, w=120, color=RED)
    pg.math("\\rho_k = \\frac{\\sum_t (y_t-\\bar{y})(y_{t-k}-\\bar{y})}{\\sum_t (y_t-\\bar{y})^2}", x0 + 30, y0 + 300,
            600, 54, 16)
    pg.text(f"Sample values: ρ₁ = {acf[0]:.2f}, ρ₂ = {acf[1]:.2f}, PACF₂ = {pacf[1]:.2f}", x0 + 30, y0 + 355, 600, 18,
            11, align="center")
    rows = [["Model", "ACF", "PACF"],
            ["<b>AR(p)</b>", "tails off gradually", "<b>cuts off after lag p</b>"],
            ["<b>MA(q)</b>", "<b>cuts off after lag q</b>", "tails off gradually"],
            ["<b>ARMA(p, q)</b>", "tails off", "tails off"]]
    pg.table(x0 + 60, y0 + 390, 540, rows, size=12, h=110)
    pg.ask("How many AR lags (p) should the model use?",
           f"p = 1: only the lag-1 PACF bar ({pacf[0]:.2f}) is outside the red band → AR(1)",
           x0 + 482, y0 + 375, x0 + 452, y0 + 424, w=170)
    pg.bullets(x0 + 20, y0 + 520, 620, [
        "ACF = correlation with lag k (including indirect effects through lags in between)",
        "PACF = direct correlation with lag k after removing lags 1 … k−1",
        "Here: PACF has one significant bar → choose <b>p = 1</b>"], head="Notes:")

    # ---------------------------------------------------------------- ④ AR / MA / ARMA formulas
    x0, y0 = 400, 920
    pg.panel(x0, y0, 640, 700, 4, "The Model Family: AR, MA, ARMA")
    pg.text("AR(p) — Autoregressive: regress on its own past values", x0 + 20, y0 + 55, 600, 20, 13, bold=True)
    pg.math("y_t = c + \\varphi_1 y_{t-1} + \\varphi_2 y_{t-2} + \\dots + \\varphi_p y_{t-p} + \\varepsilon_t", x0 + 20,
            y0 + 80, 600, 40, 17)
    pg.note("past values (lags)", x0 + 160, y0 + 125, x0 + 230, y0 + 105, w=110, color=BLUE)
    pg.note("today's shock", x0 + 480, y0 + 125, x0 + 560, y0 + 105, w=90, color=GREY)
    pg.text("MA(q) — Moving Average: regress on past forecast errors (shocks)", x0 + 20, y0 + 160, 600, 20, 13,
            bold=True)
    pg.math("y_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}", x0 + 20,
            y0 + 185, 600, 40, 17)
    pg.note("past shocks", x0 + 330, y0 + 230, x0 + 330, y0 + 210, w=80, color=ORANGE)
    _series(pg, x0 + 60, y0 + 270, 520, 80, ma[:120], (-4, 4), color=ORANGE, mean=0)
    pg.text("MA(1), θ = 0.7: shock affects only 1 step ahead (short memory)", x0 + 60, y0 + 360, 520, 18, 11,
            italic=True, color=ORANGE, align="center")
    pg.text("ARMA(p, q) — both together", x0 + 20, y0 + 395, 600, 20, 13, bold=True)
    pg.html(f'<span style="background:{HL}">$$y_t = c + \\sum_{{i=1}}^{{p}}\\varphi_i y_{{t-i}} + \\varepsilon_t + '
            f'\\sum_{{j=1}}^{{q}}\\theta_j\\varepsilon_{{t-j}}$$</span>', x0 + 20, y0 + 420, 600, 56, 17,
            align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 495, 380, [
        "\\(\\varphi_i\\) = AR coefficients, \\(\\theta_j\\) = MA coefficients, \\(\\varepsilon_t \\sim N(0,\\sigma^2)\\) white noise",
        "AR = long memory that fades; MA = shocks that last exactly q steps",
        "ARMA uses fewer parameters than a long pure AR or MA model",
        "Extensions: <b>ARIMA</b> (+ differencing), <b>SARIMA</b> (+ seasonality), <b>VAR</b> (many series), GARCH (volatility)"],
        head="Where:", h=150)
    pg.ask("Is the series driven by its past values or by past shocks?",
           "AR terms = memory of past values (fades slowly); MA terms = past surprises lasting q steps; ARMA mixes both",
           x0 + 410, y0 + 492, x0 + 520, y0 + 470, w=215)

    # ---------------------------------------------------------------- ⑤ fitting
    x0 = 1080
    pg.panel(x0, y0, 620, 700, 5, "Fitting & Diagnostics", "Background: least squares / maximum likelihood")
    ys, yl = ar[1:], ar[:-1]
    m = sum(ar) / len(ar)
    phi = sum((a - m) * (b - m) for a, b in zip(ys, yl)) / sum((b - m) ** 2 for b in yl)
    pg.text("AR(1) is just linear regression of yₜ on yₜ₋₁:", x0 + 20, y0 + 60, 580, 20, 12, bold=True)
    pg.math("\\hat{\\varphi} = \\frac{\\sum_t (y_t-\\bar{y})(y_{t-1}-\\bar{y})}{\\sum_t (y_{t-1}-\\bar{y})^2}", x0 + 20,
            y0 + 85, 580, 56, 17)
    pg.html(f'<span style="background:{HL}">\\(\\hat{{\\varphi}} = {phi:.3f}\\)</span>  (true φ = 0.8, n = {N})',
            x0 + 20, y0 + 145, 580, 26, 14, align="center", valign="middle")
    pg.text("General ARMA: maximize the likelihood (MLE) numerically; choose p, q by lowest AIC / BIC",
            x0 + 20, y0 + 180, 580, 20, 11, align="center")
    pg.math("AIC = 2k - 2\\ln(\\hat{L})", x0 + 20, y0 + 205, 580, 32, 15)
    res = [ys[i] - m - phi * (yl[i] - m) for i in range(len(ys))]
    _bars(pg, x0 + 90, y0 + 290, 420, 140, [_acf(res, k) for k in range(1, 16)],
          "Residual ACF: all bars inside the band → white noise ✔", color=GREEN)
    pg.bullets(x0 + 20, y0 + 470, 330, [
        "Residuals must look like white noise — otherwise the model missed structure",
        "<b>Ljung-Box</b> test: p &gt; 0.05 → no autocorrelation left in residuals",
        "Check residuals are roughly normal (histogram / Q-Q plot, see Distributions page)",
        "Workflow (Box-Jenkins): identify → estimate → diagnose → forecast"], head="Notes:", h=150)
    pg.ask("Did the model capture all the structure?",
           f"yes: no residual ACF bar leaves the band → white noise; φ̂ = {phi:.3f} is close to the true 0.8",
           x0 + 365, y0 + 472, x0 + 460, y0 + 368, w=235)

    # ---------------------------------------------------------------- ⑥ forecast
    x0 = 1740
    pg.panel(x0, y0, 660, 700, 6, "Forecast with Confidence Interval")
    hist = ar[140:200]
    H = 25
    yT = hist[-1]
    fc = [m + phi ** h * (yT - m) for h in range(1, H + 1)]
    sd = [math.sqrt(sum(phi ** (2 * j) for j in range(h))) for h in range(1, H + 1)]
    p = pg.axes(x0 + 70, y0 + 90, 520, 260, (0, 60 + H), (-6, 6), "t", "y_t", label_size=12, origin=(0, -6),
                xticks=(0, 20, 40, 60, 80))
    p.poly([(i, max(-6, min(6, v))) for i, v in enumerate(hist)], color=BLUE, width=1.2)
    tt = [59 + h for h in range(0, H + 1)]
    up = [yT] + [f + 1.96 * s for f, s in zip(fc, sd)]
    lo = [yT] + [f - 1.96 * s for f, s in zip(fc, sd)]
    for i in range(H):
        a, b = p.P(tt[i], up[i + 1]), p.P(tt[i] + 1, lo[i + 1])
        pg.rect(a[0], a[1], b[0] - a[0], b[1] - a[1], fill=LIGHTBLUE, stroke="none", opacity=60)
    p.poly(list(zip(tt, up)), color=GREY, dashed=True)
    p.poly(list(zip(tt, lo)), color=GREY, dashed=True)
    p.poly(list(zip(tt, [yT] + fc)), color=RED, width=2)
    p.vline(59.5, color=BLACK)
    p.hline(m, color=GREEN)
    p.text("history", 30, 5.4, w=80, size=11, color=BLUE)
    p.text("forecast", 72, 5.4, w=80, size=11, color=RED)
    pg.note("95% interval widens with horizon h", x0 + 440, y0 + 58, *p.P(78, up[18]), w=210, color=GREY)
    pg.note("forecast decays to the mean μ", x0 + 400, y0 + 362, *p.P(84, m), w=190, color=GREEN)
    pg.math("\\hat{y}_{T+h} = \\mu + \\varphi^{h}\\,(y_T - \\mu)", x0 + 30, y0 + 410, 600, 40, 18)
    pg.math("Var(e_{T+h}) = \\sigma^2\\sum_{j=0}^{h-1}\\varphi^{2j}", x0 + 30, y0 + 455, 600, 50, 16)
    pg.bullets(x0 + 20, y0 + 520, 340, [
        "ARMA forecasts are good for a few steps, then fall back to the long-run mean",
        "Uncertainty grows with every step → always show the interval, not only the line",
        "Baseline for your LSTM: if LSTM can't beat AR on returns, it isn't learning real signal"],
        head="Notes:", h=130)
    pg.ask("How far ahead can I trust the forecast?",
           f"a few steps: the 95 % band widens from ±{1.96 * sd[0]:.2f} to ±{1.96 * sd[-1]:.2f}σ by h = {H} "
           "while the line decays to μ",
           x0 + 375, y0 + 515, *p.P(80, up[21]), w=265)

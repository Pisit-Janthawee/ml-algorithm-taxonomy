"""Extra panels for the AR / MA / ARMA page: time step, lag window, decomposition, ACF by hand."""
import math
from dio import *


def sbars(pg, p, data, width, fill=BLUE, stroke=BLACK):
    """Signed bars (supports negative heights)."""
    for xc, h in data:
        xa, ya = p.P(xc - width / 2, max(h, 0))
        xb, yb = p.P(xc + width / 2, min(h, 0))
        pg.rect(xa, ya, xb - xa, max(yb - ya, 1), fill=fill, stroke=stroke)


def build(pg: Page, cat):
    r = seeded(21)
    Y0 = pg.max_y() + 60

    # ------------------------------------------------------------ ⑦ time step in tabular data
    x0, y0 = 400, Y0
    pg.panel(x0, y0, 980, 700, 7, "What is a Time Step? (time series as tabular data)",
             "Timeframe = 1h candles · Period = 1 week · each row = one time step t")
    # simulate 1h candles
    price, rows, candles = 1950.0, [], []
    for i in range(16):
        o = price
        c = o + r.gauss(0.4, 3.0)
        hi = max(o, c) + abs(r.gauss(0, 1.5))
        lo = min(o, c) - abs(r.gauss(0, 1.5))
        candles.append((o, hi, lo, c))
        price = c
    tab = [["t", "Timestamp", "Open", "High", "Low", "Close", "Volume"]]
    for i in range(8):
        o, hi, lo, c = candles[i]
        tab.append([f"<b>{i + 1}</b>", f"2026-11-02 {9 + i:02d}:00", f"{o:.1f}", f"{hi:.1f}", f"{lo:.1f}",
                    f"{c:.1f}", f"{r.randint(800, 2400)}"])
    tab.append(["…", "…", "…", "…", "…", "…", "…"])
    tab.append(["<b>120</b>", "2026-11-06 16:00", "", "", "", "", ""])
    pg.table(x0 + 20, y0 + 70, 470, tab, size=10, highlight_rows=(4,))
    pg.note("one row = one time step (t = 4)", x0 + 20, y0 + 340, x0 + 120, y0 + 158, w=200, color=RED)
    # candle chart
    lo_all = min(c[2] for c in candles) - 2
    hi_all = max(c[1] for c in candles) + 2
    p = pg.axes(x0 + 560, y0 + 90, 380, 220, (0, 17), (lo_all, hi_all), "t", "Price", label_size=12,
                xticks=(1, 4, 8, 12, 16))
    for i, (o, hi, lo, c) in enumerate(candles):
        t = i + 1
        col = GREEN if c >= o else RED
        p.poly([(t, lo), (t, hi)], color=BLACK, width=1)
        xa, ya = p.P(t - 0.3, max(o, c))
        xb, yb = p.P(t + 0.3, min(o, c))
        pg.rect(xa, ya, xb - xa, max(yb - ya, 2), fill=col, stroke=BLACK, opacity=None if i != 3 else None,
                sw=2 if t == 4 else 1)
    p.vline(4, color=RED)
    p.text("t = 4", 4, hi_all, dy=-8, size=10, color=RED)
    # timeline with timeframe / period / horizon
    ty = y0 + 400
    pg.line([(x0 + 40, ty), (x0 + 940, ty)], arrow="classic")
    pg.text("time →", x0 + 900, ty + 6, 60, 16, 10, color="#555555")
    step = 50
    for k in range(17):
        tx = x0 + 60 + k * step
        pg.line([(tx, ty - 5), (tx, ty + 5)])
    pg.rect(x0 + 60 + 9 * step, ty - 14, step, 28, fill=LIGHTBLUE, stroke=BLUE)
    pg.note("timeframe (bar size) = 1 hour", x0 + 470, ty + 30, x0 + 60 + 9.5 * step, ty + 12, w=190, color=BLUE)
    bcx, bcy = x0 + 60 + 7.5 * step, ty - 28
    pg.bracket(bcx - 6, bcy - 7.5 * step, 12, 15 * step, rotation=90)
    pg.text("period / lookback window (e.g. 1 week = 120 hourly bars)", x0 + 60 + 2.5 * step, ty - 62, 500, 18, 11,
            bold=True, align="center")
    pg.line([(x0 + 60 + 15 * step, ty - 30), (x0 + 60 + 16.4 * step, ty - 30)], arrow="classic", color=RED,
            width=2)
    pg.text("horizon h (predict ahead)", x0 + 60 + 13.3 * step, ty - 52, 170, 16, 10, color=RED, align="center")
    pg.dot(x0 + 60 + 15 * step, ty, r=5, color=RED)
    pg.text("now (t)", x0 + 60 + 14.4 * step, ty + 8, 60, 16, 10, color=RED, align="center")
    pg.ask("Which rows may the model use to predict the target?",
           "only rows up to now (t), inside the lookback period; the target is h steps ahead — never shuffle rows",
           x0 + 640, y0 + 470, x0 + 60 + 16.2 * step, ty - 32, w=320)
    pg.bullets(x0 + 20, y0 + 470, 600, [
        "<b>Time step t</b> = one observation (one row / one candle). Rows are <b>ordered</b> — shuffling them destroys the information",
        "<b>Timeframe / interval</b> = size of one bar (1m, 5m, 1h, 1D). In yfinance: <i>interval=\"1h\"</i>",
        "<b>Period</b> = how much history you download / look back (1wk, 1mo, 1y). In yfinance: <i>period=\"1wk\"</i>  → 1 week of 1h bars ≈ 5 days × 24 = 120 rows (FX)",
        "<b>Horizon h</b> = how many steps ahead the target is (h = 1 → next bar's close)",
        "Unlike normal tabular data, rows are <b>not independent</b>: y<sub>t</sub> depends on y<sub>t−1</sub>, y<sub>t−2</sub>, … → we must build lag features ⑧"],
        head="Definitions:", h=190)

    # ------------------------------------------------------------ ⑧ lag window -> supervised learning
    x0 = 1420
    pg.panel(x0, y0, 980, 700, 8, "From Time Steps to Lagged Features (sliding window, N = 3)",
             "How a sequential model 'sees' the past N values")
    ys = [10, 12, 13, 12, 15, 16, 18, 17]
    bx, by, bw = x0 + 60, y0 + 90, 70
    cols = [ORANGE, GREEN, BLUE]
    for i, v in enumerate(ys):
        pg.rect(bx + i * bw, by, bw - 6, 40, str(v), size=14, bold=True)
        pg.text(f"t={i + 1}", bx + i * bw, by + 42, bw - 6, 14, 9, align="center", color="#555555")
    for k, c in enumerate(cols):      # three windows
        sx = bx + k * bw - 3
        pg.rect(sx, by - 12 - k * 10, 3 * bw, 64 + k * 20, stroke=c, dashed=True, sw=1.5)
        pg.line([(bx + (k + 3) * bw + 29, by - 14 - k * 10), (bx + (k + 3) * bw + 29, by - 2)], color=c,
                arrow="classic", width=1.5)
    pg.text("window of N = 3 past values → target (next value); the window slides one step at a time →", bx, by + 62, 600, 16, 11, bold=True)
    tab = [["sample", "\\(y_{t-3}\\)", "\\(y_{t-2}\\)", "\\(y_{t-1}\\)", "target \\(y_t\\)"]]
    for t in range(3, len(ys)):
        tab.append([f"t={t + 1}", ys[t - 3], ys[t - 2], ys[t - 1], f"<b>{ys[t]}</b>"])
    pg.table(x0 + 40, y0 + 190, 400, tab, size=11, h=150)
    pg.note("diagonal: the same value shifts left each row (12 → 12 → 12)", x0 + 40, y0 + 350, x0 + 300,
            y0 + 255, w=380, color=RED)
    pg.text("X: 5 samples × 3 features    y: 5 targets", x0 + 40, y0 + 375, 400, 16, 11, italic=True)
    # tensor shape
    tx, tyy = x0 + 500, y0 + 200
    for k in range(3):
        pg.rect(tx + k * 10, tyy - k * 10, 120, 90, fill="#FFFFFF" if k < 2 else LIGHTBLUE, stroke=BLUE)
    pg.text("3-D input for LSTM / RNN", tx - 10, tyy + 95, 200, 16, 11, bold=True, align="center")
    pg.math("(\\text{samples},\\ N\\ \\text{time steps},\\ \\text{features}) = (5, 3, 1)", tx - 30, tyy + 112,
            300, 28, 12)
    pg.text("features > 1 when each step has e.g. Close, Volume, RSI", tx - 10, tyy + 140, 260, 30, 10,
            color="#555555", align="center")
    pg.ask("How many past values does the model see per prediction?",
           f"N = 3 lags per sample → input shape ({len(ys) - 3}, 3, 1); pick N from the PACF / AIC / validation",
           x0 + 680, y0 + 190, tx + 140, tyy + 40, w=280)
    # unrolled RNN
    ry = y0 + 440
    pg.text("Unrolled RNN / LSTM over the window", x0 + 40, ry - 30, 400, 18, 12, bold=True)
    prev = None
    for k, lab in enumerate(["y_{t-3}", "y_{t-2}", "y_{t-1}"]):
        cx = x0 + 80 + k * 170
        cell = pg.rect(cx, ry + 10, 100, 50, "LSTM cell", fill="#E1D5E7", stroke="#9673A6", size=11)
        pg.math(lab, cx + 20, ry + 95, 60, 24, 13)
        pg.line([(cx + 50, ry + 95), (cx + 50, ry + 60)], arrow="classic")
        if prev:
            pg.line([(cx - 70, ry + 35), (cx, ry + 35)], arrow="classic", color=BLUE)
            pg.math("h", cx - 48, ry + 12, 24, 20, 12, color=BLUE)
        prev = cell
    pg.line([(x0 + 520, ry + 35), (x0 + 580, ry + 35)], arrow="classic", color=RED, width=2)
    pg.math("\\hat{y}_t", x0 + 585, ry + 22, 50, 26, 16, color=RED)
    pg.text("hidden state h carries memory of earlier steps", x0 + 200, ry - 8, 300, 16, 10, color=BLUE)
    pg.math("AR(3):\\ \\hat{y}_t=c+\\phi_1y_{t-1}+\\phi_2y_{t-2}+\\phi_3y_{t-3}", x0 + 640, ry + 5, 330, 40, 13)
    pg.text("= the linear version of the same lag window", x0 + 650, ry + 45, 320, 16, 10, italic=True,
            align="center")
    pg.bullets(x0 + 20, y0 + 580, 940, [
        "Choose N with the <b>PACF</b> (last significant lag), information criteria (<b>AIC/BIC</b>) or validation error",
        "Split train / test <b>by time</b> (never shuffle) and fit scalers on the training part only — otherwise look-ahead leakage",
        "Longer N = more memory but fewer samples and more parameters; LSTM can learn longer dependencies than AR"],
        head="Choosing the lag N:", h=80)

    # ------------------------------------------------------------ ⑨ decomposition
    x0, y0 = 400, Y0 + 740
    pg.panel(x0, y0, 980, 660, 9, "Time Series Decomposition", "Series = Trend + Seasonality + Residual")
    n = 120
    trend = [50 + 0.15 * t for t in range(n)]
    seas = [5 * math.sin(2 * math.pi * t / 12) for t in range(n)]
    res = [r.gauss(0, 1.5) for _ in range(n)]
    ser = [a + b + c for a, b, c in zip(trend, seas, res)]
    comps = [("Observed  \\(y_t\\)", ser, BLACK, (40, 80)), ("Trend  \\(T_t\\)", trend, BLUE, (40, 80)),
             ("Seasonal  \\(S_t\\)  (period s = 12)", seas, GREEN, (-7, 7)), ("Residual  \\(R_t\\)", res, RED, (-5, 5))]
    for k, (lab, data, col, yr) in enumerate(comps):
        py = y0 + 70 + k * 120
        p = pg.axes(x0 + 60, py, 560, 85, (0, n), yr, "", "", arrow=False, origin=(0, yr[0]))
        if yr[0] < 0: p.hline(0)
        p.poly([(t, v) for t, v in enumerate(data)], color=col, width=1.3)
        pg.html(lab, x0 + 70, py - 16, 300, 20, 12)
        if k < 3:
            pg.text("=" if k == 0 else "+", x0 + 330, py + 92, 30, 20, 16, bold=True, align="center")
    pg.math("\\text{Additive: } y_t = T_t + S_t + R_t", x0 + 640, y0 + 70, 320, 30, 13)
    pg.math("\\text{Multiplicative: } y_t = T_t \\times S_t \\times R_t", x0 + 640, y0 + 110, 330, 30, 13)
    pg.bullets(x0 + 630, y0 + 160, 340, [
        "Additive: seasonal swings constant in size",
        "Multiplicative: swings grow with the level → log turns it additive",
        "Trend by moving average (window = s); seasonal = average deviation per position in the cycle; residual = rest",
        "STL (LOESS) is robust and lets the season change slowly",
        "Residual should look like white noise → model it (ARMA) or use it for anomaly detection"],
        head="How it works:", size=10, h=210)
    pg.ask("Is there a repeating pattern, and how long is its cycle?",
           "yes: the seasonal part repeats every s = 12 steps (±5); the trend adds +0.15 per step; the rest is noise",
           x0 + 640, y0 + 400, x0 + 620, y0 + 352, w=320)

    # ------------------------------------------------------------ ⑩ ACF by hand + reading guide
    x0 = 1420
    pg.panel(x0, y0, 980, 660, 10, "ACF & PACF by hand + how to read a correlogram")
    pg.math("r_k = \\frac{\\sum_{t=k+1}^{n}(y_t-\\bar{y})(y_{t-k}-\\bar{y})}{\\sum_{t=1}^{n}(y_t-\\bar{y})^2}", x0 + 30,
            y0 + 45, 420, 60, 16)
    pg.note("correlation of the series with itself shifted k steps", x0 + 450, y0 + 60, x0 + 360, y0 + 75,
            w=210)
    ys = [2, 4, 6, 5, 3, 4]
    m = sum(ys) / len(ys)
    d = [v - m for v in ys]
    tab = [["t", "\\(y_t\\)", "\\(d_t=y_t-\\bar{y}\\)", "\\(d_t^2\\)", "\\(d_t d_{t-1}\\)", "\\(d_t d_{t-2}\\)"]]
    for i in range(6):
        tab.append([i + 1, ys[i], f"{d[i]:g}", f"{d[i] ** 2:g}", f"{d[i] * d[i - 1]:g}" if i >= 1 else "",
                    f"{d[i] * d[i - 2]:g}" if i >= 2 else ""])
        tab[-1] = [str(c).replace("-0", "0") if str(c) == "-0" else c for c in tab[-1]]
    s2 = sum(v * v for v in d)
    s1 = sum(d[i] * d[i - 1] for i in range(1, 6))
    s_2 = sum(d[i] * d[i - 2] for i in range(2, 6))
    tab.append(["", f"\\(\\bar{{y}}={m:g}\\)", "Σ", f"<b>{s2:g}</b>", f"<b>{s1:g}</b>", f"<b>{s_2:g}</b>"])
    pg.table(x0 + 30, y0 + 115, 470, tab, size=11, h=190, highlight_rows=(7,))
    pg.html(f'\\(r_1 = {s1:g}/{s2:g} = {s1 / s2:.1f}\\) &nbsp;&nbsp; '
            f'<span style="background:{HL}">\\(r_2 = {s_2:g}/{s2:g} = {s_2 / s2:.1f}\\)</span>',
            x0 + 30, y0 + 315, 470, 30, 14, align="center", valign="middle")
    pg.text("PACF(k) = correlation at lag k after removing the effect of lags 1…k−1 (last coefficient of an AR(k) fit)",
            x0 + 30, y0 + 350, 470, 34, 11, italic=True)
    pg.ask("Is the series correlated with its own past?",
           f"r₁ = {s1 / s2:.1f} (almost none), r₂ = {s_2 / s2:.1f}: values 2 steps apart tend to move in opposite directions",
           x0 + 30, y0 + 392, x0 + 330, y0 + 340, w=250)
    # reading guide: 4 mini correlograms
    guides = [("AR(1): ACF decays, PACF cuts off after 1", [0.8 ** k for k in range(1, 9)], "ACF"),
              ("MA(1): ACF cuts off after lag 1", [0.5] + [0.03 * (-1) ** k for k in range(7)], "ACF"),
              ("Seasonal (s = 4): spikes at 4, 8", [0.1, -0.05, 0.1, 0.7, 0.05, -0.05, 0.08, 0.5], "ACF"),
              ("Non-stationary: very slow decay", [0.97 ** k for k in range(1, 9)], "ACF")]
    for k, (lab, vals, kind) in enumerate(guides):
        gx = x0 + 530 + (k % 2) * 225
        gy = y0 + 70 + (k // 2) * 170
        p = pg.axes(gx, gy + 20, 190, 100, (0, 9), (-0.4, 1), "", "", arrow=False, origin=(0, 0))
        p.hline(0.3, color=BLUE)
        p.hline(-0.3, color=BLUE)
        sbars(pg, p, [(i + 1, v) for i, v in enumerate(vals)], 0.45, fill=BLUE if k != 3 else RED)
        pg.text(lab, gx - 5, gy - 2, 210, 30, 10, bold=True, align="center")
    pg.text("dashed blue = 95 % band ±1.96/√n", x0 + 560, y0 + 410, 380, 16, 10, color=BLUE, align="center")
    pg.table(x0 + 30, y0 + 445, 920, [
        ["Model", "ACF", "PACF"],
        ["AR(p)", "tails off gradually", "<b>cuts off after lag p</b>"],
        ["MA(q)", "<b>cuts off after lag q</b>", "tails off gradually"],
        ["ARMA(p, q)", "tails off", "tails off"],
        ["Non-stationary", "very slow linear decay → difference first", "lag-1 ≈ 1"]], size=11, h=125)
    pg.text("Two series that are each non-stationary but move together → Cointegration & ECM "
            "(see Time Series Hypothesis Tests page)", x0 + 30, y0 + 600, 920, 30, 11, italic=True, color=BLUE)

"""Time Series Data Labelling page (base: Fixed-Time Horizon; Triple Barrier Method in detail)."""
import math
from dio import *


def price_path(n=220, seed=11, vol=0.01, p0=100.0):
    r = seeded(seed)
    p, out = p0, [p0]
    for i in range(n - 1):
        v = vol * (1.8 if 120 < i < 160 else 1.0)          # a volatile regime in the middle
        p *= math.exp(r.gauss(0.0004, v))
        out.append(p)
    return out


def ewma_vol(prices, span=20):
    a = 2 / (span + 1)
    rets = [0.0] + [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    var, out = rets[1] ** 2, [abs(rets[1])]
    for i in range(1, len(rets)):
        var = (1 - a) * var + a * rets[i] ** 2
        out.append(math.sqrt(var))
    return out


P = price_path()
SIG = ewma_vol(P)


def triple_barrier(t0, pt=1.0, sl=1.0, t1=15):
    """First touch: returns (touch_index, label, upper, lower). Barrier width = σ_t·√h (h-bar volatility)."""
    s = SIG[t0] * math.sqrt(t1)
    up, lo = P[t0] * (1 + pt * s), P[t0] * (1 - sl * s)
    for i in range(t0 + 1, min(t0 + t1, len(P) - 1) + 1):
        if pt and P[i] >= up: return i, 1, up, lo
        if sl and P[i] <= lo: return i, -1, up, lo
    end = min(t0 + t1, len(P) - 1)
    return end, 0, up, lo


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① look-ahead bias + tabular example
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 620, 1, "Labels come from the FUTURE: avoid look-ahead bias",
             "Features use only the past window; the label uses the next h bars")
    ty = y0 + 120
    pg.line([(x0 + 30, ty), (x0 + 610, ty)], arrow="classic", width=1.5)
    pg.math("time", x0 + 600, ty + 4, 40, 20, 11)
    tx = x0 + 330
    pg.line([(tx, ty - 50), (tx, ty + 30)], color=RED, width=2)
    pg.math("t", tx - 10, ty + 30, 20, 20, 14, color=RED)
    pg.rect(x0 + 80, ty - 40, 245, 32, "past window  t−w … t  →  FEATURES X", fill=LIGHTBLUE, stroke=BLUE, size=11)
    pg.rect(x0 + 335, ty - 40, 200, 32, "t+1 … t+h  →  LABEL y", fill="#FFE6CC", stroke=ORANGE, size=11)
    pg.text("✘ no peeking: a feature must never use data after t", x0 + 80, ty + 50, 520, 20, 12, bold=True,
            color=RED, align="center")
    # tabular example (timeframe 1h, h = 3 bars, τ = 0.3 %)
    pg.text("Tabular example — timeframe 1h, horizon h = 3 bars, threshold τ = 0.3 %", x0 + 20, y0 + 210, 600, 20,
            12, bold=True)
    rows = [["time (1h)", "close", "return", "EMA(3)", "\\(r_{t,t+3}\\)", "label y"]]
    base = 40
    ema = P[base - 1]
    hl = ()
    for k in range(7):
        i = base + k
        ema = 0.5 * P[i] + 0.5 * ema
        ret = P[i] / P[i - 1] - 1
        fwd = P[i + 3] / P[i] - 1
        lab = 1 if fwd > 0.003 else (-1 if fwd < -0.003 else 0)
        col = {1: "#009900", -1: "#FF0000", 0: "#555555"}[lab]
        rows.append([f"{9 + k:02d}:00", f"{P[i]:.2f}", f"{ret * 100:+.2f}%",
                     f"{ema:.2f}", f"{fwd * 100:+.2f}%", f'<b><font color="{col}">{lab:+d}</font></b>' if lab else '<b>0</b>'])
    pg.table(x0 + 20, y0 + 238, 600, rows, size=11, highlight_rows=(2,))
    pg.note("label of 10:00 uses the closes of 11:00, 12:00 and 13:00 (future!)", x0 + 465, y0 + 290, x0 + 447,
            y0 + 313, w=165, color=ORANGE)
    pg.bullets(x0 + 20, y0 + 485, 600, [
        "<b>Timeframe</b> = bar size (1m, 1h, 1d); <b>period</b> = how much history (e.g. 1 week of 1m bars)",
        "Each row = one time step t; features: values up to t; label: outcome over (t, t+h]",
        "The last h rows have no label → drop them (they would need future data)",
        "Never shuffle before splitting: train on the past, test on the future (see ⑥)"], head="Notes:")

    # ---------------------------------------------------------------- ② sampling: bars & CUSUM
    x0 = 1080
    pg.panel(x0, y0, 560, 620, 2, "Sampling: which rows to label?", "Bars & the CUSUM event filter")
    r = seeded(5)
    vol = [1 + 3 * math.exp(-((i - 30) / 8) ** 2) + 0.4 * r.random() for i in range(60)]
    bx, bw = x0 + 70, 440
    p = Plot(pg, bx, y0 + 70, bw, 60, (0, 60), (0, 5))
    p.bars([(i + 0.5, v) for i, v in enumerate(vol)], 0.8, fill="#DDDDDD", stroke="#999999")
    pg.text("volume", x0 + 14, y0 + 90, 56, 18, 10, color="#555555")
    pg.text("time bars", x0 + 14, y0 + 140, 56, 18, 10, bold=True)
    for i in range(0, 61, 5):
        px, _ = p.P(i, 0)
        pg.line([(px, y0 + 140), (px, y0 + 158)], color=BLUE, width=1.5)
    pg.text("dollar bars", x0 + 10, y0 + 170, 60, 18, 10, bold=True)
    acc, cum = 0, []
    for i, v in enumerate(vol):
        acc += v
        if acc >= 5:
            acc -= 5
            cum.append(i + 1)
    for i in cum:
        px, _ = p.P(i, 0)
        pg.line([(px, y0 + 170), (px, y0 + 188)], color=RED, width=1.5)
    pg.text("every 1 h (uniform, ignores activity)  vs  every $X traded (more bars when the market is busy)",
            x0 + 20, y0 + 195, 520, 30, 10, italic=True, color="#555555", align="center")
    # CUSUM
    pg.text("CUSUM filter: sample an event when the cumulative move exceeds h", x0 + 20, y0 + 235, 520, 20, 12,
            bold=True)
    seg = P[0:120]
    q = pg.axes(x0 + 60, y0 + 270, 460, 150, (0, 120), (min(seg) * 0.995, max(seg) * 1.005), "t", "P", label_size=11)
    q.poly(list(enumerate(seg)), color=BLACK, width=1)
    h, sp, sn, events = 0.025, 0, 0, []
    for i in range(1, len(seg)):
        d = math.log(seg[i] / seg[i - 1])
        sp, sn = max(0, sp + d), min(0, sn + d)
        if sp > h:
            events.append((i, 1)); sp = 0
        elif sn < -h:
            events.append((i, -1)); sn = 0
    for i, s in events:
        q.scatter([(i, seg[i])], color=GREEN if s > 0 else RED, r=4)
    pg.text(f"{len(events)} events (green: up-move, red: down-move) instead of 120 bars", x0 + 60, y0 + 425, 460, 18,
            10, align="center", color="#555555")
    pg.math("S^+_t = \\max(0,\\ S^+_{t-1} + \\Delta y_t)", x0 + 20, y0 + 450, 260, 30, 13)
    pg.math("S^-_t = \\min(0,\\ S^-_{t-1} + \\Delta y_t)", x0 + 280, y0 + 450, 260, 30, 13)
    pg.text("event if S⁺ > h or S⁻ < −h, then reset to 0   (here h = 2.5 % log-move)", x0 + 20, y0 + 482, 520, 18,
            11, align="center")
    pg.bullets(x0 + 20, y0 + 510, 520, [
        "Fewer, more informative samples; labels start at events t₀",
        "Dollar/volume bars have returns closer to i.i.d. normal than time bars",
        "h is often set from volatility (e.g. h = daily σ)"], head="Why:")

    # ---------------------------------------------------------------- ③ fixed-time horizon (base)
    x0 = 1680
    pg.panel(x0, y0, 720, 620, 3, "★ Fixed-Time Horizon labelling (base method)",
             "Look h bars ahead and compare the return with a fixed threshold τ")
    pg.math("r_{t_0,\\,t_0+h} = \\frac{P_{t_0+h}}{P_{t_0}} - 1", x0 + 20, y0 + 60, 300, 50, 17)
    pg.math("y = \\begin{cases} +1 & r \\gt \\tau \\\\ \\ \\ 0 & |r| \\le \\tau \\\\ -1 & r \\lt -\\tau \\end{cases}",
            x0 + 360, y0 + 55, 300, 80, 16)
    t0, hh = 60, 20
    seg = P[t0 - 15:t0 + hh + 6]
    xs = list(range(t0 - 15, t0 + hh + 6))
    tau = 0.02
    lo_, hi_ = min(min(seg), P[t0] * (1 - tau)) * 0.995, max(max(seg), P[t0] * (1 + tau)) * 1.005
    q = pg.axes(x0 + 60, y0 + 150, 440, 230, (xs[0], xs[-1]), (lo_, hi_), "t", "P", label_size=12)
    q.poly(list(zip(xs, seg)), color=BLACK, width=1.3)
    q.hline(P[t0] * (1 + tau), t0, t0 + hh, color=GREEN, dashed=True, width=1.5)
    q.hline(P[t0] * (1 - tau), t0, t0 + hh, color=RED, dashed=True, width=1.5)
    q.hline(P[t0], t0, t0 + hh, color=GREY)
    q.vline(t0, lo_, hi_, color=BLUE)
    q.vline(t0 + hh, lo_, hi_, color=BLUE)
    q.scatter([(t0, P[t0])], color=BLUE, r=5)
    q.scatter([(t0 + hh, P[t0 + hh])], color=ORANGE, r=5)
    rr = P[t0 + hh] / P[t0] - 1
    lab = 1 if rr > tau else (-1 if rr < -tau else 0)
    q.text("t₀", t0, lo_, dy=12, size=11, color=BLUE)
    q.text("t₀ + h", t0 + hh, lo_, dy=12, size=11, color=BLUE)
    q.text("+τ", t0 + hh, P[t0] * (1 + tau), dx=18, size=11, color=GREEN)
    q.text("−τ", t0 + hh, P[t0] * (1 - tau), dx=18, size=11, color=RED)
    pg.html(f'h = {hh} bars, τ = {tau:.0%}:  r = {rr * 100:+.2f} %  →  '
            f'<span style="background:{HL}"><b>y = {lab:+d}</b></span>', x0 + 60, y0 + 400, 440, 24, 13,
            align="center", valign="middle")
    # problems
    pg.text("Problems", x0 + 530, y0 + 150, 170, 20, 13, bold=True, color=RED)
    pg.html("<b>1. Fixed τ ignores volatility</b><br>calm market → almost all 0; volatile market → all ±1",
            x0 + 530, y0 + 172, 180, 70, 11)
    pg.html("<b>2. Path is ignored</b><br>the price may hit your stop-loss before ending above +τ, "
            "but the label still says +1", x0 + 530, y0 + 245, 180, 80, 11)
    # mini path illustrating problem 2
    m = Plot(pg, x0 + 540, y0 + 330, 160, 70, (0, 10), (-3.5, 3))
    m.hline(2, color=GREEN); m.hline(-2, color=RED); m.hline(0, color=GREY)
    path = [(0, 0), (1, -1), (2, -2.2), (3, -3), (4, -1.5), (5, 0), (6, 1), (7, 1.8), (8, 2.6), (10, 2.8)]
    m.poly(path, color=BLACK, width=1.3)
    m.scatter([(3, -3)], color=RED, r=3.5)
    m.text("stopped out here!", 3, -3, dx=40, dy=6, size=9, color=RED, w=90)
    m.text("labelled +1", 9, 2.8, dy=-10, size=9, color=GREEN, w=70)
    pg.bullets(x0 + 20, y0 + 440, 680, [
        "Simple and the most common labelling in papers (classification of the next-h-bar move)",
        "Better: dynamic threshold τₜ = k·σₜ (volatility-scaled)",
        "Best: take the path into account → <b>Triple Barrier Method</b> ④",
        "Assumes non-overlapping, independent labels — usually false (see ⑥)"], head="Notes:")

    # ---------------------------------------------------------------- ④ triple barrier method (big)
    x0, y0 = 400, 900
    pg.panel(x0, y0, 1340, 800, 4, "Triple Barrier Method (López de Prado) — label by the FIRST barrier touched",
             "Two horizontal barriers (profit-take / stop-loss) scaled by volatility + one vertical barrier (time limit)")
    H = 15
    events = [(10, "A"), (27, "B"), (41, "C")]
    seg = P[0:76]
    lows = [triple_barrier(t, 1.0, 1.0, H)[3] for t, _ in events]
    ymin, ymax = min(min(seg), min(lows)) * 0.995, max(seg) * 1.02
    q = pg.axes(x0 + 70, y0 + 90, 780, 360, (0, 75), (ymin, ymax), "t", "P", label_size=13,
                xticks=(0, 10, 20, 30, 40, 50, 60, 70))
    q.poly(list(enumerate(seg)), color=BLACK, width=1.3)
    res = []
    colors = {1: GREEN, -1: RED, 0: BLUE}
    names = {1: "upper (profit-take)", -1: "lower (stop-loss)", 0: "vertical (time)"}
    for t0, nm in events:
        ti, lab, up, lo = triple_barrier(t0, 1.0, 1.0, H)
        t1 = t0 + H
        q.hline(up, t0, t1, color=GREEN, dashed=True, width=1.5)
        q.hline(lo, t0, t1, color=RED, dashed=True, width=1.5)
        q.vline(t1, lo, up, color=BLUE, dashed=True, width=1.5)
        q.vline(t0, lo, up, color=GREY, dashed=True)
        q.scatter([(t0, P[t0])], color=BLACK, r=4.5)
        q.text(f"t₀ ({nm})", t0, up, dy=-12, size=11, bold=True, w=70)
        pg.ellipse(*[c - 8 for c in q.P(ti, P[ti])], 16, 16, stroke=colors[lab], sw=2.5)
        q.text(f"y = {lab:+d}" if lab else "y = 0", ti, P[ti], dx=34 if lab != 0 else 30, dy=(-14 if lab == 1 else 14), size=12,
               bold=True, color=colors[lab], w=60)
        res.append((nm, t0, P[t0], SIG[t0], up, lo, t1, ti, lab, P[ti] / P[t0] - 1))
    pg.note("upper barrier  \\(P_{t_0}(1 + pt\\cdot\\sigma_{t_0}\\sqrt{h})\\)", x0 + 100, y0 + 62,
            *q.P(12, res[0][4]), w=230, raw=True, color=GREEN)
    pg.note("lower barrier  \\(P_{t_0}(1 - sl\\cdot\\sigma_{t_0}\\sqrt{h})\\)", x0 + 470, y0 + 470,
            *q.P(31, res[1][5]), w=230, raw=True, color=RED)
    pg.note("vertical barrier  \\(t_1 = t_0 + h\\)", x0 + 650, y0 + 62, *q.P(56, (res[2][4] + res[2][5]) / 2),
            w=170, raw=True, color=BLUE)
    pg.text("A touches the upper barrier first → +1 · B touches the lower barrier first → −1 · "
            "C reaches t₁ without a touch → 0 (or sign of r)", x0 + 70, y0 + 500, 780, 20, 11, italic=True,
            align="center")
    # resulting labelled table
    rows = [["event", "\\(t_0\\)", "\\(P_{t_0}\\)", "\\(\\sigma_{t_0}\\)", "upper", "lower", "\\(t_1\\)",
             "first touch", "barrier", "return", "label"]]
    for nm, t0, p0, s, up, lo, t1, ti, lab, rr in res:
        col = {1: "#009900", -1: "#FF0000", 0: "#3333FF"}[lab]
        rows.append([nm, t0, f"{p0:.2f}", f"{s * 100:.2f}%", f"{up:.2f}", f"{lo:.2f}", t1, ti, names[lab],
                     f"{rr * 100:+.2f}%", f'<b><font color="{col}">{lab:+d}</font></b>' if lab else "<b>0</b>"])
    pg.text(f"Resulting labelled events (pt = sl = 1, h = {H} bars, σ = EWMA(span = 20) of log-returns)",
            x0 + 30, y0 + 535, 800, 20, 12, bold=True)
    pg.table(x0 + 30, y0 + 560, 820, rows, size=11)
    pg.bullets(x0 + 30, y0 + 680, 820, [
        "Label reflects what a real trade with take-profit, stop-loss and max holding time would do (path-dependent)",
        "Barrier width follows volatility → balanced labels in calm and volatile regimes",
        "Also returns the exit time t_touch: the label <i>span</i> [t₀, t_touch] needed for sample weights & purging ⑥"],
        head="Why it is better than the fixed horizon:")
    # right column: formulas, parameters, configurations, pseudo-code
    rx = x0 + 890
    pg.text("Daily volatility (EWMA)", rx, y0 + 85, 420, 20, 13, bold=True)
    pg.math("\\sigma_t^2 = (1-\\lambda)\\,\\sigma_{t-1}^2 + \\lambda\\, r_t^2,\\quad \\lambda=\\frac{2}{span+1}",
            rx, y0 + 108, 420, 40, 14)
    pg.text("Label rule (first touch)", rx, y0 + 155, 420, 20, 13, bold=True)
    pg.math("y=\\begin{cases}+1 & \\text{upper first}\\\\ -1 & \\text{lower first}\\\\ \\ \\ 0 \\text{ or } "
            "\\mathrm{sign}(r_{t_0,t_1}) & \\text{vertical first}\\end{cases}", rx, y0 + 178, 420, 80, 14)
    pg.table(rx, y0 + 270, 420, [
        ["parameter", "meaning", "here"],
        ["pt", "profit-take multiple of σ", "1"],
        ["sl", "stop-loss multiple of σ", "1"],
        ["h (t₁)", "max holding period", f"{H} bars"],
        ["min ret", "skip events with tiny σ", "0.5 %"],
        ["side", "long / short (from primary model)", "long"]], size=11)
    pg.text("Barrier configurations [pt, sl, t₁]", rx, y0 + 430, 420, 20, 13, bold=True)
    pg.table(rx, y0 + 455, 420, [
        ["[pt, sl, t₁]", "barriers active", "use"],
        ["[1, 1, 1]", "all three", "standard labelling"],
        ["[0, 1, 1]", "stop-loss + time", "exit at t₁ unless stopped"],
        ["[1, 0, 1]", "profit-take + time", "take profit or time out"],
        ["[1, 1, 0]", "no time limit", "rare: may never finish"]], size=11)
    pg.rect(rx, y0 + 600, 420, 180,
            "<b>Pseudo-code</b><br><font face='Courier New'>σ = ewma_std(log_returns, span=20)<br>"
            "for t0 in events:  # e.g. CUSUM<br>"
            "&nbsp;&nbsp;up = P[t0]·(1 + pt·σ[t0]·√h)<br>&nbsp;&nbsp;lo = P[t0]·(1 − sl·σ[t0]·√h)<br>"
            "&nbsp;&nbsp;path = P[t0+1 : t0+h]<br>"
            "&nbsp;&nbsp;t_up = first(path ≥ up); t_lo = first(path ≤ lo)<br>"
            "&nbsp;&nbsp;t_touch = min(t_up, t_lo, t0+h)<br>"
            "&nbsp;&nbsp;y = +1 if t_touch==t_up else −1 if t_touch==t_lo else 0</font>",
            fill="#F5F5F5", stroke="#999999", size=11, align="left", valign="top", raw=True)

    # ---------------------------------------------------------------- ⑤ meta-labelling & trend scanning
    x0 = 1780
    pg.panel(x0, y0, 620, 800, 5, "Meta-Labelling & Trend-Scanning")
    flow = [("Features X (price, indicators, news)", "#FFFFFF", BLACK),
            ("Primary model (rule / MA crossover) → side ±1", LIGHTBLUE, BLUE),
            ("Triple barrier using that side → meta-label: 1 = profit-take hit, 0 = otherwise", "#FFE6CC", ORANGE),
            ("Secondary ML model (e.g. Random Forest) on X + side → P(success)", "#E1D5E7", "#9673A6"),
            ("Bet size = f(P), e.g. 2P − 1; skip trade if P < 0.5", HL, BLACK)]
    ids = []
    for k, (lab, f, s) in enumerate(flow):
        ids.append(pg.rect(x0 + 20, y0 + 50 + k * 62, 300, 46, lab, fill=f, stroke=s, size=11))
    for a, b in zip(ids, ids[1:]):
        pg.connect(a, b, orth=False)
    pg.html("<b>Why two models?</b><br>Primary: decides the <i>side</i>, tuned for high <b>recall</b> (catch most "
            "opportunities).<br>Secondary: decides <i>whether / how much</i>, filters false positives → higher "
            "<b>precision</b>.", x0 + 340, y0 + 55, 265, 130, 11)
    pg.math("Precision=\\frac{TP}{TP+FP}", x0 + 340, y0 + 190, 260, 36, 13)
    pg.math("Recall=\\frac{TP}{TP+FN}", x0 + 340, y0 + 228, 260, 36, 13)
    pg.math("F_1=\\frac{2PR}{P+R}", x0 + 340, y0 + 266, 260, 36, 13)
    # trend scanning
    pg.text("Trend-Scanning: label = sign of the most significant forward trend", x0 + 20, y0 + 375, 580, 20, 12,
            bold=True)
    t0 = 164
    Ls = [5, 10, 20]
    seg = P[t0:t0 + 21]
    q = pg.axes(x0 + 60, y0 + 410, 300, 170, (0, 20), (min(seg) * 0.995, max(seg) * 1.005), "L", "P",
                label_size=11, xticks=(0, 5, 10, 20))
    q.poly(list(enumerate(seg)), color=BLACK, width=1.2)
    cols = [ORANGE, GREEN, BLUE]
    rows = [["window L", "slope β", "t-value"]]
    best = None
    for L, c in zip(Ls, cols):
        xs, ys = list(range(L + 1)), seg[:L + 1]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((a - mx) ** 2 for a in xs)
        b = sum((a - mx) * (yv - my) for a, yv in zip(xs, ys)) / sxx
        a0 = my - b * mx
        res_ = [yv - (a0 + b * a) for a, yv in zip(xs, ys)]
        se = math.sqrt(sum(e * e for e in res_) / (len(xs) - 2) / sxx)
        tv = b / se
        q.poly([(0, a0), (L, a0 + b * L)], color=c, width=2)
        rows.append([f'<font color="{c}"><b>{L}</b></font>', f"{b:+.3f}", f"{tv:+.2f}"])
        if best is None or abs(tv) > abs(best[1]): best = (L, tv, len(rows) - 1)
    pg.table(x0 + 390, y0 + 420, 210, rows, size=11, highlight_rows=(best[2],))
    pg.html(f'max |t| at L = {best[0]} → <span style="background:{HL}"><b>y = {1 if best[1] > 0 else -1:+d}</b>'
            f'</span><br><font color="#555555">(|t| also usable as sample weight)</font>', x0 + 390, y0 + 545, 210,
            44, 11)
    pg.math("t = \\frac{\\hat{\\beta}}{SE(\\hat{\\beta})}", x0 + 60, y0 + 605, 200, 40, 14)
    pg.text("fit P(t₀ … t₀+L) = α + β·l for several L, keep the window with the largest |t|", x0 + 260, y0 + 610,
            340, 34, 10, color="#555555")

    # ---------------------------------------------------------------- ⑥ overlap, weights & leakage-safe validation
    x0, y0 = 400, 1740
    pg.panel(x0, y0, 2000, 580, 6, "Overlapping labels, sample weights & leakage-safe validation",
             "Labels span time intervals [t₀, t_touch] → neighbouring samples share information → weight them and purge CV")
    spans = [(0, 8), (3, 10), (5, 9), (12, 20), (14, 25), (22, 28)]
    p = Plot(pg, x0 + 70, y0 + 80, 560, 190, (0, 30), (0, 7))
    for k, (a, b) in enumerate(spans):
        _, yy = p.P(0, 6.5 - k)
        xa, _ = p.P(a, 0); xb, _ = p.P(b, 0)
        pg.rect(xa, yy - 9, xb - xa, 18, f"label {k + 1}", fill=LIGHTBLUE, stroke=BLUE, size=10)
    conc = [sum(1 for a, b in spans if a <= t <= b) for t in range(31)]
    c = pg.axes(x0 + 70, y0 + 290, 560, 90, (0, 30), (0, 3.5), "t", "c_t", label_size=11, xticks=(0, 10, 20, 30),
                yticks=(1, 2, 3))
    c.bars([(t, conc[t]) for t in range(31) if conc[t]], 0.8, fill="#DDDDDD", stroke="#777777")
    rows = [["label", "span", "\\(\\bar{u}_i\\)"]]
    for k, (a, b) in enumerate(spans):
        u = sum(1 / conc[t] for t in range(a, b + 1)) / (b - a + 1)
        rows.append([k + 1, f"[{a}, {b}]", f"{u:.2f}"])
    pg.table(x0 + 660, y0 + 80, 150, rows, size=10)
    pg.math("c_t = \\sum_i 1_{t,i}", x0 + 640, y0 + 300, 190, 30, 13)
    pg.math("\\bar{u}_i = \\frac{1}{T_i}\\sum_{t \\in i}\\frac{1}{c_t}", x0 + 640, y0 + 335, 190, 46, 14)
    pg.bullets(x0 + 30, y0 + 410, 790, [
        "\\(c_t\\) = number of labels alive at time t (concurrency);  \\(\\bar{u}_i\\) = average uniqueness of label i (1 = no overlap)",
        "Use \\(\\bar{u}_i\\) (× |return|) as sample weight, and bootstrap with probability ∝ uniqueness (sequential bootstrap)",
        "Overlap breaks the i.i.d. assumption → ordinary K-fold leaks future information into training"], head="Notes:")
    # purged k-fold with embargo
    kx = x0 + 860
    pg.text("Purged K-Fold CV with Embargo (K = 5)", kx, y0 + 70, 620, 20, 13, bold=True)
    W, cols_ = 560, 5
    seg_w = W / cols_
    for k in range(cols_):
        yy = y0 + 105 + k * 46
        pg.text(f"fold {k + 1}", kx, yy + 4, 50, 20, 11)
        bx = kx + 55
        ts, te = bx + k * seg_w, bx + (k + 1) * seg_w
        purge, emb = 22, 18
        if k > 0: pg.rect(bx, yy, ts - purge - bx, 28, "train", fill=LIGHTBLUE, stroke=BLUE, size=10)
        if k > 0: pg.rect(ts - purge, yy, purge, 28, "", fill="#BBBBBB", stroke="#777777")
        pg.rect(ts, yy, seg_w, 28, "test", fill="#FFE6CC", stroke=ORANGE, size=10)
        if k < cols_ - 1:
            pg.rect(te, yy, emb, 28, "", fill=LIGHTRED, stroke=RED)
            pg.rect(te + emb, yy, bx + W - te - emb, 28, "train", fill=LIGHTBLUE, stroke=BLUE, size=10)
    ly = y0 + 345
    for k, (lab, f, s) in enumerate([("train", LIGHTBLUE, BLUE), ("test", "#FFE6CC", ORANGE),
                                     ("purged (label overlaps test)", "#BBBBBB", "#777777"),
                                     ("embargo (gap after test)", LIGHTRED, RED)]):
        pg.rect(kx + 55 + k * 150, ly, 16, 14, "", fill=f, stroke=s)
        pg.text(lab, kx + 75 + k * 150, ly - 3, 130, 20, 10)
    pg.bullets(kx, y0 + 380, 620, [
        "<b>Purge</b>: drop training samples whose label span overlaps the test period",
        "<b>Embargo</b>: also drop a small gap (e.g. 1 % of bars) right after the test set — features with memory "
        "(moving averages) would otherwise leak",
        "Use with the sample weights from the left"], head="Rules:")
    # walk-forward
    wx = x0 + 1520
    pg.text("Walk-forward (expanding window)", wx, y0 + 70, 460, 20, 13, bold=True)
    for k in range(5):
        yy = y0 + 105 + k * 46
        trw = 120 + k * 60
        pg.rect(wx, yy, trw, 28, "train (past)", fill=LIGHTBLUE, stroke=BLUE, size=10)
        pg.rect(wx + trw + 8, yy, 70, 28, "test", fill="#FFE6CC", stroke=ORANGE, size=10)
    pg.line([(wx, y0 + 350), (wx + 460, y0 + 350)], arrow="classic")
    pg.text("time →", wx + 400, y0 + 352, 60, 18, 10, color="#555555")
    pg.bullets(wx, y0 + 380, 460, [
        "Train only on the past, test on the next block, then roll forward",
        "Closest to live trading; rolling (fixed-length) window adapts to regime change",
        "Single path → noisier estimate than purged K-fold; use both"], head="Notes:")
    asks(pg)


def asks(pg):
    """Interpretation callouts ('Doing this answers ...') — one per panel, absolute positions."""
    x0, y0 = 400, 240          # ① look-ahead
    pg.ask("Is this feature row safe to train on?",
           "features use data up to t only; the label uses t+1…t+3 → never feed label columns as inputs",
           x0 + 460, y0 + 345, x0 + 427, y0 + 302, w=170)
    x0 = 1080                  # ② CUSUM
    pg.ask("Which bars become training samples?",
           "only CUSUM events (dots): 18 of 120 bars, where a real move accumulated",
           x0 + 70, y0 + 282, x0 + 125, y0 + 412, w=200)
    x0 = 1680                  # ③ fixed horizon
    pg.ask("Does the label reflect what my trade really did?",
           "not always: the path is ignored — the stop-loss can be hit first and the label still says +1 → use Triple Barrier",
           x0 + 20, y0 + 542, x0 + 635, y0 + 420, w=680)
    x0, y0 = 400, 900          # ④ triple barrier
    pg.ask("Which barrier decides each event's label?",
           "the FIRST one touched: A upper → +1, B lower → −1, C time → 0",
           x0 + 675, y0 + 585, x0 + 560, y0 + 627, w=185)
    x0 = 1780                  # ⑤ trend scanning
    pg.ask("Which trend label does trend-scanning pick?",
           "the window with the largest |t|: L = 20, t = −7.74 → y = −1 (strong down-trend)",
           x0 + 20, y0 + 700, x0 + 576, y0 + 522, w=590)
    x0, y0 = 400, 1740         # ⑥ uniqueness weights
    pg.ask("How much should each overlapping label count?",
           "weight = average uniqueness ūᵢ: label 3 (ū = 0.37) shares most bars → gets the least weight",
           x0 + 20, y0 + 502, x0 + 730, y0 + 410, w=780)

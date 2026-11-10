"""Z-Score page: normal curve & 68-95-99.7, mean/σ formulas, worked example, number line, modified Z (MAD), IQR."""
import math
from dio import *

DATA = [50, 52, 48, 51, 49, 50, 53, 47, 50, 52, 49, 85]     # e.g. daily trade volume; 85 is suspicious


def _median(v):
    s = sorted(v); n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def build(pg: Page, cat):
    n = len(DATA)
    mu = sum(DATA) / n
    sig = math.sqrt(sum((x - mu) ** 2 for x in DATA) / n)
    s_samp = math.sqrt(sum((x - mu) ** 2 for x in DATA) / (n - 1))
    z = [(x - mu) / sig for x in DATA]
    med = _median(DATA)
    mad = _median([abs(x - med) for x in DATA])
    mz = [0.6745 * (x - med) / mad for x in DATA]
    srt = sorted(DATA)
    q1, q3 = _median(srt[:n // 2]), _median(srt[n // 2:])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    # ---------------------------------------------------------------- ① normal curve & empirical rule
    x0, y0 = 400, 240
    pg.panel(x0, y0, 900, 620, 1, "Normal Distribution & the 68–95–99.7 Rule",
             "Background: the Z-score measures distance from μ in units of σ")
    p = pg.axes(x0 + 70, y0 + 190, 760, 260, (-4, 4), (0, 0.42), "z", "f(z)", label_size=13, origin=(-4, 0))
    f = lambda t: normal_pdf(t)
    p.fill_under(f, 2, 4, n=12, color=ORANGE, opacity=50)
    p.fill_under(f, -4, -2, n=12, color=ORANGE, opacity=50)
    p.fill_under(f, 3, 4, n=6, color=RED, opacity=90)
    p.fill_under(f, -4, -3, n=6, color=RED, opacity=90)
    p.curve(f, -4, 4, n=64, width=2)
    for k in range(-3, 4):
        p.vline(k, 0, f(k), color=BLACK if k == 0 else GREY, dashed=k != 0)
        lab = "μ" if k == 0 else f"μ{'+' if k > 0 else '−'}{abs(k)}σ"
        p.text(lab, k, 0, w=60, dy=14, size=11, bold=(k == 0))
        p.text(f"z = {k}", k, 0, w=50, dy=30, size=9, color="#555555")
    # stacked double arrows above the curve
    for lvl, (k, pct) in enumerate([(3, "99.73%"), (2, "95.45%"), (1, "68.27%")]):
        ya = y0 + 92 + lvl * 30
        xa, _ = p.P(-k, 0); xb, _ = p.P(k, 0)
        pg.line([(xa, ya), (xb, ya)], arrow="classic", start_arrow="classic", color=BLUE if k == 1 else BLACK)
        pg.text(pct + f"  within ±{k}σ", (xa + xb) / 2 - 70, ya - 16, 140, 16, 11, bold=True, align="center",
                bg="#FFFFFF", color=BLUE if k == 1 else BLACK)
    pg.note("|z| > 2 : unusual (4.55%)", x0 + 700, y0 + 330, *p.P(2.3, 0.02), w=150, color=ORANGE)
    pg.note("|z| > 3 : outlier (0.27%)", x0 + 700, y0 + 380, *p.P(3.2, 0.003), w=150, color=RED)
    pg.note("|z| > 3 : outlier", x0 + 20, y0 + 380, *p.P(-3.2, 0.003), w=100, color=RED)
    pg.ask("How unusual is a value more than 3σ from the mean?",
           "only 0.27% of normal data lies beyond ±3σ → such a value is almost surely an outlier",
           x0 + 90, y0 + 210, *p.P(-3.4, 0.008), w=215)
    pg.math("f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}}\\, e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}", x0 + 60, y0 + 510, 380,
            56, 18)
    pg.math("z = \\frac{x-\\mu}{\\sigma} \\;\\sim\\; N(0,\\,1)", x0 + 470, y0 + 510, 380, 56, 18)
    pg.text("Gaussian PDF (see Distributions page)", x0 + 60, y0 + 570, 380, 18, 10, align="center", color="#555555")
    pg.text("Standard normal: mean 0, σ = 1", x0 + 470, y0 + 570, 380, 18, 10, align="center", color="#555555")

    # ---------------------------------------------------------------- ② formulas
    x0 = 1340
    pg.panel(x0, y0, 520, 620, 2, "Mean (μ) & Standard Deviation (σ)", "Background: central tendency & dispersion")
    pg.text("Population", x0 + 30, y0 + 70, 220, 20, 12, bold=True, align="center")
    pg.text("Sample", x0 + 270, y0 + 70, 220, 20, 12, bold=True, align="center")
    pg.math("\\mu = \\frac{1}{N}\\sum_{i=1}^{N} x_i", x0 + 30, y0 + 95, 220, 56, 16)
    pg.math("\\bar{x} = \\frac{1}{n}\\sum_{i=1}^{n} x_i", x0 + 270, y0 + 95, 220, 56, 16)
    pg.math("\\sigma = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(x_i-\\mu)^2}", x0 + 20, y0 + 160, 240, 64, 15)
    pg.math("s = \\sqrt{\\frac{1}{n-1}\\sum_{i=1}^{n}(x_i-\\bar{x})^2}", x0 + 260, y0 + 160, 240, 64, 15)
    pg.note("n − 1 = Bessel's correction (unbiased)", x0 + 300, y0 + 235, x0 + 345, y0 + 200, w=200, color=RED)
    pg.line([(x0 + 260, y0 + 75), (x0 + 260, y0 + 225)], color="#BBBBBB", dashed=True)
    pg.text("Z-score of one observation:", x0 + 20, y0 + 270, 480, 20, 12, bold=True)
    pg.math("z_i = \\frac{x_i - \\mu}{\\sigma}", x0 + 120, y0 + 295, 280, 64, 24)
    pg.note("how far is the value from the mean …", x0 + 20, y0 + 375, x0 + 245, y0 + 315, w=210)
    pg.note("… measured in standard deviations", x0 + 290, y0 + 375, x0 + 262, y0 + 345, w=210)
    pg.bullets(x0 + 20, y0 + 420, 480, [
        "z = 0 → exactly at the mean;  z = +2 → 2σ above the mean",
        "Variance = \\(\\sigma^2\\);  σ is in the same unit as x",
        "Rule: flag \\(|z_i| > 3\\) (or 2.5 / 2 for stricter screening)",
        "Assumes the data is roughly <b>normal</b> — check the histogram first",
        "Same formula is used for <b>feature scaling</b> (standardization)"], head="Where:")
    pg.ask("Can I trust the |z| > 3 rule on any data?", "only if the data is roughly normal",
           x0 + 270, y0 + 532, x0 + 350, y0 + 503, w=240)

    # ---------------------------------------------------------------- ③ worked example
    x0 = 1900
    pg.panel(x0, y0, 500, 620, 3, "Worked Example (n = 12)", "Daily trade volume (thousand)")
    rows = [["i", "\\(x_i\\)", "\\(x_i-\\mu\\)", "\\(z_i\\)", "flag"]]
    hi = []
    for i, (x, zz) in enumerate(zip(DATA, z), 1):
        flag = "<b>outlier</b>" if abs(zz) > 3 else ""
        rows.append([i, x, f"{x - mu:+.0f}", f"{zz:+.2f}", flag])
        if abs(zz) > 3: hi.append(i)
    pg.table(x0 + 30, y0 + 65, 440, rows, size=11, highlight_rows=tuple(hi))
    pg.ask("Which day is abnormal?",
           f"day 12 (85): z = {(85 - mu) / sig:.2f} > 3 → outlier; all other days have |z| < 1",
           x0 + 340, y0 + 90, x0 + 332, y0 + 335, w=150)
    pg.math(f"\\mu = \\frac{{{sum(DATA)}}}{{{n}}} = {mu:.0f}", x0 + 20, y0 + 365, 220, 48, 15)
    pg.math(f"\\sigma = \\sqrt{{\\frac{{{sum((x - mu) ** 2 for x in DATA):.0f}}}{{{n}}}}} = {sig:.2f}", x0 + 240,
            y0 + 365, 240, 48, 15)
    zo = (85 - mu) / sig
    pg.html(f"<span style='background:{HL}'>\\(z_{{12}} = \\frac{{85 - {mu:.0f}}}{{{sig:.2f}}} = {zo:.2f} > 3\\) → "
            "outlier</span>", x0 + 20, y0 + 425, 460, 44, 15, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 480, 460, [
        f"Note: the outlier itself inflates σ (without it σ ≈ "
        f"{math.sqrt(sum((x - 50.09) ** 2 for x in DATA[:-1]) / 11):.1f}) → <b>masking</b>",
        "With small n, |z| can never be very large: \\(|z| \\le \\sqrt{n-1}\\)",
        "→ robust alternatives ⑤ ⑥ use the median instead of the mean"], head="Notes:")

    # ---------------------------------------------------------------- ④ number line
    x0, y0 = 400, 900
    pg.panel(x0, y0, 620, 600, 4, "The data on a z-scale", "Each dot = one day")
    lx0, lx1, ly = x0 + 50, x0 + 560, y0 + 230
    xmin, xmax = 20, 90
    X = lambda v: lx0 + (v - xmin) / (xmax - xmin) * (lx1 - lx0)
    for k, colr in [(3, LIGHTRED), (2, "#FFE5CC"), (1, LIGHTBLUE)]:
        pg.rect(X(mu - k * sig), ly - 110, X(mu + k * sig) - X(mu - k * sig), 100, fill=colr, stroke="none",
                opacity=70)
    for k in range(-3, 4):
        xv = mu + k * sig
        if xmin <= xv <= xmax:
            pg.line([(X(xv), ly - 115), (X(xv), ly)], color=BLACK if k == 0 else GREY, dashed=k != 0)
            pg.text("μ" if k == 0 else f"{k:+d}σ", X(xv) - 20, ly - 132, 40, 16, 10, align="center", bold=k == 0)
    pg.line([(lx0 - 10, ly), (lx1 + 10, ly)], arrow="classic")
    for t in range(20, 91, 10):
        pg.line([(X(t), ly), (X(t), ly + 5)])
        pg.text(str(t), X(t) - 15, ly + 6, 30, 14, 9, align="center", color="#555555")
    stack = {}
    for x in DATA:
        stack[x] = stack.get(x, 0) + 1
        pg.dot(X(x), ly - 8 - (stack[x] - 1) * 10, r=4, color=RED if x == 85 else BLUE)
    pg.note("85 → z = %.2f" % zo, x0 + 450, y0 + 80, X(85), ly - 10, w=110, color=RED)
    pg.text("x (thousand)", lx1 - 60, ly + 22, 100, 16, 10, align="right", color="#555555")
    pg.bullets(x0 + 20, y0 + 290, 580, [
        "<font color='#3333FF'>Blue band</font> = μ ± 1σ (≈ 68% of normal data)",
        "<font color='#FF8000'>Orange</font> = μ ± 2σ (≈ 95%),  <font color='#FF0000'>red</font> = μ ± 3σ (≈ 99.7%)",
        "Points outside μ ± 3σ are flagged as anomalies",
        "Multivariate version: Mahalanobis distance \\(\\sqrt{(x-\\mu)^T\\Sigma^{-1}(x-\\mu)}\\)",
        "Rolling z-score (moving μ, σ) for time series / price returns"], head="Read it:")
    pg.ask("Is the 85 day inside the normal range?",
           f"No: it lies beyond μ + 3σ (z = {zo:.2f}) while every other day stays within ±1σ",
           x0 + 20, y0 + 400, X(85), ly - 16, w=580)

    # ---------------------------------------------------------------- ⑤ modified z (MAD)
    x0 = 1060
    pg.panel(x0, y0, 620, 600, 5, "Robust variant: Modified Z-score (median & MAD)")
    pg.math("M_i = \\frac{0.6745\\,(x_i - \\tilde{x})}{MAD}", x0 + 40, y0 + 60, 540, 60, 20)
    pg.math("MAD = \\text{median}\\left(|x_i - \\tilde{x}|\\right)", x0 + 40, y0 + 125, 540, 40, 17)
    pg.note("x̃ = median (not pulled by outliers)", x0 + 430, y0 + 160, x0 + 400, y0 + 140, w=180)
    pg.note("0.6745 makes MAD comparable to σ", x0 + 20, y0 + 48, x0 + 290, y0 + 66, w=200)
    pg.text(f"Same data: median x̃ = {med:g},  |x − x̃| = "
            f"{', '.join(f'{v:g}' for v in sorted(abs(x - med) for x in DATA))}",
            x0 + 20, y0 + 210, 580, 34, 11, align="center")
    pg.math(f"MAD = {mad:g} \\qquad M_{{12}} = \\frac{{0.6745\\,(85-{med:g})}}{{{mad:g}}} = {mz[-1]:.1f}",
            x0 + 20, y0 + 250, 580, 50, 16)
    pg.html(f"<span style='background:{HL}'>\\(M_{{12}} = {mz[-1]:.1f} \\gg 3.5\\) → clear outlier "
            f"(normal points: |M| ≤ {max(abs(v) for v in mz[:-1]):.2f})</span>",
            x0 + 20, y0 + 310, 580, 30, 14, align="center", valign="middle")
    pg.table(x0 + 60, y0 + 355, 500, [["", "Z-score", "Modified Z"],
                                      ["centre", f"mean = {mu:.0f}", f"median = {med:g}"],
                                      ["spread", f"σ = {sig:.2f}", f"MAD = {mad:g}"],
                                      ["score of 85", f"{zo:.2f}", f"{mz[-1]:.1f}"],
                                      ["threshold", "|z| > 3", "|M| > 3.5"]], size=11)
    pg.bullets(x0 + 20, y0 + 490, 580, [
        "Median and MAD ignore extreme values → no masking",
        "Works better for small samples and skewed data"], head="Why robust:")
    pg.ask("Is 85 still an outlier with a robust score?",
           f"yes, M = {mz[-1]:.1f} ≫ 3.5; median and MAD are not pulled by the outlier",
           x0 + 360, y0 + 490, x0 + 380, y0 + 437, w=250)

    # ---------------------------------------------------------------- ⑥ IQR / box plot
    x0 = 1720
    pg.panel(x0, y0, 680, 600, 6, "IQR rule & Box plot (Tukey fences)")
    lx0, lx1, by = x0 + 60, x0 + 620, y0 + 150
    X = lambda v: lx0 + (v - 40) / (90 - 40) * (lx1 - lx0)
    pg.line([(lx0 - 10, by + 60), (lx1 + 10, by + 60)], arrow="classic")
    for t in range(40, 91, 5):
        pg.line([(X(t), by + 60), (X(t), by + 65)])
        pg.text(str(t), X(t) - 15, by + 66, 30, 14, 9, align="center", color="#555555")
    wl, wh = min(v for v in DATA if v >= lo_f), max(v for v in DATA if v <= hi_f)
    pg.rect(X(q1), by - 20, X(q3) - X(q1), 40, fill=LIGHTBLUE, stroke=BLUE, sw=1.5)
    pg.line([(X(med), by - 20), (X(med), by + 20)], color=RED, width=2)
    pg.line([(X(wl), by), (X(q1), by)]); pg.line([(X(q3), by), (X(wh), by)])
    pg.line([(X(wl), by - 10), (X(wl), by + 10)]); pg.line([(X(wh), by - 10), (X(wh), by + 10)])
    for fv, lab in [(lo_f, "lower fence"), (hi_f, "upper fence")]:
        pg.line([(X(fv), by - 40), (X(fv), by + 55)], color=ORANGE, dashed=True)
        pg.text(f"{lab} {fv:g}", X(fv) - 50, by - 58, 100, 16, 10, align="center", color=ORANGE)
    pg.dot(X(85), by, r=5, color=RED)
    pg.text("85 outlier", X(85) - 40, by + 12, 80, 16, 10, align="center", color=RED, bold=True)
    pg.text(f"Q1 = {q1:g}", X(q1) - 70, by - 38, 70, 14, 10, align="right")
    pg.text(f"Q2 = {med:g}", X(med) - 30, by + 22, 60, 14, 10, align="center", color=RED)
    pg.text(f"Q3 = {q3:g}", X(q3) + 2, by - 38, 70, 14, 10)
    pg.ask("Where does 'normal' end for this data?",
           f"at the upper fence {hi_f:g}; 85 is far outside → outlier, with no normality assumed",
           x0 + 290, y0 + 75, X(hi_f) + 2, by - 30, w=230)
    pg.math(f"IQR = Q_3 - Q_1 = {q3:g} - {q1:g} = {iqr:g}", x0 + 30, y0 + 255, 620, 36, 16)
    pg.math(f"\\text{{fences}} = [\\,Q_1 - 1.5\\,IQR,\\; Q_3 + 1.5\\,IQR\\,] = [{lo_f:g},\\ {hi_f:g}]", x0 + 30,
            y0 + 295, 620, 36, 15)
    pg.table(x0 + 30, y0 + 350, 620, [["Method", "Centre / spread", "Assumes normal?", "85 flagged?"],
                                      ["Z-score", "mean / σ", "yes", "yes (barely, z = %.2f)" % zo],
                                      ["Modified Z", "median / MAD", "no", "yes (M = %.1f)" % mz[-1]],
                                      ["IQR rule", "median / IQR", "no", "yes (85 > %g)" % hi_f]], size=11)
    pg.bullets(x0 + 20, y0 + 470, 640, [
        "Box = middle 50% of the data;  whiskers reach the last point inside the fences",
        "Q1 / Q3 = 25th / 75th percentiles (median of the lower / upper half)",
        "Multiplier 1.5 ≈ ±2.7σ for normal data;  use 3 × IQR for 'extreme' outliers"], head="Notes:")

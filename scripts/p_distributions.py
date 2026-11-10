"""Distributions & Descriptive Statistics page: normal + 68-95-99.7, central tendency, skewness, kurtosis,
dispersion + box plot, z-score, distribution gallery (continuous & discrete)."""
import math
from dio import *

N = normal_pdf
H = f"background:{HL}"


def gamma_pdf(x, k=2.0, th=1.0):
    return 0 if x <= 0 else x ** (k - 1) * math.exp(-x / th) / (math.gamma(k) * th ** k)


def t_pdf(x, v):
    return math.gamma((v + 1) / 2) / (math.sqrt(v * math.pi) * math.gamma(v / 2)) * (1 + x * x / v) ** (-(v + 1) / 2)


def chi2_pdf(x, k):
    return 0 if x <= 0 else x ** (k / 2 - 1) * math.exp(-x / 2) / (2 ** (k / 2) * math.gamma(k / 2))


def lognorm_pdf(x, mu=0.0, s=0.5):
    return 0 if x <= 0 else math.exp(-(math.log(x) - mu) ** 2 / (2 * s * s)) / (x * s * math.sqrt(2 * math.pi))


def vlab(pg, p, xd, ytop, label, color, dy=-12, w=70, raw=False):
    p.vline(xd, 0, ytop, color=color, width=1.2)
    px, py = p.P(xd, ytop)
    pg.text(label, px - w / 2, py + dy - 9, w, 18, 10, bold=True, align="center", color=color, raw=raw)


def build(pg: Page, cat):
    # ================================================================ ① normal distribution
    x0, y0 = 400, 240
    pg.panel(x0, y0, 960, 680, 1, "Normal (Gaussian) Distribution", "Symmetrical distribution (Bell Curve)")
    p = pg.axes(x0 + 60, y0 + 90, 560, 330, (-4, 4), (0, 0.44), "x", "f(x)", origin=(-4, 0))
    for (a, b), op in [((-3, -2), 12), ((2, 3), 12), ((-2, -1), 25), ((1, 2), 25), ((-1, 1), 45)]:
        p.fill_under(N, a, b, n=30, color=BLUE, opacity=op)
    p.curve(N, -4, 4, n=60, width=2.5)
    for k in range(-3, 4):
        p.vline(k, 0, N(k) if k else 0.42, color=GREY if k else BLACK, width=1)
        lab = "\\mu" if k == 0 else f"\\mu{'+' if k > 0 else '-'}{abs(k) if abs(k) > 1 else ''}\\sigma"
        p.math(lab, k, 0, w=70, h=22, size=11, dy=16)
    # sigma arrow (like the original)
    pg.line([p.P(0, N(1)), p.P(1, N(1))], arrow="classic", width=1.2)
    p.math("\\sigma", 0.5, N(1), size=13, dy=-12)
    # 68-95-99.7 brackets below
    for k, (pct, col) in enumerate([("68.27 %", BLUE), ("95.45 %", GREEN), ("99.73 %", RED)]):
        yy = y0 + 470 + k * 26
        pa, pb = p.P(-(k + 1), 0)[0], p.P(k + 1, 0)[0]
        pg.line([(pa, yy), (pb, yy)], color=col, arrow="classic", start_arrow="classic", width=1.2)
        pg.text(pct + f"  within ±{k + 1}σ", (pa + pb) / 2 - 70, yy - 18, 140, 16, 10, bold=True, align="center",
                color=col)
    p.math("\\mu", 0, 0.42, size=13, dy=-12)
    pg.note("Mean = Median = Mode", x0 + 420, y0 + 80, *p.P(0, 0.40), w=140)
    pg.note("Tails never touch the axis", x0 + 500, y0 + 360, *p.P(3.3, N(3.3)), w=150)
    # right column
    rx = x0 + 650
    pg.math("f(x)=\\frac{1}{\\sigma\\sqrt{2\\pi}}\\,e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}", rx, y0 + 90, 300, 60, 17)
    pg.math("X \\sim N(\\mu,\\ \\sigma^2)", rx, y0 + 155, 300, 34, 16)
    pg.html(f'<span style="{H}"><b>Empirical rule 68 – 95 – 99.7</b></span>', rx, y0 + 200, 300, 22, 13,
            align="center")
    pg.table(rx + 10, y0 + 230, 280, [["Range", "Area", "Outside"],
                                      ["μ ± 1σ", "68.27 %", "31.73 %"],
                                      ["μ ± 2σ", "95.45 %", "4.55 %"],
                                      ["μ ± 3σ", "99.73 %", "0.27 %"]], size=11)
    pg.bullets(rx, y0 + 345, 300, [
        "\\(\\mu\\) = mean (center / location)",
        "\\(\\sigma\\) = standard deviation (width / spread)",
        "Symmetric: skewness = 0, kurtosis = 3",
        "Total area under the curve = 1",
        "Sums / averages of many variables → normal (<b>Central Limit Theorem</b>)",
        "Assumed by Linear Regression residuals, Z-test, GDA, Naive Bayes (Gaussian)"], head="Where:", h=200)
    pg.text("Darker band = more probability mass close to μ", x0 + 60, y0 + 560, 560, 20, 11, italic=True,
            align="center", color="#555555")
    pg.bullets(x0 + 20, y0 + 590, 620, [
        "Bigger σ → flatter, wider bell;  smaller σ → taller, narrower bell (area stays 1)",
        "Probability of an exact value is 0; probability = area over an interval"], head="Notes:")
    pg.ask("Where does most of the data fall?",
           "68 % within μ ± 1σ, 95 % within ±2σ, 99.7 % within ±3σ → beyond ±3σ is rare (possible outlier)",
           x0 + 650, y0 + 552, *(p.P(2, 0)[0], y0 + 496), w=300)

    # ================================================================ ② central tendency
    x0 = 1400
    pg.panel(x0, y0, 1000, 680, 2, "Central Tendency: Mean · Median · Mode", "Where is the 'middle' of the data?")
    data = [2, 3, 3, 5, 7, 8, 9]
    pg.text("Example data (n = 7):  2, 3, 3, 5, 7, 8, 9", x0 + 20, y0 + 62, 460, 20, 12, bold=True)
    p = pg.axes(x0 + 40, y0 + 100, 420, 120, (0, 10), (0, 3), "x", "", arrow=True, xticks=range(1, 10))
    cnt = {}
    for v in data:
        cnt[v] = cnt.get(v, 0) + 1
        p.scatter([(v, cnt[v] * 0.7)], r=7)
    mean = sum(data) / len(data)
    for xv, lab, col, dy in [(mean, f"Mean = {mean:.2f}", RED, -16), (5, "Median = 5", GREEN, -34),
                             (3, "Mode = 3", ORANGE, -52)]:
        p.vline(xv, 0, 2.6, color=col, width=1.3)
        px, py = p.P(xv, 2.6)
        pg.text(lab, px - 50, py + dy + 16, 100, 16, 10, bold=True, align="center", color=col)
    tx, ty = p.P(mean, 0)
    pg.vertex("", tx - 8, ty + 2, 16, 12, "triangle;direction=north;whiteSpace=wrap;html=1;fillColor=#FF0000;"
              "strokeColor=none;")
    pg.note("mean = balance point", x0 + 300, y0 + 250, tx, ty + 10, w=130, color=RED)
    # formulas
    fx = x0 + 500
    pg.text("Mean (arithmetic average)", fx, y0 + 60, 480, 18, 12, bold=True, color=RED)
    pg.math("\\mu = \\frac{\\sum_{i=1}^{N} x_i}{N}\\ \\text{(population)}\\qquad "
            "\\bar{x} = \\frac{\\sum_{i=1}^{n} x_i}{n}\\ \\text{(sample)}", fx, y0 + 80, 480, 50, 14)
    pg.math("\\bar{x} = \\frac{2+3+3+5+7+8+9}{7} = \\frac{37}{7} = 5.29", fx, y0 + 132, 480, 40, 14)
    pg.text("Median (middle value of the SORTED data)", fx, y0 + 180, 480, 18, 12, bold=True, color=GREEN)
    pg.math("n\\ \\text{odd: } M = x_{\\left(\\frac{n+1}{2}\\right)}\\qquad n\\ \\text{even: } "
            "M = \\frac{x_{(n/2)} + x_{(n/2+1)}}{2}", fx, y0 + 200, 480, 50, 14)
    pg.text("odd n=7 → 4th value = 5   ·   add 10 (n=8) → (5 + 7)/2 = 6", fx, y0 + 252, 480, 18, 11,
            align="center")
    pg.text("Mode (most frequent value)", fx, y0 + 280, 480, 18, 12, bold=True, color=ORANGE)
    pg.text("Raw data: 3 appears twice → Mode = 3.   Grouped data:", fx, y0 + 300, 480, 18, 11)
    pg.math("Mo = L + \\frac{f_1 - f_0}{(f_1 - f_0) + (f_1 - f_2)} \\times h", fx, y0 + 320, 480, 50, 16)
    # grouped histogram (same data as the original diagram)
    cls = [(5, 5), (15, 8), (25, 12), (35, 20), (45, 10), (55, 5)]
    q = pg.axes(x0 + 60, y0 + 330, 360, 170, (0, 60), (0, 22), "x", "f", label_size=12,
                xticks=(0, 10, 20, 30, 40, 50, 60), yticks=(5, 10, 20))
    for xc, fq in cls:
        q.bars([(xc, fq)], 10, fill=ORANGE if xc == 35 else LIGHTBLUE, stroke=BLACK, labels=True)
    mo = 30 + (20 - 12) / ((20 - 12) + (20 - 10)) * 10
    q.vline(mo, 0, 21.5, color=RED, width=1.5, dashed=False)
    pg.note(f"Mo = {mo:.2f}", x0 + 300, y0 + 320, *q.P(mo, 21), w=80, color=RED)
    pg.math("Mo = 30 + \\frac{20-12}{(20-12)+(20-10)}\\times 10 = 34.44", fx, y0 + 375, 480, 44, 14)
    pg.bullets(fx, y0 + 425, 480, [
        "\\(L\\) = lower limit of the modal class (30–40),  \\(h\\) = class width (10)",
        "\\(f_1\\) = frequency of modal class (20),  \\(f_0\\) = class before (12),  \\(f_2\\) = class after (10)"],
        head="Where:")
    pg.table(x0 + 20, y0 + 530, 960, [
        ["Measure", "Best for", "Sensitive to outliers?", "Example use in ML"],
        ["Mean", "symmetric numeric data", "Yes", "feature scaling (μ), K-Means centroid, imputation"],
        ["Median", "skewed data / outliers", "No (robust)", "robust imputation, prices, income, MAE"],
        ["Mode", "categorical / discrete", "No", "fill missing categories, majority vote (k-NN, trees)"]], size=11)
    pg.ask("Which 'average' should I use for skewed data?",
           f"the median (5 here vs mean {mean:.2f}): robust to extreme values; the mean is pulled toward the tail",
           x0 + 772, y0 + 540, x0 + 762, y0 + 603, w=220)

    # ================================================================ ③ skewness
    x0, y0 = 400, 960
    pg.panel(x0, y0, 960, 620, 3, "Skewness: shape & position of Mean, Median, Mode", "Asymmetry of the distribution")
    k = 2.0
    med = 1.6783  # median of Gamma(2,1)
    plots = [("Right (positive) skew", lambda v: gamma_pdf(v, k), (1.0, med, 2.0), "Mode < Median < Mean"),
             ("Symmetric", lambda v: N(v, 4, 1.15), (4, 4, 4), "Mean = Median = Mode"),
             ("Left (negative) skew", lambda v: gamma_pdf(8 - v, k), (7.0, 8 - med, 6.0), "Mean < Median < Mode")]
    for i, (name, f, (mo, md, mn), order) in enumerate(plots):
        px0 = x0 + 40 + i * 305
        p = pg.axes(px0, y0 + 110, 250, 200, (0, 8), (0, 0.42), "", "", origin=(0, 0))
        p.curve(f, 0.01, 7.99, n=60, width=2)
        if i == 1:
            vlab(pg, p, 4, 0.40, "Mean = Median = Mode", BLACK, w=150)
        else:
            lx = px0 + 140 if i == 0 else px0 + 10
            for j, (xv, lab, col) in enumerate([(mo, "Mode", ORANGE), (md, "Median", GREEN), (mn, "Mean", RED)]):
                p.vline(xv, 0, f(xv), color=col, width=1.3)
                jj = j if i == 0 else 2 - j
                pg.note(f"{lab} = {xv:.2f}" if i == 0 else f"{lab} = {xv:.2f}", lx, y0 + 112 + jj * 22,
                        *p.P(xv, f(xv) * 0.55 + 0.02 * jj), w=95, color=col, size=10)
        pg.text(name, px0 - 10, y0 + 62, 270, 20, 13, bold=True, align="center")
        pg.text(order, px0 - 10, y0 + 325, 270, 20, 12, bold=True, align="center",
                color=[RED, BLACK, BLUE][i])
        pg.math(["g_1 \\gt 0", "g_1 = 0", "g_1 \\lt 0"][i], px0 + 80, y0 + 350, 90, 28, 15)
    pg.note("long tail pulls the mean", x0 + 170, y0 + 210, x0 + 230, y0 + 290, w=130, color=RED)
    pg.math("\\text{Skewness: } g_1 = \\frac{\\frac{1}{n}\\sum_{i=1}^{n}(x_i-\\bar{x})^3}{s^3}", x0 + 30, y0 + 395,
            440, 56, 16)
    pg.math("\\text{Pearson: } Sk = \\frac{3(\\text{Mean}-\\text{Median})}{\\sigma}", x0 + 480, y0 + 395, 440, 56,
            16)
    pg.ask("How do I measure the skew with one number?",
           "g₁ (or Pearson Sk): > 0 = long right tail, < 0 = long left tail; |g₁| > 1 is highly skewed → try a log",
           x0 + 640, y0 + 458, x0 + 700, y0 + 448, w=300)
    pg.bullets(x0 + 20, y0 + 460, 600, [
        "The <b>mean is pulled toward the long tail</b>; the median sits between mode and mean",
        "Right skew examples: income, house prices, trading volume, waiting times  →  try a <b>log transform</b>",
        "Left skew examples: exam scores with an easy test, age at retirement",
        "Rule of thumb: |g₁| &lt; 0.5 ≈ symmetric,  0.5–1 moderate,  &gt; 1 highly skewed"], head="Notes:")

    # ================================================================ ④ kurtosis
    x0 = 1400
    pg.panel(x0, y0, 1000, 620, 4, "Kurtosis: peakedness & tail weight", "Same mean and variance, different shape")
    p = pg.axes(x0 + 60, y0 + 90, 520, 300, (-4, 4), (0, 0.74), "x", "f(x)", origin=(0, 0))
    lap = lambda v, b=1 / math.sqrt(2): math.exp(-abs(v) / b) / (2 * b)
    plat = lambda v: 0.36 * math.exp(-(abs(v) / 1.45) ** 4)
    p.curve(lap, -4, 4, n=80, dashed=True, color=GREY, width=1.5)
    p.curve(N, -4, 4, n=60, width=2.5)
    p.curve(plat, -4, 4, n=80, dashed=True, color=GREY, width=1.5)
    pg.note("(Peaked) (+) Leptokurtic", x0 + 380, y0 + 85, *p.P(0.12, lap(0.12)), w=160, color=RED)
    pg.note("(0) Mesokurtic — Normal", x0 + 400, y0 + 210, *p.P(0.9, N(0.9)), w=160, color=GREEN)
    pg.note("(Flat) (−) Platykurtic", x0 + 400, y0 + 250, *p.P(1.3, plat(1.3)), w=160, color=BLUE)
    pg.note("heavy tails → more outliers", x0 + 70, y0 + 335, *p.P(-2.8, lap(-2.8)), w=160)
    pg.math("\\text{Kurtosis: } g_2 = \\frac{\\frac{1}{n}\\sum_{i=1}^{n}(x_i-\\bar{x})^4}{s^4}", x0 + 600, y0 + 90,
            380, 56, 15)
    pg.math("\\text{Excess kurtosis} = g_2 - 3", x0 + 600, y0 + 150, 380, 34, 15)
    pg.table(x0 + 610, y0 + 200, 370, [
        ["Type", "Excess", "Tails", "Example"],
        ["Leptokurtic", "&gt; 0", "heavy", "Laplace, Student t, stock returns"],
        ["Mesokurtic", "= 0", "normal", "Normal"],
        ["Platykurtic", "&lt; 0", "light", "Uniform (−1.2)"]], size=11)
    pg.ask("Do extreme values happen more often than a normal curve predicts?",
           "yes if leptokurtic (excess > 0, heavy tails), e.g. stock returns → expect more outliers",
           x0 + 610, y0 + 305, x0 + 958, y0 + 236, targets=[p.P(2.8, lap(2.8))], w=370)
    pg.bullets(x0 + 20, y0 + 420, 960, [
        "Kurtosis measures the <b>tails</b> (extreme values) more than the peak",
        "Financial returns are usually leptokurtic → extreme moves happen more often than a normal model predicts",
        "High kurtosis → check outliers (Z-score / IQR) and use robust models / losses",
        "Fourth power (vs. third for skewness, second for variance) makes far-away points dominate"], head="Notes:")

    # ================================================================ ⑤ dispersion + box plot
    x0, y0 = 400, 1620
    pg.panel(x0, y0, 960, 760, 5, "Dispersion: Variance · σ · Range · Quartiles · IQR", "How spread out is the data?")
    mu, sd = 50, 10
    q1, q3 = mu - 0.6745 * sd, mu + 0.6745 * sd
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    p = pg.axes(x0 + 60, y0 + 80, 560, 200, (10, 90), (0, 0.044), "x", "f(x)", origin=(10, 0))
    p.fill_under(lambda v: N(v, mu, sd), q1, q3, n=16, color=BLUE, opacity=30)
    p.curve(lambda v: N(v, mu, sd), 10, 90, n=60, width=2)
    for xv, lab in [(q1, "Q_1"), (mu, "Q_2"), (q3, "Q_3"), (lo, ""), (hi, "")]:
        p.vline(xv, 0, N(xv, mu, sd), color=GREY)
    p.text("50 % of data", mu, 0.012, w=90, size=10, bold=True, color=BLUE)
    # box plot aligned below
    by = y0 + 330
    bx = lambda v: p.P(v, 0)[0]
    pg.rect(bx(q1), by, bx(q3) - bx(q1), 40, fill=LIGHTBLUE, stroke=BLACK, sw=1.5)
    pg.line([(bx(mu), by), (bx(mu), by + 40)], color=RED, width=2)
    pg.line([(bx(lo), by + 20), (bx(q1), by + 20)]); pg.line([(bx(q3), by + 20), (bx(hi), by + 20)])
    pg.line([(bx(lo), by + 8), (bx(lo), by + 32)]); pg.line([(bx(hi), by + 8), (bx(hi), by + 32)])
    for v in (17, 84, 87):
        pg.ellipse(bx(v) - 4, by + 16, 8, 8, stroke=RED, sw=1.5)
    for xv in (q1, mu, q3, lo, hi):
        pg.line([(bx(xv), p.P(0, 0)[1]), (bx(xv), by)], color="#CCCCCC", dashed=True, width=0.75)
    for xv, lab in [(q1, "Q₁ (25 %)"), (mu, "Median"), (q3, "Q₃ (75 %)"), (lo, "Q₁ − 1.5·IQR"),
                    (hi, "Q₃ + 1.5·IQR")]:
        pg.text(lab, bx(xv) - 45, by + 46, 90, 16, 9, align="center", bold=True)
    pg.line([(bx(q1), by - 12), (bx(q3), by - 12)], arrow="classic", start_arrow="classic", color=BLUE)
    pg.text("IQR = Q₃ − Q₁", bx(mu) - 50, by - 30, 100, 16, 10, bold=True, align="center", color=BLUE)
    pg.note("outliers", x0 + 640, by + 6, bx(87) + 4, by + 20, w=60, color=RED)
    pg.ask("How spread out is the data, and where is the middle 50 %?",
           f"σ = {sd} around μ = {mu}; IQR = Q₃ − Q₁ = {iqr:.1f}: half of all values lie inside the box",
           x0 + 645, y0 + 90, bx(q3), by, w=295)
    # formulas
    pg.math("\\sigma^2 = \\frac{\\sum_{i=1}^{N}(x_i-\\mu)^2}{N}\\ \\ \\text{(population)}", x0 + 20, y0 + 420, 440,
            54, 15)
    pg.math("s^2 = \\frac{\\sum_{i=1}^{n}(x_i-\\bar{x})^2}{n-1}\\ \\ \\text{(sample)}", x0 + 480, y0 + 420, 460, 54,
            15)
    pg.math("\\sigma = \\sqrt{\\sigma^2},\\quad s = \\sqrt{s^2},\\quad Range = x_{max}-x_{min},\\quad "
            "IQR = Q_3 - Q_1", x0 + 20, y0 + 478, 920, 34, 15)
    pg.text("Worked example: 2, 4, 4, 4, 5, 5, 7, 9  (n = 8, mean = 5)", x0 + 20, y0 + 520, 600, 18, 12, bold=True)
    pg.table(x0 + 20, y0 + 545, 560, [
        ["x", "2", "4", "4", "4", "5", "5", "7", "9", "Σ"],
        ["\\(x-\\bar{x}\\)", "−3", "−1", "−1", "−1", "0", "0", "2", "4", "0"],
        ["\\((x-\\bar{x})^2\\)", "9", "1", "1", "1", "0", "0", "4", "16", "<b>32</b>"]], size=11)
    pg.html(f'σ² = 32 / 8 = 4 → <span style="{H}"><b>σ = 2</b></span><br>'
            f's² = 32 / 7 = 4.57 → <span style="{H}"><b>s = 2.14</b></span>',
            x0 + 600, y0 + 545, 340, 50, 13, valign="middle")
    pg.bullets(x0 + 20, y0 + 640, 920, [
        "<b>n − 1</b> (Bessel's correction) makes the sample variance an unbiased estimate of σ²",
        "σ has the same unit as the data; variance is in squared units",
        "Box plot: box = middle 50 % (IQR), line = median, whiskers to 1.5·IQR, dots = outliers"], head="Notes:")

    # ================================================================ ⑥ z-score
    x0 = 1400
    pg.panel(x0, y0, 1000, 760, 6, "Z-Score & the Standard Normal N(0, 1)", "Standardization: how many σ from the mean?")
    pg.math("z = \\frac{x-\\mu}{\\sigma}", x0 + 40, y0 + 70, 260, 60, 22)
    pg.note("distance from the mean", x0 + 300, y0 + 66, x0 + 185, y0 + 88, w=150)
    pg.note("in units of standard deviation", x0 + 300, y0 + 110, x0 + 185, y0 + 113, w=170)
    # before -> after
    p = pg.axes(x0 + 520, y0 + 70, 200, 100, (20, 80), (0, 0.045), "x", "", label_size=11, origin=(20, 0),
                xticks=(30, 50, 70))
    p.curve(lambda v: N(v, 50, 10), 20, 80, n=40, width=1.5)
    pg.text("N(μ = 50, σ = 10)", x0 + 520, y0 + 190, 200, 16, 10, align="center")
    pg.line([(x0 + 740, y0 + 120), (x0 + 790, y0 + 120)], arrow="block", width=2)
    q = pg.axes(x0 + 800, y0 + 70, 170, 100, (-3, 3), (0, 0.45), "z", "", label_size=11, origin=(0, 0),
                xticks=(-2, 0, 2))
    q.curve(N, -3, 3, n=40, width=1.5)
    pg.text("N(0, 1)", x0 + 800, y0 + 190, 170, 16, 10, align="center", bold=True)
    # hypothesis / outlier plot (like the original: red tails)
    p = pg.axes(x0 + 80, y0 + 250, 560, 230, (-4, 4), (0, 0.44), "z", "", origin=(-4, 0),
                xticks=(-3, -1.96, -1, 0, 1, 1.96, 3))
    p.fill_under(N, -4, -1.96, n=12, color=RED, opacity=70)
    p.fill_under(N, 1.96, 4, n=12, color=RED, opacity=70)
    p.curve(N, -4, 4, n=60, width=2)
    p.vline(0, 0, 0.42, color=BLACK)
    p.text("Accept H₀ / normal  (95 %)", 0, 0.2, w=170, size=11, bold=True)
    p.text("2.5 %", -2.6, 0.06, w=50, size=10, bold=True, color=RED)
    p.text("2.5 %", 2.6, 0.06, w=50, size=10, bold=True, color=RED)
    pg.note("Reject H₀ / outlier", x0 + 60, y0 + 230, *p.P(-2.4, 0.03), w=130, color=RED)
    pg.note("Reject H₀ / outlier", x0 + 520, y0 + 230, *p.P(2.4, 0.03), w=130, color=RED)
    pg.table(x0 + 690, y0 + 260, 290, [["Confidence", "α", "|z| critical"],
                                       ["90 %", "0.10", "1.645"],
                                       ["95 %", "0.05", "1.96"],
                                       ["99 %", "0.01", "2.576"],
                                       ["99.73 %", "0.0027", "3"]], size=11)
    pg.ask("Is this observation unusual?",
           "if |z| > 1.96 it is in the red 5 % tails; the example z = 3 is beyond 99.7 % of normal data → outlier",
           x0 + 680, y0 + 380, *p.P(2.6, 0.02), w=300)
    pg.html(f'<b>Example:</b> house price x = 80, μ = 50, σ = 10 → z = (80 − 50)/10 = '
            f'<span style="{H}"><b>3</b></span> → outlier (|z| ≥ 3)', x0 + 40, y0 + 520, 920, 22, 13,
            valign="middle")
    pg.bullets(x0 + 20, y0 + 560, 960, [
        "After standardization every feature has mean 0 and σ 1 → fair comparison & faster Gradient Descent",
        "Z-test (σ known) vs. T-test (σ unknown, uses s and Student's t);  one-tailed vs. two-tailed tests",
        "Percentile ↔ z:  z = 0 → 50th,  z = 1 → 84th,  z = 2 → 97.7th percentile",
        "Z-score outlier rule is the base algorithm of <b>Anomaly Detection</b> (assumes roughly normal data)"],
        head="Notes:")
    pg.text("▶ Open Anomaly Detection (Z-Score) page", x0 + 20, y0 + 700, 400, 22, 12, bold=True, color=BLUE,
            link="data:page/id,anomaly")

    # ================================================================ ⑦ distribution gallery
    x0, y0 = 400, 2420
    pg.panel(x0, y0, 2000, 1000, 7, "Distribution Gallery: types of probability distributions",
             "Continuous (curves = PDF) and discrete (bars = PMF)")
    CW, CH = 396, 470

    def cell(i, title, kind):
        cx = x0 + 10 + (i % 5) * CW
        cy = y0 + 60 + (i // 5) * CH
        pg.rect(cx + 6, cy, CW - 12, CH - 12, stroke="#DDDDDD", sw=0.5)
        pg.text(title, cx + 16, cy + 8, CW - 40, 22, 14, bold=True)
        pg.text(kind, cx + 16, cy + 30, CW - 40, 16, 10, italic=True, color="#777777")
        return cx, cy

    def info(cx, cy, params, mean, var, use):
        pg.math(params, cx + 16, cy + 268, CW - 40, 46, 13)
        pg.math(mean + "\\qquad " + var, cx + 16, cy + 318, CW - 40, 44, 13)
        pg.bullets(cx + 16, cy + 368, CW - 44, use, size=10)

    def legend(cx, cy, items):
        for j, (lab, col, dash) in enumerate(items):
            pg.line([(cx + 270, cy + 72 + j * 18), (cx + 296, cy + 72 + j * 18)], color=col, dashed=dash, width=2)
            pg.text(lab, cx + 300, cy + 63 + j * 18, 80, 16, 10, color=col, raw=True)

    def ax(cx, cy, xr, yr, xl="x", origin=None, ticks=()):
        return pg.axes(cx + 50, cy + 70, 290, 160, xr, yr, xl, "", label_size=11, origin=origin, xticks=ticks)

    # 1 uniform
    cx, cy = cell(0, "Uniform  U(a, b)", "continuous")
    p = ax(cx, cy, (0, 6), (0, 0.5), ticks=(1, 5))
    p.poly([(0, 0), (1, 0), (1, 0.25), (5, 0.25), (5, 0), (6, 0)], width=2, color=BLUE)
    p.fill_under(lambda v: 0.25, 1, 5, n=8, color=BLUE, opacity=15)
    p.text("a", 1, 0, dy=26, size=11); p.text("b", 5, 0, dy=26, size=11)
    p.math("\\frac{1}{b-a}", 3, 0.25, dy=-16, size=12)
    info(cx, cy, "f(x)=\\frac{1}{b-a},\\ a\\le x\\le b", "E[X]=\\frac{a+b}{2}", "Var=\\frac{(b-a)^2}{12}",
         ["Every value equally likely", "Random weight init, random search, simulation"])
    # 2 exponential
    cx, cy = cell(1, "Exponential  Exp(λ)", "continuous, right-skewed")
    p = ax(cx, cy, (0, 5), (0, 2.1), ticks=(1, 2, 3, 4))
    for lam, col in [(0.5, GREEN), (1, BLUE), (2, RED)]:
        p.curve(lambda v, l=lam: l * math.exp(-l * v), 0, 5, n=50, color=col, width=2)
    legend(cx, cy, [("λ = 0.5", GREEN, False), ("λ = 1", BLUE, False), ("λ = 2", RED, False)])
    info(cx, cy, "f(x)=\\lambda e^{-\\lambda x},\\ x\\ge 0", "E[X]=\\frac{1}{\\lambda}", "Var=\\frac{1}{\\lambda^2}",
         ["Time between events (memoryless)", "Waiting time, time to failure, gap between trades"])
    # 3 log-normal
    cx, cy = cell(2, "Log-Normal", "continuous, right-skewed, x > 0")
    p = ax(cx, cy, (0, 4), (0, 1.0), ticks=(1, 2, 3))
    for s, col in [(0.25, RED), (0.5, BLUE), (1.0, GREEN)]:
        p.curve(lambda v, s=s: lognorm_pdf(v, 0, s), 0.01, 4, n=70, color=col, width=2)
    legend(cx, cy, [("σ = 0.25", RED, False), ("σ = 0.5", BLUE, False), ("σ = 1", GREEN, False)])
    info(cx, cy, "\\ln X \\sim N(\\mu,\\sigma^2)", "E[X]=e^{\\mu+\\sigma^2/2}",
         "Var=(e^{\\sigma^2}-1)e^{2\\mu+\\sigma^2}",
         ["Product of many positive factors", "Stock prices, income, house prices → use log(x)"])
    # 4 student t
    cx, cy = cell(3, "Student's t  t(ν)", "continuous, heavier tails than normal")
    p = ax(cx, cy, (-4, 4), (0, 0.45), origin=(0, 0), ticks=(-2, 2))
    p.curve(N, -4, 4, n=60, color=BLACK, width=2)
    p.curve(lambda v: t_pdf(v, 3), -4, 4, n=60, color=BLUE, width=2, dashed=True)
    p.curve(lambda v: t_pdf(v, 1), -4, 4, n=60, color=RED, width=2, dashed=True)
    legend(cx, cy, [("N(0,1)", BLACK, False), ("ν = 3", BLUE, True), ("ν = 1", RED, True)])
    info(cx, cy, "\\nu = n-1\\ \\text{degrees of freedom}", "E[X]=0\\ (\\nu\\gt 1)", "Var=\\frac{\\nu}{\\nu-2}\\ (\\nu\\gt 2)",
         ["T-test when σ is unknown / small n", "ν → ∞ ⇒ becomes the normal distribution"])
    # 5 chi-square
    cx, cy = cell(4, "Chi-Square  χ²(k)", "continuous, sum of k squared N(0,1)")
    p = ax(cx, cy, (0, 12), (0, 0.5), ticks=(2, 4, 6, 8, 10))
    for kk, col in [(2, RED), (3, BLUE), (5, GREEN)]:
        p.curve(lambda v, kk=kk: chi2_pdf(v, kk), 0.05, 12, n=60, color=col, width=2)
    legend(cx, cy, [("k = 2", RED, False), ("k = 3", BLUE, False), ("k = 5", GREEN, False)])
    info(cx, cy, "Q=\\sum_{i=1}^{k} Z_i^2", "E[X]=k", "Var=2k",
         ["Chi-square test of independence / goodness-of-fit", "Feature selection (Chi² filter) for categorical data"])
    # 6 bimodal / multimodal
    cx, cy = cell(5, "Bimodal / Multimodal", "continuous, mixture of distributions")
    p = ax(cx, cy, (-4, 4), (0, 0.32), origin=(-4, 0))
    mix = lambda v: 0.5 * N(v, -1.5, 0.7) + 0.5 * N(v, 1.5, 0.7)
    p.curve(lambda v: 0.5 * N(v, -1.5, 0.7), -4, 4, n=50, color=GREY, dashed=True)
    p.curve(lambda v: 0.5 * N(v, 1.5, 0.7), -4, 4, n=50, color=GREY, dashed=True)
    p.curve(mix, -4, 4, n=70, color=BLUE, width=2)
    p.math("\\mu_1", -1.5, mix(-1.5), dy=-12, size=11); p.math("\\mu_2", 1.5, mix(1.5), dy=-12, size=11)
    info(cx, cy, "f(x)=\\sum_{k}\\pi_k\\,N(x\\mid\\mu_k,\\sigma_k^2)", "\\text{2 peaks = bimodal}",
         "\\text{3+ peaks = multimodal}",
         ["Two or more sub-populations mixed together", "Model with Gaussian Mixture Models (GMM) / clustering"])
    # 7 bernoulli
    cx, cy = cell(6, "Bernoulli  Bern(p)", "discrete, one trial")
    p = ax(cx, cy, (-0.8, 1.8), (0, 1.0), "k", origin=(-0.8, 0), ticks=(0, 1))
    p.bars([(0, 0.7), (1, 0.3)], 0.5, fill=BLUE, opacity=70, labels=True)
    pg.ask("How do I read a discrete distribution?",
           "bar height = probability of that exact value (P(up day) = 0.3); the bars sum to 1",
           cx + 215, cy + 46, *p.P(1.3, 0.2), w=170, size=9)
    info(cx, cy, "P(X=1)=p,\\ P(X=0)=1-p\\ \\ (p=0.3)", "E[X]=p", "Var=p(1-p)",
         ["Single yes / no outcome", "Click / no click, up / down day, logistic regression target"])
    # 8 binomial
    cx, cy = cell(7, "Binomial  B(n, p)", "discrete, successes in n trials")
    p = ax(cx, cy, (-0.8, 10.8), (0, 0.3), "k", origin=(-0.8, 0), ticks=(0, 5, 10))
    p.bars([(kk, math.comb(10, kk) * 0.5 ** 10) for kk in range(11)], 0.7, fill=BLUE, opacity=70)
    info(cx, cy, "P(X=k)=\\binom{n}{k}p^k(1-p)^{n-k}\\ \\ (n=10,p=0.5)", "E[X]=np", "Var=np(1-p)",
         ["Number of successes in n independent trials", "Heads in 10 flips, conversions out of n visitors"])
    # 9 poisson
    cx, cy = cell(8, "Poisson  Pois(λ)", "discrete, events per interval")
    p = ax(cx, cy, (-0.8, 12.8), (0, 0.25), "k", origin=(-0.8, 0), ticks=(0, 3, 6, 9, 12))
    p.bars([(kk, 3 ** kk * math.exp(-3) / math.factorial(kk)) for kk in range(13)], 0.7, fill=BLUE, opacity=70)
    p5 = 1 - sum(3 ** kk * math.exp(-3) / math.factorial(kk) for kk in range(5))
    pg.ask("How many orders should I expect per minute?",
           f"λ = 3 on average; 5 or more happens only ≈ {p5 * 100:.0f} % of the time",
           cx + 205, cy + 48, *p.P(3, 0.224), w=180, size=9)
    info(cx, cy, "P(X=k)=\\frac{\\lambda^k e^{-\\lambda}}{k!}\\ \\ (\\lambda=3)", "E[X]=\\lambda", "Var=\\lambda",
         ["Count of rare events in a fixed time / space", "Orders per minute, calls per hour, defects per item"])
    # 10 geometric
    cx, cy = cell(9, "Geometric  Geo(p)", "discrete, trials until first success")
    p = ax(cx, cy, (0.2, 10.8), (0, 0.35), "k", origin=(0.2, 0), ticks=(1, 5, 10))
    p.bars([(kk, 0.7 ** (kk - 1) * 0.3) for kk in range(1, 11)], 0.7, fill=BLUE, opacity=70)
    info(cx, cy, "P(X=k)=(1-p)^{k-1}p\\ \\ (p=0.3)", "E[X]=\\frac{1}{p}", "Var=\\frac{1-p}{p^2}",
         ["Number of tries until the first success", "Attempts until a sale, spins until a win"])

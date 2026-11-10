"""Inferential Statistics page (base: Central Limit Theorem)."""
import math
from dio import *


def t_pdf(x, df):
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return c * (1 + x * x / df) ** (-(df + 1) / 2)


def hist(values, lo, hi, bins):
    w = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        k = int((v - lo) / w)
        if 0 <= k < bins: counts[k] += 1
    n = len(values)
    return [(lo + (k + 0.5) * w, c / (n * w)) for k, c in enumerate(counts)], w


def build(pg: Page, cat):
    r = seeded(42)

    # ------------------------------------------------------------ ① population vs sample
    x0, y0 = 400, 240
    pg.panel(x0, y0, 620, 660, 1, "Population vs Sample", "We rarely see the population; we infer it from a sample")
    pg.ellipse(x0 + 30, y0 + 80, 250, 200, fill="none", sw=1.5)
    for _ in range(90):
        a, rr = r.uniform(0, 2 * math.pi), math.sqrt(r.random())
        pg.dot(x0 + 155 + 108 * rr * math.cos(a), y0 + 180 + 85 * rr * math.sin(a), r=2.5,
               color=BLUE if r.random() > 0.15 else RED)
    pg.text("Population (N)", x0 + 30, y0 + 285, 250, 20, 13, bold=True, align="center")
    pg.math("\\mu,\\ \\sigma,\\ p", x0 + 105, y0 + 305, 100, 24, 14)
    for k in range(3):
        sy = y0 + 80 + k * 70
        sid_x = x0 + 400
        pg.ellipse(sid_x, sy, 80, 54)
        for _ in range(6):
            pg.dot(sid_x + r.uniform(18, 62), sy + r.uniform(14, 40), r=2.5)
        pg.line([(x0 + 270, y0 + 180), (sid_x, sy + 27)], arrow="classic", dashed=True, color=GREY)
        pg.math(f"\\bar{{x}}_{k + 1},\\ s_{k + 1}", sid_x + 90, sy + 14, 80, 26, 13, align="left")
    pg.text("random sampling (n)", x0 + 280, y0 + 120, 120, 30, 10, italic=True, color="#555555")
    pg.table(x0 + 40, y0 + 350, 540, [
        ["", "Population <b>parameter</b>", "Sample <b>statistic</b>"],
        ["Size", "\\(N\\)", "\\(n\\)"],
        ["Mean", "\\(\\mu = \\frac{1}{N}\\sum x_i\\)", "\\(\\bar{x} = \\frac{1}{n}\\sum x_i\\)"],
        ["Std. dev.", "\\(\\sigma\\)", "\\(s = \\sqrt{\\frac{\\sum(x_i-\\bar{x})^2}{n-1}}\\)"],
        ["Proportion", "\\(p\\)", "\\(\\hat{p} = \\frac{x}{n}\\)"]], size=12, h=150)
    pg.bullets(x0 + 20, y0 + 520, 580, [
        "A <b>statistic</b> changes from sample to sample → it has its own distribution: the <b>sampling distribution</b>",
        "Inferential statistics = estimate parameters and test claims using that sampling distribution",
        "Samples must be <b>random</b> and <b>independent</b> (otherwise bias, not just noise)"], head="Key idea:")
    pg.ask("What can I actually compute from my data?",
           "only statistics (x̄, s, p̂); μ, σ, p stay unknown and must be estimated from them",
           x0 + 30, y0 + 588, x0 + 525, y0 + 478, w=560)

    # ------------------------------------------------------------ ② central limit theorem
    x0 = 1060
    pg.panel(x0, y0, 1340, 660, 2, "★ Central Limit Theorem (CLT)",
             "Whatever the population shape, the distribution of sample means → Normal as n grows (simulated: 4000 samples each)")
    pops = [("Uniform", lambda: r.random(), 0, 1, 0.5, math.sqrt(1 / 12)),
            ("Exponential (skewed)", lambda: r.expovariate(1.0), 0, 4, 1.0, 1.0),
            ("Bimodal", lambda: r.gauss(-2 if r.random() < 0.5 else 2, 0.6), -4, 4, 0.0, math.sqrt(4.36))]
    ns = [1, 2, 5, 30]
    cw, ch = 255, 110
    for ci, n in enumerate(ns):
        pg.text("Population  (n = 1)" if n == 1 else f"Means of n = {n}", x0 + 170 + ci * (cw + 30), y0 + 70,
                cw, 20, 13, bold=True, align="center", color=RED if n == 30 else BLACK)
    for ri, (name, draw, lo, hi, mu, sd) in enumerate(pops):
        yy = y0 + 100 + ri * 150
        pg.text(name, x0 + 20, yy + 35, 140, 40, 13, bold=True)
        pg.math(f"\\mu={mu:g},\\ \\sigma={sd:.2f}", x0 + 20, yy + 70, 140, 22, 10, align="left")
        for ci, n in enumerate(ns):
            means = [sum(draw() for _ in range(n)) / n for _ in range(4000)]
            data, bw = hist(means, lo, hi, 24)
            top = max(h for _, h in data) * 1.15
            p = pg.axes(x0 + 170 + ci * (cw + 30), yy + 10, cw, ch, (lo, hi), (0, top), "", "", arrow=False)
            p.bars(data, bw, fill=LIGHTBLUE, stroke=BLUE, opacity=80)
            p.vline(mu, 0, top * 0.95, color=RED)
            if n >= 5:
                p.curve(lambda v: normal_pdf(v, mu, sd / math.sqrt(n)), lo, hi, n=80, color=BLACK, width=1.5)
            sm = sum(means) / len(means)
            ssd = math.sqrt(sum((m - sm) ** 2 for m in means) / len(means))
            pg.text(f"SE = {ssd:.3f}  (σ/√n = {sd / math.sqrt(n):.3f})", x0 + 170 + ci * (cw + 30), yy + ch + 12,
                    cw, 16, 9, align="center", color="#555555")
    pg.math("\\bar{X} \\;\\sim\\; N\\!\\left(\\mu,\\ \\frac{\\sigma^2}{n}\\right)\\quad \\text{as } n \\to \\infty",
            x0 + 170, y0 + 560, 520, 50, 20)
    pg.math("Z = \\frac{\\bar{X}-\\mu}{\\sigma/\\sqrt{n}} \\;\\to\\; N(0,1)", x0 + 700, y0 + 560, 330, 50, 18)
    pg.ask("Can I use normal-based tests on a non-normal variable?",
           "yes, for the mean: at n = 30 even bimodal data gives a bell-shaped x̄ (black curve fits)",
           x0 + 985, y0 + 555, x0 + 170 + 3 * (cw + 30) + cw / 2, y0 + 100 + 2 * 150 + 40, w=345)
    pg.bullets(x0 + 20, y0 + 610, 1300, [
        "Rule of thumb: <b>n ≥ 30</b> is usually enough (skewed populations need more; already-normal populations need any n)  ·  "
        "Black curve = normal predicted by the CLT, red line = population mean μ"], size=11)

    # ------------------------------------------------------------ ③ standard error & law of large numbers
    x0, y0 = 400, 940
    pg.panel(x0, y0, 620, 660, 3, "Standard Error & Law of Large Numbers")
    pg.math("SE(\\bar{x}) = \\frac{\\sigma}{\\sqrt{n}} \\approx \\frac{s}{\\sqrt{n}}", x0 + 30, y0 + 50, 300, 50, 18)
    pg.note("4× more data → only 2× more precise", x0 + 340, y0 + 62, x0 + 290, y0 + 75, w=230, color=RED)
    p = pg.axes(x0 + 80, y0 + 130, 440, 170, (0, 100), (0, 1.05), "n", "SE", label_size=12,
                xticks=(1, 4, 16, 36, 64, 100), yticks=(0.5, 1))
    p.curve(lambda v: 1 / math.sqrt(v), 1, 100, n=80, color=BLUE, width=2)
    for nn in (4, 16, 64):
        p.scatter([(nn, 1 / math.sqrt(nn))], color=RED, r=4)
        p.text(f"σ/{int(math.sqrt(nn))}", nn, 1 / math.sqrt(nn), dx=22, dy=-10, size=10, color=RED)
    pg.text("(σ = 1)", x0 + 430, y0 + 130, 80, 16, 10, color="#555555")
    pg.ask("How much more data do I need for 2× precision?",
           "4× more: SE = σ/√n, so n = 4 → σ/2 and n = 16 → σ/4 (red dots)",
           x0 + 190, y0 + 150, *p.P(16, 0.25), w=320)
    pg.text("Law of Large Numbers: running mean of 500 coin flips", x0 + 20, y0 + 340, 580, 20, 13, bold=True)
    q = pg.axes(x0 + 80, y0 + 380, 440, 150, (0, 500), (0, 1), "n", "\\bar{x}", label_size=12,
                xticks=(0, 100, 200, 300, 400, 500), yticks=(0.5, 1))
    q.hline(0.5, color=RED)
    s, pts = 0, []
    for i in range(1, 501):
        s += 1 if r.random() < 0.5 else 0
        if i <= 20 or i % 5 == 0: pts.append((i, s / i))
    q.poly(pts, color=BLUE, width=1.2)
    q.text("p = 0.5", 500, 0.5, dx=30, size=10, color=RED)
    pg.bullets(x0 + 20, y0 + 565, 580, [
        "<b>LLN</b>: the sample mean x̄ converges to μ as n grows (about the <i>value</i>)",
        "<b>CLT</b>: the <i>distribution</i> of x̄ around μ is normal with spread SE = σ/√n"], head="LLN vs CLT:")

    # ------------------------------------------------------------ ④ confidence interval
    x0 = 1060
    pg.panel(x0, y0, 640, 660, 4, "Confidence Interval", "Range that captures μ in 95 % of repeated samples")
    pg.math("\\bar{x} \\pm z_{\\alpha/2}\\,\\frac{\\sigma}{\\sqrt{n}}", x0 + 30, y0 + 55, 260, 50, 20)
    pg.math("\\bar{x} \\pm t_{\\alpha/2,\\,n-1}\\,\\frac{s}{\\sqrt{n}}", x0 + 320, y0 + 55, 280, 50, 18)
    pg.text("σ known", x0 + 30, y0 + 102, 260, 16, 10, align="center", color="#555555")
    pg.text("σ unknown (use t)", x0 + 320, y0 + 102, 280, 16, 10, align="center", color="#555555")
    pg.note("margin of error", x0 + 200, y0 + 130, x0 + 200, y0 + 95, w=100, color=RED)
    # 20 simulated CIs, population N(50, 10), n = 25 -> SE = 2
    p = Plot(pg, x0 + 50, y0 + 160, 380, 300, (42, 58), (0, 21))
    p.vline(50, 0, 21, color=RED, width=1.5)
    p.text("μ = 50", 50, 21, dx=38, dy=-4, size=11, color=RED, w=60)
    miss = 0
    for k in range(20):
        xb = sum(r.gauss(50, 10) for _ in range(25)) / 25
        lo_, hi_ = xb - 1.96 * 2, xb + 1.96 * 2
        bad = not (lo_ <= 50 <= hi_)
        miss += bad
        col = RED if bad else BLUE
        p.poly([(lo_, 20 - k), (hi_, 20 - k)], color=col, width=2 if bad else 1.2)
        p.scatter([(xb, 20 - k)], color=col, r=3)
    pg.ask("Does each 95 % interval really contain μ?",
           f"{20 - miss} of 20 do here; over many samples ≈ 95 % will — that is what '95 % confidence' means",
           x0 + 445, y0 + 210, *p.P(50, 12), w=180)
    pg.line([(x0 + 50, y0 + 470), (x0 + 440, y0 + 470)], arrow="classic")
    for t in (42, 46, 50, 54, 58):
        px, _ = p.P(t, 0)
        pg.text(str(t), px - 15, y0 + 474, 30, 14, 9, align="center", color="#555555")
    pg.text(f"20 samples (n = 25): {20 - miss} intervals contain μ, {miss} miss (red) — on average 1 in 20 misses",
            x0 + 40, y0 + 495, 560, 20,
            11, bold=True, align="center")
    pg.html(f'<b>Example:</b> n = 25, x̄ = 52, σ = 10 → SE = 2 → 95 % CI = 52 ± 1.96·2 = '
            f'<span style="background:{HL}">[48.08, 55.92]</span>', x0 + 30, y0 + 525, 580, 30, 12)
    pg.bullets(x0 + 20, y0 + 560, 600, [
        "95 % is about the <i>method</i>: 95 % of such intervals contain μ (not '95 % chance μ is inside this one')",
        "Wider for higher confidence (99 % → z = 2.576) and narrower for larger n"], head="Interpretation:")

    # ------------------------------------------------------------ ⑤ hypothesis testing
    x0 = 1740
    pg.panel(x0, y0, 660, 660, 5, "Hypothesis Testing (Z-test / t-test)")
    steps = ["1. State \\(H_0: \\mu = \\mu_0\\) and \\(H_1: \\mu \\ne \\mu_0\\) (or \\(\\gt, \\lt\\))",
             "2. Choose significance level \\(\\alpha\\) (usually 0.05)",
             "3. Compute the test statistic from the sample",
             "4. Find the p-value (or compare with the critical value)",
             "5. If \\(p \\le \\alpha\\) → reject \\(H_0\\); else fail to reject"]
    pg.bullets(x0 + 20, y0 + 45, 380, steps, size=11, h=120)
    pg.math("z = \\frac{\\bar{x}-\\mu_0}{\\sigma/\\sqrt{n}}", x0 + 420, y0 + 45, 220, 50, 18)
    pg.math("t = \\frac{\\bar{x}-\\mu_0}{s/\\sqrt{n}},\\ df=n-1", x0 + 400, y0 + 100, 250, 50, 15)
    p = pg.axes(x0 + 60, y0 + 195, 540, 200, (-4, 4), (0, 0.45), "z", "", origin=(-4, 0),
                xticks=(-1.96, 0, 1.96, 2.5))
    p.fill_under(normal_pdf, 1.96, 4, n=14, color=RED, opacity=55)
    p.fill_under(normal_pdf, -4, -1.96, n=14, color=RED, opacity=55)
    p.curve(normal_pdf, -4, 4, n=80, width=2)
    p.vline(0, 0, normal_pdf(0))
    p.vline(2.5, 0, 0.25, color=BLUE, dashed=False, width=2)
    p.text("observed z = 2.5", 2.5, 0.28, size=10, color=BLUE, w=110)
    pg.ask("Is the observed difference real or just luck?",
           "z = 2.5 lands in the red tail: p = 0.012 ≤ α = 0.05 → reject H₀ (unlikely by chance)",
           x0 + 425, y0 + 148, *p.P(2.5, 0.29), w=220)
    p.text("Accept H₀ region (95 %)", 0, 0.2, w=160, size=11, bold=True)
    p.text("Reject H₀<br>α/2 = 2.5 %", -3.1, 0.09, w=90, h=30, size=10, color=RED, raw=True)
    p.text("Reject H₀<br>α/2 = 2.5 %", 3.1, 0.09, w=90, h=30, size=10, color=RED, raw=True)
    pg.html(f'<b>Example:</b> μ₀ = 50, n = 25, x̄ = 55, σ = 10 → z = (55 − 50)/2 = 2.5 → two-tailed '
            f'p = 2·(1 − Φ(2.5)) = 0.012 ≤ 0.05 → <span style="background:{HL}"><b>reject H₀</b></span>',
            x0 + 20, y0 + 430, 620, 40, 12)
    # one vs two tailed
    for k, (lab, two) in enumerate([("Two-tailed (≠)", True), ("Right-tailed (>)", False)]):
        q = pg.axes(x0 + 40 + k * 310, y0 + 500, 240, 80, (-4, 4), (0, 0.45), "", "", origin=(-4, 0), arrow=False)
        if two:
            q.fill_under(normal_pdf, 1.96, 4, n=8, color=RED, opacity=55)
            q.fill_under(normal_pdf, -4, -1.96, n=8, color=RED, opacity=55)
        else:
            q.fill_under(normal_pdf, 1.645, 4, n=8, color=RED, opacity=55)
        q.curve(normal_pdf, -4, 4, n=60, width=1.5)
        pg.text(f"{lab}: critical z = {'±1.96' if two else '1.645'} (α = 0.05)", x0 + 20 + k * 310, y0 + 590, 290,
                18, 10, bold=True, align="center")
    pg.text("Z-test: σ known or n ≥ 30   ·   t-test: σ unknown and small n (heavier tails, see ⑦)", x0 + 20,
            y0 + 620, 620, 18, 10, italic=True, color="#555555", align="center")

    # ------------------------------------------------------------ ⑥ errors and power
    x0, y0 = 400, 1640
    pg.panel(x0, y0, 980, 600, 6, "Type I / Type II Errors & Power", "One-tailed test, α = 0.05")
    p = pg.axes(x0 + 60, y0 + 100, 560, 260, (-3.5, 6), (0, 0.45), "", "", origin=(-3.5, 0))
    c = 1.645
    h1 = lambda v: normal_pdf(v, 2.5, 1)
    p.fill_under(h1, -1, c, n=18, color=LIGHTBLUE, opacity=90)
    p.fill_under(h1, c, 6, n=24, color="#CCFFCC", opacity=90)
    p.fill_under(normal_pdf, c, 3.5, n=12, color=RED, opacity=60)
    p.curve(normal_pdf, -3.5, 3.5, n=70, width=2)
    p.curve(h1, -1, 6, n=70, width=2, color=BLUE)
    p.vline(c, 0, 0.45, color=BLACK, width=1.5)
    p.text("H₀ true", 0, 0.42, size=12, bold=True)
    p.text("H₁ true", 2.5, 0.42, size=12, bold=True, color=BLUE)
    p.text("critical value", c, 0.0, dy=14, size=10)
    pg.note("α = Type I error (false positive)", x0 + 420, y0 + 380, *p.P(2.0, 0.03), w=200, color=RED)
    pg.note("β = Type II error (false negative)", x0 + 60, y0 + 380, *p.P(1.2, 0.08), w=210, color=BLUE)
    pg.note("Power = 1 − β (detect a real effect)", x0 + 470, y0 + 90, *p.P(3.2, 0.2), w=210, color=GREEN)
    pg.table(x0 + 660, y0 + 100, 300, [
        ["Decision ↓ / Truth →", "H₀ true", "H₀ false"],
        ["Reject H₀", '<font color="#FF0000">Type I error (α)</font>', '<font color="#009900">Correct (power)</font>'],
        ["Fail to reject H₀", "Correct (1 − α)", '<font color="#3333FF">Type II error (β)</font>']], size=11, h=110)
    pg.bullets(x0 + 660, y0 + 230, 300, [
        "Lower α → fewer false alarms but more misses (higher β)",
        "Power ↑ with larger n, larger effect size, higher α",
        "Typical target: power ≥ 0.8"], head="Trade-off:", size=11)
    power = 1 - 0.5 * (1 + math.erf((c - 2.5) / math.sqrt(2)))
    pg.ask("How likely am I to detect an effect that really exists?",
           f"power = green area = {power:.2f}; grow it with larger n or a bigger effect size",
           x0 + 660, y0 + 335, *p.P(3.2, 0.2), w=300)
    pg.bullets(x0 + 20, y0 + 430, 940, [
        "ML analogy: Type I = <b>false positive</b>, Type II = <b>false negative</b> (confusion matrix on the Logistic Regression page)",
        "p-value = probability of data at least this extreme <i>if H₀ is true</i> — it is not the probability that H₀ is true",
        "Statistically significant ≠ practically important: always report the effect size / confidence interval",
        "Running many tests inflates false positives → Bonferroni correction (α / m)"], head="Notes:")

    # ------------------------------------------------------------ ⑦ choosing a test + t distribution
    x0 = 1420
    pg.panel(x0, y0, 980, 600, 7, "Choosing a Test  ·  t vs Normal")
    pg.table(x0 + 20, y0 + 50, 600, [
        ["Test", "Question", "Statistic"],
        ["Z-test", "mean vs μ₀, σ known / large n", "\\(z=\\frac{\\bar{x}-\\mu_0}{\\sigma/\\sqrt{n}}\\)"],
        ["One-sample t", "mean vs μ₀, σ unknown", "\\(t=\\frac{\\bar{x}-\\mu_0}{s/\\sqrt{n}}\\)"],
        ["Two-sample t", "two group means differ?", "\\(t=\\frac{\\bar{x}_1-\\bar{x}_2}{\\sqrt{s_1^2/n_1+s_2^2/n_2}}\\)"],
        ["Paired t", "before vs after (same subjects)", "\\(t=\\frac{\\bar{d}}{s_d/\\sqrt{n}}\\)"],
        ["Chi-square", "categorical variables independent?", "\\(\\chi^2=\\sum\\frac{(O-E)^2}{E}\\)"],
        ["ANOVA", "3+ group means differ?", "\\(F=\\frac{MS_{between}}{MS_{within}}\\)"],
        ["Pearson r", "linear correlation ≠ 0?", "\\(t=\\frac{r\\sqrt{n-2}}{\\sqrt{1-r^2}}\\)"]], size=11, h=330)
    p = pg.axes(x0 + 660, y0 + 90, 290, 200, (-4, 4), (0, 0.42), "t", "", origin=(-4, 0), xticks=(-2, 0, 2))
    p.curve(normal_pdf, -4, 4, n=70, width=2)
    for df, col in [(1, RED), (3, ORANGE), (10, GREEN)]:
        p.curve(lambda v, d=df: t_pdf(v, d), -4, 4, n=70, color=col, width=1.2, dashed=True)
    pg.html('<font color="#000000"><b>— Normal</b></font><br><font color="#FF0000">- - t, df = 1</font><br>'
            '<font color="#FF8000">- - t, df = 3</font><br><font color="#009900">- - t, df = 10</font>',
            x0 + 860, y0 + 70, 120, 70, 10)
    pg.text("Smaller df → heavier tails → larger critical values; df ≥ 30 ≈ normal", x0 + 650, y0 + 310, 310, 34,
            11, align="center", italic=True)
    pg.ask("Why use t instead of z with a small sample?",
           "s is noisy at small n → t has fatter tails → wider cut-offs (2.57 at df = 5 vs 1.96)",
           x0 + 650, y0 + 348, *p.P(3, t_pdf(3, 1)), w=310)
    pg.bullets(x0 + 20, y0 + 410, 940, [
        "Non-normal data &amp; small n → non-parametric tests: Mann-Whitney U, Wilcoxon, Kruskal-Wallis",
        "In ML: compare two models' CV scores with a paired t-test; check feature–target dependence with Chi-square (feature selection)",
        "A/B testing = two-sample test on conversion rates (proportions)",
        "All of these rely on the CLT: test statistics are (approximately) normal / t because they are built from means"],
        head="Notes:")

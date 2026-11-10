"""Hypothesis Tests by Data Type page (base: Z-Test)."""
import math
from statistics import NormalDist, mean, stdev
from dio import *

ND = NormalDist()
PARAM, NONPAR, CATEG, CORR = "#CCE5FF", "#FFE6CC", "#D5E8D4", "#E1D5E7"


# ---------------------------------------------------------------- helpers
def t_pdf(x, df):
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return c * (1 + x * x / df) ** (-(df + 1) / 2)


def f_pdf(x, d1, d2):
    if x <= 0: return 0.0
    num = (d1 * x) ** d1 * d2 ** d2 / (d1 * x + d2) ** (d1 + d2)
    beta = math.gamma(d1 / 2) * math.gamma(d2 / 2) / math.gamma((d1 + d2) / 2)
    return math.sqrt(num) / (x * beta)


def chi2_pdf(x, k):
    if x <= 0: return 0.0
    return x ** (k / 2 - 1) * math.exp(-x / 2) / (2 ** (k / 2) * math.gamma(k / 2))


def sf(pdf, a, b, n=4000):
    """Numeric upper-tail area of a pdf from a to b (Simpson)."""
    h = (b - a) / n
    s = pdf(a) + pdf(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * pdf(a + i * h)
    return s * h / 3


def rhombus(pg, label, cx, cy, w=130, h=60, size=10):
    return pg.vertex(E(label), cx - w / 2, cy - h / 2, w, h,
                     f"rhombus;whiteSpace=wrap;html=1;fillColor=#FFF2CC;strokeColor=#000000;fontSize={size};"
                     f"fontFamily=Helvetica;")


def leaf(pg, label, cx, cy, fill, w=130, h=40, size=10):
    return pg.rect(cx - w / 2, cy - h / 2, w, h, label, fill=fill, size=size, raw=True)


def build(pg: Page, cat):
    flowchart(pg)
    linkage(pg)
    ztest(pg)
    ttests(pg)
    anova(pg)
    nonparam(pg)
    categorical(pg)
    correlation(pg)
    asks(pg)


def asks(pg):
    """Interpretation callouts ('Doing this answers ...') — one per panel."""
    x0, y0 = 400, 240      # ① flowchart
    pg.ask("Which test should I run on my data?",
           "follow types → groups → paired? → normal? down to one test",
           x0 + 10, y0 + 70, x0 + 454, y0 + 104, w=420)
    x0, y0 = 1760, 240     # ② linkage matrix
    pg.ask("Which test fits my outcome type and sampling design?",
           "row = outcome type, column = how samples are linked; e.g. numeric + 2 paired → paired t",
           x0 + 20, y0 + 612, x0 + 528, y0 + 130, w=600)
    x0, y0 = 400, 980      # ③ Z-test
    pg.ask("Is IQ = 105 really above the claimed 100?",
           "z = 2.00 is past 1.96 → p = 0.0455 &lt; 0.05: reject H₀ (only just)",
           x0 + 60, y0 + 126, x0 + 434, y0 + 197, w=155)
    x0, y0 = 1080, 980     # ④ t-tests
    pg.ask("Does it matter whether my samples are linked?",
           "yes: the same data gives p = 0.361 treated as independent but p = 0.0030 paired → respect the pairing",
           x0 + 20, y0 + 588, x0 + 560, y0 + 468, w=600)
    x0, y0 = 1760, 980     # ⑤ ANOVA
    pg.ask("Are the three strategy means really different?",
           "F = 11.53 is far beyond the 5 % cut-off 3.68 → p = 0.00093: at least one mean differs (Tukey says which)",
           x0 + 20, y0 + 575, x0 + 600, y0 + 452, w=560)
    x0, y0 = 400, 1680     # ⑥ non-parametric
    pg.ask("Why use ranks instead of raw values?",
           "30 just becomes rank 10: outliers lose their pull, no normality needed",
           x0 + 385, y0 + 32, x0 + 535, y0 + 140, w=230)
    x0, y0 = 1080, 1680    # ⑦ categorical
    pg.ask("What would counts be with no link?", "E = row×col/N: 52.5 vs 60 seen",
           x0 + 405, y0 + 34, x0 + 470, y0 + 118, w=230, size=9)
    x0, y0 = 1760, 1680    # ⑧ correlation
    pg.ask("Is r or ρ the right measure here?",
           "monotonic but curved data: ρ = 0.954 > r = 0.911 → Spearman fits better",
           x0 + 40, y0 + 60, x0 + 381, y0 + 111, w=195)


# ---------------------------------------------------------------- ① flowchart
def flowchart(pg):
    x0, y0 = 400, 240
    pg.panel(x0, y0, 1320, 700, 1, "Which test? — selection flowchart",
             "Decide from (a) the variable types, (b) number of groups, (c) independent or paired, (d) normality")
    start = pg.rect(x0 + 470, y0 + 70, 380, 40, "What types are your variables (outcome ~ predictor)?", bold=True,
                    sw=1.5, size=12)
    A = pg.rect(x0 + 295, y0 + 140, 200, 40, "Numeric outcome ~ Groups", fill=PARAM, bold=True, size=11)
    B = pg.rect(x0 + 835, y0 + 140, 190, 40, "Categorical ~ Categorical", fill=CATEG, bold=True, size=11)
    C = pg.rect(x0 + 1090, y0 + 140, 190, 40, "Numeric ~ Numeric", fill=CORR, bold=True, size=11)
    for t in (A, B, C):
        pg.connect(start, t)

    # --- numeric ~ groups subtree
    g = rhombus(pg, "How many groups? Paired?", x0 + 395, y0 + 225, 170, 56)
    pg.connect(A, g)
    cols = [("1 sample (vs μ₀)", "Z-test (σ known)<br>/ one-sample t", "Wilcoxon<br>signed-rank"),
            ("2 groups · independent", "Independent t<br>(Welch if σ₁≠σ₂)", "Mann-Whitney U"),
            ("2 groups · paired", "Paired t", "Wilcoxon<br>signed-rank"),
            ("3+ groups · independent", "One-way ANOVA", "Kruskal-Wallis H"),
            ("3+ groups · paired", "Repeated-measures<br>ANOVA", "Friedman")]
    for k, (lab, par, non) in enumerate(cols):
        cx = x0 + 95 + k * 150
        lb = pg.rect(cx - 68, y0 + 290, 136, 36, lab, size=10, bold=True)
        pg.connect(g, lb)
        n = rhombus(pg, "Normal? (or n ≥ 30)", cx, y0 + 378, 118, 54, 9)
        pg.connect(lb, n)
        a = leaf(pg, ("<b>★ " if k == 0 else "<b>") + par + "</b>", cx, y0 + 475, PARAM, 136, 44)
        b = leaf(pg, "<b>" + non + "</b>", cx, y0 + 560, NONPAR, 136, 40)
        pg.connect(n, a, label="yes")
        pg.line([(cx - 59, y0 + 378), (cx - 74, y0 + 378), (cx - 74, y0 + 560), (cx - 68, y0 + 560)],
                arrow="classic", label="no")

    # --- categorical subtree
    p = rhombus(pg, "Paired / same subjects?", x0 + 930, y0 + 225, 140, 56)
    pg.connect(B, p)
    e = rhombus(pg, "All expected counts ≥ 5?", x0 + 870, y0 + 345, 140, 60, 9)
    pg.connect(p, e, label="no")
    chi = leaf(pg, "<b>Chi-square test<br>of independence</b>", x0 + 870, y0 + 475, CATEG, 130, 44)
    fis = leaf(pg, "<b>Fisher's exact</b>", x0 + 870, y0 + 560, CATEG, 130, 40)
    mcn = leaf(pg, "<b>McNemar</b><br>(2×2 paired)", x0 + 1000, y0 + 475, CATEG, 110, 44)
    pg.connect(e, chi, label="yes")
    pg.line([(x0 + 800, y0 + 345), (x0 + 795, y0 + 345), (x0 + 795, y0 + 560), (x0 + 805, y0 + 560)],
            arrow="classic", label="no")
    pg.line([(x0 + 1000, y0 + 225), (x0 + 1000, y0 + 453)], arrow="classic", label="yes")
    pg.text("1 variable vs expected proportions → Chi-square goodness-of-fit / binomial test", x0 + 800, y0 + 590,
            250, 30, 9, italic=True, color="#555555")

    # --- numeric ~ numeric subtree
    r1 = rhombus(pg, "Linear & bivariate normal?", x0 + 1185, y0 + 225, 150, 60, 9)
    pg.connect(C, r1)
    pea = leaf(pg, "<b>Pearson r</b>", x0 + 1120, y0 + 475, CORR, 120, 40)
    r2 = rhombus(pg, "Small n / many ties?", x0 + 1250, y0 + 345, 112, 58, 9)
    spe = leaf(pg, "<b>Spearman ρ</b>", x0 + 1250, y0 + 475, CORR, 110, 40)
    ken = leaf(pg, "<b>Kendall τ</b>", x0 + 1250, y0 + 560, CORR, 110, 40)
    pg.line([(x0 + 1120, y0 + 248), (x0 + 1120, y0 + 455)], arrow="classic", label="yes")
    pg.connect(r1, r2, label="no")
    pg.connect(r2, spe, label="no")
    pg.line([(x0 + 1306, y0 + 345), (x0 + 1312, y0 + 345), (x0 + 1312, y0 + 560), (x0 + 1305, y0 + 560)],
            arrow="classic", label="yes")
    pg.text("Predict y from x → regression t-test on slope", x0 + 1060, y0 + 615, 250, 30, 9, italic=True,
            color="#555555")

    # legend
    ly = y0 + 645
    for k, (fill, lab) in enumerate([(PARAM, "Parametric (normal data, means)"),
                                     (NONPAR, "Non-parametric (ranks / medians)"),
                                     (CATEG, "Categorical (counts)"), (CORR, "Association (correlation)")]):
        pg.rect(x0 + 30 + k * 310, ly, 22, 16, fill=fill)
        pg.text(lab, x0 + 58 + k * 310, ly - 2, 250, 20, 11)


# ---------------------------------------------------------------- ② data linkage matrix
def linkage(pg):
    x0, y0 = 1760, 240
    pg.panel(x0, y0, 640, 700, 2, "Data Linkage Matrix", "Outcome type × how the samples are linked")
    hdr = ["Outcome ↓ / Design →", "1 sample", "2 indep.", "2 paired", "3+ indep.", "3+ paired", "Assoc."]
    rows = [hdr,
            ["<b>Numeric</b><br>(normal)", "Z / 1-sample t", "Indep. t<br>(Welch)", "Paired t", "One-way<br>ANOVA",
             "RM-ANOVA", "Pearson r"],
            ["<b>Numeric non-normal</b><br>/ <b>Ordinal</b>", "Wilcoxon<br>signed-rank", "Mann-<br>Whitney U",
             "Wilcoxon<br>signed-rank", "Kruskal-<br>Wallis", "Friedman", "Spearman ρ<br>Kendall τ"],
            ["<b>Binary /</b><br><b>Categorical</b>", "χ² GOF /<br>binomial", "χ² / Fisher<br>/ 2-prop Z",
             "McNemar", "χ² (r×c)", "Cochran's Q", "Cramér's V"]]
    pg.table(x0 + 15, y0 + 60, 610, rows, size=10, h=200)
    # independent vs paired illustration
    yy = y0 + 200
    pg.text("Independent samples", x0 + 30, yy, 260, 20, 13, bold=True, align="center")
    pg.text("Paired / related samples", x0 + 340, yy, 270, 20, 13, bold=True, align="center")
    r = seeded(11)
    for k, (cx, col, lab) in enumerate([(x0 + 95, BLUE, "Group A"), (x0 + 225, RED, "Group B")]):
        pg.ellipse(cx - 55, yy + 35, 110, 120)
        for _ in range(7):
            pg.dot(cx + r.uniform(-35, 35), yy + 95 + r.uniform(-40, 40), r=4, color=col)
        pg.text(lab, cx - 50, yy + 160, 100, 18, 11, align="center", color=col)
    pg.text("different subjects in each group", x0 + 30, yy + 180, 260, 18, 10, italic=True, align="center")
    for j in range(6):
        ya = yy + 45 + j * 20
        bx, ax = x0 + 400, x0 + 540
        a = 0.3 + r.random()
        pg.dot(bx, ya, r=4, color=BLUE)
        pg.dot(ax, ya - 6 + r.uniform(-6, 2), r=4, color=RED)
        pg.line([(bx, ya), (ax, ya - 4)], color=GREY, width=0.75)
        pg.text(f"S{j + 1}", bx - 36, ya - 8, 30, 16, 9, align="right")
    pg.text("Before", x0 + 370, yy + 165, 60, 18, 11, align="center", color=BLUE)
    pg.text("After", x0 + 510, yy + 165, 60, 18, 11, align="center", color=RED)
    pg.text("same subject measured twice (S1 … S6)", x0 + 340, yy + 180, 270, 18, 10, italic=True, align="center")
    pg.bullets(x0 + 20, y0 + 410, 600, [
        "<b>Independent</b>: different people / assets / days in each group (A/B test, two strategies on different stocks)",
        "<b>Paired</b>: same unit measured twice or matched pairs (before/after, two models on the <i>same</i> CV folds)",
        "<b>Repeated measures</b>: same unit measured 3+ times (model A, B, C on the same folds)",
        "Paired designs remove between-subject noise → more power; treating paired data as independent is a common error"],
        head="Linkage types:", size=11)


# ---------------------------------------------------------------- ③ Z-tests
def ztest(pg):
    x0, y0 = 400, 980
    pg.panel(x0, y0, 640, 660, 3, "★ Z-Test (one-sample & two-proportion)", "σ known or large n → statistic ~ N(0, 1)")
    mu0, sig, n, xb = 100, 15, 36, 105
    se = sig / math.sqrt(n)
    z = (xb - mu0) / se
    p = 2 * (1 - ND.cdf(z))
    pg.math("z = \\frac{\\bar{x}-\\mu_0}{\\sigma/\\sqrt{n}}", x0 + 20, y0 + 60, 200, 50, 18)
    pg.html(f"<b>Example:</b> IQ claim μ₀ = {mu0}, σ = {sig}, n = {n}, x̄ = {xb}<br>"
            f"SE = 15/√36 = {se:.1f} → z = (105 − 100)/2.5 = <b>{z:.2f}</b><br>"
            f'two-tailed p = 2(1 − Φ({z:.2f})) = <span style="background:{HL}"><b>{p:.4f}</b> &lt; 0.05 → reject H₀</span>',
            x0 + 230, y0 + 58, 400, 60, 11)
    c = pg.axes(x0 + 60, y0 + 135, 520, 160, (-4, 4), (0, 0.45), "z", "", origin=(-4, 0), xticks=(-1.96, 0, 1.96))
    c.fill_under(normal_pdf, 1.96, 4, n=12, color=RED, opacity=55)
    c.fill_under(normal_pdf, -4, -1.96, n=12, color=RED, opacity=55)
    c.curve(normal_pdf, -4, 4, n=80, width=2)
    c.vline(z, 0, 0.3, color=BLUE, dashed=False, width=2)
    c.text(f"observed z = {z:.2f}", z, 0.32, size=10, color=BLUE, w=120)
    c.text("α/2 = 0.025", -3.0, 0.08, size=10, color=RED)
    c.text("α/2 = 0.025", 3.0, 0.08, size=10, color=RED)
    c.text("fail-to-reject region (95 %)", 0, 0.15, size=11, bold=True, w=180)

    # two-proportion z (A/B test)
    yy = y0 + 345
    pg.text("Two-proportion Z-test: A/B test of conversion rate", x0 + 20, yy, 600, 20, 13, bold=True)
    xa, na, xbb, nb = 120, 1000, 150, 1000
    pa, pb = xa / na, xbb / nb
    pp = (xa + xbb) / (na + nb)
    se2 = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    z2 = (pb - pa) / se2
    p2 = 2 * (1 - ND.cdf(z2))
    pg.math("z = \\frac{\\hat{p}_B-\\hat{p}_A}{\\sqrt{\\hat{p}(1-\\hat{p})\\left(\\frac{1}{n_A}+\\frac{1}{n_B}\\right)}}",
            x0 + 20, yy + 30, 330, 60, 15)
    pg.note("pooled p̂ = (120 + 150)/2000", x0 + 120, yy + 98, x0 + 230, yy + 78, w=180)
    pg.html(f"A: {xa}/{na} = <b>{pa:.3f}</b> &nbsp; B: {xbb}/{nb} = <b>{pb:.3f}</b><br>"
            f"p̂ = {pp:.3f}, SE = {se2:.4f}<br>z = 0.03 / {se2:.4f} = <b>{z2:.2f}</b><br>"
            f'p = <span style="background:{HL}"><b>{p2:.4f}</b></span> → just significant at α = 0.05',
            x0 + 20, yy + 125, 330, 70, 11)
    b = pg.axes(x0 + 400, yy + 40, 200, 150, (0, 3), (0, 0.2), "", "rate", label_size=11, yticks=(0.05, 0.1, 0.15))
    for k, (lab, pr, nn, col) in enumerate([("A", pa, na, LIGHTBLUE), ("B", pb, nb, LIGHTRED)]):
        b.bars([(k + 1, pr)], 0.6, fill=col)
        e = 1.96 * math.sqrt(pr * (1 - pr) / nn)
        b.poly([(k + 1, pr - e), (k + 1, pr + e)], width=1.5)
        b.poly([(k + 0.9, pr + e), (k + 1.1, pr + e)], width=1.5)
        b.poly([(k + 0.9, pr - e), (k + 1.1, pr - e)], width=1.5)
        b.text(lab, k + 1, 0, dy=12, size=12, bold=True)
    pg.text("bars: 95 % CI of each rate", x0 + 400, yy + 210, 200, 16, 9, italic=True, align="center")
    pg.bullets(x0 + 20, y0 + 580, 600, [
        "Z when σ is known or n is large (CLT); otherwise use t (next panel)",
        "Proportions: Z valid when n·p ≥ 10 and n·(1 − p) ≥ 10 in each group"], size=11)


def tp(t, df):
    """two-tailed p-value of t with df."""
    return 2 * sf(lambda v: t_pdf(v, df), abs(t), abs(t) + 60)


A_IND = [23, 25, 28, 30, 26, 24, 27, 29]
B_IND = [30, 33, 29, 35, 31, 34, 32, 28]
BEFORE = [72, 75, 68, 80, 77, 70]
AFTER = [75, 79, 70, 82, 81, 71]


# ---------------------------------------------------------------- ④ t-tests
def ttests(pg):
    x0, y0 = 1080, 980
    pg.panel(x0, y0, 640, 660, 4, "t-Tests: independent vs paired", "σ unknown → t distribution with df")
    ma, mb, sa, sb = mean(A_IND), mean(B_IND), stdev(A_IND), stdev(B_IND)
    na, nb = len(A_IND), len(B_IND)
    se = math.sqrt(sa ** 2 / na + sb ** 2 / nb)
    t = (mb - ma) / se
    df = se ** 4 / ((sa ** 2 / na) ** 2 / (na - 1) + (sb ** 2 / nb) ** 2 / (nb - 1))
    p = tp(t, df)
    pg.text("Independent (Welch) t-test", x0 + 20, y0 + 60, 290, 20, 13, bold=True)
    c = pg.axes(x0 + 30, y0 + 95, 270, 120, (18, 42), (0, 0.22), "", "", origin=(18, 0), arrow=False,
                xticks=(20, 25, 30, 35, 40))
    c.curve(lambda v: normal_pdf(v, ma, sa), 18, 42, n=60, color=BLUE, width=2)
    c.curve(lambda v: normal_pdf(v, mb, sb), 18, 42, n=60, color=RED, width=2)
    c.vline(ma, 0, 0.2, color=BLUE)
    c.vline(mb, 0, 0.2, color=RED)
    c.scatter([(v, 0.012) for v in A_IND], color=BLUE, r=3)
    c.scatter([(v, 0.028) for v in B_IND], color=RED, r=3)
    c.text(f"x̄A = {ma:.2f}", ma, 0.215, size=10, color=BLUE, dx=-30)
    c.text(f"x̄B = {mb:.2f}", mb, 0.215, size=10, color=RED, dx=30)
    pg.math("t = \\frac{\\bar{x}_B-\\bar{x}_A}{\\sqrt{\\frac{s_A^2}{n_A}+\\frac{s_B^2}{n_B}}}", x0 + 20, y0 + 240,
            280, 60, 15)
    pg.html(f"sA = {sa:.2f}, sB = {sb:.2f}, n = 8 + 8<br>t = {mb - ma:.2f} / {se:.3f} = <b>{t:.2f}</b>, "
            f"Welch df = {df:.1f}<br>p = <span style=\"background:{HL}\"><b>{p:.5f}</b></span> → means differ",
            x0 + 20, y0 + 305, 290, 56, 11)
    d = [a - b for a, b in zip(AFTER, BEFORE)]
    md, sd, n = mean(d), stdev(d), len(d)
    tpair = md / (sd / math.sqrt(n))
    ppair = tp(tpair, n - 1)
    px = x0 + 340
    pg.text("Paired t-test (before → after)", px, y0 + 60, 290, 20, 13, bold=True)
    q = pg.axes(px + 40, y0 + 95, 200, 160, (0, 3), (65, 85), "", "", label_size=11, origin=(0, 65),
                yticks=(70, 75, 80))
    for b, a in zip(BEFORE, AFTER):
        q.poly([(1, b), (2, a)], color=GREY, width=1)
    q.scatter([(1, b) for b in BEFORE], color=BLUE, r=4)
    q.scatter([(2, a) for a in AFTER], color=RED, r=4)
    q.text("Before", 1, 65, dy=12, size=10, color=BLUE)
    q.text("After", 2, 65, dy=12, size=10, color=RED)
    pg.math("t = \\frac{\\bar{d}}{s_d/\\sqrt{n}},\\ df=n-1", px, y0 + 285, 280, 46, 15)
    pg.html(f"d = after − before = {d}<br>d̄ = {md:.2f}, s_d = {sd:.2f}, n = {n}<br>"
            f"t = <b>{tpair:.2f}</b>, df = {n - 1} → p = <span style=\"background:{HL}\"><b>{ppair:.4f}</b></span>",
            px, y0 + 332, 290, 56, 11)
    sw = math.sqrt(stdev(AFTER) ** 2 / n + stdev(BEFORE) ** 2 / n)
    tw = (mean(AFTER) - mean(BEFORE)) / sw
    pw = tp(tw, 2 * n - 2)
    pg.rect(x0 + 20, y0 + 405, 600, 56,
            f"<b>Same before/after data, wrong linkage:</b> independent t = {tw:.2f}, p = {pw:.3f} → "
            f'<font color="#FF0000"><b>not significant</b></font><br>The paired t removes the subject-to-subject spread '
            f"(baselines 68 … 80) and tests only the changes d → p = {ppair:.4f}",
            fill="#FFF8E1", stroke=ORANGE, dashed=True, size=11, align="left", raw=True)
    pg.bullets(x0 + 20, y0 + 495, 600, [
        "<b>One-sample t</b>: \\(t = (\\bar{x}-\\mu_0)/(s/\\sqrt{n})\\), df = n − 1",
        "<b>Student's t</b> pools variances (assumes σ₁ = σ₂); <b>Welch</b> does not — use Welch by default",
        "Assumptions: normal (or n ≥ 30 per group), independent observations; paired: differences normal",
        "Non-normal / ordinal → Mann-Whitney U (independent) or Wilcoxon signed-rank (paired) ⑥"],
        head="Notes:", size=11)
    pg.html(f"<b>Why linkage matters (see ④):</b> the same 6 before/after pairs give p = {ppair:.4f} with a paired "
            f"t-test but p = {pw:.3f} if wrongly treated as independent samples.", 1780, 800, 600, 40, 11)


ANOVA_G = {"Strategy A": [2.1, 2.5, 1.8, 2.9, 2.4, 2.2],
           "Strategy B": [3.0, 3.4, 2.8, 3.6, 3.1, 3.3],
           "Strategy C": [2.6, 2.2, 2.9, 2.5, 3.0, 2.4]}


# ---------------------------------------------------------------- ⑤ ANOVA
def anova(pg):
    x0, y0 = 1760, 980
    pg.panel(x0, y0, 640, 660, 5, "One-way ANOVA (3+ independent groups)",
             "Is the variation BETWEEN group means large compared with variation WITHIN groups?")
    groups = list(ANOVA_G.items())
    allv = [v for _, g in groups for v in g]
    gm = mean(allv)
    k, N = len(groups), len(allv)
    ssb = sum(len(g) * (mean(g) - gm) ** 2 for _, g in groups)
    ssw = sum((v - mean(g)) ** 2 for _, g in groups for v in g)
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw
    fp = lambda v: f_pdf(v, dfb, dfw)
    pF = sf(fp, F, F + 200, 8000)
    lo, hi = 0.5, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if sf(fp, mid, mid + 200, 4000) > 0.05: lo = mid
        else: hi = mid
    Fc = (lo + hi) / 2
    p = pg.axes(x0 + 50, y0 + 80, 250, 220, (0.4, 3.6), (1.5, 4.0), "", "return %", label_size=11,
                origin=(0.4, 1.5), yticks=(2, 2.5, 3, 3.5))
    cols = [BLUE, RED, GREEN]
    for i, ((name, g), col) in enumerate(zip(groups, cols)):
        xc = i + 1
        p.scatter([(xc + (j - 2.5) * 0.06, v) for j, v in enumerate(g)], color=col, r=3.5)
        p.poly([(xc - 0.3, mean(g)), (xc + 0.3, mean(g))], color=col, width=2.5)
        p.text(name, xc, 1.5, dy=12, size=10, color=col, w=90)
        p.poly([(xc + 0.36, gm), (xc + 0.36, mean(g))], color=ORANGE, width=1.5, arrow="classic")
    p.hline(gm, color=BLACK)
    p.text(f"grand mean = {gm:.2f}", 0.4, gm, dx=4, dy=-9, size=9, align="left", w=120)
    g0 = groups[0][1]
    p.poly([(1 - 0.15, g0[2]), (1 - 0.15, mean(g0))], color=GREY, width=1.2, arrow="classic")
    pg.note("between-group (orange)", x0 + 400, y0 + 90, *p.P(2.36, 3.0), w=150, color=ORANGE)
    pg.note("within-group (grey)", x0 + 400, y0 + 280, *p.P(0.85, 1.9), w=150, color=GREY)
    pg.math("F = \\frac{MS_{between}}{MS_{within}} = \\frac{SS_B/(k-1)}{SS_W/(N-k)}", x0 + 320, y0 + 140,
            310, 56, 14)
    pg.math("SS_B=\\sum n_j(\\bar{x}_j-\\bar{x})^2", x0 + 380, y0 + 200, 240, 30, 12)
    pg.math("SS_W=\\sum\\sum(x_{ij}-\\bar{x}_j)^2", x0 + 380, y0 + 232, 240, 30, 12)
    pg.table(x0 + 20, y0 + 330, 360, [
        ["Source", "SS", "df", "MS", "F", "p"],
        ["Between", f"{ssb:.3f}", dfb, f"{msb:.3f}", f"<b>{F:.2f}</b>", f"<b>{pF:.5f}</b>"],
        ["Within", f"{ssw:.3f}", dfw, f"{msw:.3f}", "", ""],
        ["Total", f"{ssb + ssw:.3f}", N - 1, "", "", ""]], size=10, h=90, highlight_rows=(1,))
    f = pg.axes(x0 + 410, y0 + 340, 200, 100, (0, 8), (0, 1.05), "F", "", label_size=11, origin=(0, 0),
                xticks=(0, round(Fc, 2), 8))
    f.fill_under(fp, Fc, 8, n=10, color=RED, opacity=55)
    f.curve(fp, 0.02, 8, n=60, width=2)
    pg.text(f"F(2, 15) · critical = {Fc:.2f} (α = 0.05)", x0 + 400, y0 + 458, 220, 16, 10, align="center")
    pg.text(f"observed F = {F:.2f} →", x0 + 480, y0 + 335, 130, 16, 10, color=BLUE, align="right")
    pg.bullets(x0 + 20, y0 + 490, 600, [
        f"F = {F:.2f} > {Fc:.2f} (p = {pF:.5f}) → at least one strategy mean differs",
        "ANOVA does not say <i>which</i> one → post-hoc <b>Tukey HSD</b> (or Bonferroni pairwise t-tests)",
        "Assumptions: normal residuals, equal variances (Levene), independent groups; else Welch ANOVA / Kruskal-Wallis",
        "Same subjects in every group → repeated-measures ANOVA (non-parametric: Friedman)"],
        head="Interpretation:", size=11)


def ranks(vals):
    """average ranks (ties share the mean rank)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]: j += 1
        for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))


# ---------------------------------------------------------------- ⑥ non-parametric
def nonparam(pg):
    x0, y0 = 400, 1680
    pg.panel(x0, y0, 640, 660, 6, "Non-parametric tests: work on RANKS",
             "No normality assumption · robust to outliers · for ordinal or skewed data")
    A, B = [12, 15, 9, 20, 14], [22, 18, 25, 16, 30]
    allv = A + B
    rk = ranks(allv)
    p = Plot(pg, x0 + 50, y0 + 90, 540, 60, (5, 32), (0, 1))
    pg.line([(x0 + 40, y0 + 150), (x0 + 600, y0 + 150)], arrow="classic")
    for v, r_, grp in zip(allv, rk, ["A"] * 5 + ["B"] * 5):
        col = BLUE if grp == "A" else RED
        px, _ = p.P(v, 0)
        pg.dot(px, y0 + 140, r=5, color=col)
        pg.text(str(v), px - 15, y0 + 112, 30, 14, 9, align="center", color=col)
        pg.text(f"{r_:g}", px - 15, y0 + 158, 30, 16, 11, bold=True, align="center")
    pg.text("value", x0 + 0, y0 + 112, 45, 14, 9, color="#555555")
    pg.text("rank", x0 + 0, y0 + 158, 45, 16, 10, color="#555555")
    rA = sum(r_ for r_, g in zip(rk, ["A"] * 5 + ["B"] * 5) if g == "A")
    nA, nB = len(A), len(B)
    UA = rA - nA * (nA + 1) / 2
    U = min(UA, nA * nB - UA)
    pg.text("Mann-Whitney U (2 independent groups) — A = blue, B = red", x0 + 20, y0 + 190, 600, 20, 13, bold=True)
    pg.math("U_A = R_A - \\frac{n_A(n_A+1)}{2},\\quad U=\\min(U_A,\\ n_An_B-U_A)", x0 + 20, y0 + 215, 420, 40, 14)
    pg.html(f"R_A = 1+2+3+4+7 = {rA:g} → U_A = {rA:g} − 15 = {UA:g}<br>U = <b>{U:g}</b> ≤ critical 2 (n = 5, 5, α = 0.05) "
            f'→ <span style="background:{HL}"><b>medians differ</b></span>', x0 + 20, y0 + 258, 600, 36, 11)
    # Wilcoxon signed-rank on the paired data
    d = [a - b for a, b in zip(AFTER, BEFORE)]
    rd = ranks([abs(v) for v in d])
    wp = sum(r_ for r_, v in zip(rd, d) if v > 0)
    wm = sum(r_ for r_, v in zip(rd, d) if v < 0)
    pg.text("Wilcoxon signed-rank (paired, same data as ④)", x0 + 20, y0 + 305, 600, 20, 13, bold=True)
    pg.table(x0 + 20, y0 + 330, 360, [["d", *d], ["|d| rank", *[f"{v:g}" for v in rd]],
                                      ["sign", *["+" if v > 0 else "−" for v in d]]], header=False, size=10, h=66)
    pg.html(f"W⁺ = {wp:g}, W⁻ = {wm:g} → W = min = <b>{min(wp, wm):g}</b><br>"
            f"critical (n = 6, α = 0.05) = 0 → reject H₀ (exact p = 0.031)", x0 + 395, y0 + 330, 230, 66, 11)
    # Kruskal-Wallis on the ANOVA groups
    groups = list(ANOVA_G.values())
    allg = [v for g in groups for v in g]
    rg = ranks(allg)
    N = len(allg)
    H, pos = 0.0, 0
    for g in groups:
        R = sum(rg[pos:pos + len(g)])
        H += R ** 2 / len(g)
        pos += len(g)
    H = 12 / (N * (N + 1)) * H - 3 * (N + 1)
    pk = sf(lambda v: chi2_pdf(v, 2), H, H + 100)
    pg.text("Kruskal-Wallis H (3+ independent groups, data of ⑤)", x0 + 20, y0 + 412, 600, 20, 13, bold=True)
    pg.math("H = \\frac{12}{N(N+1)}\\sum_{j}\\frac{R_j^2}{n_j} - 3(N+1)\\ \\sim\\ \\chi^2_{k-1}", x0 + 20,
            y0 + 437, 400, 44, 14)
    pg.html(f"H = <b>{H:.2f}</b>, df = 2 → p = <b>{pk:.4f}</b><br>(ANOVA gave p = 0.0009)", x0 + 430, y0 + 440,
            200, 40, 11)
    pg.table(x0 + 20, y0 + 500, 600, [
        ["Parametric", "Non-parametric equivalent", "Compares"],
        ["One-sample / paired t", "Wilcoxon signed-rank", "median of (differences)"],
        ["Independent t", "Mann-Whitney U (Wilcoxon rank-sum)", "distributions / medians"],
        ["One-way ANOVA", "Kruskal-Wallis H", "3+ distributions"],
        ["Repeated-measures ANOVA", "Friedman", "3+ linked samples"],
        ["Pearson r", "Spearman ρ / Kendall τ", "monotonic association"]], size=10, h=140)


# ---------------------------------------------------------------- ⑦ categorical
def categorical(pg):
    x0, y0 = 1080, 1680
    pg.panel(x0, y0, 640, 660, 7, "Categorical data: Chi-square · Fisher · McNemar",
             "Counts in a contingency table")
    O = [[60, 40], [45, 55]]
    R = [sum(r) for r in O]
    C = [O[0][j] + O[1][j] for j in range(2)]
    N = sum(R)
    Ex = [[R[i] * C[j] / N for j in range(2)] for i in range(2)]
    chi = sum((O[i][j] - Ex[i][j]) ** 2 / Ex[i][j] for i in range(2) for j in range(2))
    pchi = 2 * (1 - ND.cdf(math.sqrt(chi)))
    pg.text("Does a buy signal relate to the next-day direction?", x0 + 20, y0 + 60, 600, 20, 12, bold=True)
    pg.text("Observed O", x0 + 20, y0 + 85, 280, 18, 11, bold=True, align="center")
    pg.table(x0 + 20, y0 + 105, 280, [["", "Up", "Down", "Total"],
                                      ["Signal", O[0][0], O[0][1], R[0]],
                                      ["No signal", O[1][0], O[1][1], R[1]],
                                      ["Total", C[0], C[1], N]], size=11, h=100)
    pg.text("Expected E = row × col / N", x0 + 330, y0 + 85, 290, 18, 11, bold=True, align="center")
    pg.table(x0 + 330, y0 + 105, 290, [["", "Up", "Down"],
                                       ["Signal", f"{Ex[0][0]:.1f}", f"{Ex[0][1]:.1f}"],
                                       ["No signal", f"{Ex[1][0]:.1f}", f"{Ex[1][1]:.1f}"]], size=11, h=76,
             highlight_rows=(1,))
    pg.note("100 × 105 / 200 = 52.5", x0 + 470, y0 + 190, x0 + 475, y0 + 150, w=150, color=RED)
    pg.math("\\chi^2=\\sum\\frac{(O-E)^2}{E},\\quad df=(r-1)(c-1)", x0 + 20, y0 + 220, 360, 44, 15)
    pg.html(f"χ² = 2·(7.5²/52.5) + 2·(7.5²/47.5) = <b>{chi:.2f}</b>, df = 1<br>"
            f'p = <span style="background:{HL}"><b>{pchi:.4f}</b></span> &lt; 0.05 → signal and direction are related',
            x0 + 20, y0 + 268, 380, 40, 11)
    c = pg.axes(x0 + 420, y0 + 215, 190, 100, (0, 8), (0, 1.0), "\\chi^2", "", label_size=11, origin=(0, 0),
                xticks=(0, 3.84, 8))
    c.fill_under(lambda v: chi2_pdf(v, 1), 3.84, 8, n=10, color=RED, opacity=55)
    c.curve(lambda v: chi2_pdf(v, 1), 0.08, 8, n=60, width=2)
    c.vline(chi, 0, 0.5, color=BLUE, dashed=False, width=2)
    c.text(f"{chi:.2f}", chi, 0.55, size=10, color=BLUE)
    # Fisher
    a, b, cc, dd = 7, 1, 2, 6
    n1, n2, m1 = a + b, cc + dd, a + cc
    nt = n1 + n2
    def hyp(x):
        return math.comb(n1, x) * math.comb(n2, m1 - x) / math.comb(nt, m1)
    pobs = hyp(a)
    pf = sum(hyp(x) for x in range(max(0, m1 - n2), min(n1, m1) + 1) if hyp(x) <= pobs + 1e-12)
    pg.text("Fisher's exact (small counts: expected < 5)", x0 + 20, y0 + 330, 600, 20, 13, bold=True)
    pg.table(x0 + 20, y0 + 355, 200, [["", "Win", "Loss"], ["Model A", a, b], ["Model B", cc, dd]], size=11, h=72)
    pg.math("p=\\frac{\\binom{a+b}{a}\\binom{c+d}{c}}{\\binom{n}{a+c}}", x0 + 235, y0 + 350, 180, 56, 14)
    pg.html(f"sum of tables as or more extreme<br>two-sided p = <b>{pf:.4f}</b>", x0 + 420, y0 + 360, 200, 40, 11)
    # McNemar
    bb, cm = 15, 5
    chim = (bb - cm) ** 2 / (bb + cm)
    pm = 2 * (1 - ND.cdf(math.sqrt(chim)))
    pg.text("McNemar (paired yes/no: same cases, two classifiers)", x0 + 20, y0 + 445, 600, 20, 13, bold=True)
    pg.table(x0 + 20, y0 + 470, 230, [["", "B correct", "B wrong"], ["A correct", 70, f"<b>b = {bb}</b>"],
                                      ["A wrong", f"<b>c = {cm}</b>", 10]], size=11, h=72)
    pg.math("\\chi^2=\\frac{(b-c)^2}{b+c}", x0 + 270, y0 + 470, 150, 50, 15)
    pg.html(f"= (15 − 5)²/20 = <b>{chim:.1f}</b><br>p = <b>{pm:.4f}</b> → the classifiers differ<br>"
            f"(only discordant pairs b, c matter)", x0 + 420, y0 + 470, 210, 60, 11)
    pg.bullets(x0 + 20, y0 + 560, 600, [
        "Chi-square needs all expected counts ≥ 5 (else Fisher); goodness-of-fit compares one variable with expected proportions",
        "Effect size: Cramér's V = \\(\\sqrt{\\chi^2/(N\\cdot(\\min(r,c)-1))}\\)",
        "Ordinal categories → use rank tests or Chi-square for trend"], size=11)


# ---------------------------------------------------------------- ⑧ correlation + assumption checks
def correlation(pg):
    x0, y0 = 1760, 1680
    pg.panel(x0, y0, 640, 660, 8, "Correlation tests & Assumption checks")
    r = seeded(5)
    xs = list(range(1, 16))
    ys = [math.exp(0.28 * x) + r.gauss(0, 2.5) for x in xs]
    rp = pearson(xs, ys)
    rs = pearson(ranks(xs), ranks(ys))
    n = len(xs)
    tr = rp * math.sqrt(n - 2) / math.sqrt(1 - rp ** 2)
    p = pg.axes(x0 + 50, y0 + 70, 240, 170, (0, 16), (min(ys) - 5, max(ys) + 5), "x", "y", label_size=12)
    p.scatter(list(zip(xs, ys)), r=4)
    b1 = rp * stdev(ys) / stdev(xs)
    b0 = mean(ys) - b1 * mean(xs)
    p.curve(lambda v: b0 + b1 * v, 0, 16, n=2, curved=False, color=RED, dashed=True)
    pg.html(f"Monotonic but curved data<br>Pearson r = <b>{rp:.3f}</b><br>Spearman ρ = "
            f'<span style="background:{HL}"><b>{rs:.3f}</b></span><br>(ranks almost in order → ρ &gt; r)',
            x0 + 320, y0 + 70, 300, 70, 11)
    pg.math("t=\\frac{r\\sqrt{n-2}}{\\sqrt{1-r^2}},\\ df=n-2", x0 + 320, y0 + 150, 300, 46, 14)
    pg.html(f"H₀: ρ = 0 → t = {tr:.2f}, df = {n - 2}, p &lt; 0.001", x0 + 320, y0 + 200, 300, 20, 11)
    pg.math("\\rho_s = 1-\\frac{6\\sum d_i^2}{n(n^2-1)}", x0 + 320, y0 + 222, 300, 40, 13)
    # Q-Q plots
    pg.text("Assumption check: Q-Q plots (sample vs normal quantiles)", x0 + 20, y0 + 280, 600, 20, 13, bold=True)
    for k, (lab, data) in enumerate([("Normal sample → straight line ✔", sorted(r.gauss(0, 1) for _ in range(30))),
                                     ("Skewed sample → curved ✘", sorted(r.expovariate(1) - 1 for _ in range(30)))]):
        m, s = mean(data), stdev(data)
        zs = [(d - m) / s for d in data]
        th = [ND.inv_cdf((i + 0.5) / 30) for i in range(30)]
        q = pg.axes(x0 + 50 + k * 300, y0 + 315, 200, 140, (-2.6, 2.6), (-2.6, 3.4), "", "", origin=(-2.6, -2.6))
        q.poly([(-2.6, -2.6), (2.6, 2.6)], color=RED, width=1)
        q.scatter([(a, min(b, 3.3)) for a, b in zip(th, zs)], r=3, color=BLUE if k == 0 else ORANGE)
        pg.text(lab, x0 + 20 + k * 300, y0 + 462, 260, 18, 11, bold=True, color=GREEN if k == 0 else RED,
                align="center")
    pg.text("theoretical quantile →", x0 + 60, y0 + 478, 200, 14, 9, color="#555555")
    pg.table(x0 + 20, y0 + 505, 600, [
        ["Check", "H₀", "If p < 0.05"],
        ["Shapiro-Wilk", "data are normal", "use non-parametric test / transform"],
        ["Levene / Bartlett", "equal variances", "Welch t / Welch ANOVA"],
        ["Durbin-Watson / Ljung-Box", "no autocorrelation", "time-series methods (see TS tests page)"]],
        size=10, h=100)
    pg.text("Large n: tests flag tiny deviations → always look at the Q-Q plot too", x0 + 20, y0 + 615, 600, 18, 10,
            italic=True, color="#555555", align="center")

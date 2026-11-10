"""Nature of Data (Tabular) page: dataset anatomy, data types, NOIR, relationship shapes, measures, structure."""
import math
from dio import *

TYPE_COL = {"ID": "#EEEEEE", "Datetime": "#D9D9F3", "Nominal": "#FFCCCC", "Ordinal": "#FFD9B3",
            "Binary": "#E6CCFF", "Discrete": "#CCE5FF", "Continuous": "#CCFFCC"}


def build(pg: Page, cat):
    panel_anatomy(pg, 400, 240)
    panel_tree(pg, 1340, 240)
    panel_noir(pg, 400, 920)
    panel_relations(pg, 1440, 920)
    panel_matrix(pg, 400, 1600)
    panel_structure(pg, 1340, 1600)


# ---------------------------------------------------------------- ① anatomy
def panel_anatomy(pg, x0, y0):
    pg.panel(x0, y0, 900, 640, 1, "Anatomy of a Tabular Dataset", "Rows = observations · Columns = variables (features + target)")
    cols = [("ID", "ID"), ("Date", "Datetime"), ("Sector", "Nominal"), ("Rating", "Ordinal"), ("Trend", "Binary"),
            ("#Trades", "Discrete"), ("Close", "Continuous"), ("Return %", "Continuous"), ("Target: Up?", "Binary")]
    rows = [[1, "2024-01-02", "Tech", "A", "Up", 1520, 182.4, 1.2, 1],
            [2, "2024-01-03", "Energy", "BBB", "Down", 830, 76.1, -0.8, 0],
            [3, "2024-01-04", "Tech", "AA", "Up", 2210, 185.9, 1.9, 1],
            [4, "2024-01-05", "Bank", "BB", "Down", 640, 34.7, -1.5, 0],
            [5, "2024-01-08", "Energy", "A", "Up", 990, 77.3, 0.4, 1]]
    cw, ch = 88, 28
    tx, ty = x0 + 80, y0 + 130
    # feature / target brackets
    fx0, fx1 = tx + cw, tx + 8 * cw
    pg.line([(fx0 + 2, ty - 44), (fx0 + 2, ty - 50), (fx1 - 2, ty - 50), (fx1 - 2, ty - 44)], color=BLUE, width=1.5)
    pg.text("Features  X  (p = 7 columns)", fx0, ty - 74, fx1 - fx0, 20, 12, bold=True, align="center", color=BLUE)
    pg.line([(fx1 + 2, ty - 44), (fx1 + 2, ty - 50), (fx1 + cw - 2, ty - 50), (fx1 + cw - 2, ty - 44)], color=RED,
            width=1.5)
    pg.text("Target  y", fx1, ty - 74, cw, 20, 12, bold=True, align="center", color=RED)
    for j, (name, typ) in enumerate(cols):
        x = tx + j * cw
        pg.rect(x, ty - 38, cw, 22, typ, fill=TYPE_COL[typ], stroke="#999999", size=10)
        tgt = j == len(cols) - 1
        pg.rect(x, ty - 12, cw, ch, name, fill=HL if tgt else LIGHTGREY, size=11, bold=True, sw=2 if tgt else 1)
        for i, r in enumerate(rows):
            pg.rect(x, ty + 16 + i * ch, cw, ch, str(r[j]), size=11, sw=2 if tgt else 1)
    pg.rect(tx, ty + 16 + 5 * ch, 9 * cw, ch, "⋮   (n rows)", size=11, stroke="#BBBBBB", dashed=True)
    # observation brace
    pg.bracket(tx - 26, ty + 16, 14, 5 * ch)
    pg.text("rows<br>i = 1…n", x0 + 8, ty + 16 + 2.5 * ch - 18, 48, 36, 10, color="#555555", raw=True, align="center")
    pg.note("one row = one observation  \\(x_i = (x_{i1}, …, x_{ip}),\\ y_i\\)", x0 + 470, ty + 220,
            tx + 6 * cw, ty + 16 + 6 * ch, w=300, raw=True)
    pg.note("one column = one variable / feature  \\(x_{\\cdot j}\\)", x0 + 60, ty + 220,
            tx + 2 * cw, ty + 16 + 6 * ch, w=270, raw=True)
    pg.math("X \\in \\mathbb{R}^{n \\times p},\\quad y \\in \\{0,1\\}^n", x0 + 250, ty + 255, 400, 36, 16)
    # legend
    lx, ly = x0 + 40, y0 + 440
    pg.text("Column type legend:", lx, ly, 200, 20, 12, bold=True)
    for k, (t, c) in enumerate(TYPE_COL.items()):
        pg.rect(lx + (k % 4) * 205, ly + 26 + (k // 4) * 28, 195, 22, t, fill=c, stroke="#999999", size=11)
    pg.bullets(x0 + 20, y0 + 535, 860, [
        "<b>ID</b> is not a feature — it carries no pattern (and can leak information); drop it before training",
        "<b>Datetime</b> → index for time series, or extract features (day-of-week, month, hour)",
        "Qualitative columns (Sector, Rating, Trend) must be <b>encoded</b> before most ML algorithms",
        "Binary target → classification; continuous target (e.g. Return %) → regression"], head="Notes:")
    pg.ask("Which column must the model learn to predict?",
           "the target y (yellow): binary Up / Down → a classification problem",
           x0 + 645, y0 + 440, tx + 8 * cw + cw / 2, ty + 16 + 5 * ch, w=240)


# ---------------------------------------------------------------- ② data-type tree
def panel_tree(pg, x0, y0):
    pg.panel(x0, y0, 1060, 640, 2, "Types of Data (Variables)", "Qualitative = describes a quality · Quantitative = measures an amount")
    root = pg.rect(x0 + 430, y0 + 70, 200, 44, "Data (Variable)", size=15, bold=True, sw=2)
    qual = pg.rect(x0 + 140, y0 + 160, 260, 44, "Qualitative (Categorical)", fill=TYPE_COL["Nominal"], size=13,
                   bold=True)
    quan = pg.rect(x0 + 660, y0 + 160, 260, 44, "Quantitative (Numerical)", fill=TYPE_COL["Continuous"], size=13,
                   bold=True)
    pg.connect(root, qual, arrow="classic")
    pg.connect(root, quan, arrow="classic")
    pg.text("labels / groups — counting & mode", x0 + 140, y0 + 206, 260, 18, 10, italic=True, align="center",
            color="#555555")
    pg.text("numbers — arithmetic is meaningful", x0 + 660, y0 + 206, 260, 18, 10, italic=True, align="center",
            color="#555555")
    leaves = [
        (x0 + 20, qual, "Nominal", "No natural order", ["Sector: Tech, Bank, Energy", "Exchange: NYSE, SET", "Colour, country"]),
        (x0 + 200, qual, "Ordinal", "Ordered, unequal gaps", ["Credit rating: BB < BBB < A", "Signal: weak / mid / strong", "Survey 1–5 stars"]),
        (x0 + 380, qual, "Binary", "Exactly 2 categories", ["Trend: Up / Down", "Buy signal: 1 / 0", "Is holiday? yes / no"]),
        (x0 + 600, quan, "Discrete", "Countable (integers)", ["#Trades per day", "#News headlines", "Days since earnings"]),
        (x0 + 820, quan, "Continuous", "Measurable, any value", ["Close price, Return %", "Volatility σ", "Time-to-fill (s)"]),
    ]
    for lx, par, name, desc, ex in leaves:
        lid = pg.rect(lx, y0 + 270, 170, 50, f"<b>{name}</b><br><font style='font-size:10px'>{desc}</font>",
                      fill=TYPE_COL[name], size=13, raw=True)
        pg.connect(par, lid, arrow="classic")
        pg.html("<b>e.g.</b><ul style='margin:2px 0 0 0;padding-left:14px'>" + "".join(f"<li>{E(e)}</li>" for e in ex)
                + "</ul>", lx, y0 + 330, 175, 70, 11)
    # continuous -> interval / ratio
    pg.rect(x0 + 760, y0 + 420, 130, 34, "Interval (no true 0)", size=11, dashed=True)
    pg.rect(x0 + 910, y0 + 420, 130, 34, "Ratio (true 0)", size=11, dashed=True)
    pg.line([(x0 + 905, y0 + 400), (x0 + 825, y0 + 420)], color=GREY)
    pg.line([(x0 + 905, y0 + 400), (x0 + 975, y0 + 420)], color=GREY)
    pg.text("measurement scale → see ③", x0 + 760, y0 + 456, 280, 16, 10, italic=True, align="center", color="#555555")
    pg.bullets(x0 + 20, y0 + 480, 1020, [
        "<b>Discrete</b> = you <i>count</i> it (0, 1, 2…);  <b>Continuous</b> = you <i>measure</i> it (any value in a range)",
        "<b>Binary</b> is a special nominal variable with 2 levels — usually stored as 0 / 1",
        "Ordinal codes (1–5 stars) <i>look</i> numeric, but the gap 1→2 need not equal 4→5: avoid the mean, use the median",
        "Numbers are not always quantitative: zip code, ticker ID, account number are <b>nominal</b>"], head="Notes:")
    pg.ask("Which maths is allowed on this numeric column?",
           "interval: +, − and the mean; ratio (true zero, e.g. price) also × and ÷ → 'twice as much'",
           x0 + 600, y0 + 515, x0 + 1030, y0 + 454, w=440)


# ---------------------------------------------------------------- ③ NOIR
def panel_noir(pg, x0, y0):
    pg.panel(x0, y0, 1000, 640, 3, "★ Levels of Measurement (NOIR)", "Each level keeps everything from the level below and adds one property")
    fills = ["#E8F0FF", "#CCE0FF", "#B3D1FF", "#99C2FF"]
    steps = [("Nominal", "categories (=, ≠)", "Sector"), ("Ordinal", "+ order (<, >)", "Rating"),
             ("Interval", "+ equal intervals (+, −)", "Temperature °C, dates"),
             ("Ratio", "+ true zero (×, ÷)", "Price, volume")]
    base, sw = y0 + 290, 225
    for k, (name, adds, ex) in enumerate(steps):
        h = 50 + 40 * k
        sx = x0 + 50 + k * sw
        pg.rect(sx, base - h, sw, h, f"<b>{name}</b><br><font style='font-size:10px'>{E(ex)}</font>", fill=fills[k],
                size=14, raw=True, valign="bottom")
        pg.text(adds, sx, base - h - 24, sw, 20, 12, bold=True, align="center", color=BLUE)
    pg.line([(x0 + 60, y0 + 82), (x0 + 940, y0 + 82)], arrow="classic", color=RED, width=1.5)
    pg.text("more information → more mathematics allowed", x0 + 300, y0 + 60, 420, 20, 12, italic=True,
            align="center", color=RED)
    ok, no = '<font color="#009900">✔</font>', '<font color="#FF0000">✘</font>'
    rows = [["", "Nominal", "Ordinal", "Interval", "Ratio"],
            ["Categories  =, ≠", ok, ok, ok, ok],
            ["Order  <, >", no, ok, ok, ok],
            ["Differences  +, −", no, no, ok, ok],
            ["Ratios  ×, ÷  ('twice as much')", no, no, no, ok],
            ["Central tendency", "Mode", "Median (+ mode)", "Mean (+ median)", "Mean, geometric mean"],
            ["Dispersion", "# categories, entropy", "Range, IQR", "Std. dev. σ", "σ, coefficient of variation"],
            ["Typical encoding / prep", "One-hot, target enc.", "Ordinal (0,1,2…)", "Standardize", "Standardize, log"]]
    pg.table(x0 + 30, y0 + 310, 940, rows, size=11, h=215, highlight_rows=())
    pg.bullets(x0 + 20, y0 + 535, 960, [
        "20 °C is <i>not</i> 'twice as hot' as 10 °C (interval, zero is arbitrary) — but $20 <i>is</i> twice $10 (ratio)",
        "The scale decides which statistics, tests and encodings are valid (see the Hypothesis-Tests page)"], head="Why it matters:")
    pg.ask("What extra maths does each level unlock?",
           "nominal = only =, ≠; each step up adds order, then +/−, then ×/÷ (true zero)",
           x0 + 40, y0 + 96, x0 + 245, y0 + 246, w=230)


# ---------------------------------------------------------------- ④ relationship shapes
def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(order): r[i] = k + 1
    return r


def _pearson(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))


def _mi(a, b, bins=6):
    def binz(v):
        lo, hi = min(v), max(v)
        return [min(bins - 1, int((x - lo) / (hi - lo + 1e-12) * bins)) for x in v]
    ba, bb, n = binz(a), binz(b), len(a)
    from collections import Counter
    pa, pb, pab = Counter(ba), Counter(bb), Counter(zip(ba, bb))
    return sum(c / n * math.log2((c / n) / (pa[i] / n * pb[j] / n)) for (i, j), c in pab.items())


def panel_relations(pg, x0, y0):
    pg.panel(x0, y0, 960, 640, 4, "Relationship Shapes: Monotonic vs Non-monotonic",
             "Same n = 80 points each — which correlation measure sees the pattern?")
    r = seeded(11)
    xs = [r.uniform(0, 10) for _ in range(80)]
    sets = [("Linear", [2 * x + r.gauss(0, 2.2) for x in xs], "constant rate of change"),
            ("Monotonic (non-linear)", [math.exp(0.5 * x) / 15 + r.gauss(0, 0.6) for x in xs], "always ↑, rate changes"),
            ("Non-monotonic", [(x - 5) ** 2 + r.gauss(0, 2.0) for x in xs], "↓ then ↑ (U-shape)"),
            ("No relationship", [r.gauss(5, 2.5) for _ in xs], "independent")]
    for k, (name, ys, desc) in enumerate(sets):
        px = x0 + 50 + k * 228
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.08
        p = pg.axes(px, y0 + 110, 175, 150, (0, 10.5), (lo - pad, hi + pad), "x", "y", label_size=11,
                    origin=(0, lo - pad))
        p.scatter(list(zip(xs, ys)), r=2.5)
        pg.text(name, px - 10, y0 + 70, 200, 20, 12, bold=True, align="center")
        pg.text(desc, px - 10, y0 + 272, 200, 16, 10, italic=True, align="center", color="#555555")
        pr, sr, mi = _pearson(xs, ys), _pearson(_rank(xs), _rank(ys)), _mi(xs, ys)
        if k == 2: u_pr, u_mi = pr, mi
        col = lambda v, t: GREEN if abs(v) >= t else RED
        pg.html(f'Pearson r = <b><font color="{col(pr, 0.5)}">{pr:+.2f}</font></b><br>'
                f'Spearman ρ = <b><font color="{col(sr, 0.5)}">{sr:+.2f}</font></b><br>'
                f'Mutual info = <b><font color="{col(mi, 0.4)}">{mi:.2f}</font></b> bits',
                px + 10, y0 + 295, 180, 50, 11)
    pg.math("r = \\frac{\\sum(x_i-\\bar{x})(y_i-\\bar{y})}{\\sqrt{\\sum(x_i-\\bar{x})^2\\sum(y_i-\\bar{y})^2}}",
            x0 + 20, y0 + 370, 330, 56, 15)
    pg.math("\\rho = 1 - \\frac{6\\sum d_i^2}{n(n^2-1)}", x0 + 360, y0 + 370, 250, 56, 15)
    pg.math("I(X;Y) = \\sum\\sum p(x,y)\\log_2\\frac{p(x,y)}{p(x)p(y)}", x0 + 610, y0 + 370, 340, 56, 14)
    pg.text("linear only", x0 + 20, y0 + 425, 330, 16, 10, align="center", color="#555555")
    pg.text("dᵢ = rank difference → any monotonic", x0 + 360, y0 + 425, 250, 16, 10, align="center", color="#555555")
    pg.text("any dependence (incl. non-monotonic)", x0 + 610, y0 + 425, 340, 16, 10, align="center", color="#555555")
    pg.bullets(x0 + 20, y0 + 455, 920, [
        "<b>Monotonic</b>: as x increases, y never changes direction (always ↑ or always ↓) — Spearman ρ / Kendall τ capture it",
        "<b>Non-monotonic</b>: direction changes (U-shape, cycles) — r ≈ 0 and ρ ≈ 0 even though y depends strongly on x!",
        "Pearson r = 0 means <i>no linear</i> relation, <b>not</b> no relation → always plot the data, use MI or tree models",
        "Linear models need linear (or transformed) relations; trees / kNN / neural nets handle non-monotonic ones"],
        head="Interpretation:")
    pg.ask("Is there ANY relationship even when r ≈ 0?",
           f"yes: the U-shape has r = {u_pr:+.2f} but MI = {u_mi:.2f} bits → strong non-linear dependence",
           x0 + 600, y0 + 470, x0 + 780, y0 + 426, w=340)


# ---------------------------------------------------------------- ⑤ measure per pair of types
def _icon(pg, kind, x, y):
    if kind == "bar":
        for k, (h1, h2) in enumerate([(30, 14), (18, 26), (24, 10)]):
            pg.rect(x + k * 16, y + 44 - h1 - h2, 12, h2, fill="#FFCCCC", stroke=BLACK, sw=0.5)
            pg.rect(x + k * 16, y + 44 - h1, 12, h1, fill=LIGHTBLUE, stroke=BLACK, sw=0.5)
    elif kind == "box":
        for k, (a, b) in enumerate([(14, 30), (6, 22), (20, 36)]):
            cx = x + 8 + k * 16
            pg.line([(cx, y + a - 4), (cx, y + b + 6)], width=0.75)
            pg.rect(cx - 5, y + a, 10, b - a, fill=LIGHTBLUE, stroke=BLACK, sw=0.5)
    elif kind == "scatter":
        rr = seeded(5)
        for _ in range(14):
            u = rr.uniform(0, 44)
            pg.dot(x + u, y + 44 - u * 0.8 - rr.uniform(0, 8), r=1.8)
    elif kind == "heat":
        for i in range(3):
            for j in range(3):
                pg.rect(x + j * 15, y + 2 + i * 14, 15, 14, fill=["#CCE5FF", "#66B2FF", "#0066CC"][(i + j) % 3],
                        stroke="#FFFFFF", sw=0.5)


def panel_matrix(pg, x0, y0):
    pg.panel(x0, y0, 900, 600, 5, "Which Measure & Chart for Each Pair of Variable Types?")
    types = ["Categorical (nominal)", "Ordinal", "Numeric (interval / ratio)"]
    cells = {(0, 0): ("Cramér's V, χ² test", "stacked bar / heatmap", "bar"),
             (0, 1): ("Cramér's V, Kruskal-Wallis", "stacked bar", "bar"),
             (0, 2): ("η² (ANOVA), point-biserial (binary)", "box plot by group", "box"),
             (1, 1): ("Spearman ρ, Kendall τ", "heatmap / jitter", "heat"),
             (1, 2): ("Spearman ρ", "box plot per level", "box"),
             (2, 2): ("Pearson r · Spearman ρ · MI", "scatter / corr. heatmap", "scatter")}
    gx, gy, cw, ch = x0 + 190, y0 + 90, 230, 92
    for j, t in enumerate(types):
        pg.rect(gx + j * cw, gy - 36, cw, 36, t, fill=LIGHTGREY, bold=True, size=12)
        pg.rect(gx - 170, gy + j * ch, 170, ch, t, fill=LIGHTGREY, bold=True, size=12)
    for i in range(3):
        for j in range(3):
            x, y = gx + j * cw, gy + i * ch
            if (i, j) in cells:
                m, c, ic = cells[(i, j)]
                pg.rect(x, y, cw, ch, fill="#FFFFFF")
                pg.html(f"<b>{E(m)}</b><br><font color='#555555' style='font-size:10px'>{E(c)}</font>",
                        x + 8, y + 10, cw - 70, ch - 20, 11, valign="middle")
                _icon(pg, ic, x + cw - 58, y + 22)
            else:
                pg.rect(x, y, cw, ch, "(mirror)", fill="#F7F7F7", stroke="#CCCCCC", size=10, color="#AAAAAA")
    pg.ask("How do I measure the link between two numeric columns?",
           "Pearson r if linear, Spearman ρ if monotonic, MI for any shape (see ④)",
           gx + 8, gy + 2 * ch + 11, gx + 2 * cw + 40, gy + 2 * ch + 30, w=444)
    pg.math("V = \\sqrt{\\frac{\\chi^2}{n\\,(\\min(r,c)-1)}}", x0 + 30, y0 + 390, 260, 50, 15)
    pg.math("\\eta^2 = \\frac{SS_{between}}{SS_{total}}", x0 + 310, y0 + 390, 240, 50, 15)
    pg.math("r_{pb} = \\frac{\\bar{y}_1-\\bar{y}_0}{s_y}\\sqrt{pq}", x0 + 580, y0 + 390, 280, 50, 15)
    pg.bullets(x0 + 20, y0 + 455, 860, [
        "All measures lie in [0, 1] (V, η², MI ≥ 0) or [−1, 1] (r, ρ, τ, r_pb): 0 = no association",
        "The matching hypothesis test for each cell is on the <b>Hypothesis Tests by Data Type</b> page",
        "Feature selection: use these to rank features against the target (filter methods)"], head="Notes:")


# ---------------------------------------------------------------- ⑥ structure & quality
def _grid(pg, x, y, rl, cl, cw=44, ch=20, fill="#FFFFFF"):
    for j, c in enumerate(cl):
        pg.rect(x + 60 + j * cw, y, cw, ch, c, fill=LIGHTGREY, size=9, bold=True)
    for i, rlab in enumerate(rl):
        pg.rect(x, y + ch + i * ch, 60, ch, rlab, fill=LIGHTGREY, size=9)
        for j in range(len(cl)):
            pg.rect(x + 60 + j * cw, y + ch + i * ch, cw, ch, "·", fill=fill, size=9)


def panel_structure(pg, x0, y0):
    pg.panel(x0, y0, 1060, 600, 6, "Data Structure & Data Quality")
    feats = ["Close", "Vol", "Ret"]
    specs = [(30, "Cross-sectional", "many entities, one time", ["AAPL", "MSFT", "PTT", "KBANK"], "#CCFFCC"),
             (330, "Time Series", "one entity, many time steps", ["t = 1", "t = 2", "t = 3", "t = 4"], LIGHTBLUE),
             (650, "Panel (longitudinal)", "many entities × many times", ["AAPL t1", "AAPL t2", "PTT t1", "PTT t2"],
              "#E6CCFF")]
    for dx, name, desc, rl, fill in specs:
        pg.text(name, x0 + dx, y0 + 50, 260, 20, 13, bold=True)
        pg.text(desc, x0 + dx, y0 + 68, 260, 16, 10, italic=True, color="#555555")
        _grid(pg, x0 + dx, y0 + 90, rl, feats, fill=fill)
    pg.line([(x0 + 535, y0 + 112), (x0 + 535, y0 + 192)], arrow="classic", color=RED)
    pg.text("order matters!", x0 + 540, y0 + 140, 90, 16, 9, color=RED)
    pg.text("→ random shuffle-split CV is fine only for cross-sectional data; time series / panel need time-ordered splits",
            x0 + 30, y0 + 200, 1000, 18, 11, italic=True, color=RED)
    pg.text("Missing data mechanisms", x0 + 30, y0 + 232, 400, 20, 13, bold=True)
    miss = [("MCAR", "Missing Completely At Random", "Sensor glitch drops random ticks",
             "Drop rows or mean / median impute"),
            ("MAR", "Missing At Random (depends on observed data)", "Volume missing more often for small-cap stocks",
             "Model-based / KNN / iterative impute"),
            ("MNAR", "Missing Not At Random (depends on the missing value itself)", "Firms hide bad earnings → missing",
             "Add an 'is_missing' flag, domain modelling")]
    for k, (a, b, ex, fix) in enumerate(miss):
        bx = x0 + 30 + k * 340
        pg.rect(bx, y0 + 258, 320, 124,
                f"<b>{a}</b> — {E(b)}<br><br><i>e.g.</i> {E(ex)}<br><b>Handle:</b> {E(fix)}", align="left",
                valign="top", size=11, raw=True, fill=["#F5F5F5", "#FFF8E6", "#FFE6E6"][k])
    pg.text("Outliers & class imbalance", x0 + 30, y0 + 398, 400, 20, 13, bold=True)
    pg.rect(x0 + 40, y0 + 426, 190, 22, "95 % class 0", fill=LIGHTBLUE, size=10)
    pg.rect(x0 + 230, y0 + 426, 10, 22, fill=RED)
    pg.text("5 % class 1", x0 + 245, y0 + 428, 80, 18, 10, color=RED)
    pg.ask("Why is 95 % accuracy useless on this table?",
           "always predicting class 0 already scores 95 % → judge the rare class with precision / recall / F1",
           x0 + 340, y0 + 392, x0 + 300, y0 + 437, w=700, size=9)
    pg.bullets(x0 + 30, y0 + 460, 1010, [
        "Imbalance: accuracy is misleading (95 % by always predicting 0) → precision / recall / F1, class weights, resampling",
        "Outliers: detect with Z-score / IQR (Anomaly page); error → fix / drop, real extreme → keep and use robust models",
        "Data leakage: never build features from future information or from the target itself"], head="Watch out:")

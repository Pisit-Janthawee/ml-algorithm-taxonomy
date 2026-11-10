"""Bagging page: bootstrap sampling, out-of-bag 36.8 %, bias-variance dartboard, pipeline + real demo
(deep trees vs bagged average), error vs number of trees."""
import math
from dio import *

PAL = [("#F8CECC", "#B85450"), ("#DAE8FC", "#6C8EBF"), ("#D5E8D4", "#82B366"), ("#FFF2CC", "#D6B656"),
       ("#E1D5E7", "#9673A6"), ("#FFE6CC", "#D79B00"), ("#B1DDF0", "#10739E"), ("#E6E6E6", "#666666")]


def chip(pg, x, y, i, s=30):
    f, st = PAL[i % len(PAL)]
    pg.rect(x, y, s, s, str(i + 1), fill=f, stroke=st, size=12, bold=True)


def deep_tree_predict(xs, ys, x):
    """A fully grown 1-D regression tree = piecewise constant split at midpoints = nearest neighbour."""
    j = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
    return ys[j]


def mini_tree(pg, cx, y, col=BLACK):
    pts = [(cx, y), (cx - 22, y + 30), (cx + 22, y + 30), (cx - 34, y + 58), (cx - 10, y + 58)]
    for a, b in [(0, 1), (0, 2), (1, 3), (1, 4)]:
        pg.line([pts[a], pts[b]], color=col)
    for x, yy in pts:
        pg.ellipse(x - 6, yy - 6, 12, 12, fill="#FFFFFF", stroke=col)


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① bootstrap sampling
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 720, 1, "Bootstrap Sampling (sampling with replacement)",
             "Background: bootstrap (Efron, 1979)")
    n = 8
    pg.text("Original dataset D (n = 8 rows)", x0 + 30, y0 + 70, 300, 20, 12, bold=True)
    for i in range(n):
        chip(pg, x0 + 60 + i * 38, y0 + 95, i)
    r = seeded(4)
    oob_total = 0
    for b in range(3):
        yy = y0 + 175 + b * 70
        sample = [r.randrange(n) for _ in range(n)]
        oob = sorted(set(range(n)) - set(sample))
        oob_total += len(oob)
        pg.math(f"D_{b + 1}", x0 + 14, yy + 2, 40, 26, 15)
        for i, s in enumerate(sorted(sample)):
            chip(pg, x0 + 60 + i * 38, yy, s)
        dup = [str(v + 1) for v in sorted(set(sample)) if sample.count(v) > 1]
        pg.text(f"OOB: {', '.join(str(v + 1) for v in oob)}", x0 + 375, yy - 2, 240, 18, 11, bold=True, color=RED)
        pg.text(f"repeated: {', '.join(dup) or '–'}", x0 + 375, yy + 16, 240, 18, 10, color="#555555")
    pg.line([(x0 + 200, y0 + 130), (x0 + 200, y0 + 165)], arrow="classic")
    pg.text("draw n rows at random, put each back", x0 + 210, y0 + 135, 260, 20, 10, italic=True)
    pg.math("P(\\text{row never drawn}) = \\left(1-\\frac{1}{n}\\right)^{n} \\xrightarrow{n\\to\\infty} e^{-1} \\approx 0.368",
            x0 + 20, y0 + 395, 600, 50, 17)
    pg.html(f"n = 8:  \\((7/8)^8 = {(7 / 8) ** 8:.3f}\\)  ·  in our 3 samples: {oob_total} of 24 rows OOB = "
            f'<span style="background:{HL}">{oob_total / 24:.3f}</span>', x0 + 20, y0 + 448, 600, 24, 12,
            align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 490, 600, [
        "Each bootstrap sample has the same size n as the original data",
        "≈ 63.2 % unique rows appear (some twice or more), ≈ 36.8 % are left out = <b>out-of-bag (OOB)</b>",
        "Every sample is a slightly different 'version' of the data → every model learns something different",
        "Bootstrap also estimates confidence intervals of any statistic (mean, median …)"],
        head="Notes:")
    pg.ask("How many rows does each tree never see?",
           f"about 1/e ≈ 36.8 % — here {oob_total} of 24 = {oob_total / 24:.3f}; those out-of-bag rows are a free "
           "test set for that tree",
           x0 + 20, y0 + 620, x0 + 470, y0 + 472, w=600)

    # ---------------------------------------------------------------- ② OOB
    x0 = 1080
    pg.panel(x0, y0, 620, 720, 2, "Out-of-Bag (OOB) = free validation set")
    p = pg.axes(x0 + 70, y0 + 80, 300, 170, (0, 50), (0, 0.5), "n", "P(OOB)", label_size=12,
                xticks=(1, 10, 20, 30, 40, 50), yticks=(0.25, 0.368, 0.5), tick_fmt="{:g}")
    p.hline(math.exp(-1), color=RED)
    p.scatter([(k, (1 - 1 / k) ** k) for k in range(2, 51)], color=BLUE, r=2.5)
    p.text("1/e = 0.368", 44, 0.368, dy=-12, size=10, color=RED)
    pg.note("n = 2: (1/2)² = 0.25", x0 + 120, y0 + 200, *p.P(2, 0.25), w=120)
    # simulated
    r = seeded(9)
    N = 1000
    sims = [1 - len(set(r.randrange(N) for _ in range(N))) / N for _ in range(5)]
    sim = sum(sims) / len(sims)
    bx = x0 + 400
    pg.text(f"Simulation, n = {N} (5 runs)", bx, y0 + 80, 200, 20, 11, bold=True)
    pg.rect(bx, y0 + 105, 60, 150 * (1 - sim), f"in-bag<br>{100 * (1 - sim):.1f}%", fill=LIGHTBLUE, stroke=BLUE,
            size=10, raw=True)
    pg.rect(bx, y0 + 105 + 150 * (1 - sim), 60, 150 * sim, f"OOB<br>{100 * sim:.1f}%", fill=LIGHTGREY,
            stroke=GREY, size=10, raw=True)
    pg.text("runs: " + ", ".join(f"{s:.3f}" for s in sims), bx + 70, y0 + 160, 130, 60, 10, color="#555555")
    # in-bag grid
    gy = y0 + 300
    pg.text("Which rows each tree saw (6 rows × 5 trees)", x0 + 30, gy, 400, 20, 12, bold=True)
    r = seeded(13)
    bags = [[r.randrange(6) for _ in range(6)] for _ in range(5)]
    for t in range(5):
        pg.text(f"Tree {t + 1}", x0 + 110 + t * 62, gy + 26, 60, 18, 10, bold=True, align="center")
    for row in range(6):
        pg.text(f"row {row + 1}", x0 + 40, gy + 48 + row * 26, 60, 22, 10, align="right")
        for t in range(5):
            cnt = bags[t].count(row)
            pg.rect(x0 + 115 + t * 62, gy + 48 + row * 26, 50, 22, f"×{cnt}" if cnt else "OOB",
                    fill=LIGHTBLUE if cnt else "#FFFFFF", stroke=BLUE if cnt else RED, size=10,
                    color=BLACK if cnt else RED)
    row3 = [str(t + 1) for t in range(5) if bags[t].count(2) == 0]
    pg.html(f"Row 3 is predicted <b>only</b> by trees {', '.join(row3) or '–'} (never trained on it) → "
            "average of all such predictions = <b>OOB error</b> ≈ test error, no hold-out set needed",
            x0 + 430, gy + 60, 170, 150, 11)
    pg.bullets(x0 + 20, y0 + 520, 580, [
        "scikit-learn: <font face='Courier New'>BaggingClassifier(oob_score=True)</font> → <font face='Courier New'>.oob_score_</font>",
        "OOB also gives permutation feature importance in Random Forest"],
        head="Notes:")
    t3 = next((t for t in range(5) if bags[t].count(2) == 0), 0)
    pg.ask("Can I estimate test error without a hold-out set?",
           "yes: predict each row only with the trees that never saw it (OOB cells) and average → OOB error ≈ test error",
           x0 + 20, y0 + 595, x0 + 140 + t3 * 62, gy + 48 + 2 * 26 + 22, w=580)

    # ---------------------------------------------------------------- ③ bias-variance dartboard
    x0 = 1740
    pg.panel(x0, y0, 660, 720, 3, "Bias–Variance Trade-off", "Why averaging helps")
    r = seeded(2)
    cells = [("Low bias", "Low variance", (0, 0), 0.12), ("High bias", "Low variance", (0.45, -0.35), 0.12),
             ("Low bias", "High variance", (0, 0), 0.45), ("High bias", "High variance", (0.45, -0.35), 0.45)]
    R = 62
    for k, (bl, vl, (mx, my), sd) in enumerate(cells):
        cx = x0 + 230 + (k % 2) * 210
        cy = y0 + 150 + (k // 2) * 175
        for rr, fc in [(R, "#FFFFFF"), (R * 0.66, "#FFFFFF"), (R * 0.33, LIGHTRED)]:
            pg.ellipse(cx - rr, cy - rr, 2 * rr, 2 * rr, fill=fc, stroke=BLACK)
        for _ in range(9):
            dx, dy = r.gauss(mx, sd), r.gauss(my, sd)
            pg.dot(cx + dx * R, cy + dy * R, 3.5, BLUE)
    pg.text("Low bias", x0 + 170, y0 + 66, 120, 20, 12, bold=True, align="center")
    pg.text("High bias", x0 + 380, y0 + 66, 120, 20, 12, bold=True, align="center")
    pg.text("Low variance", x0 + 20, y0 + 140, 110, 20, 12, bold=True, align="right")
    pg.text("High variance", x0 + 20, y0 + 315, 110, 20, 12, bold=True, align="right")
    pg.note("single deep tree", x0 + 20, y0 + 380, x0 + 190, y0 + 345, w=110, color=RED)
    pg.note("Bagging = average of deep trees", x0 + 20, y0 + 95, x0 + 180, y0 + 125, w=120, color=GREEN)
    pg.line([(x0 + 160, y0 + 300), (x0 + 160, y0 + 200)], color=GREEN, arrow="block", width=2)
    pg.math("Var\\left(\\frac{1}{B}\\sum_{b=1}^{B} f_b\\right) = \\rho\\,\\sigma^2 + \\frac{1-\\rho}{B}\\,\\sigma^2",
            x0 + 40, y0 + 420, 580, 56, 18)
    pg.note("ρ = correlation between trees", x0 + 120, y0 + 480, x0 + 400, y0 + 456, w=170)
    pg.note("shrinks as B grows", x0 + 440, y0 + 480, x0 + 545, y0 + 460, w=120)
    pg.bullets(x0 + 20, y0 + 510, 620, [
        "Expected error = \\(Bias^2 + Variance + Noise\\)",
        "Deep trees: low bias, high variance → averaging B of them keeps the bias and cuts the variance",
        "Correlated trees limit the gain (ρσ² stays) → Random Forest de-correlates them with random features"],
        head="Notes:")
    pg.ask("Why does averaging many deep trees help?",
           "each deep tree is unbiased but noisy (scattered dots); averaging B of them keeps the low bias and pulls "
           "the prediction to the centre",
           x0 + 20, y0 + 610, x0 + 160, y0 + 250, w=620)

    # ---------------------------------------------------------------- ④ pipeline + demo
    x0, y0 = 400, 1000
    pg.panel(x0, y0, 1320, 760, 4, "Bagging = Bootstrap + Aggregating", "Parallel, homogeneous ensemble")
    D = pg.rect(x0 + 200, y0 + 75, 240, 40, "Original dataset D (n rows)", size=12, bold=True, sw=1.5)
    labels = ["1", "2", "3", "B"]
    preds = ["A", "B", "A", "A"]
    for i, (lab, pr) in enumerate(zip(labels, preds)):
        bx = x0 + 40 + i * 145 + (20 if i == 3 else 0)
        b = pg.rect(bx, y0 + 160, 100, 34, f"bootstrap D<sub>{lab}</sub>", size=11, raw=True)
        pg.connect(D, b, orth=False)
        mini_tree(pg, bx + 50, y0 + 220)
        pg.line([(bx + 50, y0 + 194), (bx + 50, y0 + 212)], arrow="classic")
        pg.text(f"Tree {i + 1 if i < 3 else 'B'}", bx, y0 + 285, 100, 18, 11, bold=True, align="center")
        pg.rect(bx + 10, y0 + 310, 80, 30, f"ŷ = {pr}", fill=LIGHTBLUE if pr == "A" else LIGHTRED,
                stroke=BLUE if pr == "A" else RED, size=12)
        pg.line([(bx + 50, y0 + 340), (x0 + 320, y0 + 395)], color=GREY, arrow="classic")
    pg.text("…", x0 + 470, y0 + 160, 40, 34, 20, bold=True, align="center")
    pg.rect(x0 + 180, y0 + 397, 280, 44, "Majority vote → A (3 of 4)", fill=HL, sw=2, size=14, bold=True)
    pg.text("each tree trained independently (in parallel)", x0 + 30, y0 + 120, 170, 34, 10, italic=True,
            color="#555555")
    pg.math("\\hat{y} = \\text{mode}\\{f_1(x), \\dots, f_B(x)\\}", x0 + 30, y0 + 455, 290, 40, 14)
    pg.math("\\hat{y} = \\frac{1}{B}\\sum_{b=1}^{B} f_b(x)", x0 + 330, y0 + 450, 290, 50, 14)
    pg.text("classification", x0 + 30, y0 + 495, 290, 16, 10, align="center", color="#555555")
    pg.text("regression", x0 + 330, y0 + 495, 290, 16, 10, align="center", color="#555555")
    # demo: deep tree vs bagged on noisy sine
    r = seeded(31)
    xs = sorted(r.uniform(0, 10) for _ in range(30))
    f = lambda v: math.sin(v * 0.8) * 1.2
    ys = [f(v) + r.gauss(0, 0.45) for v in xs]
    grid = [i * 10 / 200 for i in range(201)]
    single = [deep_tree_predict(xs, ys, g) for g in grid]
    B = 60
    boots = []
    for _ in range(B):
        idx = [r.randrange(len(xs)) for _ in range(len(xs))]
        boots.append(([xs[i] for i in idx], [ys[i] for i in idx]))
    preds_b = [[deep_tree_predict(bx_, by_, g) for g in grid] for bx_, by_ in boots]
    bag = [sum(pb[i] for pb in preds_b) / B for i in range(len(grid))]
    dx0 = x0 + 680
    P = pg.axes(dx0 + 50, y0 + 110, 540, 270, (0, 10), (-2.2, 2.2), "x", "y", origin=(0, -2.2))
    step_pts = []
    for i, g in enumerate(grid):
        if i and single[i] != single[i - 1]:
            step_pts.append((g, single[i - 1]))
        step_pts.append((g, single[i]))
    P.poly(step_pts, color=RED, width=1.2)
    P.poly(list(zip(grid, bag)), color=BLUE, width=2.5)
    P.curve(f, 0, 10, color=GREEN, dashed=True, width=1.5)
    P.scatter(list(zip(xs, ys)), color=BLACK, r=3)
    lx = dx0 + 60
    for k, (col, lab, dsh) in enumerate([(RED, "one deep tree (fits every point)", False),
                                         (BLUE, f"bagging, B = {B} deep trees", False),
                                         (GREEN, "true function", True)]):
        pg.line([(lx + k * 190, y0 + 85), (lx + 25 + k * 190, y0 + 85)], color=col, width=2, dashed=dsh)
        pg.text(lab, lx + 30 + k * 190, y0 + 76, 160, 18, 10, color=col)
    mse = lambda pr: sum((a - f(g)) ** 2 for a, g in zip(pr, grid)) / len(grid)
    pg.html(f"Error vs true function (MSE):  one tree = <b>{mse(single):.3f}</b>  →  "
            f'<span style="background:{HL}">bagging = <b>{mse(bag):.3f}</b></span>  '
            f"({100 * (1 - mse(bag) / mse(single)):.0f}% lower)", dx0 + 20, y0 + 405, 600, 24, 13, align="center",
            valign="middle")
    pg.bullets(dx0 + 20, y0 + 445, 600, [
        "A fully grown 1-D regression tree is a step function through every training point (high variance)",
        "Averaging trees grown on bootstrap samples smooths the steps → closer to the true curve",
        "Real computation: 30 noisy points of 1.2·sin(0.8x), noise σ = 0.45"],
        head="Demo:")
    pg.ask("Does averaging really beat a single deep tree?",
           f"yes: MSE {mse(single):.3f} → {mse(bag):.3f} ({100 * (1 - mse(bag) / mse(single)):.0f} % lower) — the blue "
           "average follows the true curve, the red tree chases noise",
           dx0 + 20, y0 + 560, *P.P(5, bag[100]), w=600)
    pg.bullets(x0 + 20, y0 + 560, 640, [
        "Works with any unstable model (trees, kNN with small k, neural nets);  little gain for stable models (linear regression)",
        "Embarrassingly parallel: <font face='Courier New'>n_jobs=-1</font> trains all trees at once"],
        head="Notes:")
    pg.ask("What final answer does the ensemble give?",
           "the majority of the trees' votes: 3 of 4 say A → A; for regression the trees' predictions are averaged",
           x0 + 20, y0 + 645, x0 + 320, y0 + 441, w=640)

    # ---------------------------------------------------------------- ⑤ error vs number of trees + RF
    x0 = 1760
    pg.panel(x0, y0, 640, 760, 5, "How many trees? → Random Forest")
    curve = []
    acc = [0.0] * len(grid)
    for b, pb in enumerate(preds_b, 1):
        acc = [a + v for a, v in zip(acc, pb)]
        curve.append((b, mse([a / b for a in acc])))
    ymax = 0.05 * math.ceil(max(v for _, v in curve) / 0.05 + 0.5)
    E = pg.axes(x0 + 80, y0 + 80, 460, 200, (0, B), (0, ymax), "B (trees)", "MSE", label_size=12,
                xticks=(1, 10, 20, 30, 40, 50, 60), yticks=(round(ymax / 2, 2), round(ymax, 2)))
    E.poly(curve, color=BLUE, width=2)
    E.scatter([curve[0]], color=RED, r=4)
    pg.note(f"1 tree: {curve[0][1]:.3f}", x0 + 120, y0 + 60, *E.P(*curve[0]), w=90, color=RED)
    pg.note(f"plateau ≈ {curve[-1][1]:.3f} — more trees never overfit, just cost time", x0 + 250, y0 + 120,
            *E.P(55, curve[54][1]), w=240)
    rows = [["", "Bagging", "Random Forest"],
            ["Rows", "bootstrap sample", "bootstrap sample"],
            ["Features per split", "all p", "random subset m ≈ √p"],
            ["Tree correlation ρ", "higher", "lower → more variance cut"],
            ["Base model", "any model", "decision trees"]]
    pg.table(x0 + 30, y0 + 320, 580, rows, size=12)
    pg.bullets(x0 + 20, y0 + 460, 600, [
        "<b>Hyperparameters:</b> n_estimators (B), max_samples, max_features, bootstrap, oob_score",
        "<b>Pros:</b> reduces variance, robust to noise, parallel, OOB estimate for free",
        "<b>Cons:</b> does not reduce bias, B models → more memory & slower prediction, less interpretable",
        "<b>Family:</b> Bagging → Random Forest → Extra Trees (random thresholds too)"],
        head="Summary:")
    pg.ask("How many trees are enough?",
           f"until the curve flattens: error drops from {curve[0][1]:.3f} (1 tree) to ≈ {curve[-1][1]:.3f}; "
           "extra trees never overfit, they only cost time",
           x0 + 20, y0 + 610, *E.P(*curve[19]), w=600)

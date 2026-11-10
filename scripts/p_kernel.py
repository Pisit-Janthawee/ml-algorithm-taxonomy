"""k-Nearest Neighbors page: idea & vote, distance metrics, feature scaling, effect of k (real decision
regions + leave-one-out CV curve), worked example."""
import math
from dio import *


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def knn(train, q, k, skip=None):
    ds = sorted((dist(p, q), c) for i, (p, c) in enumerate(train) if i != skip)
    votes = [c for _, c in ds[:k]]
    return max(set(votes), key=lambda c: (votes.count(c), c == votes[0]))


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① idea + vote
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 720, 1, "The Idea: ask your k nearest neighbours", "Lazy learner — no training, just store the data")
    Q = (5.0, 5.0)
    near = [((5.5, 5.4), "B"), ((4.4, 5.5), "B"), ((4.4, 4.3), "A"), ((5.8, 4.3), "A"), ((4.0, 4.6), "A"),
            ((6.0, 5.7), "A"), ((5.2, 6.4), "B")]
    r = seeded(5)
    bg = []
    while len(bg) < 34:
        if len(bg) % 2:
            p, c = (r.gauss(2.6, 1.2), r.gauss(2.8, 1.2)), "A"
        else:
            p, c = (r.gauss(7.4, 1.2), r.gauss(7.2, 1.2)), "B"
        if 0.3 < p[0] < 9.7 and 0.3 < p[1] < 9.7 and dist(p, Q) > 1.7:
            bg.append((p, c))
    pts = near + bg
    p = pg.axes(x0 + 60, y0 + 90, 360, 360, (0, 10), (0, 10), "x_1", "x_2")
    for (a, b), c in pts:
        p.scatter([(a, b)], color=BLUE if c == "A" else RED, r=4)
    ranked = sorted(pts, key=lambda t: dist(t[0], Q))
    for k, col, dsh in [(3, RED, False), (7, BLUE, True)]:
        rad = dist(ranked[k - 1][0], Q) + 0.15
        tl = p.P(Q[0] - rad, Q[1] + rad)
        pg.ellipse(tl[0], tl[1], rad * 36, rad * 36, stroke=col, dashed=dsh, sw=1.5)
    for (a, b), c in ranked[:3]:
        pg.line([p.P(*Q), p.P(a, b)], color=GREY, dashed=True, width=0.75)
    qx, qy = p.P(*Q)
    pg.ellipse(qx - 7, qy - 7, 14, 14, "?", fill=HL, stroke=BLACK, size=10, bold=True)
    pg.note("new point (query)", x0 + 300, y0 + 70, qx + 5, qy - 6, w=110)
    # legend + votes
    lx = x0 + 440
    pg.dot(lx + 6, y0 + 110, 4, BLUE); pg.text("class A", lx + 16, y0 + 101, 100, 18, 11)
    pg.dot(lx + 6, y0 + 132, 4, RED); pg.text("class B", lx + 16, y0 + 123, 100, 18, 11)
    for i, (k, col) in enumerate([(3, RED), (7, BLUE)]):
        v = [c for _, c in ranked[:k]]
        na, nb = v.count("A"), v.count("B")
        win = "A" if na > nb else "B"
        yy = y0 + 180 + i * 110
        pg.text(f"k = {k}  ({'solid' if k == 3 else 'dashed'} circle)", lx, yy, 190, 20, 12, bold=True, color=col)
        pg.text(f"A: {na}   B: {nb}", lx, yy + 22, 190, 18, 12)
        pg.html(f'<span style="background:{HL}">→ predict class {win}</span>', lx, yy + 44, 190, 22, 12,
                valign="middle")
    pg.text("Different k → different answer!", lx, y0 + 410, 190, 36, 11, italic=True, color=RED)
    pg.math("\\hat{y} = \\text{mode}\\{\\,y_i : x_i \\in N_k(x)\\,\\}", x0 + 20, y0 + 470, 300, 40, 16)
    pg.math("\\hat{y} = \\frac{1}{k}\\sum_{x_i \\in N_k(x)} y_i", x0 + 330, y0 + 465, 290, 50, 16)
    pg.text("classification (majority vote)", x0 + 20, y0 + 508, 300, 16, 10, align="center", color="#555555")
    pg.text("regression (average)", x0 + 330, y0 + 508, 290, 16, 10, align="center", color="#555555")
    pg.bullets(x0 + 20, y0 + 535, 600, [
        "\\(N_k(x)\\) = the k training points closest to x (by a distance metric ②)",
        "Weighted variant: closer neighbours vote more, weight \\(w_i = 1/d_i\\)",
        "No model is learned — all work happens at prediction time"],
        head="Where:")
    v3, v7 = [c for _, c in ranked[:3]], [c for _, c in ranked[:7]]
    pg.ask("Will the answer change if I ask more neighbours?",
           f"yes: k = 3 votes B ({v3.count('B')} of 3) but k = 7 votes A ({v7.count('A')} of 7) — k sets how "
           "local the decision is",
           x0 + 20, y0 + 630, lx + 95, y0 + 446, w=600)

    # ---------------------------------------------------------------- ② distance metrics
    x0 = 1080
    pg.panel(x0, y0, 620, 720, 2, "Distance Metrics", "Background: geometry of the feature space")
    g = pg.axes(x0 + 50, y0 + 90, 240, 240, (0, 6), (0, 6), "x_1", "x_2", label_size=12,
                xticks=(1, 2, 3, 4, 5), yticks=(1, 2, 3, 4, 5))
    for t in range(1, 7):
        g.vline(t, color="#DDDDDD", dashed=False, width=0.5)
        g.hline(t, color="#DDDDDD", dashed=False, width=0.5)
    A, B = (1, 1), (5, 4)
    g.poly([A, (5, 1), B], color=RED, width=2.5)
    g.poly([A, B], color=BLUE, width=2.5)
    g.poly([(1, 1), (1, 4), (5, 4)], color=GREEN, width=1, dashed=True)
    g.scatter([A, B], color=BLACK, r=5)
    g.text("A (1,1)", 1, 1, dx=-10, dy=16, size=10)
    g.text("B (5,4)", 5, 4, dx=12, dy=-14, size=10)
    g.text("4", 3, 1, dy=12, size=11, color=RED)
    g.text("3", 5, 2.5, dx=12, size=11, color=RED)
    g.text("5", 2.9, 2.8, dx=-8, dy=-6, size=12, color=BLUE)
    fx = x0 + 330
    rows = [("Euclidean (L2)", "d = \\sqrt{\\sum_j (a_j-b_j)^2}", "\\sqrt{4^2+3^2} = 5", BLUE),
            ("Manhattan (L1)", "d = \\sum_j |a_j-b_j|", "4 + 3 = 7", RED),
            ("Chebyshev (L∞)", "d = \\max_j |a_j-b_j|", "\\max(4,3) = 4", GREEN),
            ("Minkowski (Lp)", "d = \\left(\\sum_j |a_j-b_j|^p\\right)^{1/p}", "p=1 → L1,\\ p=2 → L2", BLACK)]
    for i, (name, f, ex, col) in enumerate(rows):
        yy = y0 + 80 + i * 66
        pg.text(name, fx, yy, 270, 18, 12, bold=True, color=col)
        pg.math(f, fx, yy + 18, 170, 40, 13, align="left")
        pg.math(ex, fx + 165, yy + 18, 110, 40, 12, align="left", color=col)
    pg.math("\\text{cosine: } 1 - \\frac{a \\cdot b}{\\lVert a\\rVert\\,\\lVert b\\rVert}", x0 + 60, y0 + 360, 500, 40, 15)
    pg.bullets(x0 + 20, y0 + 415, 580, [
        "<b>Euclidean</b> — default, straight-line distance (needs scaled features ③)",
        "<b>Manhattan</b> — grid / city-block distance, less sensitive to one large difference",
        "<b>Cosine</b> — angle between vectors; used for text (Bag-of-Words, TF-IDF)",
        "<b>Hamming</b> — number of differing positions, for categorical / binary features",
        "The metric defines what 'similar' means → it changes who the neighbours are"],
        head="When to use:")
    pg.ask("Does the choice of distance metric matter?",
           "yes: the same two points are 5 (Euclidean), 7 (Manhattan) or 4 (Chebyshev) apart, so neighbour "
           "rankings can change",
           x0 + 20, y0 + 565, *g.P(3, 2.5), w=580)

    # ---------------------------------------------------------------- ③ feature scaling
    x0 = 1740
    pg.panel(x0, y0, 660, 720, 3, "Feature Scaling (why kNN needs it)", "Background: min-max normalization / standardization")
    rows = [["Person", "Age", "Income", "Age (scaled)", "Income (scaled)"],
            ["A", "25", "50,000", "0.125", "0.375"],
            ["B", "45", "52,000", "0.625", "0.400"],
            ["C", "26", "60,000", "0.150", "0.500"]]
    pg.table(x0 + 30, y0 + 70, 600, rows, size=12)
    pg.math("x' = \\frac{x - x_{min}}{x_{max} - x_{min}}", x0 + 40, y0 + 190, 280, 46, 16)
    pg.math("z = \\frac{x - \\mu}{\\sigma}", x0 + 340, y0 + 190, 280, 46, 16)
    pg.text("min-max: Age 20–60, Income 20k–100k", x0 + 40, y0 + 236, 280, 16, 10, align="center", color="#555555")
    pg.text("standardization (z-score)", x0 + 340, y0 + 236, 280, 16, 10, align="center", color="#555555")
    dab_raw, dac_raw = math.hypot(20, 2000), math.hypot(1, 10000)
    dab_s, dac_s = math.hypot(0.5, 0.025), math.hypot(0.025, 0.125)
    rows = [["", "d(A, B)", "d(A, C)", "Nearest to A"],
            ["Raw", f"{dab_raw:,.1f}", f"{dac_raw:,.1f}", "B ✘ (age 20 years apart!)"],
            ["Scaled", f"{dab_s:.3f}", f"{dac_s:.3f}", "C ✔"]]
    pg.table(x0 + 30, y0 + 265, 600, rows, size=12, highlight_rows=(2,))
    # contribution bars for d(A,B)^2
    pg.text("Share of d(A, B)² contributed by each feature", x0 + 30, y0 + 370, 600, 20, 12, bold=True)
    for i, (lab, age, inc) in enumerate([("Raw", 20 ** 2, 2000 ** 2), ("Scaled", 0.5 ** 2, 0.025 ** 2)]):
        yy = y0 + 398 + i * 40
        tot = age + inc
        wa = max(3, 480 * age / tot)
        pg.text(lab, x0 + 30, yy + 4, 70, 20, 11, bold=True)
        pg.rect(x0 + 100, yy, wa, 28, fill=LIGHTBLUE, stroke=BLUE)
        pg.rect(x0 + 100 + wa, yy, 480 - wa, 28, fill=LIGHTRED, stroke=RED)
        pg.text(f"age {100 * age / tot:.2f}%", x0 + 100 + (8 if wa < 100 else 10), yy + 4, 120, 20, 10,
                color=BLUE if wa >= 100 else BLACK)
        pg.text(f"income {100 * inc / tot:.2f}%", x0 + 460, yy + 4, 120, 20, 10, align="right")
    pg.bullets(x0 + 20, y0 + 490, 620, [
        "Without scaling the feature with the biggest units (income) decides everything",
        "Fit the scaler on the <b>training set only</b>, then apply it to validation / test data",
        "Same rule applies to K-Means, SVM, PCA and gradient descent"],
        head="Notes:")
    pg.ask("Which feature really decides who is 'nearest'?",
           "unscaled: income (99.99 % of the distance) → B looks closest to A; after scaling age counts too "
           "and C is correctly nearest",
           x0 + 20, y0 + 610, x0 + 520, y0 + 426, w=620)

    # ---------------------------------------------------------------- ④ effect of k
    x0, y0 = 400, 1000
    pg.panel(x0, y0, 1320, 740, 4, "Choosing k: decision boundaries & cross-validation",
             "Background: bias–variance trade-off, leave-one-out cross-validation (real computation on 100 points)")
    r = seeded(21)
    train = []
    while len(train) < 100:   # class B = inside a circle, 10 % label noise
        p = (r.uniform(0.3, 9.7), r.uniform(0.3, 9.7))
        c = "B" if (p[0] - 5) ** 2 + (p[1] - 5) ** 2 < 3.0 ** 2 else "A"
        if r.random() < 0.1: c = "A" if c == "B" else "B"
        train.append((p, c))
    N = 20
    for i, k in enumerate([1, 15]):
        bx = x0 + 40 + i * 360
        d = pg.axes(bx, y0 + 100, 300, 300, (0, 10), (0, 10), "x_1", "x_2", label_size=12)
        for gx in range(N):
            for gy in range(N):
                cx, cy = (gx + 0.5) * 10 / N, (gy + 0.5) * 10 / N
                c = knn(train, (cx, cy), k)
                tl = d.P(gx * 10 / N, (gy + 1) * 10 / N)
                pg.rect(tl[0], tl[1], 300 / N, 300 / N, fill=LIGHTBLUE if c == "A" else LIGHTRED, stroke="none")
        for (a, b), c in train:
            d.scatter([(a, b)], color=BLUE if c == "A" else RED, r=3)
        title = "k = 1 → jagged boundary (overfits, high variance)" if k == 1 else \
                "k = 15 → smooth boundary (low variance)"
        pg.text(title, bx - 10, y0 + 420, 330, 20, 12, bold=True, align="center", color=RED if k == 1 else GREEN)
    # CV curve
    ks = list(range(1, 60, 2))
    train_err = [sum(knn(train, p, k) != c for p, c in train) / len(train) for k in ks]
    cv_err = [sum(knn(train, p, k, skip=i) != c for i, (p, c) in enumerate(train)) / len(train) for k in ks]
    best = ks[min(range(len(ks)), key=lambda i: (cv_err[i], ks[i]))]
    cx0 = x0 + 790
    ymax = 0.1 * math.ceil(max(cv_err + train_err) * 10 + 0.5)
    C = pg.axes(cx0 + 40, y0 + 110, 420, 260, (0, 60), (0, ymax), "k", "error", label_size=12,
                xticks=(1, 15, 30, 45, 59), yticks=tuple(round(0.1 * i, 1) for i in range(1, int(ymax * 10) + 1)))
    C.poly(list(zip(ks, train_err)), color=GREY, width=1.5, dashed=True)
    C.poly(list(zip(ks, cv_err)), color=BLUE, width=2)
    C.scatter(list(zip(ks, cv_err)), color=BLUE, r=2.5)
    be = cv_err[ks.index(best)]
    C.scatter([(best, be)], color=GREEN, r=5)
    pg.line([(cx0 + 330, y0 + 120), (cx0 + 360, y0 + 120)], color=BLUE, width=2)
    pg.text("validation error (LOO-CV)", cx0 + 365, y0 + 111, 160, 18, 10, color=BLUE)
    pg.line([(cx0 + 330, y0 + 140), (cx0 + 360, y0 + 140)], color=GREY, width=1.5, dashed=True)
    pg.text("training error", cx0 + 365, y0 + 131, 160, 18, 10, color=GREY)
    pg.note(f"best k = {best}  (error {be:.2f})", cx0 + 200, y0 + 395, *C.P(best, be), w=150, color=GREEN)
    pg.note(f"k = 1: training error 0, validation {cv_err[0]:.2f}", cx0 + 70, y0 + 82, *C.P(1, cv_err[0]),
            w=230, color=RED)
    pg.note(f"k too large → underfits ({cv_err[-1]:.2f})", cx0 + 230, y0 + 175, *C.P(ks[-1], cv_err[-1]),
            w=170, color=RED)
    pg.bullets(x0 + 20, y0 + 470, 1280, [
        "<b>Small k</b> → follows noise (low bias, high variance);  <b>large k</b> → too smooth (high bias, low variance)",
        "Pick k with cross-validation: hold out each point (or fold), predict it from the rest, average the error",
        "Use an odd k for binary problems to avoid ties;  a rule of thumb is \\(k \\approx \\sqrt{n}\\)",
        "Curse of dimensionality: with many features all points look equally far → reduce dimensions first (PCA)"],
        head="Notes:")
    pg.ask("Is a k = 1 model memorising noise?",
           f"yes: training error 0 but validation error {cv_err[0]:.2f}; the small islands around single "
           "points are fitted label noise",
           x0 + 20, y0 + 610, x0 + 190, y0 + 400, w=620)
    pg.ask("Which k generalises best?",
           f"k = {best}: lowest leave-one-out error ({be:.2f}); larger k smooths away the circle, smaller k chases noise",
           x0 + 680, y0 + 610, *C.P(best, be), w=620)

    # ---------------------------------------------------------------- ⑤ worked example
    x0 = 1760
    pg.panel(x0, y0, 640, 740, 5, "Worked Example (k = 3, Euclidean)", "Query point q = (3, 4)")
    data = [("P1", (1, 2), "A"), ("P2", (2, 5), "A"), ("P3", (4, 4), "B"), ("P4", (3, 6), "B"),
            ("P5", (5, 2), "B"), ("P6", (2, 3), "A")]
    q = (3, 4)
    ds = sorted(((dist(p, q), n) for n, p, c in data))
    rank = {n: i + 1 for i, (_, n) in enumerate(ds)}
    rows = [["Point", "\\(x_1\\)", "\\(x_2\\)", "Class", "Distance to q", "Rank"]]
    hl = []
    for i, (n, p, c) in enumerate(data):
        dd = dist(p, q)
        rows.append([n, p[0], p[1], c, f"√({(p[0]-3)**2}+{(p[1]-4)**2}) = {dd:.2f}", rank[n]])
        if rank[n] <= 3: hl.append(i + 1)
    pg.table(x0 + 30, y0 + 70, 580, rows, size=12, highlight_rows=tuple(hl))
    top = [c for _, n in ds[:3] for nn, p, c in data if nn == n]
    win = max(set(top), key=top.count)
    pg.html(f"3 nearest: {', '.join(n for _, n in ds[:3])}  →  votes A = {top.count('A')}, B = {top.count('B')}"
            f'  →  <span style="background:{HL}"><b>q is class {win}</b></span>', x0 + 30, y0 + 245, 580, 24, 13,
            align="center", valign="middle")
    w = pg.axes(x0 + 60, y0 + 300, 180, 180, (0, 6), (0, 7), "x_1", "x_2", label_size=11)
    for n, p, c in data:
        w.scatter([p], color=BLUE if c == "A" else RED, r=4)
        w.text(n, *p, dx=14, dy=-8, size=9)
    w.scatter([q], color=HL, r=5)
    for _, n in ds[:3]:
        p = [pp for nn, pp, c in data if nn == n][0]
        pg.line([w.P(*q), w.P(*p)], color=GREEN, width=1.2)
    pg.rect(x0 + 290, y0 + 300, 320, 130,
            "<b>Algorithm</b><br><font face='Courier New'>for each training point:<br>&nbsp;&nbsp;d = distance(q, point)<br>"
            "sort by d, take first k<br>return majority class<br>(or mean value)</font>", fill="#F5F5F5",
            stroke="#999999", size=12, align="left", valign="top", raw=True)
    pg.bullets(x0 + 20, y0 + 510, 600, [
        "Training cost O(1);  prediction cost O(n·p) per query → slow for big data",
        "Speed-ups: KD-tree / Ball-tree, approximate nearest neighbours (FAISS, Annoy)",
        "Hyperparameters: k, metric, weights = uniform | distance",
        "Ties: use odd k, distance weights, or the class of the nearest point"],
        head="Notes:")
    pg.ask("Which class does q = (3, 4) get?",
           f"class {win}: its 3 nearest points vote A = {top.count('A')}, B = {top.count('B')} — even though the "
           f"single closest point ({ds[0][1]}) is B",
           x0 + 20, y0 + 640, *w.P(*q), w=600)

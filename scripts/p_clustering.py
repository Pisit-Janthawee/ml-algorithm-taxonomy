"""K-Means page: iterations, Euclidean distance, centroid = mean, elbow & silhouette, algorithm & pitfalls."""
import math
from dio import *

COL = [BLUE, RED, GREEN, ORANGE, "#9933CC", "#00AAAA", "#AA5500", "#666666"]


def _data():
    r = seeded(11)
    centers = [(2.5, 7.0), (7.5, 7.5), (5.0, 2.5)]
    pts = []
    for cx, cy in centers:
        pts += [(cx + r.gauss(0, 0.8), cy + r.gauss(0, 0.8)) for _ in range(18)]
    return [(min(9.6, max(0.4, a)), min(9.6, max(0.4, b))) for a, b in pts]


def _assign(pts, cents):
    return [min(range(len(cents)), key=lambda j: (p[0] - cents[j][0]) ** 2 + (p[1] - cents[j][1]) ** 2) for p in pts]


def _update(pts, lab, k, old):
    out = []
    for j in range(k):
        m = [p for p, l in zip(pts, lab) if l == j]
        out.append((sum(a for a, _ in m) / len(m), sum(b for _, b in m) / len(m)) if m else old[j])
    return out


def _kmeans(pts, k, iters=20):
    cents = pts[::max(1, len(pts) // k)][:k]
    for _ in range(iters):
        lab = _assign(pts, cents)
        cents = _update(pts, lab, k, cents)
    lab = _assign(pts, cents)
    inertia = sum((p[0] - cents[l][0]) ** 2 + (p[1] - cents[l][1]) ** 2 for p, l in zip(pts, lab))
    return cents, lab, inertia


def _centroid(pg, px, py, color, s=7):
    pg.rect(px - s, py - s, 2 * s, 2 * s, fill=color, stroke=BLACK, sw=2)


def build(pg: Page, cat):
    pts = _data()

    # ---------------------------------------------------------------- ① iterations
    x0, y0 = 400, 240
    pg.panel(x0, y0, 1320, 580, 1, "How K-Means works (k = 3)", "Assign → Update → Repeat until centroids stop moving")
    cents = [(1.0, 1.5), (2.0, 2.0), (9.0, 2.0)]          # deliberately poor start
    states = [("Step 0: random centroids", cents, None)]
    lab = _assign(pts, cents)
    states.append(("Iteration 1: assign to nearest + move centroids", _update(pts, lab, 3, cents), cents))
    c2 = states[-1][1]
    for _ in range(10):
        c2 = _update(pts, _assign(pts, c2), 3, c2)
    states.append(("Converged: assignments stop changing", c2, states[-1][1]))
    for k, (title, cs, prev) in enumerate(states):
        px0 = x0 + 60 + k * 430
        p = pg.axes(px0, y0 + 110, 320, 260, (0, 10), (0, 10), "x_1", "x_2", label_size=12)
        if k == 0:
            p.scatter(pts, color=GREY, r=3.5)
        else:
            l = _assign(pts, prev if k == 1 else cs)
            for pt, li in zip(pts, l):
                pg.dot(*p.P(*pt), r=3.5, color=COL[li])
        if prev:
            for (a, b), (c, d) in zip(prev, cs):
                pg.line([p.P(a, b), p.P(c, d)], dashed=True, arrow="classic", color=BLACK, width=1)
                pg.rect(*[v - 6 for v in p.P(a, b)], 12, 12, stroke=GREY, dashed=True)
        for j, (a, b) in enumerate(cs):
            _centroid(pg, *p.P(a, b), COL[j])
        pg.text(title, px0 - 20, y0 + 395, 360, 20, 12, bold=True, align="center")
    p0 = Plot(pg, x0 + 60, y0 + 110, 320, 260, (0, 10), (0, 10))
    pg.note("initial centroid \\(\\mu_j\\)", x0 + 120, y0 + 78, *p0.P(*cents[0]), w=120, raw=True)
    for k in range(2):
        ax = x0 + 60 + k * 430 + 345
        pg.line([(ax, y0 + 240), (ax + 60, y0 + 240)], arrow="block", width=2)
    pg.bullets(x0 + 20, y0 + 430, 1280, [
        "<b>Assign step</b>: every point joins the cluster of its nearest centroid (Euclidean distance ②)",
        "<b>Update step</b>: every centroid moves to the mean of its points (③) — dashed arrows show the move",
        "Repeat until no point changes cluster (or centroids move less than a tolerance) — always converges, maybe to a local minimum",
        "Poor start → use <b>k-means++</b> (spread-out initial centroids) and run several times (<i>n_init</i>), keep the lowest inertia"],
        head="Where:")
    p3 = Plot(pg, x0 + 60 + 860, y0 + 110, 320, 260, (0, 10), (0, 10))
    pg.ask("Which group does each point belong to?",
           "after convergence every point joins its nearest centroid: 3 stable groups, labels stop changing",
           x0 + 1040, y0 + 436, *p3.P(*c2[0]), w=260)

    # ---------------------------------------------------------------- ② Euclidean distance
    x0 = 1760
    pg.panel(x0, y0, 640, 580, 2, "Euclidean Distance", "Background: how 'near' is measured")
    p = pg.axes(x0 + 70, y0 + 100, 300, 260, (0, 6), (0, 6), "x_1", "x_2", label_size=12,
                xticks=(1, 4), yticks=(1, 5))
    A, B = (1, 1), (4, 5)
    p.poly([A, (4, 1), B], dashed=True, color=GREEN, width=1.5)
    p.poly([A, B], color=RED, width=2)
    p.scatter([A], color=BLUE, r=5)
    _centroid(pg, *p.P(*B), RED, s=6)
    p.text("point x = (1, 1)", 1, 1, w=110, dy=16, size=10)
    p.text("centroid μ = (4, 5)", 4, 5, w=120, dy=-16, size=10)
    p.math("\\Delta x_1 = 3", 2.5, 1, dy=14, size=11, w=90, color=GREEN)
    p.math("\\Delta x_2 = 4", 4, 3, dx=44, size=11, w=90, color=GREEN)
    p.math("d = 5", 2.5, 3, dx=-30, dy=-10, size=13, w=70, color=RED)
    pg.ask("How far is this point from the cluster centre?",
           "d = 5 units; the centroid with the smallest d decides the point's cluster",
           x0 + 390, y0 + 90, *p.P(2.5, 3), w=235)
    pg.math("d(x,\\mu) = \\sqrt{\\sum_{j=1}^{p}(x_j-\\mu_j)^2}", x0 + 30, y0 + 378, 580, 56, 18)
    pg.math("= \\sqrt{3^2 + 4^2} = 5", x0 + 30, y0 + 440, 580, 30, 15)
    pg.bullets(x0 + 20, y0 + 475, 600, [
        "Manhattan: \\(\\sum|x_j-\\mu_j|\\) = 3 + 4 = 7 (city-block)",
        "Distances depend on units → <b>scale features first</b> (standardize)",
        "K-Means uses squared distance \\(\\lVert x-\\mu\\rVert^2\\) (no square root needed)"], head="Notes:")

    # ---------------------------------------------------------------- ③ centroid = mean & inertia
    x0, y0 = 400, 860
    pg.panel(x0, y0, 620, 640, 3, "Centroid = Mean  &  Objective (Inertia)", "Background: the update step")
    p = pg.axes(x0 + 70, y0 + 80, 220, 170, (0, 4), (0, 4), "x_1", "x_2", label_size=12, xticks=(1, 2, 3),
                yticks=(1, 2, 3))
    cl = [(1, 2), (2, 1), (3, 3)]
    for q in cl:
        pg.line([p.P(*q), p.P(2, 2)], color=RED, dashed=True)
    p.scatter(cl, r=5)
    _centroid(pg, *p.P(2, 2), BLUE, s=7)
    for q in cl:
        p.text(f"({q[0]}, {q[1]})", *q, w=50, dx=24, dy=-10, size=10)
    pg.note("new centroid (2, 2)", x0 + 330, y0 + 120, *p.P(2.1, 2), w=130, color=BLUE)
    pg.note("squared distances → inertia", x0 + 330, y0 + 180, *p.P(2.6, 2.6), w=170, color=RED)
    pg.ask("Where should the cluster centre move?",
           "to the mean of its points, (2, 2), the spot with the smallest squared distances",
           x0 + 360, y0 + 202, *p.P(2, 1.85), w=250)
    pg.math("\\mu_j = \\frac{1}{|C_j|}\\sum_{x_i \\in C_j} x_i", x0 + 40, y0 + 285, 540, 56, 18)
    pg.math("\\mu = \\left(\\frac{1+2+3}{3},\\ \\frac{2+1+3}{3}\\right) = (2,\\,2)", x0 + 40, y0 + 350, 540, 44, 15)
    pg.math("J = \\sum_{j=1}^{k}\\sum_{x_i \\in C_j}\\lVert x_i-\\mu_j\\rVert^2", x0 + 40, y0 + 405, 540, 60, 18)
    pg.html(f'<span style="background:{HL}">\\(J = (1+0) + (0+1) + (1+1) = 4\\)</span>', x0 + 40, y0 + 475, 540,
            30, 14, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 530, 580, [
        "\\(C_j\\) = set of points in cluster j,  \\(|C_j|\\) = how many",
        "J (inertia / WCSS) = within-cluster sum of squares — K-Means minimizes it",
        "The mean is exactly the point that minimizes the squared distances"], head="Where:")

    # ---------------------------------------------------------------- ④ elbow & silhouette
    x0 = 1060
    pg.panel(x0, y0, 700, 640, 4, "Choosing k: Elbow & Silhouette")
    ks = list(range(1, 9))
    inert = []
    for k in ks:
        best = min((_kmeans(pts[s:] + pts[:s], k)[2] for s in (0, 7, 19, 31)), default=0)
        inert.append(best)
    top = math.ceil(inert[0] / 100) * 100
    e = pg.axes(x0 + 70, y0 + 80, 300, 230, (0.5, 8.5), (0, top), "k", "J", label_size=13, origin=(0.5, 0),
                xticks=ks, yticks=(0, top // 2, top))
    e.poly(list(zip(ks, inert)), color=BLUE, width=2)
    e.scatter(list(zip(ks, inert)), r=4)
    e.scatter([(3, inert[2])], color=RED, r=6)
    pg.note("elbow → k = 3", x0 + 160, y0 + 200, *e.P(3, inert[2]), w=100, color=RED)
    pg.ask("How many clusters are really in the data?",
           f"k = 3: J drops {inert[1] - inert[2]:.0f} from k=2→3 but only {inert[2] - inert[3]:.0f} after → 3 groups",
           x0 + 195, y0 + 82, *e.P(3.05, inert[2] * 1.05), w=200)
    pg.text("Inertia always drops as k grows — pick the 'elbow' where gains flatten", x0 + 30, y0 + 335, 360, 34,
            11, align="center")
    # silhouette illustration
    sx = x0 + 410
    pg.text("Silhouette of one point i", sx, y0 + 60, 270, 20, 12, bold=True)
    own = [(sx + 60, y0 + 140), (sx + 90, y0 + 120), (sx + 75, y0 + 180), (sx + 40, y0 + 165)]
    oth = [(sx + 200, y0 + 140), (sx + 230, y0 + 170), (sx + 215, y0 + 110), (sx + 250, y0 + 135)]
    pi = (sx + 95, y0 + 155)
    for q in own:
        pg.line([pi, q], color=BLUE, dashed=True)
    for q in oth:
        pg.line([pi, q], color=RED, dashed=True, opacity=60)
    for q in own: pg.dot(*q, r=4.5, color=BLUE)
    for q in oth: pg.dot(*q, r=4.5, color=RED)
    pg.dot(*pi, r=6, color=BLACK)
    pg.text("i", pi[0] + 6, pi[1] + 2, 20, 16, 12, bold=True)
    pg.html("<font color='#3333FF'>\\(a(i)\\) = mean distance to own cluster</font><br>"
            "<font color='#FF0000'>\\(b(i)\\) = mean distance to nearest other cluster</font>", sx, y0 + 210, 280, 44,
            11)
    pg.math("s(i) = \\frac{b(i) - a(i)}{\\max\\{a(i),\\, b(i)\\}}", sx, y0 + 260, 280, 50, 17)
    pg.table(x0 + 30, y0 + 390, 640, [["s(i)", "meaning"],
                                      ["≈ +1", "well inside its cluster (good)"],
                                      ["≈ 0", "on the border between two clusters"],
                                      ["< 0", "probably in the wrong cluster"]], size=11)
    pg.bullets(x0 + 20, y0 + 500, 660, [
        "Average silhouette over all points → choose the k with the highest score",
        "Elbow is visual / subjective; silhouette gives a number; both should agree",
        "Business sense matters too (e.g. 3 customer segments are easier to act on)"], head="Notes:")

    # ---------------------------------------------------------------- ⑤ algorithm & pitfalls
    x0 = 1800
    pg.panel(x0, y0, 600, 640, 5, "Algorithm & Pitfalls")
    pg.rect(x0 + 20, y0 + 50, 560, 150,
            "<b>Pseudo-code</b><br><font face='Courier New'>scale X<br>μ = k-means++ init (k points)<br>"
            "repeat:<br>&nbsp;&nbsp;c_i = argmin_j ‖x_i − μ_j‖²&nbsp;&nbsp;# assign<br>"
            "&nbsp;&nbsp;μ_j = mean(x_i where c_i = j)&nbsp;# update<br>until c does not change</font>",
            fill="#F5F5F5", stroke="#999999", size=12, align="left", valign="top", raw=True)
    pg.text("✘ Fails on non-spherical clusters (rings)", x0 + 20, y0 + 215, 560, 20, 12, bold=True, color=RED)
    p = pg.axes(x0 + 60, y0 + 250, 200, 200, (-1.3, 1.3), (-1.3, 1.3), "", "", origin=(-1.3, -1.3))
    r = seeded(5)
    ring = []
    for i in range(48):
        t = 2 * math.pi * i / 48
        rad = (0.35 if i % 2 else 1.05) + r.gauss(0, 0.05)
        ring.append((rad * math.cos(t), rad * math.sin(t)))
    for a, b in ring:
        pg.dot(*p.P(a, b), r=3, color=BLUE if a < 0 else RED)
    pg.text("K-Means (k=2): splits left / right", x0 + 30, y0 + 460, 260, 20, 10, align="center", color=RED)
    pg.ask("Can K-Means find clusters of any shape?",
           "No: it splits by distance to centres, so rings are cut left/right; use DBSCAN or Spectral",
           x0 + 370, y0 + 58, *p.P(0.1, 1.05), w=205)
    q = pg.axes(x0 + 340, y0 + 250, 200, 200, (-1.3, 1.3), (-1.3, 1.3), "", "", origin=(-1.3, -1.3))
    for i, (a, b) in enumerate(ring):
        pg.dot(*q.P(a, b), r=3, color=BLUE if i % 2 else RED)
    pg.text("DBSCAN / Spectral: inner vs outer ✔", x0 + 310, y0 + 460, 260, 20, 10, align="center", color=GREEN)
    pg.bullets(x0 + 20, y0 + 495, 560, [
        "Must choose k in advance;  sensitive to outliers (mean is pulled) → K-Medoids",
        "Assumes round, similar-size clusters → GMM for elliptical / soft clusters",
        "Complexity ≈ O(n · k · iterations) → fast;  Mini-Batch K-Means for big data"], head="Pitfalls:")

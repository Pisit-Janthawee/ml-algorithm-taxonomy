"""PCA page, part 3: t-SNE and UMAP (panels 11-14), computed with a small pure-Python t-SNE and a
simplified UMAP (fuzzy kNN graph + UMAP-style SGD with negative sampling, PCA initialisation)."""
import math
import random
from dio import *

COLS = [BLUE, RED, GREEN, ORANGE]
_CACHE = {}


# ---------------------------------------------------------------------- data
def make_data(seed=11):
    """90 points in 10-D: 3 Gaussian clusters + 1 elongated cluster."""
    r = random.Random(seed)
    centers = [[r.gauss(0, 1.3) for _ in range(10)] for _ in range(3)]
    X, lab = [], []
    for c, (ctr, n) in enumerate(zip(centers, (25, 25, 20))):
        for _ in range(n):
            X.append([m + r.gauss(0, 1.0) for m in ctr]); lab.append(c)
    d = [r.gauss(0, 1) for _ in range(10)]
    nd = math.sqrt(sum(v * v for v in d)); d = [v / nd for v in d]
    base = [r.gauss(0, 1.3) for _ in range(10)]
    for _ in range(20):
        t = r.uniform(-4, 4)
        X.append([b + t * dv + r.gauss(0, 0.5) for b, dv in zip(base, d)]); lab.append(3)
    return X, lab


def sqd(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def pca2(X):
    n, p = len(X), len(X[0])
    mu = [sum(row[j] for row in X) / n for j in range(p)]
    Z = [[row[j] - mu[j] for j in range(p)] for row in X]
    C = [[sum(z[a] * z[b] for z in Z) / (n - 1) for b in range(p)] for a in range(p)]
    vecs, lams = [], []
    for _ in range(2):
        v = [1.0 / math.sqrt(p)] * p
        for _ in range(200):
            w = [sum(C[a][b] * v[b] for b in range(p)) for a in range(p)]
            for u in vecs:                                    # deflate
                dot = sum(wi * ui for wi, ui in zip(w, u)); w = [wi - dot * ui for wi, ui in zip(w, u)]
            nw = math.sqrt(sum(x * x for x in w)); v = [x / nw for x in w]
        lams.append(sum(v[a] * sum(C[a][b] * v[b] for b in range(p)) for a in range(p)))
        vecs.append(v)
    Y = [[sum(z[j] * u[j] for j in range(p)) for u in vecs] for z in Z]
    return Y, lams, sum(C[j][j] for j in range(p))


# ---------------------------------------------------------------------- t-SNE
def cond_P(D, perp):
    n = len(D)
    target = math.log(perp)
    P, sig = [[0.0] * n for _ in range(n)], []
    for i in range(n):
        beta, lo, hi = 1.0, 0.0, float("inf")
        for _ in range(60):
            ps = [0.0 if j == i else math.exp(-D[i][j] * beta) for j in range(n)]
            s = sum(ps) or 1e-300
            H = math.log(s) + beta * sum(D[i][j] * ps[j] for j in range(n)) / s
            if abs(H - target) < 1e-5: break
            if H > target:
                lo = beta; beta = beta * 2 if hi == float("inf") else (beta + hi) / 2
            else:
                hi = beta; beta = (beta + lo) / 2
        P[i] = [v / s for v in ps]
        sig.append(math.sqrt(1 / (2 * beta)))
    return P, sig


def tsne(X, perp=30, iters=350, seed=1, lr=50.0):
    key = ("tsne", id(X), perp, iters, seed)
    if key in _CACHE: return _CACHE[key]
    n = len(X)
    D = [[sqd(X[i], X[j]) for j in range(n)] for i in range(n)]
    Pc, sig = cond_P(D, perp)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    Pf = [max((Pc[i][j] + Pc[j][i]) / (2 * n), 1e-12) for i, j in pairs]
    r = random.Random(seed)
    Y0 = [r.gauss(0, 1e-2) for _ in range(n)]; Y1 = [r.gauss(0, 1e-2) for _ in range(n)]
    V0, V1 = [0.0] * n, [0.0] * n
    G0, G1 = [1.0] * n, [1.0] * n
    kl = []
    for it in range(iters):
        ex = 12.0 if it < 100 else 1.0
        mom = 0.5 if it < 100 else 0.8
        nums, dxs, dys, s = [], [], [], 0.0
        for i, j in pairs:
            dx = Y0[i] - Y0[j]; dy = Y1[i] - Y1[j]
            v = 1.0 / (1.0 + dx * dx + dy * dy)
            nums.append(v); dxs.append(dx); dys.append(dy); s += v
        s *= 2
        g0, g1 = [0.0] * n, [0.0] * n
        for k, (i, j) in enumerate(pairs):
            v = nums[k]
            c = 4 * (ex * Pf[k] - v / s) * v
            g0[i] += c * dxs[k]; g0[j] -= c * dxs[k]
            g1[i] += c * dys[k]; g1[j] -= c * dys[k]
        for i in range(n):
            for G, g, V, Y in ((G0, g0, V0, Y0), (G1, g1, V1, Y1)):
                G[i] = G[i] + 0.2 if (g[i] > 0) != (V[i] > 0) else max(G[i] * 0.8, 0.01)
                V[i] = mom * V[i] - lr * G[i] * g[i]
                Y[i] += V[i]
        if (it % 10 == 0 and it > 0) or it == iters - 1:
            kl.append((it, sum(p * math.log(p / max(nums[k] / s, 1e-12)) for k, p in enumerate(Pf))))
    out = ([[a, b] for a, b in zip(Y0, Y1)], kl, sig)
    _CACHE[key] = out
    return out


def knn_agree(Y, lab, k=5):
    """Share of each point's k nearest neighbours (in Y) that have the same true label."""
    n, tot = len(Y), 0
    for i in range(n):
        nb = sorted(range(n), key=lambda j: sqd(Y[i], Y[j]) if j != i else 1e18)[:k]
        tot += sum(lab[j] == lab[i] for j in nb)
    return tot / (n * k)


# ---------------------------------------------------------------------- drawing helpers
def frame_plot(pg, x, y, w, h, Y, pad=0.08):
    xs = [p[0] for p in Y]; ys = [p[1] for p in Y]
    dx = (max(xs) - min(xs)) or 1; dy = (max(ys) - min(ys)) or 1
    pg.rect(x, y, w, h, stroke="#999999", sw=0.75)
    return Plot(pg, x, y, w, h, (min(xs) - pad * dx, max(xs) + pad * dx), (min(ys) - pad * dy, max(ys) + pad * dy))


def draw_emb(pg, x, y, w, h, Y, lab, r=3.2, title=None, sub=None):
    p = frame_plot(pg, x, y, w, h, Y)
    for (a, b), l in zip(Y, lab):
        pg.dot(*p.P(a, b), r=r, color=COLS[l % len(COLS)])
    if title: pg.text(title, x, y - 22, w, 20, 12, bold=True, align="center")
    if sub: pg.text(sub, x, y + h + 4, w, 18, 10, align="center", color="#555555")
    return p


def build(pg: Page, cat):
    X, lab = make_data()
    Y, kl, sig = tsne(X, perp=30)
    Ypca, lams, tot = pca2(X)
    y0 = pg.max_y() + 60

    # ------------------------------------------------------------------ ⑪ t-SNE step by step
    x0 = 400
    pg.panel(x0, y0, 1000, 780, 11, "t-SNE step by step (computed on 90 points in 10-D)",
             "t-distributed Stochastic Neighbour Embedding: keep each point's neighbours as neighbours in 2-D")
    pg.text("1. High-D similarities (Gaussian around each point)", x0 + 20, y0 + 62, 480, 20, 12, bold=True)
    pg.math(r"p_{j|i} = \frac{\exp\left(-\lVert x_i - x_j\rVert^2 / 2\sigma_i^2\right)}"
            r"{\sum_{k\neq i}\exp\left(-\lVert x_i - x_k\rVert^2 / 2\sigma_i^2\right)}", x0 + 20, y0 + 86, 480, 64, 16)
    pg.math(r"p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}", x0 + 20, y0 + 152, 240, 44, 15)
    pg.note("σᵢ found by binary search so that 2^H(Pᵢ) = perplexity", x0 + 270, y0 + 160, x0 + 360, y0 + 132, w=230,
            color=RED)
    pg.text(f"Here perplexity = 30 → σᵢ ranges {min(sig):.2f} … {max(sig):.2f} (small σ in dense regions)",
            x0 + 20, y0 + 202, 480, 18, 10, italic=True, color="#555555")
    pg.text("2. Low-D similarities (Student-t, 1 degree of freedom)", x0 + 20, y0 + 228, 480, 20, 12, bold=True)
    pg.math(r"q_{ij} = \frac{\left(1 + \lVert y_i - y_j\rVert^2\right)^{-1}}"
            r"{\sum_{k\neq l}\left(1 + \lVert y_k - y_l\rVert^2\right)^{-1}}", x0 + 20, y0 + 252, 480, 60, 16)
    # crowding problem: normal vs t
    c = pg.axes(x0 + 60, y0 + 345, 400, 160, (-5, 5), (0, 0.42), "d", "", origin=(-5, 0), xticks=(-4, -2, 0, 2, 4))
    c.curve(normal_pdf, -5, 5, n=80, color=BLUE, width=2)
    c.curve(lambda v: 1 / (math.pi * (1 + v * v)), -5, 5, n=80, color=RED, width=2, dashed=True)
    pg.html('<font color="#3333FF"><b>— Gaussian</b> (high-D)</font><br><font color="#FF0000"><b>- - Student-t</b> '
            '(low-D)</font>', x0 + 340, y0 + 340, 140, 36, 10)
    pg.note("heavy tail: moderately far points may sit far apart in 2-D", x0 + 260, y0 + 520, *c.P(3.4, 0.025),
            w=230, color=RED)
    pg.text("Crowding problem: 2-D has far less room than 10-D; heavy tails stop clusters from collapsing together",
            x0 + 20, y0 + 552, 480, 34, 10, italic=True)
    # right column: cost, gradient, KL curve
    rx = x0 + 530
    pg.text("3. Cost: Kullback-Leibler divergence", rx, y0 + 62, 440, 20, 12, bold=True)
    pg.math(r"C = KL(P\,\|\,Q) = \sum_{i\neq j} p_{ij}\log\frac{p_{ij}}{q_{ij}}", rx, y0 + 86, 440, 50, 16)
    pg.math(r"\frac{\partial C}{\partial y_i} = 4\sum_{j}(p_{ij}-q_{ij})(y_i-y_j)\left(1+\lVert y_i-y_j\rVert^2\right)^{-1}",
            rx, y0 + 140, 440, 50, 14)
    pg.note("(p − q) > 0 → pull together, < 0 → push apart", rx + 150, y0 + 196, rx + 175, y0 + 172, w=260, color=RED)
    pg.text("4. Gradient descent (early exaggeration ×12 for 100 iterations, then momentum 0.8)", rx, y0 + 222, 440,
            34, 12, bold=True)
    top = max(v for _, v in kl) * 1.1
    k = pg.axes(rx + 50, y0 + 290, 360, 210, (0, kl[-1][0]), (0, top), "iteration", "KL", label_size=11,
                xticks=(0, 100, 200, 300), yticks=(round(top / 2, 1),))
    k.poly(kl, color=BLUE, width=2)
    k.scatter([kl[-1]], color=RED, r=4)
    k.vline(100, 0, top)
    k.text("exaggeration ends", 100, top * 0.95, dx=58, size=9, color="#555555", w=110)
    pg.ask("Has the 2-D layout finished improving?",
           f"yes: KL falls to {kl[-1][1]:.2f} and flattens; the exaggeration bump at 100 is expected",
           rx + 230, y0 + 520, *k.P(kl[-1][0], kl[-1][1]), w=230)
    pg.ask("What does 'perplexity' control?",
           "≈ how many neighbours each point listens to (5–50); it sets every σᵢ",
           x0 + 20, y0 + 600, x0 + 30, y0 + 214, w=250)
    pg.bullets(x0 + 290, y0 + 600, 690, [
        "Exact t-SNE costs O(n²) per iteration; Barnes-Hut / FFT versions make it O(n log n)",
        "Only <b>local</b> neighbourhoods are preserved: the cost barely cares about far pairs (small pᵢⱼ)",
        "Non-convex: different seeds give different (rotated / rearranged) maps",
        "This page's t-SNE is a small exact implementation (350 iterations, learning rate 50, adaptive gains)"],
        head="Notes:", size=11)

    # ------------------------------------------------------------------ ⑫ PCA vs t-SNE, perplexity
    x0 = 1440
    pg.panel(x0, y0, 960, 780, 12, "Result: PCA vs t-SNE on the same 10-D data",
             "Colours = true groups (3 round clusters + 1 elongated); the algorithms never see the colours")
    agree_pca, agree_ts = knn_agree(Ypca, lab), knn_agree(Y, lab)
    p1 = draw_emb(pg, x0 + 40, y0 + 100, 420, 300, Ypca, lab, title="PCA (linear, 2 components)",
                  sub=f"variance kept {100 * sum(lams) / tot:.0f}% · 5-NN label agreement {agree_pca:.2f}")
    p2 = draw_emb(pg, x0 + 500, y0 + 100, 420, 300, Y, lab, title="t-SNE (perplexity 30)",
                  sub=f"5-NN label agreement {agree_ts:.2f}")
    pg.ask("Do the 4 groups really separate in 10-D?",
           f"yes: t-SNE keeps neighbours together ({agree_ts:.2f} agreement); the 2-D PCA view mixes them ({agree_pca:.2f})",
           x0 + 560, y0 + 440, x0 + 700, y0 + 400, w=330)
    pg.text("Effect of perplexity (same data, same seed)", x0 + 20, y0 + 540, 600, 20, 13, bold=True)
    for i, pp in enumerate((5, 30, 50)):
        Yp = Y if pp == 30 else tsne(X, perp=pp)[0]
        draw_emb(pg, x0 + 40 + i * 305, y0 + 590, 270, 140, Yp, lab, r=2.4, sub=f"perplexity {pp}"
                 + ("  (small, local, fragmented)" if pp == 5 else "  (more global)" if pp == 50 else ""))
    pg.ask("Which perplexity should I use?",
           "try 5–50; low values split groups into fragments, higher values show the bigger picture",
           x0 + 40, y0 + 440, x0 + 175, y0 + 590, w=300)

    # ------------------------------------------------------------------ ⑬ UMAP
    y1 = y0 + 820
    x0 = 400
    pg.panel(x0, y1, 1160, 760, 13, "UMAP: fuzzy neighbour graph → 2-D layout",
             "Uniform Manifold Approximation and Projection (shown with a simplified implementation, see notes)")
    k = 8
    edges, Yu = umap_layout(X, k)
    pg.text(f"1. kNN graph in 10-D (k = {k}), drawn on PCA axes", x0 + 30, y1 + 64, 330, 20, 12, bold=True)
    pg_ = frame_plot(pg, x0 + 30, y1 + 90, 330, 250, Ypca)
    for i, j, w in edges:
        if i < j:
            pg.line([pg_.P(*Ypca[i]), pg_.P(*Ypca[j])], color="#999999", width=0.4 + 1.2 * w, opacity=60)
    for (a, b), l in zip(Ypca, lab):
        pg.dot(*pg_.P(a, b), r=3, color=COLS[l])
    agree_u = knn_agree(Yu, lab)
    draw_emb(pg, x0 + 400, y1 + 90, 330, 250, Yu, lab, title="2. Optimised 2-D layout",
             sub=f"5-NN label agreement {agree_u:.2f}")
    fx = x0 + 760
    pg.text("Fuzzy edge weight (high-D)", fx, y1 + 64, 380, 20, 12, bold=True)
    pg.math(r"w_{ij} = \exp\left(-\frac{d_{ij} - \rho_i}{\sigma_i}\right),\quad \sum_j w_{ij} = \log_2 k",
            fx, y1 + 88, 380, 44, 14)
    pg.note("ρᵢ = distance to the nearest neighbour (always weight 1)", fx + 10, y1 + 136, fx + 120, y1 + 110, w=320,
            color=RED)
    pg.math(r"w^{sym}_{ij} = w_{ij} + w_{ji} - w_{ij}w_{ji}", fx, y1 + 160, 380, 34, 14)
    pg.text("Low-D similarity and cost", fx, y1 + 204, 380, 20, 12, bold=True)
    pg.math(r"v_{ij} = \left(1 + a\,\lVert y_i - y_j\rVert^{2b}\right)^{-1}", fx, y1 + 228, 380, 40, 14)
    pg.math(r"CE = \sum w\log\frac{w}{v} + (1-w)\log\frac{1-w}{1-v}", fx, y1 + 272, 380, 44, 14)
    pg.note("1st term: attract graph neighbours", fx + 10, y1 + 322, fx + 140, y1 + 300, w=200, color=GREEN)
    pg.note("2nd term: repel non-neighbours (negative sampling)", fx + 170, y1 + 342, fx + 300, y1 + 300, w=210,
            color=RED)
    # n_neighbors effect
    pg.text("n_neighbors controls local ↔ global", x0 + 30, y1 + 380, 400, 20, 12, bold=True)
    for i, kk in enumerate((4, 20)):
        _, Yk = umap_layout(X, kk)
        draw_emb(pg, x0 + 30 + i * 210, y1 + 408, 185, 130, Yk, lab, r=2.2, sub=f"n_neighbors = {kk}")
    # min_dist target curves
    m = pg.axes(x0 + 480, y1 + 408, 250, 130, (0, 3), (0, 1.1), "d", "v", label_size=11, xticks=(0, 1, 2, 3))
    for md, col in ((0.1, BLUE), (0.8, RED)):
        m.curve(lambda d, md=md: 1.0 if d < md else math.exp(-(d - md)), 0, 3, n=60, color=col, width=2,
                curved=False)
    pg.html('<font color="#3333FF"><b>min_dist 0.1</b>: tight clumps</font><br>'
            '<font color="#FF0000"><b>min_dist 0.8</b>: spread out</font>', x0 + 600, y1 + 400, 160, 34, 10)
    pg.text("target low-D similarity UMAP fits a, b to", x0 + 470, y1 + 552, 280, 16, 9, italic=True,
            color="#555555")
    pg.ask("Does UMAP keep both the groups and their layout?",
           f"groups yes ({agree_u:.2f} agreement); positions start from a global (here PCA) layout, so arrangement is more faithful than t-SNE",
           fx, y1 + 380, x0 + 735, y1 + 215, w=380)
    pg.table(x0 + 20, y1 + 590, 1120, [
        ["", "t-SNE", "UMAP"],
        ["Preserves", "local neighbourhoods", "local + more global structure"],
        ["Speed", "slow (O(n²), Barnes-Hut O(n log n))", "fast, scales to millions"],
        ["Transform new data", "no (re-run)", "yes (.transform)"],
        ["Key parameters", "perplexity 5–50, learning rate, iterations", "n_neighbors 5–50, min_dist 0–0.99, metric"],
        ["Use as features?", "visualisation only", "sometimes (clustering after UMAP, e.g. HDBSCAN)"]], size=11, h=150)

    # ------------------------------------------------------------------ ⑭ pitfalls
    x0 = 1600
    pg.panel(x0, y1, 800, 760, 14, "Pitfalls & reading guide", "What a t-SNE / UMAP plot can and cannot tell you")
    r = random.Random(5)
    noise = [[r.gauss(0, 1) for _ in range(10)] for _ in range(60)]
    Yn = tsne(noise, perp=3, iters=300)[0]
    pn = draw_emb(pg, x0 + 30, y1 + 100, 350, 220, Yn, [0] * 60, title="Pure random noise, perplexity 3",
                  sub="no structure exists, yet small 'clumps' appear")
    sizes = [[r.gauss(0, 0.3) for _ in range(10)] for _ in range(30)] + \
            [[8 + r.gauss(0, 3.0) for _ in range(10)] for _ in range(30)]
    Ys = tsne(sizes, perp=15, iters=300)[0]
    ps = draw_emb(pg, x0 + 420, y1 + 100, 350, 220, Ys, [0] * 30 + [1] * 30, title="Tight (σ = 0.3) vs wide (σ = 3)")

    def spread(P):
        mx = sum(p[0] for p in P) / len(P); my = sum(p[1] for p in P) / len(P)
        return math.sqrt(sum((p[0] - mx) ** 2 + (p[1] - my) ** 2 for p in P) / len(P))
    ratio = spread(Ys[30:]) / spread(Ys[:30])
    pg.text(f"true spread ratio 10× → in t-SNE only {ratio:.1f}×", x0 + 420, y1 + 324, 350, 18, 10, align="center",
            color=RED)
    pg.ask("Do these clumps prove real clusters?",
           "no: t-SNE invents clumps in pure noise at low perplexity; confirm with other seeds / methods",
           x0 + 30, y1 + 360, *pn.P(*Yn[0]), w=350)
    pg.ask("Is the big cluster really more spread out?",
           f"you can't tell: t-SNE equalises density, a 10× difference shows as {ratio:.1f}×",
           x0 + 420, y1 + 360, x0 + 595, y1 + 324, w=350)
    pg.table(x0 + 30, y1 + 470, 740, [
        ["✔ Do", "✘ Don't"],
        ["Scale features; reduce to ~50 dims with PCA first", "Read cluster sizes as real spreads"],
        ["Try several perplexities / n_neighbors and seeds", "Read distances between clusters as similarity"],
        ["Colour by known labels to check what you see", "Trust a single run or a single setting"],
        ["Use for exploration & presentation", "Feed t-SNE axes into a model (it can't transform new data)"]],
        size=11, h=140)
    pg.bullets(x0 + 30, y1 + 630, 740, [
        "Use PCA when you need meaningful axes, variance and new-data transforms; t-SNE / UMAP to <i>see</i> neighbourhoods",
        "Simplified UMAP here: fuzzy kNN graph + UMAP's attract/repel SGD with negative sampling, PCA instead of spectral initialisation"],
        head="Notes:", size=11)


def umap_layout(X, k, epochs=200, seed=3, a=1.577, b=0.895):
    """Simplified UMAP (min_dist ≈ 0.1 → a, b): fuzzy kNN graph + attract/repel SGD with negative sampling."""
    key = ("umap", id(X), k, epochs, seed)
    if key in _CACHE: return _CACHE[key]
    n = len(X)
    D = [[math.sqrt(sqd(X[i], X[j])) for j in range(n)] for i in range(n)]
    W = {}
    target = math.log2(k)
    for i in range(n):
        nb = sorted((j for j in range(n) if j != i), key=lambda j: D[i][j])[:k]
        rho = D[i][nb[0]]
        lo, hi, s = 1e-4, 100.0, 1.0
        for _ in range(60):
            s = (lo + hi) / 2
            tot = sum(math.exp(-(D[i][j] - rho) / s) for j in nb)
            if tot > target: hi = s
            else: lo = s
        for j in nb:
            W[(i, j)] = math.exp(-(D[i][j] - rho) / s)
    sym = {}
    for (i, j), w in W.items():
        w2 = W.get((j, i), 0.0)
        key2 = (min(i, j), max(i, j))
        sym[key2] = w + w2 - w * w2
    edges = [(i, j, w) for (i, j), w in sym.items()] + [(j, i, w) for (i, j), w in sym.items()]
    Yp, _, _ = pca2(X)
    span = max(abs(v) for p in Yp for v in p) or 1
    Y = [[10 * p[0] / span, 10 * p[1] / span] for p in Yp]
    r = random.Random(seed)
    clip = lambda v: max(-4.0, min(4.0, v))
    for e in range(epochs):
        alpha = 1.0 - e / epochs
        for i, j, w in edges:
            dx = Y[i][0] - Y[j][0]; dy = Y[i][1] - Y[j][1]
            d2 = dx * dx + dy * dy
            if d2 > 0:
                c = -2 * a * b * d2 ** (b - 1) / (1 + a * d2 ** b) * w
                gx, gy = clip(c * dx), clip(c * dy)
                Y[i][0] += alpha * gx; Y[i][1] += alpha * gy
                Y[j][0] -= alpha * gx; Y[j][1] -= alpha * gy
            for _ in range(3):
                l = r.randrange(n)
                if l == i: continue
                dx = Y[i][0] - Y[l][0]; dy = Y[i][1] - Y[l][1]
                d2 = dx * dx + dy * dy
                c = 2 * b / ((0.001 + d2) * (1 + a * d2 ** b))
                Y[i][0] += alpha * clip(c * dx); Y[i][1] += alpha * clip(c * dy)
    out = (edges, Y)
    _CACHE[key] = out
    return out

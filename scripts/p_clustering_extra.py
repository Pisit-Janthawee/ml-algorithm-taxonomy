"""Extra K-Means panels: how to find the optimal number of clusters (and other models' hyper-parameters)."""
import math
from dio import *

COLS = [BLUE, RED, GREEN, ORANGE, "#9933FF", "#009999", "#996633", GREY]


# ---------------------------------------------------------------- small numeric helpers
def d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def kmeans(X, k, rng, restarts=8, iters=60):
    best = None
    for _ in range(restarts):
        C = [X[rng.randrange(len(X))]]
        while len(C) < k:                                    # k-means++ seeding
            w = [min(d2(x, c) for c in C) for x in X]
            t, acc = rng.random() * sum(w), 0
            for x, wi in zip(X, w):
                acc += wi
                if acc >= t:
                    C.append(x); break
        lab = [0] * len(X)
        for _ in range(iters):
            lab = [min(range(k), key=lambda j: d2(x, C[j])) for x in X]
            newC = []
            for j in range(k):
                m = [x for x, l in zip(X, lab) if l == j] or [C[j]]
                newC.append((sum(p[0] for p in m) / len(m), sum(p[1] for p in m) / len(m)))
            if newC == C: break
            C = newC
        inertia = sum(d2(x, C[l]) for x, l in zip(X, lab))
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best


def silhouette(X, lab, k):
    n = len(X)
    D = [[math.sqrt(d2(X[i], X[j])) for j in range(n)] for i in range(n)]
    s = []
    for i in range(n):
        own = [D[i][j] for j in range(n) if lab[j] == lab[i] and j != i]
        a = sum(own) / len(own) if own else 0
        b = min(sum(D[i][j] for j in range(n) if lab[j] == c) / max(1, sum(1 for j in range(n) if lab[j] == c))
                for c in range(k) if c != lab[i] and any(l == c for l in lab))
        s.append(0 if not own else (b - a) / max(a, b))
    return s


def ch_db(X, lab, C, k):
    n = len(X)
    g = (sum(p[0] for p in X) / n, sum(p[1] for p in X) / n)
    W = sum(d2(x, C[l]) for x, l in zip(X, lab))
    B = sum(sum(1 for l in lab if l == j) * d2(C[j], g) for j in range(k))
    ch = (B / (k - 1)) / (W / (n - k))
    sc = [sum(math.sqrt(d2(x, C[j])) for x, l in zip(X, lab) if l == j) / max(1, lab.count(j)) for j in range(k)]
    db = sum(max((sc[i] + sc[j]) / math.sqrt(d2(C[i], C[j])) for j in range(k) if j != i) for i in range(k)) / k
    return ch, db


def knee(xs, ys):
    """Index of max distance to the chord between first and last point (normalised axes)."""
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    best, bi = -1, 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        u, v = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0 or 1)
        dist = abs(u - v) / math.sqrt(2)
        if dist > best: best, bi = dist, i
    return bi


def data():
    r = seeded(11)
    centers = [(2, 2), (7, 2.5), (3, 7.5), (8, 8)]
    X = [(cx + r.gauss(0, 0.8), cy + r.gauss(0, 0.8)) for cx, cy in centers for _ in range(20)]
    return X


def build(pg: Page, cat):
    X = data()
    rng = seeded(5)
    ks = list(range(1, 9))
    res = {k: kmeans(X, k, rng) for k in ks}
    inertia = [res[k][0] for k in ks]

    y0 = pg.max_y() + 60
    pg.text("Finding the optimal number of clusters  (same data set: 80 points, 4 true blobs, k = 1 … 8)",
            400, y0 - 40, 1600, 26, 16, bold=True)

    # ------------------------------------------------------------ ⑥ elbow
    x0 = 400
    pg.panel(x0, y0, 640, 660, 6, "Elbow Method", "Inertia (WCSS) always falls as k grows — look for the bend")
    pg.math("WCSS(k) = \\sum_{j=1}^{k}\\sum_{x_i \\in C_j}\\lVert x_i-\\mu_j\\rVert^2", x0 + 40, y0 + 60, 560, 46, 17)
    top = inertia[0] * 1.08
    p = pg.axes(x0 + 90, y0 + 140, 460, 280, (1, 8), (0, top), "k", "WCSS", label_size=13, origin=(1, 0),
                xticks=ks, yticks=(round(top / 2, -1), round(top * 0.9, -1)))
    p.poly(list(zip(ks, inertia)), color=BLUE, width=2)
    p.poly([(1, inertia[0]), (8, inertia[-1])], color=GREY, dashed=True)
    ki = knee(ks, inertia)
    kk = ks[ki]
    # perpendicular from knee to chord (in screen space)
    ax, ay = p.P(1, inertia[0]); bx, by = p.P(8, inertia[-1]); px, py = p.P(kk, inertia[ki])
    t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax) ** 2 + (by - ay) ** 2)
    pg.line([(px, py), (ax + t * (bx - ax), ay + t * (by - ay))], color=RED, dashed=True, width=1.5)
    p.scatter(list(zip(ks, inertia)), r=4)
    p.scatter([(kk, inertia[ki])], color=RED, r=6)
    pg.note(f"elbow (knee) at k = {kk}: max distance to the chord", x0 + 330, y0 + 170, px, py, w=230, color=RED)
    pg.note("chord: first → last point", x0 + 400, y0 + 330, *p.P(6.5, inertia[0] + (inertia[-1] - inertia[0]) * 5.5 / 7),
            w=150)
    pg.ask("Where do extra clusters stop paying off?",
           f"at k = {kk}: WCSS falls {inertia[kk - 2] - inertia[kk - 1]:.0f} to reach k={kk}, "
           f"then only {inertia[kk - 1] - inertia[kk]:.0f} more",
           x0 + 380, y0 + 194, px + 4, py - 4, w=220)
    rows = [["k"] + [str(k) for k in ks], ["WCSS"] + [f"{v:.0f}" for v in inertia],
            ["drop"] + ["–"] + [f"{inertia[i - 1] - inertia[i]:.0f}" for i in range(1, len(ks))]]
    pg.table(x0 + 30, y0 + 455, 580, rows, size=10, header=True, h=70)
    pg.bullets(x0 + 20, y0 + 535, 600, [
        f"Big drops until k = {kk}, then small gains → adding clusters only splits real groups",
        "Knee rule (Kneedle): normalise both axes, pick the point farthest from the chord",
        "Weakness: the bend is often vague → confirm with Silhouette / Gap ⑦ ⑧"], head="Read it:")

    # ------------------------------------------------------------ ⑦ silhouette
    x0 = 1080
    pg.panel(x0, y0, 1320, 660, 7, "Silhouette Method", "How well each point sits in its own cluster vs the nearest other cluster")
    pg.math("s(i) = \\frac{b(i) - a(i)}{\\max\\{a(i),\\ b(i)\\}}", x0 + 30, y0 + 60, 300, 50, 18)
    pg.bullets(x0 + 340, y0 + 58, 560, [
        "\\(a(i)\\) = mean distance to points in its <b>own</b> cluster (cohesion)",
        "\\(b(i)\\) = mean distance to points of the <b>nearest other</b> cluster (separation)"], size=11, h=44)
    pg.html('<b>Range:</b> <font color="#009900">+1 well clustered</font> · 0 on the border · '
            '<font color="#FF0000">−1 probably in the wrong cluster</font>', x0 + 900, y0 + 58, 400, 44, 11)
    avg = {}
    for k in range(2, 9):
        avg[k] = sum(silhouette(X, res[k][1], k)) / len(X)
    for ci, k in enumerate((3, 4, 5)):
        lab = res[k][1]
        s = silhouette(X, lab, k)
        gx, gy, gw, gh = x0 + 60 + ci * 300, y0 + 150, 230, 330
        q = Plot(pg, gx, gy, gw, gh, (-0.2, 1.0), (0, len(X) + 4 * k))
        pg.line([q.P(0, 0), q.P(0, len(X) + 4 * k)], width=1)
        pg.line([q.P(-0.2, 0), q.P(1.0, 0)], arrow="classic")
        for t in (0, 0.5, 1.0):
            tx, ty = q.P(t, 0)
            pg.text(f"{t:g}", tx - 15, ty + 2, 30, 14, 9, align="center", color="#555555")
        yy = 2
        for j in range(k):
            vals = sorted((v for v, l in zip(s, lab) if l == j), reverse=True)
            start = yy
            for v in vals:
                pg.line([q.P(0, yy), q.P(v, yy)], color=COLS[j], width=2.2)
                yy += 1
            q.text(str(j + 1), -0.1, (start + yy) / 2, w=20, size=10, bold=True, color=COLS[j])
            yy += 4
        q.vline(avg[k], 0, len(X) + 4 * k, color=RED, width=1.5)
        pg.text(f"k = {k}   avg s = {avg[k]:.3f}", gx - 10, gy - 26, gw + 20, 20, 12, bold=True, align="center",
                bg=HL if k == 4 else None)
    pg.text("knife plot: one bar per point, sorted inside each cluster; red dashed = average", x0 + 40, y0 + 500, 900,
            18, 10, italic=True, color="#555555")
    best_k = max(avg, key=avg.get)
    c = pg.axes(x0 + 1000, y0 + 170, 270, 220, (2, 8), (0, 0.8), "k", "\\bar{s}", label_size=12, origin=(2, 0),
                xticks=range(2, 9), yticks=(0.25, 0.5, 0.75))
    c.poly([(k, avg[k]) for k in range(2, 9)], color=BLUE, width=2)
    c.scatter([(k, avg[k]) for k in range(2, 9)], r=4)
    c.scatter([(best_k, avg[best_k])], color=RED, r=6)
    pg.note(f"max at k = {best_k} → choose k = {best_k}", x0 + 1030, y0 + 130, *c.P(best_k, avg[best_k]), w=200,
            color=RED)
    pg.text("Average silhouette vs k", x0 + 1000, y0 + 420, 280, 20, 12, bold=True, align="center")
    pg.ask("Which k separates the groups best?",
           f"k = {best_k}: highest mean s = {avg[best_k]:.2f} (> 0.7 = strong structure)",
           x0 + 1005, y0 + 442, *c.P(best_k, avg[best_k] - 0.03), w=290)
    pg.bullets(x0 + 20, y0 + 530, 1280, [
        "Pick the k with the <b>highest average silhouette</b>, and check the knife plot: all clusters should pass the average and have similar thickness",
        "Thin clusters or many negative bars (k = 5) → a real group was split; one very thick cluster (k = 3) → two groups were merged",
        "Rule of thumb: s̄ > 0.7 strong, 0.5–0.7 reasonable, 0.25–0.5 weak, < 0.25 no real structure  ·  cost O(n²) → sample for big data"],
        head="Read it:")

    # ------------------------------------------------------------ ⑧ gap statistic, CH, DB
    y1 = y0 + 700
    x0 = 400
    pg.panel(x0, y1, 1000, 760, 8, "Gap Statistic · Calinski-Harabasz · Davies-Bouldin",
             "Three more scores on the same k-means runs")
    xs = [p_[0] for p_ in X]
    ys = [p_[1] for p_ in X]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    rr = seeded(21)
    B = 8
    gap, sk = {}, {}
    for k in ks:
        refs = []
        for _ in range(B):
            R = [(rr.uniform(lo_x, hi_x), rr.uniform(lo_y, hi_y)) for _ in X]
            refs.append(math.log(kmeans(R, k, rr, restarts=2, iters=30)[0]))
        m = sum(refs) / B
        sd = math.sqrt(sum((v - m) ** 2 for v in refs) / B)
        gap[k] = m - math.log(res[k][0])
        sk[k] = sd * math.sqrt(1 + 1 / B)
    kg = next((k for k in ks[:-1] if gap[k] >= gap[k + 1] - sk[k + 1]), ks[-1])
    chs, dbs = {}, {}
    for k in range(2, 9):
        chs[k], dbs[k] = ch_db(X, res[k][1], res[k][2], k)
    kch, kdb = max(chs, key=chs.get), min(dbs, key=dbs.get)

    cols = [("Gap statistic (1-SE rule)", "Gap(k)=\\frac{1}{B}\\sum_b \\log W^*_{kb} - \\log W_k",
             [(k, gap[k]) for k in ks], kg, (1, 8), "Gap"),
            ("Calinski-Harabasz (higher = better)", "CH=\\frac{B_k/(k-1)}{W_k/(n-k)}",
             [(k, chs[k]) for k in range(2, 9)], kch, (2, 8), "CH"),
            ("Davies-Bouldin (lower = better)",
             "DB=\\frac{1}{k}\\sum_i \\max_{j\\ne i}\\frac{s_i+s_j}{d(\\mu_i,\\mu_j)}",
             [(k, dbs[k]) for k in range(2, 9)], kdb, (2, 8), "DB")]
    for ci, (ttl, tex, pts, kbest, xr, yl) in enumerate(cols):
        gx = x0 + 40 + ci * 320
        pg.text(ttl, gx - 10, y1 + 60, 300, 20, 12, bold=True, align="center")
        pg.math(tex, gx - 20, y1 + 85, 320, 50, 13)
        vmax = max(v for _, v in pts) * 1.15
        vmin = min(0, min(v for _, v in pts))
        q = pg.axes(gx + 30, y1 + 170, 240, 200, xr, (vmin, vmax), "k", yl, label_size=12, origin=(xr[0], vmin),
                    xticks=range(xr[0], 9))
        if ci == 0:
            for k in ks:
                q.poly([(k, gap[k] - sk[k]), (k, gap[k] + sk[k])], color=GREY, width=1)
        q.poly(pts, color=BLUE, width=2)
        q.scatter(pts, r=3.5)
        vb = dict(pts)[kbest]
        q.scatter([(kbest, vb)], color=RED, r=6)
        q.text(f"k = {kbest}", kbest, vb, dy=-16, size=11, bold=True, color=RED, w=60)
    pg.text("grey bars = ±s_k (simulation error)", x0 + 60, y1 + 395, 280, 16, 9, italic=True, color="#555555")
    pg.table(x0 + 40, y1 + 425, 920, [
        ["k"] + [str(k) for k in ks],
        ["Gap"] + [f"{gap[k]:.2f}" for k in ks],
        ["CH"] + ["–"] + [f"{chs[k]:.0f}" for k in range(2, 9)],
        ["DB"] + ["–"] + [f"{dbs[k]:.2f}" for k in range(2, 9)]], size=10, h=90)
    agree = f"Elbow {kk} · Silhouette {best_k} · Gap {kg} · CH {kch} · DB {kdb}"
    pg.bullets(x0 + 20, y1 + 530, 960, [
        "<b>Gap</b>: compares the clustering with <i>uniform random</i> data of the same range (B reference sets); "
        "choose the smallest k with \\(Gap(k) \\ge Gap(k+1) - s_{k+1}\\); it can also answer k = 1 (no clusters)",
        "<b>Calinski-Harabasz</b> (variance ratio): between-cluster dispersion \\(B_k\\) / within-cluster dispersion \\(W_k\\); fast, favours convex blobs",
        "<b>Davies-Bouldin</b>: average similarity of each cluster to its most similar one; 0 is best, no upper limit",
        f'Result: <span style="background:{HL}"><b>{agree}</b></span>; when they disagree prefer silhouette + domain knowledge'],
        head="Read it:", h=150)
    same = len({kk, best_k, kg, kch, kdb}) == 1
    pg.ask("Do the different scores agree on the number of clusters?",
           (f"Yes: all five methods pick k = {kk} → high confidence in {kk} clusters" if same else
            "Not fully → trust silhouette + domain knowledge, check stability across seeds"),
           x0 + 380, y1 + 614, x0 + 90, y1 + 597, w=600)

    # ------------------------------------------------------------ ⑨ other models
    x0 = 1440
    pg.panel(x0, y1, 960, 760, 9, "Optimal hyper-parameters for other clustering models")
    sub = [X[i] for i in (0, 3, 7, 20, 24, 28, 40, 44, 47, 60, 63, 66)]
    clusters = {i: [i] for i in range(len(sub))}
    hgt = {i: 0.0 for i in range(len(sub))}
    merges, nid = [], len(sub)
    while len(clusters) > 1:
        keys = list(clusters)
        best = None
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                A, Bc = clusters[keys[a]], clusters[keys[b]]
                dd = sum(math.sqrt(d2(sub[i], sub[j])) for i in A for j in Bc) / (len(A) * len(Bc))
                if best is None or dd < best[0]:
                    best = (dd, keys[a], keys[b])
        dd, a, b = best
        clusters[nid] = clusters.pop(a) + clusters.pop(b)
        hgt[nid] = dd
        merges.append((a, b, nid, dd))
        nid += 1
    order = clusters[nid - 1]
    child = {m[2]: (m[0], m[1]) for m in merges}
    maxh = merges[-1][3]
    dp = Plot(pg, x0 + 70, y1 + 90, 260, 200, (-0.5, len(sub) - 0.5), (0, maxh * 1.1))
    pos = {leaf: order.index(leaf) for leaf in order}

    def xpos(n):
        if n in pos:
            return pos[n]
        a_, b_ = child[n]
        return (xpos(a_) + xpos(b_)) / 2
    for a, b, n, dd in merges:
        xa, xb = xpos(a), xpos(b)
        dp.poly([(xa, hgt[a]), (xa, dd), (xb, dd), (xb, hgt[b])], color=BLACK, width=1.2)
    for leaf in order:
        dp.scatter([(pos[leaf], 0)], color=COLS[leaf // 3], r=3.5)
    cut = (merges[-3][3] + merges[-4][3]) / 2
    dp.hline(cut, color=RED, width=1.5)
    pg.line([dp.P(-0.5, 0), dp.P(-0.5, maxh * 1.1)], arrow="classic")
    pg.text("distance", x0 + 20, y1 + 70, 80, 16, 10, color="#555555")
    pg.text("Hierarchical: cut the dendrogram", x0 + 40, y1 + 52, 320, 18, 12, bold=True, align="center")
    pg.note("cut in the largest vertical gap → 4 clusters", x0 + 190, y1 + 298, *dp.P(8, cut), w=200, color=RED)

    def gmm_ll(k):
        _, lab, C = res[k]
        mu = [list(c) for c in C]
        var = [[1.0, 1.0] for _ in range(k)]
        w = [max(lab.count(j) / len(X), 1e-3) for j in range(k)]
        ll = 0
        for _ in range(40):
            Rr, ll = [], 0
            for x in X:
                ps = [w[j] * math.exp(-0.5 * sum((x[d] - mu[j][d]) ** 2 / var[j][d] for d in range(2)))
                      / (2 * math.pi * math.sqrt(var[j][0] * var[j][1])) for j in range(k)]
                s = sum(ps) or 1e-300
                ll += math.log(s)
                Rr.append([v / s for v in ps])
            for j in range(k):
                nj = sum(r_[j] for r_ in Rr) or 1e-9
                w[j] = nj / len(X)
                mu[j] = [sum(r_[j] * x[d] for r_, x in zip(Rr, X)) / nj for d in range(2)]
                var[j] = [max(0.05, sum(r_[j] * (x[d] - mu[j][d]) ** 2 for r_, x in zip(Rr, X)) / nj)
                          for d in range(2)]
        return ll
    bic, aic = {}, {}
    for k in ks:
        ll = gmm_ll(k)
        npar = 5 * k - 1
        bic[k] = -2 * ll + npar * math.log(len(X))
        aic[k] = -2 * ll + 2 * npar
    kb = min(bic, key=bic.get)
    vmin = min(min(aic.values()), min(bic.values())) * 0.97
    vmax = max(max(aic.values()), max(bic.values())) * 1.02
    g = pg.axes(x0 + 470, y1 + 90, 240, 200, (1, 8), (vmin, vmax), "k", "", label_size=12, origin=(1, vmin),
                xticks=ks)
    g.poly([(k, bic[k]) for k in ks], color=BLUE, width=2)
    g.poly([(k, aic[k]) for k in ks], color=ORANGE, width=1.5, dashed=True)
    g.scatter([(kb, bic[kb])], color=RED, r=6)
    g.text(f"k = {kb}", kb, bic[kb], dy=-16, size=11, bold=True, color=RED, w=60)
    pg.text("GMM: lowest BIC / AIC", x0 + 460, y1 + 52, 280, 18, 12, bold=True, align="center")
    pg.html('<font color="#3333FF"><b>— BIC</b></font><br><font color="#FF8000"><b>- - AIC</b></font>', x0 + 740,
            y1 + 90, 70, 34, 10)
    pg.math("BIC=-2\\ln\\hat{L}+p\\ln n", x0 + 730, y1 + 140, 210, 30, 12, align="left")
    pg.math("AIC=-2\\ln\\hat{L}+2p", x0 + 730, y1 + 175, 210, 30, 12, align="left")
    pg.ask("How many Gaussian components fit best?",
           f"BIC is lowest at k = {kb}: best fit after the penalty for extra parameters",
           x0 + 725, y1 + 212, *g.P(kb, bic[kb]), w=225)

    mp = 4
    kd = sorted(sorted(math.sqrt(d2(a, b)) for b in X if b is not a)[mp - 1] for a in X)
    idx = list(range(len(kd)))
    ki2 = knee(idx, kd)
    e = pg.axes(x0 + 70, y1 + 390, 300, 170, (0, len(kd)), (0, max(kd) * 1.1), "", "d_4",
                label_size=11, origin=(0, 0))
    e.poly(list(zip(idx, kd)), color=BLUE, width=2)
    e.scatter([(ki2, kd[ki2])], color=RED, r=5)
    e.hline(kd[ki2], color=RED)
    pg.text("points sorted by distance to 4th neighbour", x0 + 70, y1 + 565, 300, 16, 9, align="center",
            color="#555555")
    pg.text(f"DBSCAN: k-distance plot (minPts = {mp} = 2·dims)", x0 + 20, y1 + 345, 400, 18, 12, bold=True,
            align="center")
    pg.note(f"knee → eps ≈ {kd[ki2]:.2f}", x0 + 110, y1 + 392, *e.P(ki2, kd[ki2]), w=130, color=RED)
    pg.table(x0 + 440, y1 + 345, 500, [
        ["Model", "Hyper-parameter", "Technique", "Rule"],
        ["K-Means / K-Medoids", "k", "Elbow, Silhouette, Gap, CH, DB", "knee / max / 1-SE / max / min"],
        ["Hierarchical", "cut height / k", "Dendrogram", "largest vertical gap"],
        ["GMM", "n_components", "BIC / AIC", "minimum (BIC stricter)"],
        ["DBSCAN", "eps, minPts", "k-distance plot", "knee; minPts ≥ dims + 1 (≈ 2·dims)"],
        ["HDBSCAN", "min_cluster_size", "stability / DBCV", "smallest meaningful group"],
        ["Spectral", "k", "eigengap of Laplacian", "largest gap λₖ₊₁ − λₖ"]], size=10, h=200)
    pg.bullets(x0 + 20, y1 + 600, 920, [
        "Always scale features first: every method above depends on distances",
        "Internal scores only measure geometry; if true labels exist also check external scores (ARI, NMI)",
        "Final choice = scores + stability across seeds / subsamples + business meaning of the clusters"],
        head="Notes:")

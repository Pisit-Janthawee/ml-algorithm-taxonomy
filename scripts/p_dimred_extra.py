"""Extra PCA-page panels: curse of dimensionality, selection vs extraction, method comparison, choosing k."""
import math
from dio import *


def ball_cube_ratio(d):
    return math.pi ** (d / 2) / (math.gamma(d / 2 + 1) * 2 ** d)


def jacobi_eig(A, sweeps=60):
    n = len(A)
    A = [row[:] for row in A]
    for _ in range(sweeps):
        off = sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-12: break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-12: continue
                th = 0.5 * math.atan2(2 * A[p][q], A[q][q] - A[p][p])
                c, s = math.cos(th), math.sin(th)
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
    return sorted((A[i][i] for i in range(n)), reverse=True)


def build(pg: Page, cat):
    y0 = pg.max_y() + 60
    pg.text("Dimensionality: why reduce, and which method", 400, y0 - 40, 1200, 26, 16, bold=True)

    # ------------------------------------------------------------ ⑦ curse of dimensionality
    x0 = 400
    pg.panel(x0, y0, 1000, 660, 7, "Curse of Dimensionality", "More features → space becomes empty and distances lose meaning")
    # (a) hypersphere / hypercube
    pg.text("(a) Ball inside a cube: share of volume", x0 + 20, y0 + 60, 300, 18, 12, bold=True)
    p = pg.axes(x0 + 60, y0 + 100, 240, 180, (1, 12), (0, 1.05), "d", "", label_size=11,
                origin=(1, 0), xticks=(1, 2, 3, 5, 8, 12), yticks=(0.5, 1))
    pg.math("V_{ball}/V_{cube}", x0 + 180, y0 + 95, 120, 24, 12)
    ds = list(range(1, 13))
    p.poly([(d, ball_cube_ratio(d)) for d in ds], color=BLUE, width=2)
    p.scatter([(d, ball_cube_ratio(d)) for d in ds], r=3)
    for d in (2, 3, 10):
        p.text(f"{ball_cube_ratio(d) * 100:.1f} %", d, ball_cube_ratio(d), dx=26, dy=-8, size=9, color=RED, w=60)
    pg.text("almost all volume sits in the corners → data points are far from the centre and from each other",
            x0 + 20, y0 + 300, 310, 44, 10, italic=True, color="#555555")
    # (b) distance concentration (computed)
    r = seeded(9)
    dims = [1, 2, 5, 10, 20, 50, 100, 500]
    conc = []
    for d in dims:
        pts = [[r.random() for _ in range(d)] for _ in range(120)]
        q = [r.random() for _ in range(d)]
        dist = [math.sqrt(sum((a - b) ** 2 for a, b in zip(pt, q))) for pt in pts]
        conc.append((max(dist) - min(dist)) / min(dist))
    lc = [math.log10(c) for c in conc]
    pg.text("(b) Distances concentrate (120 random points)", x0 + 360, y0 + 60, 320, 18, 12, bold=True)
    q = pg.axes(x0 + 400, y0 + 100, 250, 180, (0, math.log10(500)), (min(lc) - 0.3, max(lc) + 0.3), "d", "",
                label_size=11, origin=(0, min(lc) - 0.3))
    q.poly([(math.log10(d), v) for d, v in zip(dims, lc)], color=BLUE, width=2)
    q.scatter([(math.log10(d), v) for d, v in zip(dims, lc)], r=3.5)
    for d, c in zip(dims, conc):
        if d in (1, 10, 100, 500):
            q.text(f"d={d}: {c:.2f}" if c < 10 else f"d={d}: {c:.0f}", math.log10(d), math.log10(c), dy=-13,
                   dx=50 if d == 1 else 0, size=9, w=90, color=RED)
    pg.math("\\frac{d_{max}-d_{min}}{d_{min}} \\to 0", x0 + 670, y0 + 120, 160, 40, 14)
    pg.text("(log scale)", x0 + 670, y0 + 165, 120, 16, 9, color="#555555")
    pg.ask("Why do kNN and K-Means break with many features?",
           f"at d = 500 the farthest point is only {conc[-1] * 100:.0f}% farther than the nearest: 'near' loses meaning",
           x0 + 700, y0 + 200, *q.P(math.log10(500), lc[-1]), w=270)
    pg.text("nearest and farthest neighbour become almost equally far → kNN, K-Means, DBSCAN degrade",
            x0 + 360, y0 + 300, 330, 44, 10, italic=True, color="#555555")
    # (c) data needed: grid cells
    pg.text("(c) Cells to cover with 10 bins per feature", x0 + 20, y0 + 360, 400, 18, 12, bold=True)
    bx, by = x0 + 40, y0 + 420
    for i in range(11):
        pg.line([(bx + i * 16, by + 60), (bx + i * 16, by + 70)], width=1)
    pg.line([(bx, by + 65), (bx + 160, by + 65)], width=1)
    pg.text("1-D: 10 cells", bx, by + 85, 160, 18, 11, bold=True, align="center")
    gx = x0 + 280
    for i in range(11):
        pg.line([(gx + i * 14, by), (gx + i * 14, by + 140)], width=0.5, color=GREY)
        pg.line([(gx, by + i * 14), (gx + 140, by + i * 14)], width=0.5, color=GREY)
    for _ in range(25):
        pg.dot(gx + r.uniform(2, 138), by + r.uniform(2, 138), r=2.5)
    pg.text("2-D: 100 cells", gx, by + 150, 140, 18, 11, bold=True, align="center")
    cx, cy, s, dz = x0 + 520, by + 30, 110, 35
    for i in range(11):
        pg.line([(cx + i * s / 10, cy), (cx + i * s / 10, cy + s)], width=0.5, color=GREY)
        pg.line([(cx, cy + i * s / 10), (cx + s, cy + i * s / 10)], width=0.5, color=GREY)
        pg.line([(cx + i * s / 10, cy), (cx + i * s / 10 + dz, cy - dz)], width=0.5, color=GREY)
        pg.line([(cx + s, cy + i * s / 10), (cx + s + dz, cy + i * s / 10 - dz)], width=0.5, color=GREY)
    pg.line([(cx + dz, cy - dz), (cx + s + dz, cy - dz), (cx + s + dz, cy + s - dz)], width=1)
    pg.rect(cx, cy, s, s, sw=1)
    for _ in range(25):
        pg.dot(cx + r.uniform(2, s - 2), cy + r.uniform(2, s - 2), r=2.5)
    pg.text("3-D: 1 000 cells", cx, by + 150, 150, 18, 11, bold=True, align="center")
    pg.html(f'<b>d-D: 10<sup>d</sup> cells</b><br>the same 25 points cover 25 % → 2.5 % of cells<br>'
            f'<span style="background:{HL}">data needed grows exponentially</span>', x0 + 700, by + 20, 280, 70, 11)
    pg.bullets(x0 + 20, y0 + 600, 960, [
        "Hughes phenomenon: with fixed n, accuracy first rises then falls as features are added → reduce dimensions (⑧ ⑨) or get more data"],
        size=11)

    # ------------------------------------------------------------ ⑧ selection vs extraction
    x0 = 1440
    pg.panel(x0, y0, 960, 660, 8, "Feature Selection vs Feature Extraction")
    feats = ["x₁", "x₂", "x₃", "x₄", "x₅", "x₆"]
    keep = {1, 4}
    pg.text("Selection: keep a subset of the original columns", x0 + 20, y0 + 55, 440, 18, 12, bold=True)
    for i, f in enumerate(feats):
        k = i in keep
        pg.rect(x0 + 30 + i * 52, y0 + 85, 46, 120, f, fill=HL if k else "#F5F5F5", stroke=BLACK if k else GREY,
                size=13, bold=k, color=BLACK if k else GREY, valign="top")
        if not k:
            pg.line([(x0 + 33 + i * 52, y0 + 200), (x0 + 73 + i * 52, y0 + 90)], color=RED, width=1)
    pg.line([(x0 + 350, y0 + 145), (x0 + 380, y0 + 145)], arrow="classic")
    for i, f in enumerate(["x₂", "x₅"]):
        pg.rect(x0 + 390 + i * 52, y0 + 85, 46, 120, f, fill=HL, size=13, bold=True, valign="top")
    pg.text("same meaning, still interpretable", x0 + 360, y0 + 210, 160, 30, 10, italic=True, align="center")
    pg.ask("Which original columns actually matter?",
           "selection keeps x₂ and x₅ unchanged → you can still say which indicator drives the price",
           x0 + 15, y0 + 214, x0 + 105, y0 + 172, w=330)
    pg.text("Extraction: build new axes from ALL columns", x0 + 520, y0 + 55, 430, 18, 12, bold=True)
    ins = []
    for i, f in enumerate(feats):
        ins.append(pg.ellipse(x0 + 540, y0 + 82 + i * 26, 40, 22, f, size=11))
    outs = [pg.ellipse(x0 + 760, y0 + 105 + j * 70, 56, 34, f"z{'₁₂'[j]}", fill=HL, bold=True, size=13)
            for j in range(2)]
    for a in ins:
        for b in outs:
            pg.connect(a, b, orth=False, color=GREY, arrow=None)
    pg.math("z_1 = 0.6x_1 + 0.5x_2 + \\dots + 0.1x_6", x0 + 610, y0 + 240, 300, 26, 12)
    pg.text("new meaning: combinations (PCA, LDA, autoencoder)", x0 + 600, y0 + 268, 320, 18, 10, italic=True,
            align="center")
    pg.table(x0 + 20, y0 + 305, 920, [
        ["Approach", "How", "Examples", "Pros", "Cons"],
        ["Filter (selection)", "score each feature alone", "correlation, χ², ANOVA F, mutual information, variance threshold",
         "fast, model-free", "ignores interactions"],
        ["Wrapper (selection)", "search subsets with a model", "RFE, forward / backward selection",
         "finds interactions", "slow, overfit risk"],
        ["Embedded (selection)", "selection inside training", "Lasso (L1), tree / RF importance",
         "efficient, model-aware", "model-specific"],
        ["Extraction", "new combined features", "PCA, LDA, ICA, t-SNE, UMAP, autoencoder",
         "compact, removes redundancy", "harder to interpret"]], size=10, h=170)
    pg.bullets(x0 + 20, y0 + 495, 920, [
        "Need to explain which <b>original</b> variables matter (e.g. which indicator drives the price) → <b>selection</b>",
        "Many correlated features, only prediction matters (e.g. 50 technical indicators) → <b>extraction</b>",
        "Fit any selection / PCA on the <b>training fold only</b>, then apply to validation / test (avoid leakage)"],
        head="Which one?")

    # ------------------------------------------------------------ ⑨ method comparison sketches
    y1 = y0 + 700
    x0 = 400
    pg.panel(x0, y1, 1320, 720, 9, "Method Comparison on Toy Data", "Linear vs supervised vs non-linear vs neural")
    # (i) PCA vs LDA
    th = math.radians(25)
    A, Bc = [], []
    for k in range(40):
        for cls, off in ((A, 0.9), (Bc, -0.9)):
            u, v = r.gauss(0, 1.8), r.gauss(off, 0.35)
            cls.append((u * math.cos(th) - v * math.sin(th), u * math.sin(th) + v * math.cos(th)))
    allp = A + Bc
    mx_ = sum(p_[0] for p_ in allp) / len(allp); my_ = sum(p_[1] for p_ in allp) / len(allp)
    sxx = sum((p_[0] - mx_) ** 2 for p_ in allp); syy = sum((p_[1] - my_) ** 2 for p_ in allp)
    sxy = sum((p_[0] - mx_) * (p_[1] - my_) for p_ in allp)
    ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
    pca_dir = (math.cos(ang), math.sin(ang))

    def mean(L): return (sum(p_[0] for p_ in L) / len(L), sum(p_[1] for p_ in L) / len(L))
    mA, mB = mean(A), mean(Bc)
    sw = [[0, 0], [0, 0]]
    for L, m in ((A, mA), (Bc, mB)):
        for p_ in L:
            dx_, dy_ = p_[0] - m[0], p_[1] - m[1]
            sw[0][0] += dx_ * dx_; sw[0][1] += dx_ * dy_; sw[1][0] += dx_ * dy_; sw[1][1] += dy_ * dy_
    det = sw[0][0] * sw[1][1] - sw[0][1] ** 2
    inv = [[sw[1][1] / det, -sw[0][1] / det], [-sw[1][0] / det, sw[0][0] / det]]
    dm = (mA[0] - mB[0], mA[1] - mB[1])
    w = (inv[0][0] * dm[0] + inv[0][1] * dm[1], inv[1][0] * dm[0] + inv[1][1] * dm[1])
    nw = math.hypot(*w); lda_dir = (w[0] / nw, w[1] / nw)
    pg.text("(i) PCA vs LDA on two classes", x0 + 20, y1 + 60, 380, 18, 12, bold=True)
    pp = Plot(pg, x0 + 40, y1 + 90, 340, 260, (-5, 5), (-3.8, 3.8))
    pg.rect(x0 + 40, y1 + 90, 340, 260, stroke="#DDDDDD", sw=0.5)
    pp.scatter(A, color=BLUE, r=3); pp.scatter(Bc, color=RED, r=3)
    for d_, col, lab in ((pca_dir, GREEN, "PCA: max variance"), (lda_dir, ORANGE, "LDA: max separation")):
        pg.line([pp.P(mx_ - 4.6 * d_[0], my_ - 4.6 * d_[1]), pp.P(mx_ + 4.6 * d_[0], my_ + 4.6 * d_[1])],
                color=col, width=2.5, arrow="classic")
    pp.text("PCA direction", mx_ + 3.6 * pca_dir[0], my_ + 3.6 * pca_dir[1] + 0.6, size=10, bold=True,
            color=GREEN, w=100)
    pp.text("LDA direction", mx_ + 3.2 * lda_dir[0] + 1.3, my_ + 3.2 * lda_dir[1], size=10, bold=True,
            color=ORANGE, w=100)
    for k, (d_, col, lab) in enumerate(((pca_dir, GREEN, "projected on PCA → classes overlap"),
                                        (lda_dir, ORANGE, "projected on LDA → classes separate"))):
        sy = y1 + 385 + k * 60
        projA = [(p_[0] - mx_) * d_[0] + (p_[1] - my_) * d_[1] for p_ in A]
        projB = [(p_[0] - mx_) * d_[0] + (p_[1] - my_) * d_[1] for p_ in Bc]
        lim = max(abs(v) for v in projA + projB) * 1.05
        sp = Plot(pg, x0 + 40, sy, 340, 20, (-lim, lim), (0, 1))
        pg.line([sp.P(-lim, 0.5), sp.P(lim, 0.5)], color=col, width=1.5)
        sp.scatter([(v, 0.75) for v in projA], color=BLUE, r=2.5)
        sp.scatter([(v, 0.25) for v in projB], color=RED, r=2.5)
        pg.text(lab, x0 + 40, sy + 24, 340, 16, 10, bold=True, align="center", color=col)
    pg.text("PCA ignores labels: the long axis mixes the classes; LDA uses labels", x0 + 30, y1 + 510, 360, 30, 10,
            italic=True, color="#555555", align="center")

    # (ii) manifold unrolling
    cols5 = ["#3333FF", "#00AACC", "#00AA00", "#FF9900", "#FF0000"]

    def tcol(f):
        return cols5[min(4, int(f * 5))]
    sp_pts = []
    for i in range(70):
        f = i / 69
        t = 1.5 * math.pi + f * 3 * math.pi
        sp_pts.append((t * math.cos(t) + r.gauss(0, 0.25), t * math.sin(t) + r.gauss(0, 0.25), f))
    pg.text("(ii) Non-linear manifold: t-SNE / UMAP", x0 + 460, y1 + 60, 400, 18, 12, bold=True)
    mp = Plot(pg, x0 + 480, y1 + 90, 240, 240, (-15, 15), (-15, 15))
    pg.rect(x0 + 480, y1 + 90, 240, 240, stroke="#DDDDDD", sw=0.5)
    for xv, yv, f in sp_pts:
        mp.scatter([(xv, yv)], color=tcol(f), r=3)
    pg.text("spiral in 2-D (colour = position along the curve)", x0 + 470, y1 + 335, 260, 30, 10, italic=True,
            color="#555555", align="center")
    pg.line([(x0 + 735, y1 + 210), (x0 + 775, y1 + 210)], arrow="classic", width=1.5)
    for k, (lab, key) in enumerate((("t-SNE / UMAP: neighbours kept → unrolled, colours in order", "f"),
                                    ("PCA (1-D): colours mixed → structure lost", "x"))):
        sy = y1 + 150 + k * 110
        vals = [(f if key == "f" else xv, f) for xv, yv, f in sp_pts]
        lo_, hi_ = min(v for v, _ in vals), max(v for v, _ in vals)
        sp = Plot(pg, x0 + 790, sy, 200, 20, (lo_, hi_), (0, 1))
        pg.line([sp.P(lo_, 0.5), sp.P(hi_, 0.5)], color=GREY)
        for v, f in vals:
            sp.scatter([(v, 0.5)], color=tcol(f), r=3)
        pg.text(lab, x0 + 780, sy + 24, 230, 30, 10, bold=True, align="center",
                color=GREEN if key == "f" else RED)
    pg.text("t-SNE: preserves local neighbourhoods (visualisation only, no transform for new data)  ·  "
            "UMAP: faster, keeps more global structure, can transform new data", x0 + 460, y1 + 400, 560, 44, 10,
            italic=True, color="#555555")
    pg.ask("Which direction keeps the two classes apart?",
           "LDA (orange): projected classes separate; PCA (green) keeps variance but mixes them",
           x0 + 470, y1 + 452, x0 + 335, y1 + 455, w=540)

    # (iii) autoencoder
    pg.text("(iii) Autoencoder (neural, non-linear)", x0 + 1050, y1 + 60, 260, 18, 12, bold=True)
    layers = [6, 4, 2, 4, 6]
    ids = []
    for li, n in enumerate(layers):
        col_ids = []
        for j in range(n):
            cy = y1 + 210 - (n - 1) * 17 + j * 34
            cx = x0 + 1060 + li * 55
            col_ids.append(pg.ellipse(cx, cy, 22, 22, fill=HL if li == 2 else "#FFFFFF", sw=1.2))
        ids.append(col_ids)
    for a_, b_ in zip(ids, ids[1:]):
        for s_ in a_:
            for t_ in b_:
                pg.connect(s_, t_, orth=False, color="#BBBBBB", arrow=None)
    pg.text("x (6)", x0 + 1045, y1 + 320, 50, 16, 10, align="center")
    pg.text("z (2)", x0 + 1155, y1 + 320, 50, 16, 10, align="center", bold=True)
    pg.text("x̂ (6)", x0 + 1265, y1 + 320, 50, 16, 10, align="center")
    pg.text("Encoder", x0 + 1055, y1 + 345, 110, 18, 11, bold=True, align="center", color=BLUE)
    pg.text("Decoder", x0 + 1170, y1 + 345, 110, 18, 11, bold=True, align="center", color=RED)
    pg.note("bottleneck = reduced dimensions", x0 + 1090, y1 + 92, x0 + 1181, y1 + 186, w=180)
    pg.math("L = \\lVert x - \\hat{x}\\rVert^2", x0 + 1060, y1 + 380, 240, 30, 15)
    pg.text("linear activations + MSE ≈ PCA; non-linear activations learn curved manifolds", x0 + 1050, y1 + 415,
            260, 44, 10, italic=True, color="#555555", align="center")
    pg.bullets(x0 + 20, y1 + 560, 1280, [
        "Linear structure → PCA (unsupervised) or LDA (supervised, at most C − 1 components)",
        "Curved structure → Kernel PCA, Isomap, UMAP (modelling) or t-SNE (plots only)",
        "Very large / complex data (images, many sensors) → autoencoders; all methods need scaled inputs"],
        head="Rule of thumb:")

    # ------------------------------------------------------------ ⑩ comparison table + choosing k
    x0 = 1760
    pg.panel(x0, y1, 640, 720, 10, "Compare Methods & Choose k")
    pg.table(x0 + 20, y1 + 50, 600, [
        ["Method", "Linear", "Supervised", "Preserves", "New data", "Cost"],
        ["PCA", "✔", "✘", "global variance", "✔", "low"],
        ["LDA", "✔", "✔", "class separation", "✔", "low"],
        ["Kernel PCA", "✘", "✘", "non-linear variance", "✔", "O(n²)"],
        ["t-SNE", "✘", "✘", "local neighbours", "✘", "high"],
        ["UMAP", "✘", "✘", "local + some global", "✔", "medium"],
        ["Autoencoder", "✘", "✘", "reconstruction", "✔", "high (GPU)"]], size=10, h=170)
    # 6 standardized features from 2 latent factors
    n = 200
    rows_ = []
    for _ in range(n):
        f1, f2 = r.gauss(0, 1), r.gauss(0, 1)
        load = [(0.9, 0.1), (0.8, 0.2), (0.85, 0.0), (0.1, 0.9), (0.2, 0.8), (0.3, 0.3)]
        rows_.append([a * f1 + b * f2 + r.gauss(0, 0.45) for a, b in load])
    m6 = [sum(rw[j] for rw in rows_) / n for j in range(6)]
    s6 = [math.sqrt(sum((rw[j] - m6[j]) ** 2 for rw in rows_) / (n - 1)) for j in range(6)]
    Z = [[(rw[j] - m6[j]) / s6[j] for j in range(6)] for rw in rows_]
    R = [[sum(z[i] * z[j] for z in Z) / (n - 1) for j in range(6)] for i in range(6)]
    lam = jacobi_eig(R)
    tot = sum(lam)
    err = [1 - sum(lam[:k]) / tot for k in range(0, 7)]
    pg.text("Reconstruction error vs k", x0 + 20, y1 + 245, 290, 18, 12, bold=True, align="center")
    e = pg.axes(x0 + 60, y1 + 280, 230, 170, (0, 6), (0, 1.05), "k", "", label_size=11, origin=(0, 0),
                xticks=range(0, 7), yticks=(0.5, 1))
    e.poly([(k, err[k]) for k in range(7)], color=BLUE, width=2)
    e.scatter([(k, err[k]) for k in range(7)], r=3.5)
    e.hline(0.05, color=RED)
    e.text("5 %", 6, 0.05, dx=14, dy=-8, size=9, color=RED, w=30)
    e.text(f"k=2: {err[2] * 100:.0f} %", 2, err[2], dx=34, dy=-10, size=10, bold=True, color=RED, w=80)
    pg.math("err(k)=1-\\frac{\\sum_{i\\le k}\\lambda_i}{\\sum_i\\lambda_i}", x0 + 30, y1 + 475, 270, 40, 12)
    pg.text("Kaiser rule (λ > 1)", x0 + 330, y1 + 245, 290, 18, 12, bold=True, align="center")
    kz = pg.axes(x0 + 370, y1 + 280, 230, 170, (0.5, 6.5), (0, max(lam) * 1.15), "PC", "\\lambda", label_size=11,
                 origin=(0.5, 0), xticks=range(1, 7), yticks=(1,))
    kz.bars([(i + 1, v) for i, v in enumerate(lam)], 0.6, fill=LIGHTBLUE, stroke=BLUE)
    kz.hline(1, color=RED, width=1.5)
    nk = sum(1 for v in lam if v > 1)
    kz.text(f"{nk} components with λ > 1", 4.5, max(lam) * 0.8, size=10, bold=True, color=RED, w=150)
    k5 = next(k for k in range(7) if err[k] <= 0.05)
    pg.ask("How many components do I need?",
           f"k = {nk} keeps {(1 - err[nk]) * 100:.0f}% (Kaiser); ≤ 5% error needs k = {k5}",
           x0 + 120, y1 + 285, *e.P(nk, err[nk]), w=170)
    pg.html(f'Data: 6 standardized features built from 2 hidden factors → eigenvalues '
            f'{", ".join(f"{v:.2f}" for v in lam)}<br>'
            f'<span style="background:{HL}"><b>Kaiser → k = {nk}</b> ({(1 - err[nk]) * 100:.0f} % kept)</span> · '
            f'error ≤ 5 % → k = {k5}', x0 + 330, y1 + 470, 290, 60, 10)
    pg.bullets(x0 + 20, y1 + 545, 600, [
        "Reconstruction error = share of variance thrown away; strict thresholds keep more PCs than Kaiser",
        "Kaiser: on standardized data keep PCs that explain more than one original feature (λ > 1)",
        "Or: cumulative variance ≥ 90–95 % (scree, ⑤) · or tune k by downstream CV score"], head="Choosing k:",
        size=10)

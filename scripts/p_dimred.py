"""PCA page: standardization, covariance & eigen-decomposition, principal components, projection, scree plot."""
import math
from dio import *


def _data():
    r = seeded(21)
    pts = []
    for _ in range(40):
        size = r.gauss(55, 15)                       # x1: house size (m²)
        rooms = 0.08 * size + r.gauss(0, 0.7)        # x2: rooms, correlated
        pts.append((size, rooms))
    return pts


def _stats(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))
    return m, s


def build(pg: Page, cat):
    raw = _data()
    m1, s1 = _stats([a for a, _ in raw])
    m2, s2 = _stats([b for _, b in raw])
    Z = [((a - m1) / s1, (b - m2) / s2) for a, b in raw]
    n = len(Z)
    a = sum(x * x for x, _ in Z) / (n - 1)
    b = sum(x * y for x, y in Z) / (n - 1)
    c = sum(y * y for _, y in Z) / (n - 1)
    l1 = (a + c) / 2 + math.sqrt(((a - c) / 2) ** 2 + b * b)
    l2 = (a + c) / 2 - math.sqrt(((a - c) / 2) ** 2 + b * b)
    v1 = (b, l1 - a); nv = math.hypot(*v1); v1 = (v1[0] / nv, v1[1] / nv)
    v2 = (-v1[1], v1[0])

    # ---------------------------------------------------------------- ① standardization
    x0, y0 = 400, 240
    pg.panel(x0, y0, 620, 620, 1, "Standardize first", "Background: mean 0, standard deviation 1")
    p = pg.axes(x0 + 60, y0 + 100, 220, 200, (0, 100), (0, 10), "x_1", "x_2", label_size=12,
                xticks=(0, 50, 100), yticks=(0, 5, 10))
    p.scatter([(min(99, max(1, u)), min(9.8, max(0.2, v))) for u, v in raw], r=3)
    pg.text("Raw: size (m²) vs rooms — different scales", x0 + 30, y0 + 330, 280, 20, 10, align="center")
    q = pg.axes(x0 + 360, y0 + 100, 220, 200, (-3, 3), (-3, 3), "z_1", "z_2", label_size=12, origin=(0, 0),
                xticks=(-2, 2), yticks=(-2, 2))
    q.scatter(Z, r=3)
    pg.text("Standardized: centred at (0, 0), same scale", x0 + 330, y0 + 330, 280, 20, 10, align="center")
    pg.line([(x0 + 295, y0 + 200), (x0 + 345, y0 + 200)], arrow="block", width=2)
    pg.math("z = \\frac{x - \\mu}{\\sigma}", x0 + 160, y0 + 370, 300, 56, 22)
    pg.note("mean of the feature", x0 + 40, y0 + 440, x0 + 330, y0 + 390, w=130)
    pg.note("standard deviation", x0 + 420, y0 + 440, x0 + 320, y0 + 410, w=130)
    pg.html(f"Example: μ₁ = {m1:.1f} m², σ₁ = {s1:.1f} m²  →  a 70 m² house has "
            f"<span style='background:{HL}'>z = (70 − {m1:.1f}) / {s1:.1f} = {(70 - m1) / s1:.2f}</span>",
            x0 + 20, y0 + 475, 580, 36, 11, align="center")
    pg.bullets(x0 + 20, y0 + 520, 580, [
        "Without scaling, the feature with the biggest numbers (m²) dominates the variance",
        "PCA finds directions of <b>maximum variance</b> → features must be comparable"], head="Why:")
    pg.ask("Is a 70 m² house big or small here?",
           f"z = {(70 - m1) / s1:.2f}: about {(70 - m1) / s1:.1f} standard deviations above the average house",
           x0 + 15, y0 + 352, x0 + 392, y0 + 488, w=210)

    # ---------------------------------------------------------------- ② covariance & eigen
    x0 = 1060
    pg.panel(x0, y0, 620, 620, 2, "Covariance Matrix & Eigen-decomposition",
             "Background: how features vary together")
    pg.math("\\Sigma = \\frac{1}{n-1} Z^{T} Z", x0 + 20, y0 + 70, 580, 46, 19)
    pg.math(f"\\Sigma = \\begin{{bmatrix}} Var(z_1) & Cov(z_1,z_2) \\\\ Cov(z_2,z_1) & Var(z_2) \\end{{bmatrix}}"
            f" = \\begin{{bmatrix}} {a:.2f} & {b:.2f} \\\\ {b:.2f} & {c:.2f} \\end{{bmatrix}}",
            x0 + 10, y0 + 125, 600, 60, 13)
    pg.note("diagonal = variance (= 1 after standardizing)", x0 + 20, y0 + 195, x0 + 410, y0 + 150, w=260)
    pg.note("off-diagonal = covariance (= correlation here)", x0 + 330, y0 + 200, x0 + 545, y0 + 150, w=260)
    pg.math("\\Sigma\\, v = \\lambda\\, v \\qquad \\det(\\Sigma - \\lambda I) = 0", x0 + 20, y0 + 245, 580, 40, 18)
    pg.note("eigenvector v = direction of a principal component", x0 + 20, y0 + 295, x0 + 215, y0 + 272, w=280)
    pg.note("eigenvalue λ = variance along v", x0 + 330, y0 + 295, x0 + 255, y0 + 272, w=200)
    pg.math(f"\\lambda_{{1,2}} = \\frac{{a+c}}{{2}} \\pm \\sqrt{{\\left(\\frac{{a-c}}{{2}}\\right)^2 + b^2}}",
            x0 + 20, y0 + 330, 580, 56, 16)
    pg.html(f"<span style='background:{HL}'>\\(\\lambda_1 = {l1:.2f},\\ \\ v_1 = ({v1[0]:.2f},\\ {v1[1]:.2f})\\)</span>"
            f"<br>\\(\\lambda_2 = {l2:.2f},\\ \\ v_2 = ({v2[0]:.2f},\\ {v2[1]:.2f})\\)",
            x0 + 20, y0 + 395, 580, 54, 15, align="center", valign="middle")
    pg.ask("Which direction holds most of the spread?",
           f"v₁ = ({v1[0]:.2f}, {v1[1]:.2f}), λ₁ = {l1:.2f} → {100 * l1 / (l1 + l2):.0f}% of all variance",
           x0 + 452, y0 + 414, x0 + 428, y0 + 412, w=163)
    pg.bullets(x0 + 20, y0 + 465, 580, [
        "\\(\\lambda_1 + \\lambda_2 = \\) total variance \\(= 2\\) (number of standardized features)",
        f"PC1 explains \\({l1:.2f}/2 = {100 * l1 / (l1 + l2):.0f}\\%\\) of the variance → keep 1 dimension instead of 2",
        "Eigenvectors are orthogonal (90°) → new features are uncorrelated",
        "In practice computed with SVD of Z (more stable)"], head="Where:")

    # ---------------------------------------------------------------- ③ principal components
    x0 = 1720
    pg.panel(x0, y0, 680, 620, 3, "Principal Components (new axes)", "Rotate the axes to the directions of spread")
    p = pg.axes(x0 + 90, y0 + 90, 400, 400, (-3, 3), (-3, 3), "z_1", "z_2", label_size=13, origin=(0, 0))
    p.poly([(-3 * v1[0], -3 * v1[1]), (3 * v1[0], 3 * v1[1])], dashed=True, color="#BBBBBB")
    sel = sorted(Z, key=lambda t: -abs(t[0] * v2[0] + t[1] * v2[1]))[:7]
    for pt in sel:
        t = pt[0] * v1[0] + pt[1] * v1[1]
        pr = (t * v1[0], t * v1[1])
        p.poly([pt, pr], dashed=True, color=RED, width=0.75)
        p.scatter([pr], color=RED, r=3)
    p.scatter(Z, r=3.5)
    L1, L2 = 2 * math.sqrt(l1), 2 * math.sqrt(l2)
    pg.line([p.P(0, 0), p.P(L1 * v1[0], L1 * v1[1])], color=RED, width=3, arrow="block")
    pg.line([p.P(0, 0), p.P(L2 * v2[0], L2 * v2[1])], color=GREEN, width=3, arrow="block")
    p.text("PC1", L1 * v1[0], L1 * v1[1], w=50, dx=24, dy=-6, size=13, color=RED, bold=True)
    p.text("PC2", L2 * v2[0], L2 * v2[1], w=50, dx=-26, dy=-6, size=13, color=GREEN, bold=True)
    pg.note("projection of a point onto PC1 (dashed ⟂)", x0 + 430, y0 + 470, *p.P(*sel[0]), w=230, color=RED)
    pg.ask("How much information does PC1 keep?",
           f"PC1 carries {100 * l1 / (l1 + l2):.0f}% of the variance: the cloud is almost a line",
           x0 + 95, y0 + 88, *p.P(0.7 * L1 * v1[0], 0.7 * L1 * v1[1]), w=190)
    pg.bullets(x0 + 20, y0 + 510, 640, [
        "<b>PC1</b> = direction of the largest variance (longest spread of the cloud)",
        "<b>PC2</b> ⟂ PC1, captures what is left;  arrow length ∝ \\(\\sqrt{\\lambda}\\)",
        "Projection = keep the shadow of every point on PC1 → 2D becomes 1D"], head="Where:")

    # ---------------------------------------------------------------- ④ projection
    x0, y0 = 400, 900
    pg.panel(x0, y0, 620, 600, 4, "Project: Z = X · Wₖ", "Reduce dimensions by keeping the top k components")
    pg.math("Z_{(n\\times k)} = X_{(n\\times p)}\\; W_{(p\\times k)}", x0 + 20, y0 + 70, 580, 44, 19)
    pg.note("standardized data", x0 + 40, y0 + 125, x0 + 290, y0 + 100, w=120)
    pg.note("top-k eigenvectors as columns", x0 + 380, y0 + 125, x0 + 400, y0 + 100, w=190)
    ex = Z[0]
    t0 = ex[0] * v1[0] + ex[1] * v1[1]
    pg.math(f"PC1_i = v_1 \\cdot z_i = ({v1[0]:.2f})({ex[0]:.2f}) + ({v1[1]:.2f})({ex[1]:.2f}) = {t0:.2f}",
            x0 + 20, y0 + 160, 580, 36, 14)
    pg.text("All 40 houses on the new 1-D axis (PC1 score):", x0 + 20, y0 + 215, 580, 20, 12, bold=True)
    s = [z[0] * v1[0] + z[1] * v1[1] for z in Z]
    lx0, lx1, ly = x0 + 60, x0 + 560, y0 + 290
    pg.line([(lx0 - 10, ly), (lx1 + 10, ly)], arrow="classic")
    for k in range(-3, 4):
        px = lx0 + (k + 3) / 6 * (lx1 - lx0)
        pg.line([(px, ly), (px, ly + 5)])
        pg.text(str(k), px - 10, ly + 6, 20, 14, 9, align="center", color="#555555")
    for v in s:
        pg.dot(lx0 + (max(-3, min(3, v)) + 3) / 6 * (lx1 - lx0), ly - 8, r=3.5, color=RED)
    pg.text("PC1", lx1 + 14, ly - 10, 40, 20, 12, bold=True, color=RED)
    pg.text("small, few rooms", lx0 - 20, ly + 24, 140, 16, 10, color="#555555")
    pg.text("big, many rooms", lx1 - 110, ly + 24, 140, 16, 10, color="#555555", align="right")
    pg.html(f"<span style='background:{HL}'>2 features → 1 feature, keeping {100 * l1 / (l1 + l2):.0f}% of the information"
            "</span>", x0 + 20, y0 + 340, 580, 24, 13, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 385, 580, [
        "New features (PC scores) are uncorrelated → removes multicollinearity",
        "Reconstruction: \\(\\hat{X} = Z W^{T}\\) (lossy: the PC2 part is thrown away)",
        "Fit μ, σ and W on training data only, then apply to valid / test (no leakage)",
        "Used in your EA pipeline: <i>Data Reduction</i> step before LSTM"], head="Notes:")
    smax = max(s)
    pg.ask("Where does each house land on the single new axis?",
           f"its PC1 score: left = small / few rooms, right = big / many rooms (house 1 → {t0:.2f})",
           x0 + 400, y0 + 372, lx0 + (min(3, smax) + 3) / 6 * (lx1 - lx0), ly - 12, w=210)

    # ---------------------------------------------------------------- ⑤ scree plot
    x0 = 1060
    pg.panel(x0, y0, 700, 600, 5, "How many components? Explained Variance", "Example with 6 standardized features")
    lam = [2.9, 1.6, 0.8, 0.4, 0.2, 0.1]
    ratio = [v / sum(lam) for v in lam]
    cum = [sum(ratio[:i + 1]) for i in range(6)]
    p = pg.axes(x0 + 80, y0 + 90, 440, 280, (0.4, 6.6), (0, 1.05), "PC", "", label_size=12, origin=(0.4, 0),
                xticks=(1, 2, 3, 4, 5, 6), yticks=(0, 0.5, 0.95), tick_fmt="{:g}")
    p.bars([(i + 1, r) for i, r in enumerate(ratio)], 0.6, fill=LIGHTBLUE, stroke=BLUE)
    for i, r in enumerate(ratio):
        p.text(f"{100 * r:.0f}%", i + 1, r, w=40, dy=-10, size=10, color=BLUE)
    p.poly([(i + 1, v) for i, v in enumerate(cum)], color=RED, width=2)
    p.scatter([(i + 1, v) for i, v in enumerate(cum)], color=RED, r=4)
    for i, v in enumerate(cum):
        if i: p.text(f"{100 * v:.0f}%", i + 1, v, w=40, dx=-18, dy=-12, size=9, color=RED)
    p.hline(0.95, color=GREEN)
    p.text("95% threshold", 0.5, 0.95, w=90, align="left", dy=-10, size=10, color=GREEN)
    p.scatter([(4, cum[3])], color=GREEN, r=7)
    pg.ask("How many components should I keep?",
           f"k = 4: first point above the 95% line ({100 * cum[3]:.0f}% kept)",
           x0 + 500, y0 + 112, *p.P(4.1, cum[3]), w=190)
    pg.html("<font color='#3333FF'>■ explained variance ratio \\(\\lambda_j / \\sum\\lambda\\)</font><br>"
            "<font color='#FF0000'>● cumulative</font>", x0 + 545, y0 + 240, 150, 70, 11)
    pg.math("\\text{explained}_j = \\frac{\\lambda_j}{\\sum_{i=1}^{p}\\lambda_i}", x0 + 30, y0 + 400, 640, 50, 18)
    pg.html(f"<span style='background:{HL}'>Keep k = 4 components → {100 * cum[3]:.0f}% of the variance with 4 instead "
            "of 6 features</span>", x0 + 20, y0 + 460, 660, 24, 13, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 495, 660, [
        "Rules of thumb: cumulative ≥ 90–95%, or the 'elbow' of the bars, or λ > 1 (Kaiser)",
        "For visualization keep k = 2 or 3"], head="Choose k:")

    # ---------------------------------------------------------------- ⑥ algorithm & notes
    x0 = 1800
    pg.panel(x0, y0, 600, 600, 6, "Algorithm & Trade-offs")
    pg.rect(x0 + 20, y0 + 50, 560, 170,
            "<b>Pseudo-code</b><br><font face='Courier New'>Z = (X − μ) / σ&nbsp;&nbsp;# ① standardize<br>"
            "Σ = Zᵀ Z / (n − 1)&nbsp;&nbsp;# ② covariance<br>λ, V = eig(Σ)&nbsp;&nbsp;# ② eigen<br>"
            "sort λ descending, reorder V<br>W = V[:, :k]&nbsp;&nbsp;# ⑤ choose k<br>"
            "Z_k = Z · W&nbsp;&nbsp;# ④ project</font>",
            fill="#F5F5F5", stroke="#999999", size=12, align="left", valign="top", raw=True)
    pg.bullets(x0 + 20, y0 + 240, 560, [
        "Removes multicollinearity, speeds up training, reduces noise",
        "Unsupervised: no labels needed;  fast (one eigen / SVD)",
        "Good for 2-D / 3-D visualization of high-dimensional data"], head="✔ Pros", h=80)
    pg.bullets(x0 + 20, y0 + 335, 560, [
        "Components are mixes of features → hard to interpret",
        "Linear only → curved structure needs Kernel PCA, t-SNE, UMAP",
        "Sensitive to scaling and outliers;  high variance ≠ useful for the target"], head="✘ Cons", h=80)
    pg.ask("Does more variance mean more predictive power?",
           "Not necessarily: PCA ignores the target; a small PC can still predict y (try LDA / supervised)",
           x0 + 365, y0 + 88, x0 + 340, y0 + 392, w=210)
    pg.bullets(x0 + 20, y0 + 430, 560, [
        "<b>LDA</b>: supervised — maximizes class separation instead of variance",
        "<b>Truncated SVD</b>: PCA without centring, for sparse text matrices",
        "<b>Autoencoder</b>: non-linear PCA with a neural network"], head="Related:", h=80)

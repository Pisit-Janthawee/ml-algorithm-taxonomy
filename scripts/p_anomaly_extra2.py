"""Z-Score / Anomaly page, part 3: ML anomaly detectors on one 2-D dataset (panels 9-14)."""
import math
from dio import *

# ---------------------------------------------------------------- shared dataset (seeded)
_r = seeded(2024)
BLOB_A = [(_r.gauss(3.0, 0.45), _r.gauss(3.0, 0.45)) for _ in range(60)]      # dense
BLOB_B = [(_r.gauss(7.0, 1.0), _r.gauss(6.5, 1.0)) for _ in range(60)]        # sparse
GLOBAL_ANOM = [(0.5, 8.5), (9.6, 1.0), (9.5, 9.6), (0.6, 0.6), (5.0, 9.6), (5.2, 0.4), (0.8, 5.6)]
LOCAL_ANOM = (3.0, 4.7)                                                        # near the dense blob
NORMAL = BLOB_A + BLOB_B
ANOM = GLOBAL_ANOM + [LOCAL_ANOM]
X = NORMAL + ANOM
IS_ANOM = [False] * len(NORMAL) + [True] * len(ANOM)
N = len(X)
I_LOCAL = N - 1


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cmap(t):
    """0 → blue, 1 → red."""
    t = max(0.0, min(1.0, t))
    r, g, b = int(51 + t * (255 - 51)), int(51 + (1 - abs(2 * t - 1)) * 120), int(255 - t * (255 - 51))
    return f"#{r:02X}{g:02X}{b:02X}"


def c_n(n):
    if n <= 1: return 0.0
    return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


# ---------------------------------------------------------------- isolation forest
def _itree(pts, depth, limit, rng):
    if depth >= limit or len(pts) <= 1:
        return ("leaf", len(pts))
    f = rng.randrange(2)
    lo, hi = min(p[f] for p in pts), max(p[f] for p in pts)
    if lo == hi: return ("leaf", len(pts))
    s = rng.uniform(lo, hi)
    return ("node", f, s, _itree([p for p in pts if p[f] < s], depth + 1, limit, rng),
            _itree([p for p in pts if p[f] >= s], depth + 1, limit, rng))


def _path(t, x, d=0):
    if t[0] == "leaf": return d + c_n(t[1])
    _, f, s, l, r = t
    return _path(l if x[f] < s else r, x, d + 1)


def iforest_scores(n_trees=150, psi=64, seed=7):
    rng = seeded(seed)
    limit = math.ceil(math.log2(psi))
    trees = [_itree(rng.sample(X, psi), 0, limit, rng) for _ in range(n_trees)]
    eh = [sum(_path(t, x) for t in trees) / n_trees for x in X]
    return [2 ** (-e / c_n(psi)) for e in eh], eh


def isolate_trace(target, seed):
    """Random splits on the full data until `target` is alone. Returns list of (feature, split, box)."""
    rng = seeded(seed)
    pts = list(X)
    box = [0.0, 10.0, 0.0, 10.0]
    cuts = []
    while len(pts) > 1 and len(cuts) < 30:
        f = rng.randrange(2)
        lo, hi = min(p[f] for p in pts), max(p[f] for p in pts)
        if lo == hi: continue
        s = rng.uniform(lo, hi)
        cuts.append((f, s, list(box)))
        if target[f] < s:
            pts = [p for p in pts if p[f] < s]
            box[2 * f + 1] = s
        else:
            pts = [p for p in pts if p[f] >= s]
            box[2 * f] = s
    return cuts


# ---------------------------------------------------------------- LOF
def lof_scores(k=10):
    D = [[dist(a, b) for b in X] for a in X]
    nbrs, kd = [], []
    for i in range(N):
        order = sorted((D[i][j], j) for j in range(N) if j != i)[:k]
        nbrs.append([j for _, j in order])
        kd.append(order[-1][0])
    lrd = []
    for i in range(N):
        reach = [max(kd[j], D[i][j]) for j in nbrs[i]]
        lrd.append(1.0 / (sum(reach) / k))
    lof = [sum(lrd[j] for j in nbrs[i]) / (k * lrd[i]) for i in range(N)]
    return lof, kd, lrd, nbrs


# ---------------------------------------------------------------- KDE (stand-in for the OCSVM RBF boundary)
def kde(p, h=0.6, pts=None):
    pts = pts or NORMAL
    return sum(math.exp(-dist(p, q) ** 2 / (2 * h * h)) for q in pts) / (len(pts) * 2 * math.pi * h * h)


# ---------------------------------------------------------------- DBSCAN
def dbscan(eps=0.8, min_pts=5):
    nb = [[j for j in range(N) if dist(X[i], X[j]) <= eps] for i in range(N)]
    core = [len(n) >= min_pts for n in nb]
    label = [-1] * N
    cid = 0
    for i in range(N):
        if label[i] != -1 or not core[i]: continue
        stack = [i]
        label[i] = cid
        while stack:
            u = stack.pop()
            if not core[u]: continue
            for v in nb[u]:
                if label[v] == -1:
                    label[v] = cid
                    stack.append(v)
        cid += 1
    return label, core


# ---------------------------------------------------------------- PCA reconstruction (linear autoencoder stand-in)
def pca_recon():
    mx = sum(p[0] for p in NORMAL) / len(NORMAL)
    my = sum(p[1] for p in NORMAL) / len(NORMAL)
    sxx = sum((p[0] - mx) ** 2 for p in NORMAL)
    syy = sum((p[1] - my) ** 2 for p in NORMAL)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in NORMAL)
    th = 0.5 * math.atan2(2 * sxy, sxx - syy)
    u = (math.cos(th), math.sin(th))
    err = []
    for x, y in X:
        dx, dy = x - mx, y - my
        z = dx * u[0] + dy * u[1]
        err.append(math.hypot(dx - z * u[0], dy - z * u[1]))
    return err, (mx, my), u


def zscore_global():
    mx = sum(p[0] for p in X) / N
    my = sum(p[1] for p in X) / N
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in X) / N)
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in X) / N)
    return [max(abs(p[0] - mx) / sx, abs(p[1] - my) / sy) for p in X]


def rank_of(scores, i):
    return 1 + sum(1 for s in scores if s > scores[i])


def prec_at_k(scores, k):
    top = sorted(range(N), key=lambda i: -scores[i])[:k]
    return sum(IS_ANOM[i] for i in top) / k


def scatter_scored(pg, p, scores, r=3.5, flag=None):
    lo, hi = min(scores), max(scores)
    for i, (x, y) in enumerate(X):
        cx, cy = p.P(x, y)
        pg.dot(cx, cy, r=r, color=cmap((scores[i] - lo) / (hi - lo + 1e-12)))
    if flag:
        for i in flag:
            cx, cy = p.P(*X[i])
            pg.ellipse(cx - 7, cy - 7, 14, 14, stroke=BLACK, sw=1.2)


# ======================================================================================
def build(pg: Page, cat):
    Y = pg.max_y() + 60
    pg.title("Beyond Z-scores: machine-learning anomaly detectors (same 2-D data for every method)", 400, Y, 1800, 22)
    pg.text(f"Data: dense blob A ({len(BLOB_A)} pts), sparse blob B ({len(BLOB_B)} pts), "
            f"{len(GLOBAL_ANOM)} global anomalies on the edges + 1 LOCAL anomaly just above blob A. "
            "Everything below is computed in pure Python (seeded).", 400, Y + 34, 2000, 20, 12, italic=True,
            color="#555555")
    y0 = Y + 70
    if_s, eh = iforest_scores()
    lof, kd, lrd, nbrs = lof_scores()
    zg = zscore_global()

    # ------------------------------------------------------------------ ⑨ families + anomaly types
    x0 = 400
    pg.panel(x0, y0, 760, 720, 9, "Families of anomaly detectors & types of anomalies")
    center = pg.rect(x0 + 280, y0 + 210, 200, 50, "Anomaly detection", size=14, bold=True, sw=2)
    fams = [("Statistical", "Z, IQR, MAD (①–⑧)", -250, -130), ("Distance", "kNN distance", 0, -140),
            ("Density", "LOF, KDE (⑪)", 250, -130), ("Isolation", "Isolation Forest (⑩)", 270, 0),
            ("Boundary", "One-Class SVM (⑫)", 250, 120), ("Reconstruction", "Autoencoder, PCA (⑬)", 0, 125),
            ("Time-series", "ARIMA residuals, rolling Z", -250, 120), ("Clustering", "DBSCAN noise (⑫)", -270, 0)]
    for name, ex, dx, dy in fams:
        b = pg.rect(x0 + 300 + dx, y0 + 215 + dy, 170, 44, f"<b>{name}</b><br><font style='font-size:10px'>{ex}</font>",
                    fill="#F5F5F5", stroke="#999999", raw=True, size=12)
        pg.connect(center, b, orth=False, arrow=None, color="#999999")
    # three types of anomalies
    ty = y0 + 445
    pg.text("Three kinds of anomalies (on a price-like series)", x0 + 20, ty - 26, 720, 20, 13, bold=True)
    r2 = seeded(5)
    kinds = [("Point", "one value far from all others"), ("Contextual", "normal value, wrong time (season)"),
             ("Collective", "each value normal, the run is not")]
    for k, (name, desc) in enumerate(kinds):
        gx = x0 + 30 + k * 245
        p = pg.axes(gx, ty + 10, 200, 110, (0, 40), (-3, 3), "", "", arrow=False)
        ys = []
        for t in range(41):
            base = 1.6 * math.sin(2 * math.pi * t / 20) if k == 1 else 0
            v = base + r2.gauss(0, 0.35)
            if k == 0 and t == 26: v = 2.8
            if k == 1 and t == 15: v = 1.4          # high value at a seasonal trough
            if k == 2 and 22 <= t <= 30: v = 0.05 * r2.random()
            ys.append(v)
        p.poly(list(enumerate(ys)), color=BLUE, width=1.2)
        hi = {0: [26], 1: [15], 2: list(range(22, 31))}[k]
        p.scatter([(t, ys[t]) for t in hi], color=RED, r=3)
        pg.text(name, gx, ty + 128, 200, 18, 12, bold=True, align="center", color=RED)
        pg.text(desc, gx - 10, ty + 146, 220, 18, 10, align="center", color="#555555")
    pg.ask("Which detector family fits my problem?",
           "point anomalies in tabular data → isolation / density; contextual → model the time pattern first; "
           "collective → windowed or sequence models",
           x0 + 470, y0 + 625, x0 + 560, ty + 160, w=270)
    pg.bullets(x0 + 20, y0 + 620, 440, ["Unsupervised (no labels) is the usual case; anomalies are rare and varied",
                                       "Score first, then pick a threshold from the expected contamination"],
               head="Key idea:", size=11)

    # ------------------------------------------------------------------ ⑩ isolation forest
    x0 = 1200
    pg.panel(x0, y0, 1200, 720, 10, "Isolation Forest: anomalies are easy to isolate",
             "Random feature, random split value, repeat → short path = anomaly")
    targets = [(GLOBAL_ANOM[2], "anomaly (9.5, 9.6)", RED, 3), (BLOB_A[0], "normal point in blob A", GREEN, 11)]
    for k, (pt, lab, col, seed) in enumerate(targets):
        gx = x0 + 30 + k * 300
        p = pg.axes(gx, y0 + 90, 250, 250, (0, 10), (0, 10), "x_1", "x_2", label_size=11)
        p.scatter(X, color="#BBBBBB", r=2.2)
        cuts = isolate_trace(pt, seed)
        for f, s, box in cuts:
            if f == 0:
                p.poly([(s, box[2]), (s, box[3])], color=col, width=1)
            else:
                p.poly([(box[0], s), (box[1], s)], color=col, width=1)
        p.scatter([pt], color=col, r=5)
        pg.text(f"{lab}: isolated after {len(cuts)} cuts", gx - 10, y0 + 352, 280, 18, 11, bold=True, align="center",
                color=col)
    # path length histogram
    hx = x0 + 650
    pg.text("Average path length E[h(x)] over 150 trees", hx, y0 + 70, 520, 18, 12, bold=True)
    hp = pg.axes(hx + 40, y0 + 100, 440, 150, (2, 10), (0, 40), "E[h]", "", label_size=11, xticks=(2, 4, 6, 8, 10))
    bins = [0] * 16
    abins = [0] * 16
    for i, e in enumerate(eh):
        b = min(15, max(0, int((e - 2) / 0.5)))
        (abins if IS_ANOM[i] else bins)[b] += 1
    hp.bars([(2.25 + 0.5 * b, c) for b, c in enumerate(bins) if c], 0.45, fill=LIGHTBLUE, stroke=BLUE)
    hp.bars([(2.25 + 0.5 * b, c) for b, c in enumerate(abins) if c], 0.45, fill=RED, stroke=RED, opacity=80)
    pg.text("red = injected anomalies (short paths) · blue = normal points", hx + 40, y0 + 262, 440, 16, 10,
            italic=True, align="center")
    pg.math(r"s(x,n) = 2^{-\frac{E[h(x)]}{c(n)}},\quad c(n)=2H(n-1)-\frac{2(n-1)}{n}", hx, y0 + 290, 520, 44, 15)
    pg.note("E[h] small → s → 1 (anomaly)", hx + 10, y0 + 340, hx + 175, y0 + 300, w=190, color=RED)
    pg.note("c(n): average path of an unsuccessful BST search (normaliser)", hx + 220, y0 + 340, hx + 400, y0 + 315, w=290)
    # score map
    k8 = len(ANOM)
    top = sorted(range(N), key=lambda i: -if_s[i])[:k8]
    sp = pg.axes(x0 + 60, y0 + 420, 250, 220, (0, 10), (0, 10), "x_1", "x_2", label_size=11)
    scatter_scored(pg, sp, if_s, flag=top)
    pg.text("score map (blue low → red high), circles = top-8", x0 + 30, y0 + 650, 320, 18, 10, italic=True,
            align="center")
    rows = [["point", "E[h]", "score s", "anomaly?"]]
    for i in top[:6]:
        rows.append([f"({X[i][0]:.1f}, {X[i][1]:.1f})", f"{eh[i]:.2f}", f"{if_s[i]:.3f}", "yes" if IS_ANOM[i] else "no"])
    pg.table(x0 + 360, y0 + 420, 300, rows, size=10, h=150)
    pg.html(f"precision@8 = <b>{prec_at_k(if_s, k8):.2f}</b> · local anomaly rank = <b>{rank_of(if_s, I_LOCAL)}</b> "
            f"of {N}", x0 + 360, y0 + 580, 320, 34, 11)
    pg.ask("Which points are unusual without any labels?",
           f"top Isolation-Forest scores ≈ {if_s[top[0]]:.2f}; normal points sit near "
           f"{sorted(if_s)[N // 2]:.2f}. Isolation is global: the local anomaly ranks only #{rank_of(if_s, I_LOCAL)}",
           x0 + 700, y0 + 450, *sp.P(*X[top[0]]), w=280)
    pg.bullets(x0 + 700, y0 + 580, 480, ["n_estimators ≈ 100–300, max_samples ψ = 256 (64 here)",
                                         "contamination only sets the threshold, not the scores",
                                         "fast O(n log ψ), no distances, works in high dimensions"],
               head="Settings:", size=10)

    # ------------------------------------------------------------------ ⑪ LOF
    x0, y0 = 400, y0 + 760
    pg.panel(x0, y0, 980, 720, 11, "Local Outlier Factor (LOF): compare density with the neighbours'",
             "k = 10 nearest neighbours")
    p = pg.axes(x0 + 60, y0 + 90, 360, 360, (0, 10), (0, 10), "x_1", "x_2", label_size=12)
    scatter_scored(pg, p, [min(v, 3.0) for v in lof], r=3.5)
    lx, ly = p.P(*LOCAL_ANOM)
    kr = kd[I_LOCAL] / 10 * 360
    pg.ellipse(lx - kr, ly - kr, 2 * kr, 2 * kr, dashed=True, stroke=RED)
    for j in nbrs[I_LOCAL]:
        pg.line([(lx, ly), p.P(*X[j])], color="#FF9999", width=0.7)
    pg.note(f"local anomaly: LOF = {lof[I_LOCAL]:.2f}", x0 + 60, y0 + 70, lx, ly - 6, w=170, color=RED)
    pg.text("colour = LOF (capped at 3)", x0 + 100, y0 + 462, 300, 16, 10, italic=True, align="center")
    fx = x0 + 450
    pg.math(r"\text{reach-dist}_k(a,b)=\max\{k\text{-dist}(b),\ d(a,b)\}", fx, y0 + 70, 500, 34, 13)
    pg.math(r"lrd_k(a)=\left(\frac{1}{k}\sum_{b\in N_k(a)}\text{reach-dist}_k(a,b)\right)^{-1}", fx, y0 + 110,
            500, 50, 13)
    pg.math(r"LOF_k(a)=\frac{1}{k}\sum_{b\in N_k(a)}\frac{lrd_k(b)}{lrd_k(a)}", fx, y0 + 168, 500, 50, 15)
    pg.note("≈ 1: as dense as neighbours · > 1.5: much sparser → outlier", fx + 120, y0 + 222, fx + 330, y0 + 200,
            w=330, color=RED)
    rk_if, rk_lof, rk_z = rank_of(if_s, I_LOCAL), rank_of(lof, I_LOCAL), rank_of(zg, I_LOCAL)
    ia = min(range(len(BLOB_A)), key=lambda i: dist(X[i], (3.0, 3.0)))                   # most central A point
    ib = min(range(len(BLOB_A), len(NORMAL)), key=lambda i: dist(X[i], (7.0, 6.5)))       # most central B point
    pg.table(fx, y0 + 255, 500, [
        ["point", "LOF", "rank LOF", "rank IF", "rank global Z"],
        [f"local ({LOCAL_ANOM[0]}, {LOCAL_ANOM[1]})", f"{lof[I_LOCAL]:.2f}", rk_lof, rk_if, rk_z],
        ["blob A centre point", f"{lof[ia]:.2f}", rank_of(lof, ia), rank_of(if_s, ia), rank_of(zg, ia)],
        ["blob B (sparse) centre point", f"{lof[ib]:.2f}", rank_of(lof, ib), rank_of(if_s, ib), rank_of(zg, ib)],
        [f"global ({GLOBAL_ANOM[2][0]}, {GLOBAL_ANOM[2][1]})", f"{lof[len(NORMAL) + 2]:.2f}",
         rank_of(lof, len(NORMAL) + 2), rank_of(if_s, len(NORMAL) + 2), rank_of(zg, len(NORMAL) + 2)]],
        size=10, h=120, highlight_rows=(1,))
    pg.bullets(fx, y0 + 395, 500, [
        "The local point is only ~1.7 units from blob A — globally unremarkable",
        "But blob A is very dense (σ = 0.45), so relative to its neighbours it is very sparse",
        "Points in the sparse blob B get LOF ≈ 1: sparse but <i>as sparse as their neighbours</i>"],
        head="Why LOF catches it:", size=11)
    pg.ask("Is this point odd compared with its own neighbourhood?",
           f"yes: LOF = {lof[I_LOCAL]:.2f} ranks it #{rk_lof} for LOF but only #{rk_if} for Isolation Forest "
           f"and #{rk_z} for a global Z rule",
           x0 + 30, y0 + 500, lx, ly + 6, w=300)
    pg.bullets(x0 + 360, y0 + 560, 600, ["k (n_neighbors) ≈ 10–50: small k = very local, large k = smoother",
                                         "O(n²) distances → slow for big data; scale features first",
                                         "novelty=True in sklearn to score new points"], head="Settings:", size=10)

    # ------------------------------------------------------------------ ⑫ OCSVM (KDE stand-in) + DBSCAN
    x0 = 1420
    pg.panel(x0, y0, 980, 720, 12, "Boundary methods: One-Class SVM & DBSCAN noise",
             "Learn the region of 'normal', flag what falls outside")
    nu = 0.08
    dens_pts = sorted(kde(p_) for p_ in NORMAL)
    thr = dens_pts[int(nu * len(dens_pts))]
    p = pg.axes(x0 + 50, y0 + 90, 380, 380, (0, 10), (0, 10), "x_1", "x_2", label_size=12)
    G = 38
    for gi in range(G):
        for gj in range(G):
            cx_, cy_ = (gi + 0.5) * 10 / G, (gj + 0.5) * 10 / G
            if kde((cx_, cy_)) >= thr:
                a, b = p.P(gi * 10 / G, (gj + 1) * 10 / G)
                pg.rect(a, b, 380 / G, 380 / G, fill="#CCFFCC", stroke="none", opacity=80)
    inside = [kde(x) >= thr for x in X]
    for i, x in enumerate(X):
        pg.dot(*p.P(*x), r=3.2, color=BLUE if inside[i] else RED)
    out_anom = sum(1 for i in range(N) if not inside[i] and IS_ANOM[i])
    out_tot = sum(1 for i in range(N) if not inside[i])
    pg.text(f"green = learned 'normal' region (ν = {nu}); red = outside: {out_tot} points, {out_anom} true anomalies",
            x0 + 20, y0 + 480, 440, 34, 10, align="center", italic=True)
    pg.html("<b>Stand-in:</b> boundary drawn from an RBF kernel-density level set (bandwidth 0.6) — the same "
            "shape an RBF One-Class SVM learns; a full OCSVM solver was not implemented here.",
            x0 + 20, y0 + 515, 440, 50, 10)
    pg.math(r"\min_{w,\rho,\xi}\ \tfrac12\lVert w\rVert^2+\tfrac{1}{\nu n}\sum_i\xi_i-\rho", x0 + 20, y0 + 570,
            440, 40, 13)
    pg.text("ν = upper bound on the fraction of training points outside (≈ contamination)", x0 + 20, y0 + 612, 440, 30,
            10, align="center", color=RED)
    # DBSCAN
    lab, core = dbscan()
    q = pg.axes(x0 + 520, y0 + 90, 400, 400, (0, 10), (0, 10), "x_1", "x_2", label_size=12)
    cols = [BLUE, GREEN, ORANGE, "#9673A6", "#00AAAA"]
    for i, x in enumerate(X):
        if lab[i] >= 0:
            q_ = q.P(*x)
            pg.dot(*q_, r=3, color=cols[lab[i] % len(cols)])
    e_px = 0.8 / 10 * 400
    noise = [i for i in range(N) if lab[i] == -1]
    for i in noise:
        cx_, cy_ = q.P(*X[i])
        pg.ellipse(cx_ - e_px, cy_ - e_px, 2 * e_px, 2 * e_px, dashed=True, stroke=RED, sw=0.8)
        pg.dot(cx_, cy_, r=3.5, color=RED)
    n_anom_noise = sum(IS_ANOM[i] for i in noise)
    pg.text(f"DBSCAN eps = 0.8, minPts = 5 → {max(lab) + 1} clusters, {len(noise)} noise points "
            f"({n_anom_noise} true anomalies)", x0 + 500, y0 + 500, 460, 34, 11, bold=True, align="center")
    pg.text("dashed circle = eps radius: fewer than minPts neighbours → noise", x0 + 500, y0 + 536, 460, 18, 10,
            italic=True, align="center")
    pg.ask("Where does 'normal' end?",
           f"outside the learned region: OCSVM-style boundary flags {out_tot} points ({out_anom} real); DBSCAN "
           f"labels {len(noise)} points as noise ({n_anom_noise} real) — sparse blob B edges cause false alarms",
           x0 + 500, y0 + 580, *q.P(*X[noise[0]]) if noise else (None, None), w=440)

    # ------------------------------------------------------------------ ⑬ reconstruction error
    x0, y0 = 400, y0 + 760
    pg.panel(x0, y0, 900, 660, 13, "Reconstruction error: Autoencoder (PCA as linear stand-in)",
             "Compress → decompress; normal data reconstructs well, anomalies don't")
    # network sketch
    layers = [(2, BLUE, "input x"), (4, "#999999", "encoder"), (1, RED, "code z"), (4, "#999999", "decoder"),
              (2, GREEN, "output x̂")]
    nodes = []
    for li, (n, col, name) in enumerate(layers):
        lx_ = x0 + 50 + li * 80
        ys_ = [y0 + 150 + (j - (n - 1) / 2) * 34 for j in range(n)]
        nodes.append([(lx_, yy) for yy in ys_])
        pg.text(name, lx_ - 35, y0 + 230, 70, 16, 10, align="center", color="#555555")
    for a, b in zip(nodes, nodes[1:]):
        for (ax, ay) in a:
            for (bx, by) in b:
                pg.line([(ax + 9, ay), (bx - 9, by)], color="#CCCCCC", width=0.6)
    for li, layer in enumerate(nodes):
        for (cx_, cy_) in layer:
            pg.ellipse(cx_ - 9, cy_ - 9, 18, 18, fill=layers[li][1], stroke=layers[li][1], opacity=70)
    pg.math(r"\text{score}(x)=\lVert x-\hat{x}\rVert^2", x0 + 30, y0 + 260, 380, 36, 15)
    err, (mx, my), u = pca_recon()
    p = pg.axes(x0 + 450, y0 + 80, 400, 260, (0, 10), (0, 10), "x_1", "x_2", label_size=11)
    t0, t1 = -8, 8
    p.poly([(mx + t0 * u[0], my + t0 * u[1]), (mx + t1 * u[0], my + t1 * u[1])], color=BLACK, width=1.5, dashed=True)
    for i, x in enumerate(X):
        dx, dy = x[0] - mx, x[1] - my
        z = dx * u[0] + dy * u[1]
        xh = (mx + z * u[0], my + z * u[1])
        if IS_ANOM[i]:
            p.poly([x, xh], color=RED, width=1)
    scatter_scored(pg, p, err, r=3)
    pg.text("dashed line = 1-D code (PC1); red segments = reconstruction error of anomalies", x0 + 430, y0 + 350, 440,
            18, 10, italic=True, align="center")
    # histogram
    thr_e = sorted(err[:len(NORMAL)])[int(0.95 * len(NORMAL))]
    h = pg.axes(x0 + 70, y0 + 410, 380, 150, (0, 6), (0, 50), "error", "count", label_size=11, xticks=(0, 2, 4, 6))
    nb_, ab_ = [0] * 12, [0] * 12
    for i, e in enumerate(err):
        b = min(11, int(e / 0.5))
        (ab_ if IS_ANOM[i] else nb_)[b] += 1
    h.bars([(0.25 + 0.5 * b, c) for b, c in enumerate(nb_) if c], 0.45, fill=LIGHTBLUE, stroke=BLUE)
    h.bars([(0.25 + 0.5 * b, c) for b, c in enumerate(ab_) if c], 0.45, fill=RED, stroke=RED, opacity=80)
    h.vline(thr_e, 0, 50, color=RED, dashed=True, width=1.5)
    h.text(f"threshold = 95th pct of normal = {thr_e:.2f}", thr_e, 46, w=220, size=10, color=RED, align="left", dx=6)
    caught = sum(1 for i in range(N) if IS_ANOM[i] and err[i] > thr_e)
    pg.ask("Does the model fail to rebuild this point?",
           f"{caught} of {len(ANOM)} anomalies exceed the threshold; points lying ALONG the main axis rebuild well "
           "even if far away — a linear 1-D code is too weak, a real autoencoder adds non-linearity",
           x0 + 490, y0 + 420, *h.P(4.5, 2), w=380)
    pg.bullets(x0 + 490, y0 + 560, 390, ["Train only on normal data, threshold on validation errors",
                                         "Real autoencoders (MLP / LSTM) handle images, logs, sequences"],
               head="Practice:", size=10)

    # ------------------------------------------------------------------ ⑭ comparison
    x0 = 1340
    pg.panel(x0, y0, 1060, 660, 14, "Which detector? Results on this data & selection guide")
    k8 = len(ANOM)
    dbs = [1.0 if lab[i] == -1 else 0.0 for i in range(N)]
    neg_d = [-kde(x) for x in X]
    res = [("Global Z (max |z|)", zg), ("Isolation Forest", if_s), ("LOF (k = 10)", lof),
           ("KDE / OCSVM-like", neg_d), ("PCA reconstruction", err)]
    rows = [["method", "precision@8", "local anomaly rank", "assumption", "local / global"]]
    assume = ["one normal blob", "anomalies are few & different", "density varies by region",
              "smooth normal region", "normal data lies on a low-dim subspace"]
    scope = ["global", "global", "local", "global", "global"]
    best = 0
    for k, (name, sc) in enumerate(res):
        pk = prec_at_k(sc, k8)
        best = max(best, pk)
        rows.append([name, f"{pk:.2f}", rank_of(sc, I_LOCAL), assume[k], scope[k]])
    db_prec = sum(IS_ANOM[i] for i in noise) / max(1, len(noise))
    rows.append(["DBSCAN noise", f"{db_prec:.2f} ({len(noise)} flagged)", "noise" if lab[I_LOCAL] == -1 else "in cluster",
                 "clusters of similar density", "local-ish"])
    hl = [1 + k for k, (_, sc) in enumerate(res) if prec_at_k(sc, k8) == best]
    pg.table(x0 + 20, y0 + 50, 1020, rows, size=11, h=180, highlight_rows=tuple(hl))
    pg.table(x0 + 20, y0 + 260, 640, [
        ["situation", "start with", "why"],
        ["tabular, many features, no labels", "Isolation Forest", "fast, scales, no distance metric"],
        ["clusters of different density", "LOF", "compares with local neighbourhood"],
        ["clean training set of normal data only", "One-Class SVM / Autoencoder", "learns the normal region (novelty detection)"],
        ["images, logs, sequences", "Autoencoder / LSTM-AE", "reconstruction in a learned space"],
        ["time series", "model residuals + rolling Z (⑦)", "remove trend / season first"],
        ["some labelled anomalies", "supervised classifier + class weights", "labels beat unsupervised guesses"]],
        size=11, h=210)
    pg.ask("Which method should I trust here?",
           f"best precision@8 here = {best:.2f}; no method wins everywhere — LOF is the only one that ranks the "
           f"local anomaly near the top (#{rank_of(lof, I_LOCAL)}). Ensemble several scores in practice",
           x0 + 690, y0 + 290, x0 + 478, y0 + 130, w=350)
    pg.bullets(x0 + 20, y0 + 490, 600, ["Scale features before distance / density methods",
                                        "Evaluate with injected or labelled anomalies (precision@k, PR-AUC)",
                                        "Threshold = business decision (alerts per day you can review)"],
               head="Notes:", size=11)

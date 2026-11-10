"""Versatile Uses (Hidden Tricks) page - base: KNN Pixel Inpainting."""
import math
from dio import *


def hexc(rgb):
    r, g, b = (max(0, min(255, int(round(v)))) for v in rgb)
    return "#%02X%02X%02X" % (r, g, b)


def make_image(n=16):
    """Synthetic scene: sky gradient, sun, hill, ground."""
    img = []
    for r in range(n):
        row = []
        for c in range(n):
            if r < 10:                                   # sky gradient
                t = r / 9
                col = (90 + 110 * t, 150 + 70 * t, 235 - 10 * t)
            else:                                        # ground gradient
                t = (r - 10) / 5
                col = (70 - 30 * t, 160 - 50 * t, 60 - 20 * t)
            if (r - 3.5) ** 2 + (c - 11.5) ** 2 <= 6.5:  # sun
                col = (255, 210, 60)
            hill = 10 - 3.2 * math.exp(-((c - 4) / 3.2) ** 2)
            if r >= hill and r < 10:                     # hill
                col = (110, 170 - 4 * (r - hill) * 10 / 3, 80)
            row.append(col)
        img.append(row)
    return img


def draw_img(pg, x, y, img, cell, missing=None, title=None, fills=None):
    n = len(img)
    for r in range(n):
        for c in range(len(img[0])):
            if missing and (r, c) in missing:
                pg.rect(x + c * cell, y + r * cell, cell, cell, "×", fill="#FFFFFF", stroke="#DDDDDD", size=8,
                        color="#999999")
            else:
                col = fills[r][c] if fills else hexc(img[r][c])
                pg.rect(x + c * cell, y + r * cell, cell, cell, fill=col, stroke="none")
    pg.rect(x, y, cell * len(img[0]), cell * n, stroke=BLACK)
    if title:
        pg.text(title, x - 20, y - 24, cell * len(img[0]) + 40, 20, 12, bold=True, align="center")


def knn_fill(img, missing, k=4):
    n = len(img)
    known = [(r, c) for r in range(n) for c in range(n) if (r, c) not in missing]
    out = [row[:] for row in img]
    detail = {}
    for (r, c) in missing:
        nb = sorted(known, key=lambda q: ((q[0] - r) ** 2 + (q[1] - c) ** 2, q))[:k]
        avg = tuple(sum(img[a][b][i] for a, b in nb) / k for i in range(3))
        out[r][c] = avg
        detail[(r, c)] = nb
    return out, detail


def build(pg: Page, cat):
    r = seeded(25)
    img = make_image()
    N = 16

    # ------------------------------------------------------------------ ① KNN pixel inpainting
    x0, y0 = 400, 240
    pg.panel(x0, y0, 1300, 720, 1, "★ KNN Pixel Inpainting: fill missing pixels from the closest colours",
             "The same nearest-neighbour idea as classification, but the 'label' is a colour and the features are (row, col)")
    missing = {(a, b) for a in range(N) for b in range(N) if r.random() < 0.22}
    missing |= {(a, b) for a in range(6, 9) for b in range(2, 6)}           # scratched block
    filled, detail = knn_fill(img, missing, k=4)
    cell, gx, gy = 12, x0 + 30, y0 + 100
    step = N * cell + 48
    draw_img(pg, gx, gy, img, cell, title="① Original")
    draw_img(pg, gx + step, gy, img, cell, missing=missing, title=f"② Damaged ({len(missing)} px missing)")
    draw_img(pg, gx + 2 * step, gy, filled, cell, title="③ Restored by kNN (k = 4)")
    errs = {}
    for (a, b) in missing:
        errs[(a, b)] = sum(abs(filled[a][b][i] - img[a][b][i]) for i in range(3)) / 3
    emax = max(errs.values())
    efill = [["#FFFFFF" if (a, b) not in missing else hexc((255, 255 - 220 * errs[(a, b)] / emax,
                                                             255 - 220 * errs[(a, b)] / emax))
              for b in range(N)] for a in range(N)]
    draw_img(pg, gx + 3 * step, gy, img, cell, title="④ Error map (red = large)", fills=efill)
    for k in range(3):
        ax = gx + (k + 1) * step - 44
        pg.line([(ax, gy + 96), (ax + 38, gy + 96)], arrow="classic")
    mae = sum(errs.values()) / len(errs)
    big = [errs[p] for p in errs if 6 <= p[0] <= 8 and 2 <= p[1] <= 5]
    small = [errs[p] for p in errs if not (6 <= p[0] <= 8 and 2 <= p[1] <= 5)]
    pg.html(f"Mean absolute colour error on missing pixels: <b>{mae:.1f}</b> / 255  ·  scattered holes "
            f"<b>{sum(small) / len(small):.1f}</b>  vs  scratched block <b>{sum(big) / len(big):.1f}</b> "
            f"(neighbours are farther away → blurrier)", gx, gy + N * cell + 14, 4 * step - 48, 36, 11)
    pg.math(r"\hat{x}_p = \frac{1}{k}\sum_{q \in N_k(p)} x_q", gx, gy + N * cell + 60, 300, 50, 18)
    pg.math(r"\hat{x}_p = \frac{\sum_{q} x_q / d_{pq}}{\sum_{q} 1 / d_{pq}}", gx + 320, gy + N * cell + 60, 300,
            50, 16)
    pg.text("plain average", gx, gy + N * cell + 108, 300, 16, 10, align="center", color="#555555")
    pg.text("distance-weighted (sharper)", gx + 320, gy + N * cell + 108, 300, 16, 10, align="center",
            color="#555555")
    pg.bullets(gx, gy + N * cell + 135, 840, [
        "\\(p\\) = missing pixel, \\(N_k(p)\\) = its k closest <b>known</b> pixels by Euclidean distance on (row, col), "
        "\\(x_q\\) = RGB colour",
        "Works because images are <b>smooth</b>: neighbouring pixels usually have similar colours (the kNN assumption)",
        "Small k keeps edges sharp but is noisy; large k blurs. Weighting by 1/d helps at edges",
        "Real tools add colour/texture similarity (patch matching) and diffusion — kNN is the simplest baseline"],
        head="Where & why:", size=11)
    # zoom-in on one missing pixel
    def spread(p):   # pick a pixel whose neighbours have different colours (most instructive)
        cs = [img[a][b] for a, b in detail[p]]
        return sum(max(c[i] for c in cs) - min(c[i] for c in cs) for i in range(3))
    cands = [p for p in missing if 2 <= p[0] <= 13 and 2 <= p[1] <= 13 and not (6 <= p[0] <= 8 and 2 <= p[1] <= 5)]
    tgt = max(sorted(cands), key=spread)
    tr, tc = tgt
    zx, zy, zc = x0 + 975, y0 + 100, 44
    pg.text(f"Zoom: pixel ({tr}, {tc}) and its 4 nearest known pixels", zx - 10, zy - 30, 340, 20, 12, bold=True)
    nb = detail[tgt]
    for a in range(-2, 3):
        for b in range(-2, 3):
            rr, cc = tr + a, tc + b
            if not (0 <= rr < N and 0 <= cc < N): continue
            px, py = zx + (b + 2) * zc + 40, zy + (a + 2) * zc
            if (rr, cc) == tgt:
                pg.rect(px, py, zc, zc, "?", fill="#FFFFFF", stroke=RED, sw=3, size=18, bold=True, color=RED)
            elif (rr, cc) in missing:
                pg.rect(px, py, zc, zc, "×", fill="#FFFFFF", stroke="#DDDDDD", size=11, color="#999999")
            else:
                pg.rect(px, py, zc, zc, fill=hexc(img[rr][cc]), stroke=BLACK if (rr, cc) in nb else "#FFFFFF",
                        sw=3 if (rr, cc) in nb else 1)
    rows = [["neighbour (r, c)", "distance", "R", "G", "B"]]
    for (a, b) in nb:
        d = math.hypot(a - tr, b - tc)
        rows.append([f"({a}, {b})", f"{d:.2f}", *[f"{img[a][b][i]:.0f}" for i in range(3)]])
    est, true = filled[tr][tc], img[tr][tc]
    rows.append(["<b>average → estimate</b>", "", *[f"<b>{est[i]:.0f}</b>" for i in range(3)]])
    rows.append(["true colour", "", *[f"{true[i]:.0f}" for i in range(3)]])
    pg.table(zx, zy + 236, 330, rows, size=10, h=150, highlight_rows=(5,))
    pg.rect(zx + 30, zy + 400, 60, 30, fill=hexc(est), stroke=BLACK)
    pg.text("estimate", zx + 20, zy + 432, 80, 16, 10, align="center")
    pg.rect(zx + 130, zy + 400, 60, 30, fill=hexc(true), stroke=BLACK)
    pg.text("true", zx + 120, zy + 432, 80, 16, 10, align="center")
    pg.text("Thick black border = the k = 4 nearest known pixels", zx, zy + 460, 340, 18, 10, italic=True,
            color="#555555")

    # ------------------------------------------------------------------ ② KNN imputer (tabular)
    x0 = 1740
    pg.panel(x0, y0, 660, 720, 2, "KNN Imputer: fill missing table values from similar rows",
             "Same trick on tabular data: distance on the observed columns")
    rows_ = []
    for i in range(10):
        vol = 0.8 + 2.2 * r.random()
        ret = r.gauss(0, 1.0)
        rsi = 50 + 9 * ret + r.gauss(0, 2)
        volume = 900 + 520 * vol + r.gauss(0, 60)
        rows_.append([round(rsi, 1), round(vol, 2), round(ret, 2), round(volume)])
    cols = ["RSI", "Volatility %", "Return %", "Volume"]
    miss = {(2, 3), (6, 0)}                                 # (row, col) hidden
    truth = {p: rows_[p[0]][p[1]] for p in miss}
    tbl = [["#"] + cols]
    for i, rw in enumerate(rows_):
        tbl.append([i + 1] + [('<span style="background:#FFCCCC"><b>NaN</b></span>' if (i, j) in miss else f"{v:g}")
                              for j, v in enumerate(rw)])
    pg.text("Table with missing cells (true values hidden)", x0 + 20, y0 + 70, 600, 20, 12, bold=True)
    pg.table(x0 + 20, y0 + 94, 380, tbl, size=10, h=230)
    # scaling stats from observed values
    stats = []
    for j in range(4):
        vals = [rows_[i][j] for i in range(10) if (i, j) not in miss]
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        stats.append((m, s))

    def impute(i, j, k=3):
        obs = [c for c in range(4) if c != j]
        cand = [t for t in range(10) if t != i and (t, j) not in miss]
        d = []
        for t in cand:
            dd = math.sqrt(sum(((rows_[i][c] - rows_[t][c]) / stats[c][1]) ** 2 for c in obs))
            d.append((dd, t))
        d.sort()
        nb = d[:k]
        return sum(rows_[t][j] for _, t in nb) / k, nb, d

    (ri, cj) = (2, 3)
    val, nb, dall = impute(ri, cj)
    pg.text(f"Row {ri + 1}: distance to other rows on scaled RSI, Volatility, Return", x0 + 20, y0 + 340, 620, 20, 12,
            bold=True)
    drows = [["row", "distance", "Volume", "used?"]]
    for dd, t in dall[:5]:
        drows.append([t + 1, f"{dd:.2f}", f"{rows_[t][3]:g}", "✔" if (dd, t) in nb else ""])
    pg.table(x0 + 20, y0 + 364, 300, drows, size=10, h=120, highlight_rows=(1, 2, 3))
    pg.math(r"d(i,t)=\sqrt{\sum_{c\,\in\,obs}\left(\frac{x_{ic}-x_{tc}}{s_c}\right)^2}", x0 + 330, y0 + 370, 310, 50, 13)
    pg.text(f"k = 3 → Volume = mean of rows {', '.join(str(t + 1) for _, t in nb)} = {val:.0f}", x0 + 330, y0 + 430,
            310, 34, 11, bold=True, color=BLUE)
    res = [["cell", "true (hidden)", "kNN (k = 3)", "column mean"]]
    errk = errm = 0
    for (i, j) in sorted(miss):
        v, _, _ = impute(i, j)
        m = stats[j][0]
        res.append([f"row {i + 1}, {cols[j]}", f"{truth[(i, j)]:g}", f'<span style="background:{HL}">{v:.1f}</span>',
                    f"{m:.1f}"])
        errk += abs(v - truth[(i, j)]) / stats[j][1]
        errm += abs(m - truth[(i, j)]) / stats[j][1]
    pg.table(x0 + 20, y0 + 500, 620, res, size=10, h=70)
    pg.html(f"Average error in σ units: kNN <b>{errk / 2:.2f}σ</b> vs mean imputation <b>{errm / 2:.2f}σ</b> — "
            "kNN uses the relationships between columns; the mean ignores them.", x0 + 20, y0 + 580, 620, 34, 11)
    pg.bullets(x0 + 20, y0 + 622, 620, [
        "Scale columns first (otherwise Volume dominates the distance)",
        "Fit on training rows only; sklearn: <i>KNNImputer(n_neighbors=3)</i>"], size=10)

    panel_quantize(pg, img, 400, 1000)
    panel_pca(pg, 1440, 1000)
    panel_tools(pg, 400, 1700)
    panel_trading(pg, 1440, 1700)
    panel_summary(pg, 400, 2430)

    # ------------------------------------------------------------ interpretation callouts
    pg.ask("What colour should the missing pixel be?",
           "the average of its 4 nearest known pixels → RGB (101, 172, 100), close to the true green (70, 160, 60)",
           440, 760, 1478, 685, w=560)
    pg.ask("Is kNN better than filling with the column mean?",
           f"yes here: error {errk / 2:.2f}σ vs {errm / 2:.2f}σ, because it uses how the columns relate",
           1763, 896, 2104, 773, w=600, size=9)
    pg.ask("How few colours can I keep before the image looks wrong?",
           "k = 4: 12× smaller with colour MSE 186; k = 2 loses the sun (MSE 1516)",
           890, 1560, 818, 1243, w=490)
    pg.ask("How many components hold the real signal?",
           "k = 2: error drops 0.201 → 0.024 (88 % less noise); extra components add noise back",
           2165, 1460, 2155, 1304, w=225)
    pg.ask("Can simple rules explain what the black box does?",
           "yes: a depth-2 tree reproduces 88 % of its predictions",
           430, 1995, 831, 1947, w=470)
    pg.ask("Do these two indicators predict a bounce better together?",
           "yes: RSI &lt; 30 &amp; Close &lt; lower BB → confidence 0.77, lift 2.15 vs ~1.85 alone; validate out-of-sample",
           1950, 2290, 2283, 2186, w=420)
    pg.ask("Why does an algorithm work outside its usual job?",
           "the hidden use relies on the same core idea, e.g. nearby points are similar, centroids summarise groups",
           440, 2500, 999, 2516, w=430)


# ---------------------------------------------------------------------- ⑤ model-as-a-tool helpers
def gini_split(S, feats):
    best = None
    for f in feats:
        for th in sorted({round(p[f], 2) for p, _ in S})[1:]:
            L = [y for p, y in S if p[f] < th]
            R = [y for p, y in S if p[f] >= th]
            if not L or not R: continue
            g = sum(len(G) * (1 - (sum(G) / len(G)) ** 2 - (1 - sum(G) / len(G)) ** 2) for G in (L, R)) / len(S)
            if best is None or g < best[0]: best = (g, f, th)
    return best


def majority(S):
    return 1 if sum(y for _, y in S) * 2 >= len(S) else 0


def panel_tools(pg, x0, y0):
    pg.panel(x0, y0, 1000, 690, 5, "Model-as-a-Tool gallery", "Use a model for a job other than its usual prediction")
    rnd = seeded(5)
    # (a) tree surrogate of a black box
    bb = lambda a, b: 1 if a + 0.18 * math.sin(7 * b) > 0.55 or b > 0.82 else 0
    S = [((rnd.random(), rnd.random()), 0) for _ in range(400)]
    S = [(p, bb(*p)) for p, _ in S]
    g, f, th = gini_split(S, (0, 1))
    L = [s for s in S if s[0][f] < th]
    R = [s for s in S if s[0][f] >= th]
    gl, fl, tl = gini_split(L, (0, 1))
    gr, fr, tr_ = gini_split(R, (0, 1))

    def sur(a, b):
        p = (a, b)
        if p[f] < th:
            return majority([s for s in L if s[0][fl] < tl]) if p[fl] < tl else majority([s for s in L if s[0][fl] >= tl])
        return majority([s for s in R if s[0][fr] < tr_]) if p[fr] < tr_ else majority([s for s in R if s[0][fr] >= tr_])
    grid = [(i / 29, j / 29) for i in range(30) for j in range(30)]
    fid = sum(sur(*p) == bb(*p) for p in grid) / len(grid)
    ax, ay = x0 + 30, y0 + 70
    pg.text("(a) Tree surrogate explains a black box", ax, ay, 460, 20, 12, bold=True)
    p = pg.axes(ax + 30, ay + 54, 170, 140, (0, 1), (0, 1), "x_1", "x_2", label_size=11)
    for (a, b) in grid[::2]:
        pg.dot(*p.P(a, b), r=2, color=RED if bb(a, b) else BLUE)
    nm = ["x₁", "x₂"]
    if f == 0:
        p.vline(th, color=BLACK, width=2)
        p.poly([(0, tl), (th, tl)] if fl == 1 else [(tl, 0), (tl, 1)], dashed=True, color=BLACK)
        p.poly([(th, tr_), (1, tr_)] if fr == 1 else [(tr_, 0), (tr_, 1)], dashed=True, color=BLACK)
    else:
        p.hline(th, color=BLACK, width=2)
    tx = ax + 240
    n0 = pg.rect(tx + 60, ay + 40, 110, 30, f"{nm[f]} < {th:.2f}?", size=10, bold=True)
    n1 = pg.rect(tx, ay + 105, 110, 30, f"{nm[fl]} < {tl:.2f}?", size=10)
    n2 = pg.rect(tx + 125, ay + 105, 110, 30, f"{nm[fr]} < {tr_:.2f}?", size=10)
    pg.connect(n0, n1, orth=False, label="yes"); pg.connect(n0, n2, orth=False, label="no")
    pg.text("leaves = black-box majority", tx, ay + 145, 240, 16, 9, italic=True, color="#555555")
    pg.html(f'Fidelity (agreement with black box): <span style="background:{HL}"><b>{fid * 100:.0f} %</b></span><br>'
            "Train the tree on the <i>model's predictions</i>, not the labels", tx - 10, ay + 165, 250, 40, 10)

    # (b) random-forest proximity imputation
    bx, by = x0 + 520, y0 + 70
    pg.text("(b) Forest proximity → impute a missing value", bx, by, 460, 20, 12, bold=True)
    data = [(1.0, 2.1), (1.2, 2.4), (3.0, 6.2), (3.3, 6.8), (1.1, None), (2.9, 5.9)]
    ys = [0, 0, 1, 1, 0, 1]
    filled_init = sum(v for _, v in data if v is not None) / 5
    tgt, T, est, hist = 4, 60, filled_init, [filled_init]
    for _ in range(3):                       # iterate: proximity -> impute -> refit
        X = [(a, b if b is not None else est) for a, b in data]
        prox = [[0] * 6 for _ in range(6)]
        for _ in range(T):                   # supervised stumps on bootstrap samples (target = class y)
            boot = [rnd.randrange(6) for _ in range(6)]
            fe = rnd.randrange(2)
            best = None
            for th1 in sorted({X[i][fe] for i in boot})[1:]:
                Lb = [ys[i] for i in boot if X[i][fe] < th1]
                Rb = [ys[i] for i in boot if X[i][fe] >= th1]
                g = sum(len(G) * (1 - (sum(G) / len(G)) ** 2 - (1 - sum(G) / len(G)) ** 2) for G in (Lb, Rb))
                if best is None or g < best[0]: best = (g, th1)
            if best is None: continue
            leaf = [x[fe] < best[1] for x in X]
            for i in range(6):
                for j in range(6):
                    prox[i][j] += (leaf[i] == leaf[j]) / T
        w = [prox[tgt][j] for j in range(6) if j != tgt]
        vals = [data[j][1] for j in range(6) if j != tgt]
        est = sum(a * b for a, b in zip(w, vals)) / sum(w)
        hist.append(est)
    cell = 34
    for i in range(6):
        pg.text(f"r{i + 1}", bx + 4, by + 52 + i * cell, 26, cell, 9, align="center")
        pg.text(f"r{i + 1}", bx + 34 + i * cell, by + 30, cell, 20, 9, align="center")
        for j in range(6):
            v = prox[i][j]
            pg.rect(bx + 34 + j * cell, by + 50 + i * cell, cell, cell, f"{v:.2f}",
                    fill=hexc((255, 255 - 150 * v, 255 - 150 * v)), stroke=BLACK if i == tgt else "#CCCCCC",
                    sw=2 if i == tgt else 0.5, size=8)
    pg.html("<b>Proximity</b> = share of trees where two rows land in the same leaf<br>"
            f"Row 5 has x₂ = <b>NaN</b> (start: column mean {filled_init:.2f})", bx + 250, by + 50, 210, 70, 10)
    pg.math(r"\hat{x}_{5} = \frac{\sum_j prox_{5j}\,x_j}{\sum_j prox_{5j}}", bx + 250, by + 120, 210, 46, 13)
    pg.html(f'iterations: {" → ".join(f"{h:.2f}" for h in hist[:-1])} → '
            f'<span style="background:{HL}"><b>{est:.2f}</b></span><br>'
            "row 5 shares leaves mostly with r1, r2 (same class y = 0, x₂ ≈ 2.1–2.4)", bx + 250, by + 168, 210, 56, 10)
    pg.text("trees are trained on the target y, so 'similar' means similar for the prediction", bx + 34, by + 262,
            420, 16, 9, italic=True, color="#555555")

    # (c) autoencoder anomaly
    cx, cy = x0 + 30, y0 + 370
    pg.text("(c) Autoencoder: anomaly = large reconstruction error", cx, cy, 460, 20, 12, bold=True)
    normal = [abs(rnd.gauss(0, 1)) ** 2 * 0.05 + 0.02 for _ in range(400)]
    anom = [0.35 + rnd.random() * 0.3 for _ in range(8)]
    thr = sorted(normal)[int(0.99 * len(normal))]
    edges = [i * 0.025 for i in range(29)]
    cnt = [sum(e <= v < e + 0.025 for v in normal) for e in edges]
    cna = [sum(e <= v < e + 0.025 for v in anom) for e in edges]
    q = pg.axes(cx + 40, cy + 54, 400, 160, (0, 0.72), (0, max(cnt) * 1.1), "error", "count", label_size=10)
    q.bars([(e + 0.0125, c) for e, c in zip(edges, cnt) if c], 0.022, fill=LIGHTBLUE, stroke=BLUE)
    q.bars([(e + 0.0125, c * 6) for e, c in zip(edges, cna) if c], 0.022, fill=LIGHTRED, stroke=RED)
    q.vline(thr, 0, max(cnt) * 1.05, color=RED, dashed=True, width=1.5)
    q.text(f"threshold = 99th pct of normal = {thr:.2f}", thr, max(cnt) * 1.0, dx=95, size=9, color=RED, w=190)
    pg.text("blue = normal bars (train on normal data only)  ·  red = anomalies (counts ×6 to be visible)", cx, cy + 245,
            470, 30, 9, italic=True, color="#555555")

    # (d) Platt scaling
    dx_, dy_ = x0 + 520, y0 + 370
    pg.text("(d) Platt scaling: raw scores → calibrated probability", dx_, dy_, 460, 20, 12, bold=True)
    S2 = [(rnd.gauss(1.0, 1.0), 1) for _ in range(150)] + [(rnd.gauss(-1.0, 1.0), 0) for _ in range(350)]
    A, B = 0.0, 0.0
    for _ in range(600):
        gA = gB = 0
        for s, yv in S2:
            pr = 1 / (1 + math.exp(max(-50, min(50, A * s + B))))
            gA += (yv - pr) * s          # d logL / dA = -(y - p)·s  for p = 1/(1+e^{As+B})
            gB += (yv - pr)
        A -= 0.5 * gA / len(S2); B -= 0.5 * gB / len(S2)
    q = pg.axes(dx_ + 40, dy_ + 54, 280, 160, (-3.5, 3.5), (0, 1), "score s", "P(y=1)", label_size=10,
                yticks=(0.5, 1))
    for b0 in range(-3, 4):
        mem = [yv for s, yv in S2 if b0 - 0.5 <= s < b0 + 0.5]
        if len(mem) >= 5: q.scatter([(b0, sum(mem) / len(mem))], color=BLUE, r=4)
    q.curve(lambda s: 1 / (1 + math.exp(A * s + B)), -3.5, 3.5, n=50, color=RED, width=2)
    pg.math(r"P(y=1\mid s)=\frac{1}{1+e^{As+B}}", dx_ + 330, dy_ + 60, 150, 44, 12)
    pg.text(f"fitted A = {A:.2f}, B = {B:.2f}", dx_ + 330, dy_ + 110, 150, 18, 10, bold=True)
    pg.text("dots = observed share of positives per score bin; red = logistic fit", dx_ + 330, dy_ + 135, 150, 60, 9,
            italic=True, color="#555555")
    pg.text("Logistic regression on ONE feature (the model's score) turns SVM margins or boosted scores into "
            "probabilities usable for bet sizing", dx_, dy_ + 245, 470, 30, 9, italic=True, color="#555555")


# ---------------------------------------------------------------------- ⑥ trading tricks
def panel_trading(pg, x0, y0):
    pg.panel(x0, y0, 960, 690, 6, "Statistical trading tricks", "Regression, z-score, ARMA and Apriori used as signals")
    rnd = seeded(8)
    n = 160
    X, e, Y = [50.0], 0.0, []
    for t in range(1, n): X.append(X[-1] + rnd.gauss(0, 0.8))
    for t in range(n):
        e = 0.85 * e + rnd.gauss(0, 0.9)
        Y.append(10 + 1.5 * X[t] + e)
    mx, my = sum(X) / n, sum(Y) / n
    beta = sum((a - mx) * (b - my) for a, b in zip(X, Y)) / sum((a - mx) ** 2 for a in X)
    alpha = my - beta * mx
    # (a) prices
    ax, ay = x0 + 30, y0 + 70
    pg.text("(a) Regression hedge ratio β", ax, ay, 420, 20, 12, bold=True)
    lo, hi = min(X + Y) - 2, max(X + Y) + 2
    p = pg.axes(ax + 40, ay + 54, 360, 140, (0, n - 1), (lo, hi), "t", "price", label_size=10)
    p.poly(list(enumerate(X)), color=BLUE, width=1.5)
    p.poly(list(enumerate(Y)), color=ORANGE, width=1.5)
    pg.html(f'<font color="#3333FF"><b>X</b></font>, <font color="#FF8000"><b>Y</b></font>: OLS '
            f'Y = {alpha:.1f} + <span style="background:{HL}"><b>{beta:.2f}</b></span>·X  → hold 1 Y vs β X',
            ax, ay + 205, 430, 20, 10)
    # (b) z-score of spread
    spread = [b - beta * a - alpha for a, b in zip(X, Y)]
    sm = sum(spread) / n
    ss = math.sqrt(sum((v - sm) ** 2 for v in spread) / (n - 1))
    z = [(v - sm) / ss for v in spread]
    bx, by = x0 + 500, y0 + 70
    pg.text("(b) Z-score of the spread: mean reversion", bx, by, 440, 20, 12, bold=True)
    q = pg.axes(bx + 40, by + 54, 380, 140, (0, n - 1), (-3.2, 3.2), "t", "z", label_size=10, origin=(0, 0),
                yticks=(-2, 2))
    q.hline(2, color=RED); q.hline(-2, color=RED); q.hline(0, color=GREY)
    q.poly(list(enumerate(z)), color=BLACK, width=1.2)
    pos, ent = 0, []
    for t in range(1, n):
        if pos == 0 and z[t] > 2: pos = -1; ent.append((t, z[t], RED))
        elif pos == 0 and z[t] < -2: pos = 1; ent.append((t, z[t], GREEN))
        elif pos != 0 and z[t - 1] * z[t] <= 0: pos = 0; ent.append((t, z[t], BLUE))
    for t, v, c in ent: q.scatter([(t, v)], color=c, r=4.5)
    pg.html('<font color="#FF0000">● short spread (z &gt; 2)</font> &nbsp; <font color="#009900">● long (z &lt; −2)</font> '
            '&nbsp; <font color="#3333FF">● exit at 0</font>', bx, by + 205, 440, 20, 10)
    # (c) ARMA residual alerts
    cx, cy = x0 + 30, y0 + 380
    pg.text("(c) AR(1) residual alerts", cx, cy, 420, 20, 12, bold=True)
    r_, shocks = [0.0], {45: 4.5, 98: -5.0, 130: 4.0}
    for t in range(1, n): r_.append(0.4 * r_[-1] + rnd.gauss(0, 1) + shocks.get(t, 0))
    phi = sum(a * b for a, b in zip(r_[:-1], r_[1:])) / sum(a * a for a in r_[:-1])
    res = [r_[t] - phi * r_[t - 1] for t in range(1, n)]
    sd = math.sqrt(sum(v * v for v in res) / len(res))
    q = pg.axes(cx + 40, cy + 54, 360, 140, (0, n - 1), (-6, 6), "t", "e_t", label_size=10, origin=(0, 0))
    q.hline(3 * sd, color=RED); q.hline(-3 * sd, color=RED)
    q.poly([(t + 1, v) for t, v in enumerate(res)], color=GREY, width=1)
    flag = [(t + 1, v) for t, v in enumerate(res) if abs(v) > 3 * sd]
    q.scatter(flag, color=RED, r=4.5)
    pg.html(f"φ̂ = {phi:.2f}; residual eₜ = rₜ − φ̂rₜ₋₁; <b>{len(flag)}</b> alerts beyond ±3σ "
            f"(injected shocks at t = 45, 98, 130)", cx, cy + 205, 430, 34, 10)
    # (d) apriori on indicator events
    dx_, dy_ = x0 + 500, y0 + 380
    pg.text("(d) Apriori: indicators that fire together", dx_, dy_, 440, 20, 12, bold=True)
    days = []
    for _ in range(1000):
        A = rnd.random() < 0.12
        Bv = rnd.random() < (0.6 if A else 0.06)
        C = rnd.random() < (0.72 if A and Bv else 0.45 if A or Bv else 0.30)
        days.append((A, Bv, C))
    def rule(cond):
        sup_x = sum(cond(d) for d in days) / len(days)
        sup_xy = sum(cond(d) and d[2] for d in days) / len(days)
        sup_y = sum(d[2] for d in days) / len(days)
        return sup_xy, sup_xy / sup_x, sup_xy / sup_x / sup_y
    rows = [["rule (→ bounce within 5 bars)", "support", "confidence", "lift"]]
    for name, cond in [("{RSI &lt; 30}", lambda d: d[0]), ("{Close &lt; lower BB}", lambda d: d[1]),
                       ("{RSI &lt; 30, Close &lt; lower BB}", lambda d: d[0] and d[1])]:
        s, c, l = rule(cond)
        rows.append([name, f"{s:.3f}", f"{c:.2f}", f"{l:.2f}"])
    pg.table(dx_, dy_ + 34, 440, rows, size=10, h=95, highlight_rows=(3,))
    base = sum(d[2] for d in days) / len(days)
    pg.bullets(dx_, dy_ + 140, 440, [
        f"1,000 simulated days; base bounce rate {base:.2f}",
        "Each indicator condition = an 'item', each day = a 'basket'",
        "The pair has the highest confidence & lift → candidate rule for a strategy (validate out-of-sample!)"],
        size=10)


def panel_summary(pg, x0, y0):
    pg.panel(x0, y0, 2000, 360, 7, "Summary: familiar algorithms, hidden jobs")
    rows = [["Algorithm", "Usual job", "Hidden use(s)", "Why it works"],
            ["★ k-Nearest Neighbors", "classification / regression", "pixel inpainting, KNN imputer, anomaly score, recommendation",
             "nearby points (pixels, rows) are similar"],
            ["K-Means", "clustering", "colour quantization, compression, cluster features, regime labels",
             "centroid = best single representative of a group"],
            ["PCA", "dimensionality reduction", "denoising, eigen-images, decorrelation, factor models",
             "signal concentrates in few directions, noise spreads over all"],
            ["Decision tree", "classification", "surrogate explanations, rule extraction, supervised binning",
             "axis-aligned rules are human readable"],
            ["Random forest", "classification / regression", "proximity imputation & outliers, uncertainty from tree spread",
             "rows sharing leaves are similar for the target"],
            ["Autoencoder (MLP)", "representation learning", "anomaly detection, denoising, compression",
             "it only learns to rebuild 'normal' patterns"],
            ["Logistic regression", "binary classification", "Platt calibration, meta-label bet sizing, propensity scores",
             "maps any score to a probability"],
            ["Linear regression / Z-score / ARMA", "prediction / standardization / forecasting",
             "hedge ratio, mean-reversion entries, residual alerts", "deviations from a fitted relation are informative"],
            ["Apriori", "market-basket rules", "co-firing indicator rules, log-pattern mining",
             "support / confidence / lift work on any set of events"]]
    pg.table(x0 + 20, y0 + 50, 1960, rows, size=11, h=215, highlight_rows=(1,))
    pg.bullets(x0 + 20, y0 + 285, 1960, [
        "Every hidden use must still be <b>validated</b>: hide known values (pixels, cells, labels) and measure how well the trick recovers them",
        "These tricks also appear as the green '💡 Also used for' box in the left column of every algorithm page"],
        size=11)


def kmeans(pts, k, rnd, iters=25):
    cents = [pts[rnd.randrange(len(pts))]]
    while len(cents) < k:                                   # k-means++ seeding
        d2 = [min(sum((p[i] - c[i]) ** 2 for i in range(3)) for c in cents) for p in pts]
        t, acc = rnd.random() * sum(d2), 0
        for p, d in zip(pts, d2):
            acc += d
            if acc >= t:
                cents.append(p); break
    for _ in range(iters):
        lab = [min(range(k), key=lambda j: sum((p[i] - cents[j][i]) ** 2 for i in range(3))) for p in pts]
        for j in range(k):
            mem = [p for p, l in zip(pts, lab) if l == j]
            if mem: cents[j] = tuple(sum(m[i] for m in mem) / len(mem) for i in range(3))
    return cents, lab


def panel_quantize(pg, img, x0, y0):
    pg.panel(x0, y0, 1000, 660, 3, "K-Means Colour Quantization: an image with only k colours",
             "Cluster the pixels in RGB space; replace every pixel by its centroid colour")
    N = len(img)
    pts = [img[r][c] for r in range(N) for c in range(N)]
    rnd = seeded(3)
    cell, gx, gy = 8, x0 + 30, y0 + 100
    draw_img(pg, gx, gy, img, cell, title="Original (24-bit)")
    rows = [["k", "bits / pixel", "size vs 24-bit", "colour MSE"]]
    res4 = None
    for i, k in enumerate((2, 4, 8)):
        cents, lab = kmeans(pts, k, rnd)
        q = [[cents[lab[r * N + c]] for c in range(N)] for r in range(N)]
        xx = gx + (i + 1) * (N * cell + 34)
        draw_img(pg, xx, gy, q, cell, title=f"k = {k}")
        for j, ce in enumerate(sorted(cents, key=lambda c: -sum(c))):
            pg.rect(xx + j * (N * cell / k), gy + N * cell + 8, N * cell / k, 16, fill=hexc(ce), stroke=BLACK)
        mse = sum(sum((pts[t][i2] - cents[lab[t]][i2]) ** 2 for i2 in range(3)) / 3 for t in range(len(pts))) / len(pts)
        bpp = math.log2(k)
        rows.append([k, f"{bpp:g}", f"{24 / bpp:.0f}× smaller", f"{mse:.0f}"])
        if k == 4: res4 = (cents, lab)
    pg.text("palette (centroids)", gx + N * cell + 34, gy + N * cell + 28, 3 * (N * cell + 34), 16, 10,
            italic=True, align="center", color="#555555")
    # RGB scatter (R vs B) for k = 4
    cents, lab = res4
    p = pg.axes(x0 + 80, y0 + 330, 360, 230, (0, 260), (0, 260), "R", "B", label_size=12, xticks=(0, 128, 255),
                yticks=(128, 255))
    seen = set()
    for t, pt in enumerate(pts):
        key = (round(pt[0] / 6), round(pt[2] / 6))
        if key in seen: continue
        seen.add(key)
        pg.dot(*p.P(pt[0], pt[2]), r=4, color=hexc(pt), stroke="#666666")
    for ce in cents:
        cx, cy = p.P(ce[0], ce[2])
        pg.rect(cx - 8, cy - 8, 16, 16, fill=hexc(ce), stroke=BLACK, sw=2.5)
    pg.text("pixels in colour space (R vs B); squares = k = 4 centroids", x0 + 60, y0 + 590, 420, 18, 10,
            italic=True, align="center", color="#555555")
    pg.table(x0 + 520, y0 + 340, 450, rows, size=11, h=110)
    pg.math(r"J=\sum_{i}\lVert \mathbf{x}_i-\boldsymbol{\mu}_{c(i)}\rVert^2,\quad \mathbf{x}_i=(R,G,B)",
            x0 + 520, y0 + 460, 450, 44, 15)
    pg.bullets(x0 + 520, y0 + 510, 460, [
        "Same K-Means as page 6 — the 'points' are pixels, the features are colours",
        "Storage: palette (k colours) + log₂k bits per pixel instead of 24",
        "Also used for vector quantization of audio, embeddings and GIF palettes"], size=10)


def pca_top(X, k):
    n, d = len(X), len(X[0])
    mu = [sum(row[j] for row in X) / n for j in range(d)]
    C = [[sum((X[i][a] - mu[a]) * (X[i][b] - mu[b]) for i in range(n)) / (n - 1) for b in range(d)] for a in range(d)]
    vecs, vals = [], []
    for t in range(k):
        v = [1.0 / math.sqrt(d) * (1 + 0.01 * j * (t + 1)) for j in range(d)]
        for _ in range(300):
            w = [sum(C[a][b] * v[b] for b in range(d)) for a in range(d)]
            nv = math.sqrt(sum(x * x for x in w)) or 1
            v = [x / nv for x in w]
        lam = sum(v[a] * sum(C[a][b] * v[b] for b in range(d)) for a in range(d))
        vecs.append(v); vals.append(lam)
        C = [[C[a][b] - lam * v[a] * v[b] for b in range(d)] for a in range(d)]
    return mu, vecs, vals


def recon(x, mu, vecs, k):
    z = [sum((x[j] - mu[j]) * v[j] for j in range(len(x))) for v in vecs[:k]]
    return [mu[j] + sum(z[t] * vecs[t][j] for t in range(k)) for j in range(len(x))]


def panel_pca(pg, x0, y0):
    pg.panel(x0, y0, 960, 660, 4, "PCA Denoising & Compression",
             "Keep the few directions with signal, drop the many with noise")
    rnd = seeded(11)
    D, M, sig = 24, 60, 0.45
    clean, noisy = [], []
    for _ in range(M):
        a, b = rnd.gauss(0, 1.2), rnd.gauss(0, 1.2)
        s = [a * math.sin(2 * math.pi * t / D) + b * math.cos(2 * math.pi * t / D) for t in range(D)]
        clean.append(s)
        noisy.append([v + rnd.gauss(0, sig) for v in s])
    mu, vecs, vals = pca_top(noisy, 8)
    mses = []
    for k in range(0, 9):
        e = sum(sum((recon(noisy[i], mu, vecs, k)[t] - clean[i][t]) ** 2 for t in range(D)) / D for i in range(M)) / M
        mses.append(e)
    raw = sum(sum((noisy[i][t] - clean[i][t]) ** 2 for t in range(D)) / D for i in range(M)) / M
    kbest = min(range(1, 9), key=lambda k: mses[k])
    ex = max(range(M), key=lambda i: sum(v * v for v in clean[i]))
    p = pg.axes(x0 + 60, y0 + 100, 480, 230, (0, D - 1), (-3.2, 3.2), "t", "x", label_size=12, origin=(0, 0))
    p.poly(list(enumerate(clean[ex])), color=BLACK, width=2)
    p.scatter(list(enumerate(noisy[ex])), color=GREY, r=3)
    p.poly(list(enumerate(recon(noisy[ex], mu, vecs, kbest))), color=RED, width=2, dashed=True)
    pg.html('<b>— clean</b> &nbsp; <font color="#808080">● noisy</font> &nbsp; '
            f'<font color="#FF0000"><b>- - PCA reconstruction (k = {kbest})</b></font>', x0 + 60, y0 + 345, 480, 20, 11)
    q = pg.axes(x0 + 640, y0 + 100, 270, 200, (-0.5, 8.5), (0, max(mses) * 1.1), "k", "MSE", label_size=12,
                xticks=tuple(range(0, 9)))
    q.bars([(k, mses[k]) for k in range(9)], 0.6, fill=LIGHTBLUE, stroke=BLUE)
    q.bars([(kbest, mses[kbest])], 0.6, fill=HL, stroke=BLACK)
    q.hline(raw, color=RED)
    q.text("raw noise", 8, raw, dy=-10, size=9, color=RED, w=70)
    pg.text(f"error vs clean signal: raw {raw:.3f} → k = {kbest}: {mses[kbest]:.3f} "
            f"({(1 - mses[kbest] / raw) * 100:.0f} % less noise)", x0 + 600, y0 + 320, 350, 34, 11, bold=True,
            align="center")
    pg.math(r"\hat{\mathbf{x}} = \boldsymbol{\mu} + \sum_{j=1}^{k} (\mathbf{v}_j^{\top}(\mathbf{x}-\boldsymbol{\mu}))\,\mathbf{v}_j",
            x0 + 40, y0 + 380, 480, 50, 16)
    ev = ", ".join(f"{v:.2f}" for v in vals[:5])
    pg.bullets(x0 + 40, y0 + 440, 900, [
        f"60 noisy signals of length 24; the true signals live in only 2 dimensions (sin & cos) → eigenvalues {ev} …",
        f"k too small loses signal, k too large keeps noise — the error is lowest at k = {kbest}",
        f"Compression: store {kbest} scores + {kbest} shared components instead of 24 values per signal "
        f"({24 / kbest:.0f}× smaller per signal)",
        "Same idea on images: eigen-faces / eigen-images; on returns: factor models (market, sector)"],
        head="Why it works:", size=11)

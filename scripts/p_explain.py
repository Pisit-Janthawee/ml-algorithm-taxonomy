"""Explainability (XAI) page - base: Permutation Importance.
One simulated trading dataset + a small pure-Python random forest so every number is real."""
import math
from dio import *

FEATS = ["EMA_gap", "RSI", "RSI_copy", "Volatility", "Day_of_week", "Noise"]
SHORT = ["EMA", "RSI", "RSIc", "Vol", "DoW", "Noise"]
_CACHE = {}


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def make_data(n, seed):
    r = seeded(seed)
    X, y = [], []
    for _ in range(n):
        ema = r.gauss(0, 1)
        rsi = r.uniform(20, 80)
        rc = rsi + r.gauss(0, 2)
        vol = r.uniform(0.5, 2.0)
        dow = r.randint(0, 4)
        nz = r.gauss(0, 1)
        z = 1.3 * ema * vol - 1.6 * math.tanh((rsi - 55) / 8) + 0.3
        y.append(1 if r.random() < sigmoid(z) else 0)
        X.append([ema, rsi, rc, vol, dow, nz])
    return X, y


def build_tree(X, y, idx, depth, r, imp, max_depth=5, min_leaf=6, mtry=3):
    n = len(idx)
    s = sum(y[i] for i in idx)
    m = s / n
    if depth == max_depth or n < 2 * min_leaf or s in (0, n):
        return ("leaf", m)
    base = s - s * s / n
    best = None
    for f in r.sample(range(len(FEATS)), mtry):
        vals = sorted(X[i][f] for i in idx)
        cands = sorted(set(vals[int(len(vals) * q / 13)] for q in range(1, 13)))
        for t in cands:
            L = [i for i in idx if X[i][f] <= t]
            nl = len(L)
            if nl < min_leaf or n - nl < min_leaf: continue
            sl = sum(y[i] for i in L)
            sr, nr = s - sl, n - nl
            gain = base - (sl - sl * sl / nl + sr - sr * sr / nr)
            if best is None or gain > best[0]:
                best = (gain, f, t)
    if best is None or best[0] <= 1e-9:
        return ("leaf", m)
    gain, f, t = best
    imp[f] += gain
    L = [i for i in idx if X[i][f] <= t]
    R = [i for i in idx if X[i][f] > t]
    return ("node", f, t, build_tree(X, y, L, depth + 1, r, imp), build_tree(X, y, R, depth + 1, r, imp))


def tree_pred(node, x):
    while node[0] == "node":
        node = node[3] if x[node[1]] <= node[2] else node[4]
    return node[1]


def model():
    """Train once; returns dict with data, forest, predict function and importances."""
    if _CACHE: return _CACHE
    X, y = make_data(450, 11)
    Xtr, ytr, Xte, yte = X[:300], y[:300], X[300:], y[300:]
    r = seeded(5)
    imp = [0.0] * len(FEATS)
    trees = []
    for _ in range(40):
        boot = [r.randrange(len(Xtr)) for _ in range(len(Xtr))]
        trees.append(build_tree(Xtr, ytr, boot, 0, r, imp))
    tot = sum(imp)
    pred = lambda x: sum(tree_pred(t, x) for t in trees) / len(trees)
    acc = lambda XX, yy: sum((pred(x) >= 0.5) == bool(t) for x, t in zip(XX, yy)) / len(yy)
    base = acc(Xte, yte)

    def perm(cols, reps=10, seed=3):
        rr = seeded(seed)
        drops = []
        for _ in range(reps):
            Xp = [row[:] for row in Xte]
            order = list(range(len(Xte)))
            rr.shuffle(order)
            for c in cols:
                for i, j in enumerate(order):
                    Xp[i][c] = Xte[j][c]
            drops.append(base - acc(Xp, yte))
        mu = sum(drops) / reps
        sd = math.sqrt(sum((d - mu) ** 2 for d in drops) / (reps - 1))
        return mu, sd

    pi = [perm([f]) for f in range(len(FEATS))]
    _CACHE.update(dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, trees=trees, pred=pred, acc=acc, base=base,
                       imp=[v / tot for v in imp], pi=pi, perm=perm,
                       train_acc=acc(Xtr, ytr)))
    return _CACHE


def hbars(pg, x, y, w, labels, vals, errs=None, vmax=None, color=BLUE, fill=LIGHTBLUE, row=30, fmt="{:.3f}",
          hl=None):
    """Horizontal bar chart drawn with rectangles (labels left, value right)."""
    vmax = vmax or max(max(vals), 1e-9) * 1.15
    lw = 90
    pg.line([(x + lw, y - 6), (x + lw, y + row * len(vals) + 2)])
    for k, (lab, v) in enumerate(zip(labels, vals)):
        yy = y + k * row
        pg.text(lab, x, yy, lw - 6, row - 6, 11, align="right", bold=(hl == k))
        bw = max(0.0, v) / vmax * (w - lw - 70)
        pg.rect(x + lw, yy + 3, max(bw, 1), row - 12, fill=HL if hl == k else fill, stroke=color)
        if errs:
            e = errs[k] / vmax * (w - lw - 70)
            cy = yy + row / 2 - 3
            pg.line([(x + lw + bw - e, cy), (x + lw + bw + e, cy)], width=1.2)
            pg.line([(x + lw + bw - e, cy - 4), (x + lw + bw - e, cy + 4)], width=1.2)
            pg.line([(x + lw + bw + e, cy - 4), (x + lw + bw + e, cy + 4)], width=1.2)
        pg.text(fmt.format(v), x + lw + bw + (errs[k] / vmax * (w - lw - 70) if errs else 0) + 6, yy, 70, row - 6,
                10, color="#555555")


def build(pg: Page, cat):
    M = model()

    # ------------------------------------------------------------------ ① why explain
    x0, y0 = 400, 240
    pg.panel(x0, y0, 620, 700, 1, "Why explain a model?", "Global (whole model) vs local (one prediction)")
    pg.html("<b>Features</b><br>EMA_gap · RSI · RSI_copy<br>Volatility · Day_of_week · Noise", x0 + 20, y0 + 80, 170,
            70, 11, align="center", valign="middle")
    bb = pg.rect(x0 + 230, y0 + 85, 150, 60, "Trained model<br>(black box)", fill="#333333", stroke="#333333",
                 color="#FFFFFF", bold=True, raw=True)
    pg.line([(x0 + 190, y0 + 115), (x0 + 230, y0 + 115)], arrow="classic")
    pg.line([(x0 + 380, y0 + 115), (x0 + 420, y0 + 115)], arrow="classic")
    pg.text("P(close ↑) = 0.71", x0 + 425, y0 + 100, 180, 30, 13, bold=True, color=GREEN)
    g = pg.rect(x0 + 40, y0 + 195, 250, 54, "<b>Global</b>: which features matter<br>for the model overall?",
                fill=LIGHTBLUE, stroke=BLUE, raw=True, size=11)
    l = pg.rect(x0 + 330, y0 + 195, 250, 54, "<b>Local</b>: why did the model say<br>0.71 for THIS bar?",
                fill="#FFE6CC", stroke=ORANGE, raw=True, size=11)
    pg.connect(bb, g, orth=True)
    pg.connect(bb, l, orth=True)
    # mini icons: global bars, local waterfall
    for k, v in enumerate([0.9, 0.6, 0.35, 0.15]):
        pg.rect(x0 + 60, y0 + 265 + k * 16, 180 * v, 11, fill=LIGHTBLUE, stroke=BLUE)
    wx, wy = x0 + 350, y0 + 265
    steps = [(0.50, 0.66, GREEN), (0.66, 0.60, RED), (0.60, 0.71, GREEN)]
    for k, (a, b, col) in enumerate(steps):
        xa, xb = wx + (a - 0.45) * 700, wx + (b - 0.45) * 700
        pg.rect(min(xa, xb), wy + k * 18, abs(xb - xa), 13, fill=col, stroke=col, opacity=70)
    pg.text("base 0.50 → +EMA → −RSI → +Vol = 0.71", wx - 10, wy + 56, 240, 16, 9, color="#555555")
    pg.table(x0 + 20, y0 + 345, 580, [
        ["", "Model-agnostic (any model)", "Model-specific"],
        ["<b>Global</b>", f'<span style="background:{HL}">Permutation importance</span>, PDP, global SHAP',
         "Coefficients, Gini importance"],
        ["<b>Local</b>", "SHAP (Kernel), LIME, ICE, counterfactuals", "Tree SHAP, attention, Grad-CAM"]],
        size=11, h=90)
    pg.text("In your EA pipeline (original diagram):", x0 + 20, y0 + 450, 580, 20, 12, bold=True)
    chain = ["Training", "Evaluation", "Explainability", "Feature importance → close-price direction",
             "Feature selection Tier 1 / 2 / 3"]
    cx, ids = x0 + 20, []
    for k, c in enumerate(chain):
        w = [70, 80, 95, 160, 120][k]
        ids.append(pg.rect(cx, y0 + 478, w, 44, c, size=9, fill=HL if k == 2 else "none", bold=(k == 2)))
        cx += w + 14
    for a, b in zip(ids, ids[1:]):
        pg.connect(a, b, orth=False)
    pg.bullets(x0 + 20, y0 + 545, 580, [
        "<b>Trust</b>: does the model use sensible signals (trend, momentum) or noise?",
        "<b>Debug</b>: a 'too important' feature often reveals <b>data leakage</b>",
        "<b>Feature selection</b>: drop features with ~0 importance → simpler, more robust model",
        "<b>Communicate</b>: explain each trade signal to stakeholders / regulators"], head="Why bother:", size=11)

    # ------------------------------------------------------------------ ② permutation importance
    x0 = 1060
    pg.panel(x0, y0, 1340, 700, 2, "★ Permutation Importance",
             f"Random forest (40 trees) on 300 training bars; scored on 150 held-out bars · baseline accuracy "
             f"s₀ = {M['base']:.3f}")
    flow = ["1. Fit model on TRAIN", "2. Score on TEST → s₀", "3. Shuffle one column j<br>(breaks its link to y)",
            "4. Re-score → s_j", "5. I_j = s₀ − s_j<br>(repeat K times, average)"]
    ids = []
    for k, f in enumerate(flow):
        ids.append(pg.rect(x0 + 20 + k * 262, y0 + 70, 232, 50, f, raw=True, size=11,
                           fill=HL if k == 2 else "none", bold=(k == 2)))
    for a, b in zip(ids, ids[1:]):
        pg.connect(a, b, orth=False)
    pg.line([(ids and x0 + 20 + 4 * 262 + 116, y0 + 120), (x0 + 20 + 4 * 262 + 116, y0 + 138),
             (x0 + 20 + 2 * 262 + 116, y0 + 138), (x0 + 20 + 2 * 262 + 116, y0 + 120)], dashed=True, color=GREY,
            arrow="classic")
    pg.text("next feature", x0 + 20 + 3 * 262 + 60, y0 + 138, 120, 16, 9, italic=True, color="#555555")
    # shuffle illustration
    sx, sy = x0 + 20, y0 + 175
    pg.text("Shuffling the RSI column of the test set", sx, sy, 380, 20, 12, bold=True)
    rows = [(r[0], r[1], t) for r, t in zip(M["Xte"][:5], M["yte"][:5])]
    order = [2, 0, 4, 1, 3]
    hdr = ["EMA_gap", "RSI", "y", "", "RSI (shuffled)"]
    for c, h in enumerate(hdr):
        if h: pg.text(h, sx + [0, 80, 150, 0, 260][c], sy + 26, 90, 18, 10, bold=True, align="center")
    cols = ["#CCE5FF", "#FFE6CC", "#D5E8D4", "#F8CECC", "#E1D5E7"]
    for i, (e, rs, t) in enumerate(rows):
        yy = sy + 46 + i * 26
        pg.rect(sx, yy, 90, 24, f"{e:+.2f}", size=10)
        pg.rect(sx + 90, yy, 70, 24, f"{rs:.0f}", size=10, fill=cols[i])
        pg.rect(sx + 160, yy, 40, 24, str(t), size=10)
        j = order[i]
        pg.rect(sx + 270, yy, 70, 24, f"{rows[j][1]:.0f}", size=10, fill=cols[j])
        pg.line([(sx + 205, sy + 58 + j * 26), (sx + 268, yy + 12)], color=GREY, width=0.75, arrow="classic")
    pg.text("Same values, random rows → RSI can no longer help predict y. If the score drops a lot, the model "
            "relied on RSI.", sx, sy + 182, 380, 46, 11, italic=True)
    # worked table
    order_f = sorted(range(len(FEATS)), key=lambda f: -M["pi"][f][0])
    trows = [["Feature", "s₀", "mean s_j", "I_j = s₀ − s_j", "± std", "rank"]]
    for rk, f in enumerate(order_f, 1):
        mu, sd = M["pi"][f]
        trows.append([FEATS[f], f"{M['base']:.3f}", f"{M['base'] - mu:.3f}", f"<b>{mu:+.3f}</b>", f"{sd:.3f}",
                      rk])
    pg.text("Worked result (K = 10 shuffles per feature)", x0 + 430, sy, 470, 20, 12, bold=True)
    pg.table(x0 + 430, sy + 26, 470, trows, size=11, h=200, highlight_rows=(1,))
    # bar chart
    pg.text("Importance (accuracy drop) ± std", x0 + 940, sy, 380, 20, 12, bold=True)
    hbars(pg, x0 + 930, sy + 40, 390, [FEATS[f] for f in order_f], [max(0, M["pi"][f][0]) for f in order_f],
          errs=[M["pi"][f][1] for f in order_f], hl=0)
    pg.text("≈ 0 → the model does not use it (Day_of_week, Noise)", x0 + 940, sy + 235, 380, 18, 10, italic=True,
            color=RED)
    # formula + pseudo-code
    pg.math(r"I_j = s_0 - \frac{1}{K}\sum_{k=1}^{K} s_{j,k}", x0 + 30, y0 + 445, 360, 50, 18)
    pg.note("score with column j shuffled (k-th repeat)", x0 + 60, y0 + 505, x0 + 300, y0 + 478, w=230)
    pg.note("baseline score on held-out data", x0 + 20, y0 + 420, x0 + 125, y0 + 462, w=200)
    pg.rect(x0 + 430, y0 + 430, 470, 150,
            "<b>Pseudo-code</b><br><font face='Courier New'>s0 = score(model, X_test, y_test)<br>"
            "for j in features:<br>&nbsp;&nbsp;for k in range(K):<br>&nbsp;&nbsp;&nbsp;&nbsp;Xp = X_test.copy(); "
            "shuffle(Xp[:, j])<br>&nbsp;&nbsp;&nbsp;&nbsp;s[j, k] = score(model, Xp, y_test)<br>"
            "&nbsp;&nbsp;I[j] = s0 − mean(s[j])</font>", fill="#F5F5F5", stroke="#999999", size=11, align="left",
            valign="top", raw=True)
    pg.bullets(x0 + 930, y0 + 440, 390, [
        "Model-agnostic: needs only predict() and a metric",
        "Use <b>held-out</b> data: on train it rewards overfitting",
        "No retraining (cheap) — K × p extra predictions",
        f"Train accuracy {M['train_acc']:.2f} vs test {M['base']:.2f}"], head="Notes:", size=11)
    pg.bullets(x0 + 20, y0 + 600, 1300, [
        "Metric choice changes the ranking: use the metric you care about (accuracy, AUC, log loss, Sharpe of the signal) · "
        "negative I_j = shuffling helped by chance → treat as 0"], size=11)
    build_part2(pg, M)
    build_part3(pg, M)

    # ------------------------------------------------------------ interpretation callouts
    pg.ask("Should I trust the signals this model uses?",
           "check the top features are sensible (trend, momentum); a surprising top feature often means leakage",
           862, 795, 646, 762, w=150)
    pg.ask("Which feature does the model rely on most?",
           "EMA_gap: shuffling it drops accuracy by 0.194; Noise and Day_of_week ≈ 0 → candidates to drop",
           1980, 864, 2358, 470, w=400)
    pg.ask("Is a high Gini importance proof that a feature matters?",
           "no: Noise gets 7 % of Gini importance (trees split on randomness) but 0 % permutation on held-out data",
           430, 1565, 435, 1434, w=600)
    pg.ask("Does the effect of the trend depend on volatility?",
           "yes: at Volatility 2.0 P(up) climbs 0.14 → 0.87 across EMA_gap, at 0.5 only 0.33 → 0.81 → interaction",
           1850, 1495, 2324, 1355, w=500)
    pg.ask("Why did the model say SELL for this bar?",
           "EMA_gap (−0.167), RSI_copy (−0.159) and RSI 72 (−0.145) push P(up) from 0.636 down to 0.193",
           1537, 1825, 1488, 1838, w=158)
    pg.ask("What is the smallest change that would flip the signal?",
           "RSI 72 → 53 (all else equal) lifts P(up) from 0.19 to 0.54: SELL becomes BUY",
           1825, 1915, 2099, 1912, w=250)


def heat(v):
    """0 -> blue, 0.5 -> white, 1 -> red."""
    v = max(0.0, min(1.0, v))
    if v < 0.5:
        t = v / 0.5
        r, g, b = int(51 + t * 204), int(51 + t * 204), 255
    else:
        t = (v - 0.5) / 0.5
        r, g, b = 255, int(255 - t * 204), int(255 - t * 204)
    return f"#{r:02X}{g:02X}{b:02X}"


def build_part2(pg, M):
    # ------------------------------------------------------------------ ③ pitfalls
    x0, y0 = 400, 980
    pg.panel(x0, y0, 760, 680, 3, "Pitfalls: correlated & noisy features",
             "Permutation can split importance; impurity importance can reward noise")
    Xte = M["Xte"]
    a = [r[1] for r in Xte]
    b = [r[2] for r in Xte]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    corr = sum((u - ma) * (v - mb) for u, v in zip(a, b)) / math.sqrt(
        sum((u - ma) ** 2 for u in a) * sum((v - mb) ** 2 for v in b))
    g_mu, g_sd = M["perm"]([1, 2])
    pg.text(f"(a) Correlated features: RSI and RSI_copy (r = {corr:.3f})", x0 + 20, y0 + 64, 720, 20, 12, bold=True)
    hbars(pg, x0 + 20, y0 + 96, 360, ["RSI alone", "RSI_copy alone", "both together"],
          [M["pi"][1][0], M["pi"][2][0], g_mu], errs=[M["pi"][1][1], M["pi"][2][1], g_sd], hl=2)
    pg.html(f"Shuffling only RSI hurts little: the model still reads the same information from RSI_copy. "
            f"Shuffled as a <b>group</b> the drop is <b>{g_mu:.3f}</b> — larger than either alone "
            f"({M['pi'][1][0]:.3f}, {M['pi'][2][0]:.3f}).", x0 + 400, y0 + 92, 340, 90, 11)
    # impurity vs permutation
    pg.text("(b) Impurity (Gini) importance vs permutation importance", x0 + 20, y0 + 210, 720, 20, 12, bold=True)
    pis = [max(0.0, m) for m, _ in M["pi"]]
    tot = sum(pis) or 1
    order_f = sorted(range(len(FEATS)), key=lambda f: -M["imp"][f])
    pg.text("Gini importance (train, share)", x0 + 20, y0 + 238, 340, 18, 11, italic=True, align="center")
    hbars(pg, x0 + 20, y0 + 262, 350, [FEATS[f] for f in order_f], [M["imp"][f] for f in order_f], vmax=0.6,
          color=ORANGE, fill="#FFE6CC", row=28, fmt="{:.2f}")
    pg.text("Permutation (test, share)", x0 + 390, y0 + 238, 340, 18, 11, italic=True, align="center")
    hbars(pg, x0 + 390, y0 + 262, 350, [FEATS[f] for f in order_f], [pis[f] / tot for f in order_f], vmax=0.6,
          row=28, fmt="{:.2f}")
    nz = FEATS.index("Noise")
    pg.html(f'<font color="#FF0000">Noise</font> gets <b>{M["imp"][nz]:.0%}</b> of the Gini importance (trees can '
            f'always find splits on a continuous random column) but only <b>{pis[nz] / tot:.0%}</b> of the '
            f'permutation importance on held-out data.', x0 + 20, y0 + 440, 720, 40, 11)
    pg.bullets(x0 + 20, y0 + 495, 720, [
        "<b>Group permutation</b>: shuffle correlated features together (cluster by |r| &gt; 0.8 first)",
        "<b>Drop-column importance</b>: retrain without feature j and compare scores (exact but p retrainings)",
        "<b>Conditional permutation</b>: shuffle j only within groups of similar correlated features",
        "Prefer permutation / SHAP on <b>test</b> data over impurity importance (biased to high-cardinality, "
        "computed on train)"], head="Fixes:", size=11)

    # ------------------------------------------------------------------ ④ PDP & ICE
    x0 = 1200
    pg.panel(x0, y0, 1200, 680, 4, "Partial Dependence (PDP) & ICE curves",
             "How the prediction changes when one feature is varied, everything else kept as in the data")
    pred = M["pred"]
    rows = Xte[:30]
    grid = list(range(20, 81, 3))
    ice = []
    for r0 in rows:
        line = []
        for v in grid:
            x = r0[:]
            x[1] = v
            x[2] = v + (r0[2] - r0[1])
            line.append(pred(x))
        ice.append(line)
    pdp = [sum(l[k] for l in ice) / len(ice) for k in range(len(grid))]
    p = pg.axes(x0 + 70, y0 + 100, 480, 300, (20, 80), (0, 1), "RSI", "P(up)", label_size=12,
                xticks=(20, 30, 50, 70, 80), yticks=(0.5, 1))
    for l in ice:
        p.poly(list(zip(grid, l)), color="#9999FF", width=0.6)
    p.poly(list(zip(grid, pdp)), color=RED, width=3)
    p.hline(0.5)
    p.vline(55, color=GREY)
    pg.note("PDP = average of all ICE lines", x0 + 380, y0 + 85, *p.P(66, pdp[grid.index(65)]), w=190, color=RED)
    pg.note("one ICE line = one bar", x0 + 90, y0 + 420, *p.P(26, ice[0][2]), w=150, color=BLUE)
    pg.text("overbought (RSI > 55) → model lowers P(up)", x0 + 300, y0 + 410, 260, 18, 10, italic=True)
    pg.math(r"\hat{f}_{S}(x_S) = \frac{1}{n}\sum_{i=1}^{n} \hat{f}\left(x_S,\ x_{C}^{(i)}\right)", x0 + 60, y0 + 450,
            480, 50, 16)
    pg.note("feature(s) we vary", x0 + 60, y0 + 515, x0 + 250, y0 + 488, w=120)
    pg.note("other features keep each row's real values", x0 + 300, y0 + 515, x0 + 400, y0 + 488, w=250)
    # 2-D PDP heatmap EMA x Vol
    ex = [-2 + 0.5 * k for k in range(9)]
    vx = [0.5 + 0.25 * k for k in range(7)]
    bg = Xte[:60]
    hx, hy, cw, ch = x0 + 700, y0 + 110, 50, 40
    pg.text("2-D PDP: EMA_gap × Volatility (interaction)", hx - 60, y0 + 70, 480, 20, 12, bold=True)
    for j, vv in enumerate(reversed(vx)):
        pg.text(f"{vv:.2f}", hx - 46, hy + j * ch + 12, 40, 16, 9, align="right")
        for i, ee in enumerate(ex):
            s = 0.0
            for r0 in bg:
                x = r0[:]
                x[0], x[3] = ee, vv
                s += pred(x)
            v = s / len(bg)
            pg.rect(hx + i * cw, hy + j * ch, cw, ch, f"{v:.2f}", fill=heat(v), stroke="#FFFFFF", size=9)
    for i, ee in enumerate(ex):
        pg.text(f"{ee:+.1f}", hx + i * cw, hy + 7 * ch + 2, cw, 16, 9, align="center")
    pg.text("EMA_gap →", hx + 150, hy + 7 * ch + 18, 160, 16, 11, bold=True, align="center")
    pg.text("Volatility ↑", hx - 70, hy - 22, 90, 16, 11, bold=True)
    for k in range(11):
        pg.rect(hx + 60 + k * 26, hy + 7 * ch + 46, 26, 12, fill=heat(k / 10), stroke="none")
    pg.text("P(up): 0", hx, hy + 7 * ch + 42, 60, 18, 9, align="right")
    pg.text("1", hx + 60 + 11 * 26 + 4, hy + 7 * ch + 42, 30, 18, 9)
    pg.html("The colour changes <b>faster along EMA_gap when volatility is high</b> → the effect of the trend "
            "depends on volatility (interaction the data was built with: 1.3·EMA·Vol).", hx - 60, y0 + 470, 500, 50,
            11)
    pg.bullets(x0 + 20, y0 + 560, 1160, [
        "PDP shows the <b>average</b> effect; spread-out or crossing ICE lines reveal interactions hidden by the average",
        "Here RSI and RSI_copy are moved together; moving one alone would create unrealistic rows (correlation pitfall ③)",
        "Use ALE plots when features are strongly correlated; PDP assumes the varied feature is independent of the others"],
        head="Reading:", size=11)


def shap_row(M, x, bg, players=None):
    """Exact interventional Shapley values. players = list of feature-index groups (default: each feature)."""
    players = players or [[f] for f in range(len(FEATS))]
    n = len(players)
    pred = M["pred"]
    v = {}
    for mask in range(1 << n):
        s = 0.0
        for b in bg:
            z = b[:]
            for k in range(n):
                if mask >> k & 1:
                    for f in players[k]:
                        z[f] = x[f]
            s += pred(z)
        v[mask] = s / len(bg)
    phi = []
    for k in range(n):
        tot = 0.0
        for mask in range(1 << n):
            if mask >> k & 1: continue
            sz = bin(mask).count("1")
            w = math.factorial(sz) * math.factorial(n - sz - 1) / math.factorial(n)
            tot += w * (v[mask | 1 << k] - v[mask])
        phi.append(tot)
    return phi, v


def with_rsi(x, val):
    z = x[:]
    z[1], z[2] = val, val + (x[2] - x[1])
    return z


def pick_row(M):
    """A bar with high RSI and a SELL prediction that flips to BUY when RSI is lowered (counterfactual)."""
    pred = M["pred"]
    best = None
    for x in M["Xte"]:
        if not (64 <= x[1] <= 80) or pred(x) >= 0.45: continue
        for v in range(int(x[1]), 19, -1):
            if pred(with_rsi(x, v)) >= 0.5:
                score = abs(x[1] - 72)
                if best is None or score < best[0]:
                    best = (score, x, v)
                break
    return best[1], best[2]


def build_part3(pg, M):
    import itertools
    pred = M["pred"]
    bg = M["Xtr"][:30]
    xs, cf = pick_row(M)
    # ------------------------------------------------------------------ ⑤ SHAP
    x0, y0 = 400, 1700
    pg.panel(x0, y0, 1300, 780, 5, "SHAP (SHapley Additive exPlanations)",
             "Each feature's fair share of the gap between the average prediction and this prediction")
    pg.math(r"\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}"
            r"\left[v(S\cup\{j\}) - v(S)\right]", x0 + 20, y0 + 64, 620, 56, 16)
    pg.note("weight: share of orderings with S before j", x0 + 30, y0 + 128, x0 + 280, y0 + 105, w=250)
    pg.note("marginal contribution of j to coalition S", x0 + 360, y0 + 128, x0 + 520, y0 + 105, w=260)
    pg.math(r"f(x) = \phi_0 + \sum_j \phi_j,\quad \phi_0 = E[f(X)]", x0 + 120, y0 + 150, 420, 34, 14)
    players = [[0], [1, 2], [3]]
    names = ["EMA", "RSI", "Vol"]
    phi3, v3 = shap_row(M, xs, bg, players)
    pg.text(f"Worked example: 3 players (EMA, RSI group, Vol) for one bar: EMA_gap = {xs[0]:+.2f}, "
            f"RSI = {xs[1]:.0f}, Vol = {xs[3]:.2f}", x0 + 20, y0 + 196, 640, 34, 12, bold=True)
    rows = [["Coalition S", "v(S) = avg prediction"]]
    for mask in range(8):
        lab = "{" + ", ".join(nm for k, nm in enumerate(names) if mask >> k & 1) + "}"
        rows.append([lab if mask else "∅ (base)", f"{v3[mask]:.3f}"])
    pg.table(x0 + 20, y0 + 236, 260, rows, size=10, h=190, highlight_rows=(1, 8))
    orows = [["Order of joining", "Δ EMA", "Δ RSI", "Δ Vol"]]
    for perm in itertools.permutations(range(3)):
        mask, contrib = 0, [0, 0, 0]
        for k in perm:
            contrib[k] = v3[mask | 1 << k] - v3[mask]
            mask |= 1 << k
        orows.append([" → ".join(names[k] for k in perm)] + [f"{c:+.3f}" for c in contrib])
    orows.append(["<b>φ = average</b>"] + [f"<b>{p:+.3f}</b>" for p in phi3])
    pg.table(x0 + 300, y0 + 236, 360, orows, size=10, h=190, highlight_rows=(7,))
    pg.html(f"Check (efficiency): base {v3[0]:.3f} + ({phi3[0]:+.3f}) + ({phi3[1]:+.3f}) + ({phi3[2]:+.3f}) = "
            f"<span style=\"background:{HL}\"><b>{v3[7]:.3f}</b></span> = v(all). Each coalition value averages "
            f"the model over 30 background bars for the features outside S.", x0 + 20, y0 + 440, 640, 44, 11)
    # waterfall, all 6 features
    phi, v = shap_row(M, xs, bg)
    base, fx = v[0], v[(1 << len(FEATS)) - 1]
    order_f = sorted(range(len(FEATS)), key=lambda f: -abs(phi[f]))
    wx, wy, ww = x0 + 720, y0 + 122, 520
    pg.text(f"Waterfall for this bar (all 6 features): {base:.3f} → {fx:.3f}", wx - 20, y0 + 64, 560, 20, 12,
            bold=True)
    sc = lambda q: wx + 130 + q * (ww - 150)
    pg.line([(sc(0), wy - 6), (sc(1), wy - 6)], color=GREY)
    for t in (0, 0.25, 0.5, 0.75, 1):
        pg.text(f"{t:g}", sc(t) - 15, wy - 22, 30, 14, 9, align="center", color="#555555")
    cur = base
    for k, f in enumerate(order_f):
        yy = wy + k * 32
        nxt = cur + phi[f]
        col = GREEN if phi[f] > 0 else RED
        val = f"{int(xs[f])}" if f == 4 else f"{xs[f]:.2f}"
        pg.text(f"{FEATS[f]} = {val}", wx, yy + 2, 125, 22, 10, align="right")
        pg.rect(min(sc(cur), sc(nxt)), yy + 4, max(abs(sc(nxt) - sc(cur)), 1), 20, fill=col, stroke=col,
                opacity=70)
        pg.text(f"{phi[f]:+.3f}", max(sc(cur), sc(nxt)) + 4, yy + 4, 60, 20, 10, color=col)
        cur = nxt
    pg.line([(sc(base), wy), (sc(base), wy + 6 * 32)], dashed=True, color=GREY)
    pg.line([(sc(fx), wy), (sc(fx), wy + 6 * 32 + 8)], color=BLACK, width=1.5)
    pg.text(f"E[f] = {base:.3f}", sc(base) - 50, wy + 6 * 32 + 4, 100, 16, 10, align="center")
    pg.text(f"f(x) = {fx:.3f} → SELL", sc(fx) - 70, wy + 6 * 32 + 20, 140, 16, 10, bold=True, color=RED,
            align="center")
    # beeswarm
    rows30 = M["Xte"][:30]
    allphi = [shap_row(M, x, bg[:20])[0] for x in rows30]
    lim = max(abs(q) for ph in allphi for q in ph) * 1.1
    bx, by, bw = x0 + 720, y0 + 470, 520
    pg.text("Summary (beeswarm): SHAP of 30 test bars, colour = feature value", bx - 20, y0 + 440, 560, 20, 12,
            bold=True)
    gorder = sorted(range(len(FEATS)), key=lambda f: -sum(abs(ph[f]) for ph in allphi))
    xc = lambda q: bx + 130 + (q + lim) / (2 * lim) * (bw - 150)
    pg.line([(xc(0), by - 6), (xc(0), by + 6 * 30)], color=GREY, dashed=True)
    rr = seeded(9)
    for k, f in enumerate(gorder):
        yy = by + k * 30 + 12
        pg.text(FEATS[f], bx, yy - 9, 125, 18, 10, align="right")
        vals = [row[f] for row in rows30]
        lo, hi = min(vals), max(vals)
        for row, ph in zip(rows30, allphi):
            t = (row[f] - lo) / (hi - lo) if hi > lo else 0.5
            pg.dot(xc(ph[f]), yy + rr.uniform(-7, 7), r=3, color=heat(t))
    pg.text("← lowers P(up)          SHAP value          raises P(up) →", bx + 130, by + 6 * 30 + 4, bw - 150,
            16, 9, align="center", color="#555555")
    pg.text("blue = low feature value · red = high feature value", bx + 130, by + 6 * 30 + 20, bw - 150, 16, 9,
            align="center", color="#555555")
    pg.bullets(x0 + 20, y0 + 500, 660, [
        "Guarantees: <b>efficiency</b> (sums to f(x) − E[f]), symmetry, dummy (unused feature → 0)",
        "Exact cost: 2<sup>p</sup> coalitions → KernelSHAP samples them; <b>TreeSHAP</b> is exact &amp; fast for trees",
        "Global importance = mean |φ_j| over many rows (top-to-bottom order of the beeswarm)",
        "High RSI (red dots) sits on the left → overbought lowers P(up); high EMA_gap on the right",
        "Correlated features share credit (RSI vs RSI_copy) — same caution as ③"], head="Notes:", size=11)

    # ------------------------------------------------------------------ ⑥ LIME & counterfactuals
    x0 = 1740
    pg.panel(x0, y0, 660, 780, 6, "LIME & Counterfactuals", "Explain one prediction with a simple local model")
    grid = list(range(20, 81, 2))
    curve = [pred(with_rsi(xs, g)) for g in grid]
    p = pg.axes(x0 + 70, y0 + 90, 520, 260, (20, 80), (0, 1), "RSI", "P(up)", label_size=12,
                xticks=(20, 30, 40, 50, 60, 70, 80), yticks=(0.5, 1))
    p.poly(list(zip(grid, curve)), color=BLUE, width=2)
    p.hline(0.5, color=GREY)
    rr = seeded(21)
    samples = []
    for _ in range(40):
        g = min(80, max(20, rr.gauss(xs[1], 12)))
        samples.append((g, pred(with_rsi(xs, g))))

    def wls(sigma):
        sw = sx = sy = sxx = sxy = 0.0
        for g, fy in samples:
            w = math.exp(-((g - xs[1]) / sigma) ** 2)
            sw += w; sx += w * g; sy += w * fy; sxx += w * g * g; sxy += w * g * fy
        b_ = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
        return (sy - b_ * sx) / sw, b_

    a, bslope = wls(12)
    _, narrow = wls(4)
    for g, fy in samples:
        w = math.exp(-((g - xs[1]) / 12) ** 2)
        pg.dot(*p.P(g, fy), r=2 + 4 * w, color=ORANGE)
    lo, hi = max(20, xs[1] - 18), min(80, xs[1] + 8)
    p.poly([(lo, a + bslope * lo), (hi, a + bslope * hi)], color=RED, width=2.5)
    p.scatter([(xs[1], pred(xs))], color=BLACK, r=5)
    cfp = pred(with_rsi(xs, cf))
    p.vline(cf, 0, 1, color=GREEN, width=1.5)
    p.scatter([(cf, cfp)], color=GREEN, r=5)
    pg.note(f"this bar: RSI = {xs[1]:.0f}, P(up) = {pred(xs):.2f} → SELL", x0 + 330, y0 + 372,
            *p.P(xs[1], pred(xs)), w=260)
    pg.note(f"LIME line (kernel σ = 12): slope {bslope * 10:+.3f} per 10 RSI points", x0 + 330, y0 + 56,
            *p.P(lo + 2, a + bslope * (lo + 2)), w=280, color=RED)
    pg.text(f"σ = 4 gives {narrow * 10:+.3f}: the answer depends on kernel width. LIME is local: near RSI "
            f"{xs[1]:.0f} small nudges barely matter; it misses the big drop below ≈ {cf + 2} that the "
            f"counterfactual finds.", x0 + 330, y0 + 74, 300, 52, 9, italic=True, color=RED)
    pg.text("orange = perturbed samples (bigger = closer = more weight) · blue = true model · green = counterfactual",
            x0 + 20, y0 + 405, 620, 18, 10, italic=True, color="#555555")
    pg.math(r"\xi(x) = \arg\min_{g \in G}\ \mathcal{L}(f, g, \pi_x) + \Omega(g)", x0 + 60, y0 + 436, 540, 40, 15)
    pg.text("fit a simple model g (linear) to f near x, weighted by proximity π_x, keeping g simple (Ω)", x0 + 20,
            y0 + 478, 620, 18, 10, italic=True, align="center")
    pg.html(f"<b>Counterfactual:</b> \"If RSI were <span style=\"background:{HL}\"><b>{cf}</b></span> instead of "
            f"<b>{xs[1]:.0f}</b> (everything else equal), P(up) = {cfp:.2f} → the signal flips "
            f"<font color=\"#FF0000\">SELL</font> → <font color=\"#009900\">BUY</font>.\"",
            x0 + 20, y0 + 505, 620, 40, 12)
    pg.table(x0 + 20, y0 + 560, 620, [
        ["Method", "Scope", "Agnostic?", "Cost", "Watch out"],
        ["Permutation", "global", "yes", "low", "correlated features"],
        ["PDP / ICE", "global / local", "yes", "medium", "unrealistic rows"],
        ["SHAP", "local → global", "yes (Tree: exact)", "high", "background choice"],
        ["LIME", "local", "yes", "medium", "unstable, kernel width"],
        ["Counterfactual", "local", "yes", "medium", "must be realistic"]], size=10, h=150, highlight_rows=(1,))

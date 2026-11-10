"""Model Evaluation & Validation page (base: Hold-out train / validation / test)."""
import math
from dio import *

TR, VA, TE, PU, EM = "#CCE5FF", "#FFE6CC", "#D5E8D4", "#BBBBBB", "#F8CECC"


def build(pg: Page, cat):
    r = seeded(21)

    # ------------------------------------------------------------ ① hold-out
    x0, y0 = 400, 240
    pg.panel(x0, y0, 900, 660, 1, "★ Hold-out: Train / Validation / Test",
             "Never grade a model on data it has seen")
    bx, by, bw = x0 + 40, y0 + 90, 820
    parts = [("Train 60 %", 0.6, TR, "fit the parameters (β, weights, splits)"),
             ("Validation 20 %", 0.2, VA, "tune hyperparameters, pick the model"),
             ("Test 20 %", 0.2, TE, "final report — touched ONCE")]
    xx = bx
    for lab, f, col, role in parts:
        w = bw * f
        pg.rect(xx, by, w, 50, lab, fill=col, stroke=BLACK, size=13, bold=True)
        pg.text(role, xx, by + 56, w, 34, 11, align="center", color="#333333")
        xx += w
    pg.text("Dataset (n rows, shuffled — or ordered by time for time series)", bx, by - 26, bw, 20, 12,
            italic=True, color="#555555")
    # flow
    fy = y0 + 240
    boxes = [("Train set", TR), ("Fit model", "#FFFFFF"), ("Validation score", VA), ("Best hyper-\nparameters", HL),
             ("Refit on train + val", "#FFFFFF"), ("Test score (report)", TE)]
    ids, fx = [], x0 + 30
    for lab, col in boxes:
        ids.append(pg.rect(fx, fy, 125, 54, lab.replace("\n", "<br>"), fill=col, size=11, bold=True, raw=True))
        fx += 142
    for a, b in zip(ids, ids[1:]):
        pg.connect(a, b, orth=False)
    pg.line([(x0 + 92 + 2 * 142, fy + 54), (x0 + 92 + 2 * 142, fy + 80), (x0 + 92 + 142, fy + 80),
             (x0 + 92 + 142, fy + 56)], color=BLUE, dashed=True, arrow="classic")
    pg.text("repeat for each candidate setting", x0 + 150, fy + 82, 260, 16, 10, italic=True, color=BLUE)
    pg.html(f'<span style="background:{HL}"><b>Golden rule:</b> the test set is used exactly once, at the very end. '
            'Every look at it to make a decision turns it into a validation set and the reported score becomes optimistic.</span>',
            x0 + 30, fy + 110, 840, 40, 12)
    # leakage examples
    ly = fy + 170
    pg.text("Data leakage — information from validation / test / future sneaks into training:", x0 + 30, ly, 840, 20,
            13, bold=True, color=RED)
    leaks = [("Scaler fit on the FULL dataset", "μ, σ include test rows", "fit scaler on train only (Pipeline)"),
             ("Feature uses the future", "e.g. tomorrow's close, centred moving average", "use only data ≤ t (shift lags)"),
             ("Random split of time series", "train on 2024, test on 2023", "split by time / purged CV"),
             ("Duplicate / same customer in both", "model memorises the entity", "group K-fold by entity")]
    for k, (a, b, fix) in enumerate(leaks):
        yy = ly + 30 + k * 34
        pg.text("✘", x0 + 30, yy, 24, 24, 18, bold=True, color=RED)
        pg.html(f"<b>{a}</b> — {b}", x0 + 58, yy + 2, 470, 22, 11)
        pg.html(f'<font color="#009900">✔ {fix}</font>', x0 + 540, yy + 2, 330, 22, 11)

    # ------------------------------------------------------------ ② CV schemes
    x0 = 1340
    pg.panel(x0, y0, 1060, 660, 2, "Cross-validation schemes", "Use more of the data for validation, more reliably")

    def fold_grid(gx, gy, title, rows, note, cell_w=34, k=5):
        pg.text(title, gx, gy, 5 * cell_w * 1.0 + 40, 20, 12, bold=True)
        for i, row in enumerate(rows):
            pg.text(f"{i + 1}", gx - 16, gy + 28 + i * 26, 14, 20, 10, color="#555555")
            for j, kind in enumerate(row):
                col = {"T": TR, "V": VA, "P": PU, "E": EM, "-": "#FFFFFF"}[kind]
                pg.rect(gx + j * cell_w, gy + 28 + i * 26, cell_w, 20, fill=col, stroke="#666666")
        pg.html(note, gx - 16, gy + 28 + len(rows) * 26 + 6, 5 * cell_w + 60, 70, 10)

    K = 5
    kfold = [["V" if j == i else "T" for j in range(K)] for i in range(K)]
    tss = [["T"] * (i + 1) + ["V"] + ["-"] * (K - i - 2) for i in range(K - 1)]
    purged = []
    for i in range(K):
        row = []
        for j in range(K):
            row.append("V" if j == i else "T")
        purged.append(row)
    gy = y0 + 70
    fold_grid(x0 + 50, gy, "K-Fold (K = 5)", kfold,
              "every row validated once; score = mean ± std of 5 folds<br><b>use:</b> i.i.d. data")
    # stratified: show class mix bars
    sx = x0 + 300
    fold_grid(sx, gy, "Stratified K-Fold", kfold,
              "each fold keeps the class ratio (e.g. 10 % positives)<br><b>use:</b> classification, imbalance")
    for i in range(K):
        pg.rect(sx + 5 * 34 + 6, gy + 28 + i * 26 + 4, 30, 12, fill=LIGHTBLUE, stroke="#666666")
        pg.rect(sx + 5 * 34 + 6, gy + 28 + i * 26 + 4, 3, 12, fill=RED, stroke="none")
    pg.text("10 % ■", sx + 5 * 34 + 2, gy + 6, 50, 18, 9, color=RED)
    fold_grid(x0 + 550, gy, "Time-Series Split", tss,
              "expanding window: train on the past only<br><b>use:</b> forecasting, trading signals")
    # purged: draw manually with purge + embargo cells
    px = x0 + 800
    pg.text("Purged K-Fold + Embargo", px, gy, 220, 20, 12, bold=True)
    cw = 34
    for i in range(K):
        pg.text(f"{i + 1}", px - 16, gy + 28 + i * 26, 14, 20, 10, color="#555555")
        for j in range(K):
            col = VA if j == i else TR
            pg.rect(px + j * cw, gy + 28 + i * 26, cw, 20, fill=col, stroke="#666666")
        if i > 0:
            pg.rect(px + i * cw - 8, gy + 28 + i * 26, 8, 20, fill=PU, stroke="none")
        if i < K - 1:
            pg.rect(px + (i + 1) * cw, gy + 28 + i * 26, 10, 20, fill=EM, stroke="none")
    pg.html("drop train rows whose labels overlap the test fold (purge) + a gap after it (embargo)<br>"
            "<b>use:</b> overlapping financial labels", px - 16, gy + 28 + K * 26 + 6, 230, 70, 10)
    # legend
    lgy = y0 + 330
    for k, (lab, col) in enumerate([("train", TR), ("validation / test fold", VA), ("purged", PU), ("embargo", EM)]):
        pg.rect(x0 + 50 + k * 190, lgy, 18, 14, fill=col, stroke="#666666")
        pg.text(lab, x0 + 74 + k * 190, lgy - 3, 160, 20, 11)
    # nested CV
    ny = y0 + 380
    pg.text("Nested CV — tune AND estimate without bias", x0 + 30, ny, 500, 22, 13, bold=True)
    pg.rect(x0 + 30, ny + 32, 480, 150, stroke="#666666", rounded=True)
    pg.text("Outer loop (K = 5): estimate generalization", x0 + 40, ny + 36, 400, 18, 11, bold=True)
    for j in range(5):
        pg.rect(x0 + 50 + j * 88, ny + 60, 84, 22, "test" if j == 2 else "train", fill=VA if j == 2 else TR, size=10)
    pg.rect(x0 + 50, ny + 100, 436, 70, stroke=BLUE, dashed=True)
    pg.text("Inner loop (K = 3) on the outer-train part: grid search", x0 + 58, ny + 103, 420, 18, 10, bold=True,
            color=BLUE)
    for j in range(3):
        pg.rect(x0 + 60 + j * 140, ny + 130, 130, 22, "val" if j == 0 else "train", fill=VA if j == 0 else TR,
                size=10)
    pg.bullets(x0 + 540, ny + 30, 500, [
        "Score = <b>mean ± std</b> over folds → report the uncertainty, not one number",
        "K = 5 or 10 is the usual bias / cost trade-off; LOOCV for tiny data",
        "Tuning on the same folds you report → optimistic; nested CV fixes it",
        "Group K-Fold when rows share an entity (customer, ticker, patient)",
        "Final model: refit on all training data with the chosen hyperparameters"], head="Notes:", size=11)

    # interpretation callouts (panels 1-2)
    pg.ask("How will the model perform on data it has never seen?",
           "only the Test 20 % score answers this — so read it once, at the very end (golden rule)",
           1125, 680, 1178, 626, w=165)
    pg.ask("How do I tune hyperparameters AND get an honest score?",
           "nested CV: inner folds pick the settings, outer test folds give the unbiased score",
           1370, 835, 1600, 815, w=540)

    _panel3(pg, r)
    _panel4(pg, r)
    _panel5(pg, r)
    _panel6(pg, r)


def _roc_pr(pos, neg):
    """Return ROC points, PR points, AUC, AP from score lists."""
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], reverse=True)
    P, N = len(pos), len(neg)
    tp = fp = 0
    roc, pr = [(0.0, 0.0)], []
    ap, prev_r = 0.0, 0.0
    for s, y in allv:
        if y: tp += 1
        else: fp += 1
        rec, prec = tp / P, tp / (tp + fp)
        roc.append((fp / N, rec))
        pr.append((rec, prec))
        if y:
            ap += (rec - prev_r) * prec
            prev_r = rec
    auc = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(roc, roc[1:]))
    return roc, pr, auc, ap


def _thin(pts, k=60):
    step = max(1, len(pts) // k)
    return pts[::step] + [pts[-1]]


def _panel5(pg, r):
    # ------------------------------------------------------------ ⑤ ROC / PR / calibration
    x0, y0 = 400, 1620
    pg.panel(x0, y0, 1100, 700, 5, "ROC, Precision-Recall & calibration",
             "Threshold-free evaluation · simulated scores: 100 positives, 900 negatives")
    pos_a = [r.gauss(1.6, 1) for _ in range(100)]
    neg = [r.gauss(0, 1) for _ in range(900)]
    pos_b = [r.gauss(0.8, 1) for _ in range(100)]
    roc_a, pr_a, auc_a, ap_a = _roc_pr(pos_a, neg)
    roc_b, pr_b, auc_b, ap_b = _roc_pr(pos_b, neg)
    # ROC
    p = pg.axes(x0 + 70, y0 + 90, 260, 260, (0, 1), (0, 1), "FPR", "TPR", label_size=11, xticks=(0.5, 1),
                yticks=(0.5, 1))
    p.poly([(0, 0), (1, 1)], color=GREY, dashed=True)
    p.poly(_thin(roc_a), color=BLUE, width=2)
    p.poly(_thin(roc_b), color=ORANGE, width=2)
    thr = 1.0
    fpr_t = sum(s > thr for s in neg) / 900
    tpr_t = sum(s > thr for s in pos_a) / 100
    p.scatter([(fpr_t, tpr_t)], color=RED, r=5)
    p.text(f"threshold = {thr:g}", fpr_t, tpr_t, dx=50, dy=10, size=9, color=RED, w=90)
    p.text("random (AUC 0.5)", 0.7, 0.6, size=9, color=GREY, w=100)
    pg.html(f'<font color="#3333FF"><b>Model A AUC = {auc_a:.3f}</b></font><br>'
            f'<font color="#FF8000"><b>Model B AUC = {auc_b:.3f}</b></font>', x0 + 190, y0 + 300, 150, 40, 10)
    pg.text("ROC curve", x0 + 70, y0 + 380, 260, 18, 12, bold=True, align="center")
    # PR
    q = pg.axes(x0 + 420, y0 + 90, 260, 260, (0, 1), (0, 1), "Recall", "Precision", label_size=11,
                xticks=(0.5, 1), yticks=(0.5, 1))
    q.hline(0.1, color=GREY)
    q.poly(_thin(pr_a), color=BLUE, width=2)
    q.poly(_thin(pr_b), color=ORANGE, width=2)
    q.text("baseline = 10 % positives", 0.7, 0.1, dy=-10, size=9, color=GREY, w=140)
    pg.html(f'<font color="#3333FF"><b>A: AP = {ap_a:.3f}</b></font><br>'
            f'<font color="#FF8000"><b>B: AP = {ap_b:.3f}</b></font>', x0 + 560, y0 + 100, 120, 40, 10)
    pg.text("Precision-Recall curve", x0 + 420, y0 + 380, 260, 18, 12, bold=True, align="center")
    pg.bullets(x0 + 30, y0 + 420, 680, [
        "ROC: TPR vs FPR over all thresholds; AUC = P(random positive scores higher than random negative)",
        "With many negatives, a small FPR still means many false alarms → ROC looks great, PR shows the truth",
        "Prefer <b>PR-AUC / AP</b> for rare events (crashes, fraud); ROC-AUC for balanced classes",
        "Pick the operating threshold afterwards from cost of FP vs FN (red point)"], head="Reading:", size=11)
    # calibration
    cx = x0 + 760
    n = 3000
    qs = [r.random() for _ in range(n)]
    ys = [1 if r.random() < qv else 0 for qv in qs]
    def logit(v): return math.log(max(v, 1e-6) / max(1 - v, 1e-6))
    over = [1 / (1 + math.exp(-3 * logit(qv))) for qv in qs]
    def bins(pred):
        out = []
        for b in range(10):
            idx = [i for i, v in enumerate(pred) if b / 10 <= v < (b + 1) / 10]
            if idx: out.append((sum(pred[i] for i in idx) / len(idx), sum(ys[i] for i in idx) / len(idx)))
        return out
    c = pg.axes(cx + 50, y0 + 90, 260, 260, (0, 1), (0, 1), r"\hat{p}", r"\text{obs.}", label_size=11,
                xticks=(0.5, 1), yticks=(0.5, 1))
    c.poly([(0, 0), (1, 1)], color=GREY, dashed=True)
    for pred, col in [(qs, GREEN), (over, RED)]:
        bp = bins(pred)
        c.poly(bp, color=col, width=2)
        c.scatter(bp, color=col, r=3.5)
    brier = lambda pred: sum((p_ - y) ** 2 for p_, y in zip(pred, ys)) / n
    ll = lambda pred: -sum(y * math.log(max(p_, 1e-9)) + (1 - y) * math.log(max(1 - p_, 1e-9))
                           for p_, y in zip(pred, ys)) / n
    pg.html(f'<font color="#009900"><b>calibrated</b>: Brier {brier(qs):.3f}, log loss {ll(qs):.3f}</font><br>'
            f'<font color="#FF0000"><b>over-confident</b>: Brier {brier(over):.3f}, log loss {ll(over):.3f}</font>',
            cx + 20, y0 + 380, 320, 40, 10)
    pg.text("Reliability diagram", cx + 50, y0 + 360, 260, 18, 12, bold=True, align="center")
    pg.math(r"Brier=\frac{1}{n}\sum(p_i-y_i)^2", cx + 20, y0 + 440, 300, 36, 13)
    pg.bullets(cx + 10, y0 + 490, 320, [
        "Points on the diagonal = '70 %' really means 70 %",
        "Same ranking (same AUC) can still be badly calibrated",
        "Fix: Platt scaling / isotonic regression on validation data",
        "Needed when probabilities set bet size"], size=10)
    pg.ask("Can I trust the predicted probabilities as real chances?",
           f"green hugs the diagonal (Brier {brier(qs):.3f}); red is over-confident ({brier(over):.3f}) "
           "→ recalibrate before bet sizing", 800, 2210, 1221, 2032, w=300)


def _polyfit(xs, ys, deg, lam=1e-6):
    m = deg + 1
    A = [[sum(x ** (i + j) for x in xs) + (lam if i == j else 0) for j in range(m)] for i in range(m)]
    b = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(m)]
    for i in range(m):
        piv = max(range(i, m), key=lambda k: abs(A[k][i]))
        A[i], A[piv], b[i], b[piv] = A[piv], A[i], b[piv], b[i]
        for k in range(i + 1, m):
            f = A[k][i] / A[i][i]
            for j in range(i, m): A[k][j] -= f * A[i][j]
            b[k] -= f * b[i]
    w = [0.0] * m
    for i in range(m - 1, -1, -1):
        w[i] = (b[i] - sum(A[i][j] * w[j] for j in range(i + 1, m))) / A[i][i]
    return w


def _pv(w, x): return sum(c * x ** i for i, c in enumerate(w))


def _panel6(pg, r):
    # ------------------------------------------------------------ ⑥ bias-variance & learning curves
    x0, y0 = 1540, 1620
    pg.panel(x0, y0, 860, 700, 6, "Bias–variance trade-off & learning curves",
             "Computed: polynomial fits (n = 30) to y = sin(2.5x) + 0.5x² + noise (σ = 0.3)")
    f = lambda x: math.sin(2.5 * x) + 0.5 * x * x
    grid = [-0.9 + 1.8 * i / 40 for i in range(41)]
    degs = list(range(0, 10))
    b2s, vs = [], []
    for d in degs:
        preds = []
        for _ in range(80):
            xs = [r.uniform(-1, 1) for _ in range(30)]
            ys = [f(x) + r.gauss(0, 0.3) for x in xs]
            w = _polyfit(xs, ys, d)
            preds.append([_pv(w, g) for g in grid])
        mean = [sum(p[i] for p in preds) / len(preds) for i in range(len(grid))]
        b2s.append(sum((mean[i] - f(g)) ** 2 for i, g in enumerate(grid)) / len(grid))
        vs.append(sum(sum((p[i] - mean[i]) ** 2 for p in preds) / len(preds) for i in range(len(grid))) / len(grid))
    noise = 0.09
    tot = [b + v + noise for b, v in zip(b2s, vs)]
    top = 0.8

    def upto(vals):  # keep points inside the plot; stop the line where it leaves (→ off the chart)
        pts = []
        for d, v in zip(degs, vals):
            if v > top:
                pts.append((d - 1 + (top - pts[-1][1]) / (v - pts[-1][1]) if pts else d, top)); break
            pts.append((d, v))
        return pts
    p = pg.axes(x0 + 70, y0 + 90, 340, 230, (0, 9), (0, top), "degree", "error", label_size=11,
                xticks=(0, 3, 6, 9))
    p.poly(upto(b2s), color=BLUE, width=2)
    p.poly(upto(vs), color=ORANGE, width=2)
    p.poly(upto(tot), color=BLACK, width=2.5)
    p.hline(noise, color=GREY)
    best = min(range(len(tot)), key=lambda i: tot[i])
    p.scatter([(best, tot[best])], color=RED, r=5)
    p.text(f"best degree = {best}", best, tot[best], dy=-16, size=10, color=RED, w=100)
    pg.ask("How complex should the model be?",
           f"degree {best} gives the lowest total error; left of it = underfit (bias), right = overfit (variance)",
           1660, 1745, *p.P(best, tot[best]), w=175)
    pg.html('<font color="#3333FF">— bias²</font><br><font color="#FF8000">— variance</font><br>'
            '<b>— total</b><br><font color="#808080">- - noise σ²</font>', x0 + 330, y0 + 90, 90, 70, 10)
    pg.math(r"E[(y-\hat{f})^2]=Bias^2+Variance+\sigma^2", x0 + 40, y0 + 335, 400, 36, 14)
    pg.text("← underfit (high bias)          overfit (high variance) →", x0 + 70, y0 + 372, 340, 18, 10,
            italic=True, align="center")
    # learning curves
    sizes = [12, 20, 35, 60, 100, 160]
    for k, (d, title) in enumerate([(1, "degree 1: high bias"), (9, "degree 9: high variance")]):
        tr, va = [], []
        for n in sizes:
            te_, tv_ = 0.0, 0.0
            for _ in range(12):
                xs = [r.uniform(-1, 1) for _ in range(n)]
                ys = [f(x) + r.gauss(0, 0.3) for x in xs]
                w = _polyfit(xs, ys, d)
                te_ += sum((_pv(w, x) - y) ** 2 for x, y in zip(xs, ys)) / n
                xv = [r.uniform(-1, 1) for _ in range(200)]
                tv_ += sum((_pv(w, x) - f(x) - r.gauss(0, 0.3)) ** 2 for x in xv) / 200
            tr.append(te_ / 12); va.append(min(tv_ / 12, 0.6))
        gx = x0 + 470
        gy = y0 + 80 + k * 160
        g = pg.axes(gx, gy, 300, 110, (0, 170), (0, 0.6), "n", "error", label_size=10, xticks=(0, 80, 160))
        g.poly(list(zip(sizes, tr)), color=BLUE, width=2)
        g.poly(list(zip(sizes, va)), color=RED, width=2)
        g.hline(noise, color=GREY)
        pg.text(title, gx + 40, gy - 14, 260, 16, 11, bold=True, align="center")
    pg.html('<font color="#3333FF">— train error</font>  <font color="#FF0000">— validation error</font>',
            x0 + 500, y0 + 400, 300, 18, 10)
    pg.table(x0 + 30, y0 + 440, 800, [
        ["Symptom", "Diagnosis", "Fixes"],
        ["train high ≈ validation high", "high bias (underfit)", "more features / complexity, less regularization, boosting"],
        ["train low ≪ validation high", "high variance (overfit)", "more data, regularization, simpler model, bagging, early stopping"],
        ["gap shrinks as n grows", "variance — data helps", "collect more data"],
        ["both flat & high as n grows", "bias — data won't help", "change model / features"]], size=11, h=150)
    pg.text("Learning curves tell you whether to spend effort on more data or on a better model.", x0 + 30,
            y0 + 610, 800, 20, 11, italic=True, color=RED)


def _panel3(pg, r):
    # ------------------------------------------------------------ ③ regression metrics
    x0, y0 = 400, 940
    pg.panel(x0, y0, 900, 640, 3, "Regression metrics (worked example)", "6 predictions of next-day price")
    ys = [100, 102, 98, 105, 110, 95]
    yh = [101, 100, 99, 103, 104, 97]
    n = len(ys)
    e = [a - b for a, b in zip(ys, yh)]
    mae = sum(abs(v) for v in e) / n
    mse = sum(v * v for v in e) / n
    rmse = math.sqrt(mse)
    ybar = sum(ys) / n
    sst = sum((v - ybar) ** 2 for v in ys)
    r2 = 1 - sum(v * v for v in e) / sst
    mape = 100 * sum(abs(v) / a for v, a in zip(e, ys)) / n
    rows = [[r"\(y\)", r"\(\hat{y}\)", r"\(e=y-\hat{y}\)", r"\(|e|\)", r"\(e^2\)", r"\(|e|/y\) %"]]
    for a, b, v in zip(ys, yh, e):
        rows.append([a, b, f"{v:+d}", abs(v), v * v, f"{100 * abs(v) / a:.2f}"])
    rows.append(["", "", "mean →", f"<b>{mae:.2f}</b>", f"<b>{mse:.2f}</b>", f"<b>{mape:.2f}</b>"])
    pg.table(x0 + 30, y0 + 70, 420, rows, size=11, highlight_rows=(7,))
    mets = [(r"MAE=\frac{1}{n}\sum|y_i-\hat{y}_i|", f"= {mae:.2f}"),
            (r"MSE=\frac{1}{n}\sum(y_i-\hat{y}_i)^2", f"= {mse:.2f}"),
            (r"RMSE=\sqrt{MSE}", f"= {rmse:.2f}"),
            (r"R^2=1-\frac{SS_{res}}{SS_{tot}}", f"= 1 − {sum(v*v for v in e)}/{sst:.1f} = {r2:.3f}"),
            (r"MAPE=\frac{100}{n}\sum\frac{|y_i-\hat{y}_i|}{|y_i|}", f"= {mape:.2f} %")]
    for k, (tex, val) in enumerate(mets):
        pg.math(tex, x0 + 470, y0 + 70 + k * 44, 260, 40, 13, align="left")
        pg.text(val, x0 + 735, y0 + 80 + k * 44, 150, 20, 12, bold=True, color=BLUE)
    # residual plot
    p = pg.axes(x0 + 60, y0 + 330, 360, 140, (0.5, 6.5), (-7, 7), "i", "e", label_size=11, origin=(0.5, 0))
    p.bars([(i + 1, v) for i, v in enumerate(e)], 0.5, fill=LIGHTRED, stroke=RED, labels=True)
    pg.text("residuals: look for patterns, not just size", x0 + 60, y0 + 485, 360, 18, 10, italic=True,
            color="#555555", align="center")
    # outlier effect
    e2 = e[:-1] + [e[-1] + 20]
    mae2 = sum(abs(v) for v in e2) / n
    rmse2 = math.sqrt(sum(v * v for v in e2) / n)
    pg.text("One big miss (error +22 instead of −2):", x0 + 470, y0 + 330, 410, 20, 12, bold=True)
    pg.table(x0 + 470, y0 + 356, 400, [
        ["", "MAE", "RMSE"], ["original", f"{mae:.2f}", f"{rmse:.2f}"],
        ["with outlier", f"{mae2:.2f} (×{mae2 / mae:.1f})", f'<span style="background:{HL}">{rmse2:.2f} (×{rmse2 / rmse:.1f})</span>']],
        size=11, h=80)
    pg.bullets(x0 + 470, y0 + 450, 410, [
        "RMSE squares errors → punishes large misses; MAE treats all equally",
        "R² compares with always predicting ȳ (can be negative!)",
        "MAPE explodes when y ≈ 0 (returns!) → use MAE / RMSE or sMAPE"], head="Choose:", size=11)
    pg.text("For trading, also judge by direction accuracy and P&L, not only price error.", x0 + 30, y0 + 600, 840,
            20, 11, italic=True, color=RED)
    pg.ask("Which prediction hurts the error score the most?",
           "row 5 (error +6): 36 of the 50 squared-error units → it alone drives RMSE up",
           420, 1470, 729, 1282, w=280)


def _panel4(pg, r):
    # ------------------------------------------------------------ ④ classification metrics
    x0, y0 = 1340, 940
    pg.panel(x0, y0, 1060, 640, 4, "Classification metrics from the confusion matrix",
             "Example: 200 days, 60 real 'up' days")
    TP, FP, FN, TN = 40, 10, 20, 130
    _mcc = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    pg.ask("Is the model good even though 'up' days are the minority?",
           f"MCC = {_mcc:.2f} (−1…+1): solid; unlike accuracy {(TP + TN) / (TP + FP + FN + TN):.2f} it is not "
           "inflated by the many 'down' days", 1900, 1225, 2041, 1190, w=470)
    cx, cy, cs = x0 + 110, y0 + 110, 110
    pg.text("Actual", cx, cy - 46, 2 * cs, 20, 12, bold=True, align="center")
    pg.text("Up (1)", cx, cy - 24, cs, 20, 11, align="center")
    pg.text("Down (0)", cx + cs, cy - 24, cs, 20, 11, align="center")
    pg.text("Predicted", cx - 100, cy + cs - 10, 60, 20, 12, bold=True)
    pg.text("Up (1)", cx - 50, cy + cs / 2 - 10, 48, 20, 11, align="right")
    pg.text("Down (0)", cx - 60, cy + cs * 1.5 - 10, 58, 20, 11, align="right")
    for (i, j, lab, v, col) in [(0, 0, "TP", TP, "#D5E8D4"), (0, 1, "FP", FP, "#F8CECC"),
                                (1, 0, "FN", FN, "#F8CECC"), (1, 1, "TN", TN, "#D5E8D4")]:
        pg.rect(cx + j * cs, cy + i * cs, cs, cs, f"<b>{lab}</b><br><font style='font-size:20px'>{v}</font>",
                fill=col, raw=True, size=12)
    pg.text("Type I error", cx + cs, cy + cs - 18, cs, 16, 9, align="center", color=RED)
    pg.text("Type II error", cx, cy + 2 * cs - 18, cs, 16, 9, align="center", color=RED)
    acc = (TP + TN) / 200
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    spec = TN / (TN + FP)
    f1 = 2 * prec * rec / (prec + rec)
    mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    rows = [["Metric", "Formula", "Value", "Question it answers"],
            ["Accuracy", r"\(\frac{TP+TN}{N}\)", f"{acc:.3f}", "how often correct overall?"],
            ["Precision", r"\(\frac{TP}{TP+FP}\)", f"{prec:.3f}", "when I say 'up', am I right?"],
            ["Recall (TPR)", r"\(\frac{TP}{TP+FN}\)", f"{rec:.3f}", "how many real 'ups' did I catch?"],
            ["Specificity", r"\(\frac{TN}{TN+FP}\)", f"{spec:.3f}", "how many 'downs' did I reject?"],
            ["F1", r"\(\frac{2PR}{P+R}\)", f"{f1:.3f}", "balance of precision & recall"],
            ["MCC", r"\(\frac{TP\,TN-FP\,FN}{\sqrt{\cdots}}\)", f"{mcc:.3f}", "correlation, robust to imbalance"]]
    pg.table(x0 + 380, y0 + 70, 650, rows, size=11, h=240)
    # accuracy paradox
    ay = y0 + 380
    pg.text("Accuracy paradox (imbalanced data: 1 000 days, 20 crashes)", x0 + 30, ay, 700, 22, 13, bold=True,
            color=RED)
    pg.table(x0 + 30, ay + 30, 600, [
        ["Model", "Accuracy", "Recall (crash)", "Precision", "F1"],
        ["'Never crash' (always 0)", '<span style="background:#FFFF00">98.0 %</span>', "0.00", "—", "0.00"],
        ["Real model (TP 15, FP 45, FN 5)", "95.0 %", "0.75", "0.25", "0.38"]], size=11, h=80)
    pg.bullets(x0 + 650, ay + 26, 390, [
        "High accuracy can be worthless on rare events",
        "Use precision / recall / F1, PR-AUC or MCC instead",
        "Choose the threshold by the cost of FP vs FN (panel ⑤)"], head="Lesson:", size=11)
    pg.bullets(x0 + 30, ay + 130, 1000, [
        "Multi-class: macro average (each class equal) vs weighted / micro average (each sample equal)",
        "Trading: precision ≈ win rate of taken trades; recall ≈ share of opportunities captured"],
        head="Notes:", size=11)

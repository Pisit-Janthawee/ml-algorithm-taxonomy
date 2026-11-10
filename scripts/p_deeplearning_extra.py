"""MLP page extras: data illustration (tabular → scaled → mini-batch → layers → probabilities)
and hyperparameter search (LR range test, early stopping, search space)."""
import math
from dio import *

FEATS = ["RSI", "EMA gap %", "Volume (M)", "Return t−1 %"]


def make_data(n=160, seed=5, noise=0.6, flip=0.0):
    r = seeded(seed)
    X, raw, y = [], [], []
    for _ in range(n):
        z = [r.gauss(0, 1) for _ in range(4)]
        s = 1.5 * z[0] - 1.0 * z[1] + 0.9 * z[0] * z[2] + r.gauss(0, noise)
        lab = 1 if s > 0 else 0
        if flip and r.random() < flip: lab = 1 - lab
        raw.append([50 + 15 * z[0], 0.5 * z[1], 1.0 + 0.3 * z[2], 1.0 * z[3]])
        y.append(lab)
    # standardize from the raw table (what a real pipeline does)
    mu = [sum(row[j] for row in raw) / n for j in range(4)]
    sd = [math.sqrt(sum((row[j] - mu[j]) ** 2 for row in raw) / n) for j in range(4)]
    X = [[(row[j] - mu[j]) / sd[j] for j in range(4)] for row in raw]
    return raw, X, y, mu, sd


def sig(v):
    return 1 / (1 + math.exp(-max(-30, min(30, v))))


class MLP:
    """4 → H (tanh) → 1 (sigmoid), trained with mini-batch gradient descent on log loss."""

    def __init__(self, H=6, seed=1):
        r = seeded(seed)
        self.H = H
        self.W1 = [[r.gauss(0, 1 / math.sqrt(4)) for _ in range(4)] for _ in range(H)]
        self.b1 = [0.0] * H
        self.W2 = [r.gauss(0, 1 / math.sqrt(H)) for _ in range(H)]
        self.b2 = 0.0

    def forward(self, x):
        h = [math.tanh(sum(w * xi for w, xi in zip(self.W1[k], x)) + self.b1[k]) for k in range(self.H)]
        return h, sig(sum(w * hk for w, hk in zip(self.W2, h)) + self.b2)

    def loss(self, X, y):
        tot = 0
        for x, t in zip(X, y):
            p = min(max(self.forward(x)[1], 1e-7), 1 - 1e-7)
            tot -= t * math.log(p) + (1 - t) * math.log(1 - p)
        return tot / len(X)

    def step(self, Xb, yb, lr):
        H = self.H
        gW1 = [[0.0] * 4 for _ in range(H)]; gb1 = [0.0] * H; gW2 = [0.0] * H; gb2 = 0.0
        for x, t in zip(Xb, yb):
            h, p = self.forward(x)
            d2 = p - t
            gb2 += d2
            for k in range(H):
                gW2[k] += d2 * h[k]
                d1 = d2 * self.W2[k] * (1 - h[k] ** 2)
                gb1[k] += d1
                for j in range(4):
                    gW1[k][j] += d1 * x[j]
        m = len(Xb)
        for k in range(H):
            self.W2[k] -= lr * gW2[k] / m
            self.b1[k] -= lr * gb1[k] / m
            for j in range(4):
                self.W1[k][j] -= lr * gW1[k][j] / m
        self.b2 -= lr * gb2 / m


def build(pg: Page, cat):
    raw, X, y, mu, sd = make_data()
    net = MLP(H=6, seed=1)
    for ep in range(60):
        for b in range(0, len(X), 16):
            net.step(X[b:b + 16], y[b:b + 16], 0.3)
    acc = sum((net.forward(x)[1] > 0.5) == bool(t) for x, t in zip(X, y)) / len(X)

    # ---------------------------------------------------------------- ⑦ data illustration
    x0, y0 = 400, pg.max_y() + 60
    pg.panel(x0, y0, 2000, 700, 7, "Data Illustration: one mini-batch flowing through the MLP",
             f"Tabular trading features → target = next bar up (1) / down (0) · n = {len(X)} rows · "
             f"network 4 → 6 (tanh) → 1 (sigmoid) · trained 60 epochs, batch 16 → train accuracy {acc:.0%}")
    B = 4
    # ① raw table
    pg.text("1. Raw tabular data (first 4 rows = one mini-batch)", x0 + 20, y0 + 64, 520, 20, 12, bold=True)
    rows = [["#"] + FEATS + ["y"]]
    for i in range(B):
        rows.append([i + 1, f"{raw[i][0]:.1f}", f"{raw[i][1]:+.2f}", f"{raw[i][2]:.2f}", f"{raw[i][3]:+.2f}", y[i]])
    rows.append(["μ", f"{mu[0]:.1f}", f"{mu[1]:+.2f}", f"{mu[2]:.2f}", f"{mu[3]:+.2f}", ""])
    rows.append(["σ", f"{sd[0]:.1f}", f"{sd[1]:.2f}", f"{sd[2]:.2f}", f"{sd[3]:.2f}", ""])
    pg.table(x0 + 20, y0 + 90, 520, rows, size=11, highlight_rows=(5, 6))
    pg.note("very different scales → standardize", x0 + 300, y0 + 290, x0 + 330, y0 + 255, w=230, color=RED)
    # ② scaled batch matrix
    pg.text("2. Standardized mini-batch  \\(X_b\\)  (4 × 4)", x0 + 20, y0 + 330, 520, 20, 12, bold=True)
    rows = [["#"] + [f"z({f})" for f in FEATS]]
    for i in range(B):
        rows.append([i + 1] + [f"{v:+.2f}" for v in X[i]])
    pg.table(x0 + 20, y0 + 356, 520, rows, size=11)
    pg.math("z = \\frac{x-\\mu}{\\sigma}", x0 + 180, y0 + 490, 200, 40, 15)
    pg.text("μ, σ computed on TRAIN data only, reused for validation / test", x0 + 20, y0 + 530, 520, 18, 10,
            italic=True, color="#555555")

    # ③ layer shapes diagram
    lx = x0 + 600
    pg.text("3. Shapes through the layers", lx, y0 + 64, 560, 20, 12, bold=True)
    blocks = [("X_b", "4 × 4", LIGHTBLUE), ("W^{(1)T}", "4 × 6", "#FFFFFF"), ("H = tanh(XW+b)", "4 × 6", "#E1D5E7"),
              ("W^{(2)T}", "6 × 1", "#FFFFFF"), ("\\hat{p} = \\sigma(HW+b)", "4 × 1", "#FFF2CC")]
    bx = lx
    ids = []
    for name, shp, fill in blocks:
        w = 150 if "=" in name else 90
        ids.append(pg.rect(bx, y0 + 100, w, 60, f"$${name}$$<br><font style='font-size:10px'>{shp}</font>",
                           fill=fill, size=12, raw=True))
        bx += w + 22
    for a, b in zip(ids, ids[1:]):
        pg.connect(a, b, orth=False)
    pg.text("batch × features  →  batch × hidden  →  batch × 1", lx, y0 + 168, 560, 18, 11, align="center",
            color="#555555")
    # hidden activations of the batch
    pg.text("Hidden activations H (4 rows × 6 neurons)", lx, y0 + 200, 560, 20, 12, bold=True)
    rows = [["#"] + [f"h{k + 1}" for k in range(6)]]
    outs = []
    for i in range(B):
        h, p = net.forward(X[i])
        outs.append(p)
        rows.append([i + 1] + [f"{v:+.2f}" for v in h])
    pg.table(lx, y0 + 226, 560, rows, size=11)
    # small network drawing with the batch
    nx = lx + 40
    layers = [4, 6, 1]
    pos = []
    for li, cnt in enumerate(layers):
        col = []
        for k in range(cnt):
            cx = nx + li * 210
            cy = y0 + 470 + (k - (cnt - 1) / 2) * 34
            col.append((cx, cy))
        pos.append(col)
    for a, b in zip(pos, pos[1:]):
        for p1 in a:
            for p2 in b:
                pg.line([p1, p2], color="#BBBBBB", width=0.5)
    for li, col in enumerate(pos):
        for (cx, cy) in col:
            pg.ellipse(cx - 11, cy - 11, 22, 22, fill=[LIGHTBLUE, "#E1D5E7", "#FFF2CC"][li])
    for k, f in enumerate(FEATS):
        pg.text(f, nx - 120, pos[0][k][1] - 8, 105, 16, 9, align="right")
    pg.text("input (4)", nx - 30, y0 + 600, 60, 16, 10, align="center")
    pg.text("hidden (6)", nx + 180, y0 + 600, 60, 16, 10, align="center")
    pg.text("output (1)", nx + 390, y0 + 600, 60, 16, 10, align="center")
    pg.text(f"parameters: 4·6 + 6 + 6·1 + 1 = 37", lx, y0 + 625, 560, 18, 11, align="center", italic=True)

    # ④ outputs
    ox = x0 + 1240
    pg.text("4. Output probabilities for the mini-batch", ox, y0 + 64, 720, 20, 12, bold=True)
    rows = [["#", "\\(\\hat{p}=P(y=1)\\)", "prediction (≥ 0.5)", "true y", "log-loss term"]]
    hl = ()
    tot = 0
    for i in range(B):
        p = outs[i]
        pred = int(p >= 0.5)
        l = -(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))
        tot += l
        if pred != y[i]: hl += (i + 1,)
        rows.append([i + 1, f"{p:.3f}", "↑ up (1)" if pred else "↓ down (0)", y[i], f"{l:.3f}"])
    rows.append(["", "", "", "batch loss", f"<b>{tot / B:.3f}</b>"])
    pg.table(ox, y0 + 90, 720, rows, size=12, highlight_rows=hl)
    for i in range(B):
        bw = outs[i] * 300
        yy = y0 + 270 + i * 34
        pg.text(f"row {i + 1}", ox, yy, 60, 24, 11)
        pg.rect(ox + 70, yy + 2, 300, 20, stroke="#999999")
        pg.rect(ox + 70, yy + 2, bw, 20, fill=GREEN if outs[i] >= 0.5 else RED, stroke="none", opacity=60)
        pg.text(f"{outs[i]:.2f}", ox + 380, yy, 60, 24, 11)
    pg.line([(ox + 220, y0 + 262), (ox + 220, y0 + 410)], dashed=True, color=BLACK)
    pg.text("threshold 0.5", ox + 180, y0 + 412, 80, 16, 10, align="center")
    wr = hl[0] if hl else max(range(1, B + 1), key=lambda i: abs(outs[i - 1] - y[i - 1]))
    pw = outs[wr - 1]
    lw = -(y[wr - 1] * math.log(pw) + (1 - y[wr - 1]) * math.log(1 - pw))
    pg.ask("Which row did the model get wrong?",
           f"row {wr}: p̂ = {pw:.2f} → predicted {'up' if pw >= 0.5 else 'down'}, truth "
           f"{'up' if y[wr - 1] else 'down'} → largest loss ({lw:.2f}), biggest gradient push",
           ox + 455, y0 + 262, ox + 70 + pw * 300 + 2, y0 + 270 + (wr - 1) * 34 + 12, w=280)
    pg.math("L_b = -\\frac{1}{|b|}\\sum_{i\\in b}\\left[y_i\\log\\hat{p}_i+(1-y_i)\\log(1-\\hat{p}_i)\\right]",
            ox, y0 + 445, 720, 50, 16)
    pg.bullets(ox, y0 + 510, 720, [
        "One <b>mini-batch</b> = a few rows processed together as a matrix → one gradient step",
        "Rows are shuffled every epoch; 1 epoch = n / batch = 160 / 16 = 10 steps",
        "Backprop uses \\(L_b\\) of this batch only (see panel ⑤) → noisy but fast updates",
        "Highlighted row = misclassified in this batch → contributes the largest loss"], head="Notes:")

    # ---------------------------------------------------------------- ⑧ hyperparameter search
    y0 = y0 + 740
    pg.panel(x0, y0, 2000, 600, 8, "Finding the optimal hyperparameters",
             "Learning-rate range test → early stopping on validation loss → search the rest (batch, width, depth, regularization)")
    # (a) LR range test: one pass, lr grows ×1.12 per mini-batch step
    lrt = MLP(H=6, seed=3)
    lr, lrs, losses, sm = 1e-4, [], [], None
    step = 0
    while lr < 30 and step < 120:
        b = (step * 16) % len(X)
        lrt.step(X[b:b + 16], y[b:b + 16], lr)
        l = lrt.loss(X[:64], y[:64])
        sm = l if sm is None else 0.8 * sm + 0.2 * l
        lrs.append(math.log10(lr)); losses.append(sm)
        lr *= 1.12; step += 1
        if sm > 4 * losses[0]: break
    slopes = [(losses[i + 3] - losses[i - 3]) / (lrs[i + 3] - lrs[i - 3]) for i in range(3, len(lrs) - 3)]
    ist = 3 + min(range(len(slopes)), key=lambda i: slopes[i])
    imin = min(range(len(losses)), key=lambda i: losses[i])
    top = min(max(losses) * 1.05, losses[0] * 2.5)
    p = pg.axes(x0 + 80, y0 + 130, 560, 270, (-4, lrs[-1] + 0.2), (0, top), r"\log_{10}\alpha", "loss",
                label_size=12, origin=(-4, 0), xticks=(-4, -3, -2, -1, 0, 1), yticks=(0.2, 0.4, 0.6))
    p.poly([(a, min(b_, top)) for a, b_ in zip(lrs, losses)], color=BLUE, width=2)
    p.scatter([(lrs[ist], losses[ist])], color=GREEN, r=5)
    p.scatter([(lrs[imin], losses[imin])], color=RED, r=5)
    pg.note(f"steepest descent: α ≈ {10 ** lrs[ist]:.3g}  ← pick here", x0 + 140, y0 + 420,
            *p.P(lrs[ist], losses[ist]), w=230, color=GREEN)
    pg.note(f"minimum loss α ≈ {10 ** lrs[imin]:.2g} (too late: pick ~10× smaller)", x0 + 380, y0 + 70,
            *p.P(lrs[imin], losses[imin]), w=260, color=RED)
    pg.text("(a) Learning-rate range test", x0 + 20, y0 + 64, 300, 20, 13, bold=True)
    pg.ask("Which learning rate should I start with?",
           f"α ≈ {10 ** lrs[ist]:.2g} at the steepest drop; the minimum is too late (loss explodes soon after)",
           x0 + 100, y0 + 150, *p.P(lrs[ist], losses[ist]), w=300)
    pg.bullets(x0 + 20, y0 + 450, 640, [
        "One short run: start α tiny, multiply by 1.12 every mini-batch, record (smoothed) loss",
        "Loss flat → α too small · steep drop → good · explodes → too big",
        "Choose α at the steepest drop (or min-loss α ÷ 10)"], size=11)

    # (b) early stopping with a bigger net on a small, noisy train set
    raw2, X2, y2, *_ = make_data(n=200, seed=9, noise=1.0, flip=0.12)
    Xtr, ytr, Xva, yva = X2[:50], y2[:50], X2[50:], y2[50:]
    big = MLP(H=12, seed=4)
    tr, va = [], []
    for ep in range(240):
        for b in range(0, len(Xtr), 10):
            big.step(Xtr[b:b + 10], ytr[b:b + 10], 0.15)
        if ep % 4 == 0:
            tr.append((ep, big.loss(Xtr, ytr))); va.append((ep, big.loss(Xva, yva)))
    ebest = min(va, key=lambda t: t[1])
    pat = 40
    estop = next((e for e, l in va if e >= ebest[0] + pat), va[-1][0])
    vtop = max(max(l for _, l in va), max(l for _, l in tr)) * 1.08
    ex = x0 + 720
    pg.text("(b) Early stopping (patience)", ex, y0 + 64, 400, 20, 13, bold=True)
    q = pg.axes(ex + 60, y0 + 130, 520, 270, (0, 240), (0, vtop), "epoch", "loss", label_size=12,
                xticks=(0, 60, 120, 180, 240), yticks=(0.25, 0.5, 0.75, 1.0))
    q.poly(tr, color=BLUE, width=2)
    q.poly(va, color=ORANGE, width=2)
    q.vline(ebest[0], 0, vtop * 0.95, color=GREEN, width=1.5)
    q.vline(estop, 0, vtop * 0.95, color=RED)
    q.scatter([ebest], color=GREEN, r=5)
    q.text("train", 240, tr[-1][1], dx=26, size=11, color=BLUE, bold=True)
    q.text("validation", 240, va[-1][1], dx=36, size=11, color=ORANGE, bold=True)
    pg.note(f"best epoch {ebest[0]} (val loss {ebest[1]:.3f}) → restore these weights", ex + 330, y0 + 70,
            *q.P(ebest[0], ebest[1]), w=270, color=GREEN)
    pg.note(f"stop at epoch {estop} after {pat} epochs without improvement", ex + 300, y0 + 420,
            *q.P(estop, vtop * 0.5), w=290, color=RED)
    pg.bullets(ex, y0 + 450, 600, [
        f"Small noisy train set (50 rows, 12 % flipped labels) + 12 hidden units → the net starts to memorize",
        "Validation loss rises while train loss keeps falling = <b>overfitting</b>",
        "Keep the weights of the best validation epoch; patience = how long to wait"], size=11)

    # (c) search space + regularization
    sx = x0 + 1380
    pg.text("(c) Search the remaining hyperparameters", sx, y0 + 64, 600, 20, 13, bold=True)
    pg.table(sx, y0 + 92, 590, [
        ["Hyperparameter", "Typical range", "Effect"],
        ["Learning rate α", "from range test (1e-4 … 1)", "speed / stability"],
        ["Batch size", "16, 32, 64, 128", "small = noisy, regularizing"],
        ["Hidden width", "8 … 512", "capacity"],
        ["Depth (layers)", "1 … 4 for tabular", "capacity, harder to train"],
        ["Dropout p", "0 … 0.5", "randomly zero units → less co-adaptation"],
        ["Weight decay λ", "1e-6 … 1e-2", r"L2 penalty \(\lambda\lVert W\rVert^2\)"],
        ["Epochs", "early stopping decides", "—"]], size=11, h=230)
    pg.math(r"J = L + \lambda\sum W^2,\qquad h_{drop} = h\odot m,\ m_k\sim Bernoulli(1-p)", sx, y0 + 335,
            590, 44, 14)
    pg.bullets(sx, y0 + 390, 590, [
        "Order: fix α (range test) → early stopping → random search over the rest",
        "<b>Random search</b> beats grid search when few hyperparameters matter; <b>Bayesian optimization</b> (Optuna) uses past trials",
        "Evaluate every trial on the same validation split (time-ordered for trading data!)",
        "Report the final score once on an untouched test set"], head="Procedure:", size=11)

"""MLP page, part 3: ANN family built on the MLP base - CNN, RNN, LSTM/GRU, Transformer (panels 9-14)."""
import math
from dio import *

ORANGE_L, GREEN_L, PURPLE_L = "#FFE6CC", "#D5E8D4", "#E1D5E7"


def grid(pg, x, y, vals, cell=30, hl=None, hl_color=HL, size=11, fill="#FFFFFF"):
    """Draw a matrix of numbers as cells. hl = set of (r, c) to highlight."""
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            f = hl_color if hl and (r, c) in hl else fill
            pg.rect(x + c * cell, y + r * cell, cell, cell, f"{v:g}", fill=f, size=size)
    return x + len(vals[0]) * cell, y + len(vals) * cell


def block(pg, x, y, w, h, depth, label, fill, sub=None):
    """Pseudo-3D feature-map block (stack of offset rectangles)."""
    for k in range(min(depth, 5) - 1, -1, -1):
        pg.rect(x + k * 6, y - k * 6, w, h, fill=fill, stroke="#666666", opacity=90)
    pg.text(label, x - 20, y + h + 6, w + 60, 18, 10, bold=True, align="center")
    if sub: pg.text(sub, x - 20, y + h + 22, w + 60, 16, 9, align="center", color="#555555")


def build(pg: Page, cat):
    Y = pg.max_y() + 60
    pg.title("Beyond the MLP: the ANN family (CNN, RNN, LSTM / GRU, Transformer)", 400, Y, 1600, 22)
    pg.text("Every architecture keeps the MLP core  z = Wx + b → g(z), trained by backpropagation + gradient descent; "
            "they only change HOW neurons are connected and which weights are shared.", 400, Y + 34, 2000, 20, 12,
            italic=True, color="#555555")
    y0 = Y + 70

    # ------------------------------------------------------------------ ⑨ family tree
    x0 = 400
    pg.panel(x0, y0, 760, 760, 9, "ANN family tree: ★ MLP is the base", "Each child adds one idea to the MLP")
    root = pg.rect(x0 + 230, y0 + 70, 300, 44, "Artificial Neural Network (ANN)", size=13, bold=True)
    mlp = pg.rect(x0 + 205, y0 + 150, 350, 54, "★ MLP — feed-forward, fully connected", fill=HL, sw=2, size=13,
                  bold=True)
    pg.connect(root, mlp, orth=False, label="simplest deep ANN")
    cnn = pg.rect(x0 + 20, y0 + 290, 220, 64, "<b>CNN</b><br><font style='font-size:10px'>local, shared filters "
                  "(space)</font>", fill=ORANGE_L, stroke="#D79B00", raw=True)
    rnn = pg.rect(x0 + 270, y0 + 290, 220, 64, "<b>RNN</b><br><font style='font-size:10px'>hidden state loop "
                  "(time)</font>", fill=LIGHTBLUE, stroke=BLUE, raw=True)
    ae = pg.rect(x0 + 520, y0 + 290, 220, 64, "<b>Autoencoder</b><br><font style='font-size:10px'>MLP encoder → "
                 "bottleneck → decoder</font>", fill=PURPLE_L, stroke="#9673A6", raw=True)
    for t, lab in [(cnn, "+ convolution"), (rnn, "+ recurrence"), (ae, "+ bottleneck")]:
        pg.connect(mlp, t, orth=True, label=lab)
    res = pg.rect(x0 + 20, y0 + 420, 220, 60, "<b>ResNet / U-Net</b><br><font style='font-size:10px'>+ skip "
                  "connections (very deep)</font>", fill=ORANGE_L, stroke="#D79B00", raw=True)
    lstm = pg.rect(x0 + 270, y0 + 420, 220, 60, "<b>LSTM / GRU</b><br><font style='font-size:10px'>+ gates → "
                   "long memory</font>", fill=LIGHTBLUE, stroke=BLUE, raw=True)
    vae = pg.rect(x0 + 520, y0 + 420, 220, 60, "<b>VAE / GAN</b><br><font style='font-size:10px'>generative "
                  "models</font>", fill=PURPLE_L, stroke="#9673A6", raw=True)
    pg.connect(cnn, res, orth=False)
    pg.connect(rnn, lstm, orth=False)
    pg.connect(ae, vae, orth=False)
    trf = pg.rect(x0 + 230, y0 + 540, 300, 60, "<b>Transformer</b> (BERT, GPT)<br><font style='font-size:10px'>"
                  "+ self-attention, drops recurrence</font>", fill=GREEN_L, stroke="#82B366", raw=True)
    pg.connect(lstm, trf, orth=False, label="attention replaces the loop")
    pg.ask("What does the Transformer add to the MLP?",
           "self-attention: every position weighs all others → no loop, fully parallel training",
           x0 + 555, y0 + 498, x0 + 531, y0 + 570, w=190)
    pg.text("(Vision Transformer = image patches + attention: CNN and Transformer ideas meet)", x0 + 200, y0 + 605,
            360, 16, 9, italic=True, color="#555555", align="center")
    pg.bullets(x0 + 20, y0 + 625, 720, [
        "Shared by all: neurons, activations ②, forward pass ④, loss, backprop + gradient descent ⑤",
        "CNN shares weights across <b>space</b>; RNN shares weights across <b>time</b>; Transformer learns which positions to attend to",
        "'ANN' = the whole family; 'MLP' = its plain fully-connected member (also called a dense / feed-forward network)"],
        head="What stays the same, what changes:")

    # ------------------------------------------------------------------ ⑩ convolution worked
    x0 = 1200
    pg.panel(x0, y0, 1200, 760, 10, "CNN: the convolution operation (worked example)",
             "A small filter slides over the input; the same 9 weights are used everywhere")
    img = [[10, 10, 10, 0, 0]] * 5
    ker = [[1, 0, -1]] * 3
    out = [[sum(img[i + m][j + n] * ker[m][n] for m in range(3) for n in range(3)) for j in range(3)]
           for i in range(3)]
    gx, gy = x0 + 40, y0 + 110
    pg.text("Input image 5×5 (bright | dark)", gx - 10, gy - 30, 200, 20, 12, bold=True)
    grid(pg, gx, gy, img, cell=34, hl={(r, c) for r in range(3) for c in range(1, 4)})
    pg.text("*", gx + 190, gy + 70, 30, 30, 24, bold=True, align="center")
    kx = gx + 235
    pg.text("Kernel 3×3 (vertical edge)", kx - 20, gy - 30, 200, 20, 12, bold=True)
    grid(pg, kx, gy + 17, ker, cell=34, fill=ORANGE_L)
    pg.text("=", kx + 115, gy + 70, 30, 30, 24, bold=True, align="center")
    ox = kx + 165
    pg.text("Feature map 3×3", ox - 10, gy - 30, 200, 20, 12, bold=True)
    grid(pg, ox, gy + 17, out, cell=40, hl={(0, 1)})
    pg.note("edge found here (large value)", ox - 10, gy + 150, ox + 60, gy + 137, w=170, color=RED)
    calc = " + ".join(f"{img[m][1 + n]}·{ker[m][n]}" for m in range(3) for n in range(3))
    pg.html(f"<b>Yellow window → yellow output:</b><br>{calc} = <b>{out[0][1]}</b>", gx, gy + 190, 560, 44, 11)
    pg.math("S(i,j) = (I * K)(i,j) = \\sum_{m}\\sum_{n} I(i+m,\\,j+n)\\,K(m,n)", gx, gy + 245, 560, 44, 16)
    pg.math("\\text{output size} = \\frac{n + 2p - k}{s} + 1 = \\frac{5 + 0 - 3}{1} + 1 = 3", gx, gy + 295, 560, 44,
            15)
    pg.bullets(gx, gy + 350, 560, ["n = input size, k = kernel size, p = padding (zeros around), s = stride (step)",
                                   "'same' padding p = (k−1)/2 keeps the size; stride 2 halves it",
                                   "Each filter learns one pattern (edge, corner, texture); a layer has many filters"],
               head="Where:", size=11)
    pg.ask("Why is the feature map smaller than the input?",
           "a 3×3 filter fits only 3 positions per row → (5 − 3)/1 + 1 = 3; add padding p = 1 to keep 5",
           gx, gy + 452, x0 + 505, gy + 333, w=560)
    # 1-D convolution on prices
    px = x0 + 650
    pg.text("1-D convolution on a price series", px, y0 + 80, 520, 20, 13, bold=True)
    series = [100, 101, 103, 106, 104, 101, 99, 100]
    k1 = [-1, 0, 1]
    conv = [sum(series[i + j] * k1[j] for j in range(3)) for i in range(len(series) - 2)]
    pg.text("prices", px, y0 + 115, 60, 20, 10, color="#555555")
    grid(pg, px + 60, y0 + 110, [series], cell=46)
    pg.text("kernel", px, y0 + 165, 60, 20, 10, color="#555555")
    grid(pg, px + 60, y0 + 160, [k1], cell=46, fill=ORANGE_L)
    pg.text("output", px, y0 + 215, 60, 20, 10, color="#555555")
    grid(pg, px + 60 + 46, y0 + 210, [conv], cell=46, hl={(0, 1)})
    pg.text("= momentum (price t+1 − price t−1): + rising, − falling", px, y0 + 262, 520, 20, 11, italic=True)
    p = pg.axes(px + 40, y0 + 310, 460, 150, (0, 7), (96, 108), "t", "price", label_size=11)
    p.poly(list(enumerate(series)), color=BLUE, width=2)
    p.scatter(list(enumerate(series)), r=3)
    q = pg.axes(px + 40, y0 + 500, 460, 110, (0, 7), (-6, 6), "t", "conv", label_size=11, origin=(0, 0))
    q.bars([(i + 1, v) for i, v in enumerate(conv)], 0.6, fill=ORANGE_L, stroke="#D79B00", labels=True)
    pg.text("1-D CNNs (and TCNs) learn such filters automatically from OHLCV windows", px, y0 + 640, 520, 20, 11,
            italic=True, color="#555555", align="center")

    # ------------------------------------------------------------------ ⑪ CNN architecture + pooling + sharing
    x0, y0 = 400, y0 + 800
    pg.panel(x0, y0, 1000, 720, 11, "CNN architecture, pooling & parameter sharing",
             "Conv → Pool → Conv → Pool → Flatten → MLP head")
    stages = [("Input", 28, 1, "#FFFFFF"), ("Conv 3×3 ×8", 26, 8, ORANGE_L), ("MaxPool 2×2", 13, 8, "#FFF2CC"),
              ("Conv 3×3 ×16", 11, 16, ORANGE_L), ("MaxPool 2×2", 5, 16, "#FFF2CC")]
    bx = x0 + 40
    for name, s, d, col in stages:
        h = 30 + s * 3.2
        block(pg, bx, y0 + 220 - h / 2, 22 + s * 1.2, h, d, name, col, f"{s}×{s}×{d}")
        bx += 22 + s * 1.2 + 70
        pg.line([(bx - 62, y0 + 220), (bx - 14, y0 + 220)], arrow="classic")
    for name, n in [("Flatten", 400), ("Dense", 64), ("Output", 10)]:
        h = {400: 180, 64: 110, 10: 50}[n]
        pg.rect(bx, y0 + 220 - h / 2, 26, h, fill=LIGHTBLUE if name != "Output" else HL, stroke=BLUE)
        pg.text(f"{name}<br>{n}", bx - 22, y0 + 220 + h / 2 + 6, 70, 30, 10, align="center", raw=True, bold=True)
        bx += 26 + 45
        if name != "Output":
            pg.line([(bx - 41, y0 + 220), (bx - 6, y0 + 220)], arrow="classic")
    pg.text("feature extractor (convolution blocks)", x0 + 200, y0 + 345, 300, 18, 11, italic=True)
    pg.text("MLP head (classifier)", x0 + 720, y0 + 345, 220, 18, 11, italic=True, color=BLUE)
    pg.ask("Why use convolution instead of dense layers?",
           "80 shared weights vs 4.2 M: far fewer parameters, less overfitting",
           x0 + 830, y0 + 80, x0 + 656, y0 + 465, w=160)
    # max pooling worked
    mp = [[1, 3, 2, 1], [4, 6, 5, 2], [7, 2, 9, 1], [3, 1, 4, 8]]
    pooled = [[max(mp[2 * i + a][2 * j + b] for a in (0, 1) for b in (0, 1)) for j in (0, 1)] for i in (0, 1)]
    mx_, my_ = x0 + 40, y0 + 410
    pg.text("Max pooling 2×2, stride 2", mx_, my_ - 26, 260, 20, 12, bold=True)
    colors = {(0, 0): "#CCE5FF", (0, 1): "#FFE6CC", (1, 0): "#D5E8D4", (1, 1): "#F8CECC"}
    for r in range(4):
        for c in range(4):
            pg.rect(mx_ + c * 32, my_ + r * 32, 32, 32, str(mp[r][c]), fill=colors[(r // 2, c // 2)], size=11,
                    bold=mp[r][c] == pooled[r // 2][c // 2])
    pg.line([(mx_ + 140, my_ + 64), (mx_ + 180, my_ + 64)], arrow="classic")
    for r in range(2):
        for c in range(2):
            pg.rect(mx_ + 190 + c * 40, my_ + 24 + r * 40, 40, 40, str(pooled[r][c]), fill=colors[(r, c)], size=13,
                    bold=True)
    pg.text("keeps the strongest response, halves size, adds small shift-invariance", mx_, my_ + 136, 300, 34, 10,
            italic=True)
    # parameter sharing table
    dense = 28 * 28 * 26 * 26 * 8 + 26 * 26 * 8
    convp = 3 * 3 * 1 * 8 + 8
    pg.text("Why convolution? Parameter sharing", x0 + 400, y0 + 384, 560, 20, 12, bold=True)
    pg.table(x0 + 400, y0 + 410, 560, [
        ["Layer: 28×28×1 → 26×26×8", "Weights", "Idea"],
        ["Fully connected (MLP)", f"{dense:,}", "every pixel → every neuron"],
        ["Convolution 3×3, 8 filters", f'<span style="background:{HL}">{convp}</span>', "9 weights + 1 bias per filter, reused"]],
        size=11, h=90)
    pg.math("\\text{params}_{conv} = (k \\cdot k \\cdot C_{in} + 1)\\cdot C_{out} = (3\\cdot3\\cdot1+1)\\cdot 8 = 80",
            x0 + 400, y0 + 510, 560, 40, 14)
    pg.bullets(x0 + 400, y0 + 560, 580, [
        f"≈ {dense // convp:,}× fewer weights → less overfitting, works on big images",
        "Local connectivity: each neuron sees only a small patch (receptive field), which grows with depth",
        "Translation equivariance: the same pattern is detected anywhere in the input"], head="Effects:", size=11)

    # ------------------------------------------------------------------ ⑫ RNN
    x0 = 1440
    pg.panel(x0, y0, 960, 720, 12, "RNN: an MLP with a loop over time",
             "The hidden state hₜ carries memory; the same weights are used at every time step")
    # folded
    fx, fy = x0 + 40, y0 + 110
    xin = pg.ellipse(fx + 20, fy + 190, 50, 40, "xₜ", fill=LIGHTBLUE, stroke=BLUE)
    hc = pg.rect(fx + 10, fy + 90, 70, 50, "h", fill="#FFF2CC", stroke=BLACK, size=14, bold=True)
    yo = pg.ellipse(fx + 20, fy, 50, 40, "yₜ", fill=GREEN_L, stroke="#82B366")
    pg.connect(xin, hc, orth=False, label="W_xh")
    pg.connect(hc, yo, orth=False, label="W_hy")
    pg.line([(fx + 80, fy + 105), (fx + 115, fy + 105), (fx + 115, fy + 70), (fx + 45, fy + 70)], color=RED,
            width=1.5, arrow="classic")
    pg.text("W_hh (loop)", fx + 95, fy + 50, 90, 16, 10, color=RED)
    pg.text("folded", fx, fy + 245, 90, 18, 11, italic=True, align="center")
    pg.text("=", fx + 140, fy + 105, 30, 30, 22, bold=True)
    # unrolled 4 steps with numbers
    W_xh, W_hh = 0.5, 0.8
    xs = [1.0, 2.0, 0.0, -1.0]
    hs, h = [], 0.0
    for xv in xs:
        h = math.tanh(W_xh * xv + W_hh * h)
        hs.append(h)
    ux = fx + 190
    prev = None
    for t, (xv, hv) in enumerate(zip(xs, hs)):
        cx = ux + t * 150
        xi = pg.ellipse(cx + 20, fy + 190, 50, 40, f"x{t + 1}={xv:g}", fill=LIGHTBLUE, stroke=BLUE, size=10)
        hi = pg.rect(cx, fy + 90, 90, 50, f"h{t + 1}={hv:.3f}", fill="#FFF2CC", size=11, bold=True)
        pg.connect(xi, hi, orth=False)
        if prev: pg.connect(prev, hi, orth=False, color=RED, label="W_hh")
        prev = hi
    yl = pg.ellipse(ux + 3 * 150 + 20, fy, 50, 40, "ŷ", fill=GREEN_L, stroke="#82B366")
    pg.connect(prev, yl, orth=False, label="many-to-one")
    pg.text("unrolled (BPTT trains through these copies)", ux, fy + 245, 560, 18, 11, italic=True, align="center")
    pg.math("h_t = \\tanh\\left(W_{xh}\\,x_t + W_{hh}\\,h_{t-1} + b_h\\right)", x0 + 40, y0 + 380, 440, 40, 16)
    pg.math("\\hat{y}_t = W_{hy}\\,h_t + b_y", x0 + 520, y0 + 380, 300, 40, 16)
    rows = [["t", "\\(x_t\\)", "\\(W_{xh}x_t + W_{hh}h_{t-1}\\)", "\\(h_t=\\tanh(\\cdot)\\)"]]
    hp = 0.0
    for t, (xv, hv) in enumerate(zip(xs, hs)):
        rows.append([t + 1, f"{xv:g}", f"0.5·{xv:g} + 0.8·{hp:.3f} = {W_xh * xv + W_hh * hp:.3f}", f"{hv:.3f}"])
        hp = hv
    pg.text("Worked example (scalar, W_xh = 0.5, W_hh = 0.8, h₀ = 0, b = 0):", x0 + 40, y0 + 432, 600, 20, 12, bold=True)
    pg.table(x0 + 40, y0 + 456, 520, rows, size=11, h=140)
    pg.bullets(x0 + 590, y0 + 450, 350, [
        "<b>many-to-one</b>: price window → next return (this example)",
        "<b>many-to-many</b>: label every step (e.g. POS tags, per-bar signals)",
        "<b>one-to-many</b>: caption / sequence generation",
        "Inputs come from the lag window of the AR/MA page ⑧: shape (samples, N steps, features)"],
        head="Sequence patterns:", size=11)
    pg.ask("Does the RNN still remember the first input at step 4?",
           f"weakly: h₄ = {hs[3]:.3f} mixes all inputs, but x₁'s share shrinks by ≈0.8× per step",
           x0 + 550, y0 + 548, x0 + 418, y0 + 557, w=380)
    pg.text("h₄ still contains a trace of x₁ — that is the memory; but it fades quickly (see ⑬)", x0 + 40, y0 + 615,
            880, 20, 11, italic=True, color=RED)

    # ------------------------------------------------------------------ ⑬ vanishing gradient + LSTM/GRU
    x0, y0 = 400, y0 + 760
    pg.panel(x0, y0, 1000, 680, 13, "Vanishing gradients → LSTM / GRU gates",
             "Why plain RNNs forget, and how gates fix it")
    pg.math("\\frac{\\partial h_T}{\\partial h_t} = \\prod_{k=t+1}^{T} W_{hh}\\,\\tanh'(\\cdot) \\approx "
            "(w\\,\\bar{g})^{T-t}", x0 + 30, y0 + 60, 470, 50, 16)
    p = pg.axes(x0 + 70, y0 + 150, 400, 220, (0, 20), (0, 2.0), "T-t", "|grad|", label_size=11,
                xticks=(0, 5, 10, 15, 20), yticks=(1, 2))
    for f_, col, lab in [(0.75, BLUE, "w·g = 0.75 → vanishes"), (1.0, GREEN, "= 1.0 → stable"),
                         (1.04, RED, "= 1.04 → explodes")]:
        p.curve(lambda v, f_=f_: f_ ** v, 0, 20, n=40, color=col, width=2)
    p.text("0.75 → vanishes", 17, 0.10, size=10, color=BLUE, w=110)
    pg.ask("How much signal survives 15 steps back?",
           f"factor 0.75 → {0.75 ** 15:.3f} (≈ 1%): a plain RNN forgets",
           x0 + 250, y0 + 268, *p.P(15, 0.75 ** 15), w=220)
    p.text("1.0 → stable", 16, 1.1, size=10, color=GREEN, w=100)
    p.text("1.04 → explodes", 15, 1.95, size=10, color=RED, w=110)
    pg.bullets(x0 + 30, y0 + 400, 460, [
        "Gradient is multiplied by the same factor at every step back in time",
        "Factor < 1 → early inputs get ~0 gradient → RNN can't learn long dependencies",
        "Factor > 1 → exploding gradients → fix with <b>gradient clipping</b>",
        "Fix for vanishing: <b>gated</b> cells with an additive memory path (LSTM, GRU)"], head="Reading:", size=11)
    # LSTM cell diagram
    lx, ly = x0 + 530, y0 + 80
    pg.rect(lx, ly, 440, 270, stroke="#999999", rounded=True)
    pg.text("LSTM cell", lx + 10, ly + 4, 100, 18, 11, bold=True)
    pg.line([(lx - 20, ly + 40), (lx + 460, ly + 40)], color=BLACK, width=2.5, arrow="classic")
    pg.text("cell state c (memory highway)", lx + 260, ly + 16, 180, 18, 10, italic=True)
    pg.math("c_{t-1}", lx - 60, ly + 28, 50, 24, 12)
    pg.math("c_t", lx + 455, ly + 18, 40, 24, 12)
    for i, (gname, col, gx_) in enumerate([("f: forget", RED, 40), ("i: input", GREEN, 150), ("c̃: candidate", ORANGE, 250),
                                           ("o: output", BLUE, 350)]):
        g = pg.rect(lx + gx_, ly + 170, 80, 36, gname, fill="#FFFFFF", stroke=col, sw=2, size=10, bold=True)
    for cx_, sym in [(lx + 80, "×"), (lx + 210, "×"), (lx + 250, "+")]:
        pg.ellipse(cx_ - 12, ly + 28, 24, 24, sym, fill="#FFFFFF", size=13, bold=True)
    pg.line([(lx + 80, ly + 170), (lx + 80, ly + 52)], color=RED, arrow="classic")
    pg.line([(lx + 190, ly + 170), (lx + 205, ly + 52)], color=GREEN, arrow="classic")
    pg.line([(lx + 290, ly + 170), (lx + 250, ly + 52)], color=ORANGE, arrow="classic")
    pg.line([(lx + 390, ly + 170), (lx + 390, ly + 110), (lx + 430, ly + 110)], color=BLUE, arrow="classic")
    pg.text("× tanh(c) → hₜ", lx + 330, ly + 120, 110, 18, 10, color=BLUE)
    pg.line([(lx - 20, ly + 240), (lx + 400, ly + 240)], color="#555555", width=1)
    pg.text("[hₜ₋₁, xₜ] feeds every gate (each gate = a small MLP layer with σ)", lx + 20, ly + 244, 400, 18, 10,
            italic=True)
    pg.math("f_t=\\sigma(W_f[h_{t-1},x_t]+b_f),\\ \\ i_t=\\sigma(W_i[\\cdot]+b_i),\\ \\ o_t=\\sigma(W_o[\\cdot]+b_o)",
            x0 + 520, y0 + 370, 470, 36, 13)
    pg.math("c_t = f_t\\odot c_{t-1} + i_t\\odot\\tanh(W_c[\\cdot]+b_c),\\quad h_t = o_t\\odot\\tanh(c_t)",
            x0 + 520, y0 + 410, 470, 36, 13)
    pg.note("'+' path: gradient flows without repeated squashing", x0 + 700, y0 + 455, lx + 250, ly + 52, w=290,
            color=RED)
    pg.html("<b>GRU</b> (simpler, 2 gates): \\(z_t\\) update and \\(r_t\\) reset gate, no separate cell state: "
            "\\(h_t=(1-z_t)\\odot h_{t-1}+z_t\\odot\\tilde{h}_t\\) — fewer parameters, often similar accuracy.",
            x0 + 520, y0 + 490, 470, 50, 11)
    pg.html("<b>Parameters</b> per layer (input d, hidden h): RNN \\(h(h+d+1)\\), GRU \\(3h(h+d+1)\\), "
            "LSTM \\(4h(h+d+1)\\).  e.g. d = 5, h = 32 → RNN 1,216 · GRU 3,648 · LSTM 4,864",
            x0 + 520, y0 + 550, 470, 50, 11)

    # ------------------------------------------------------------------ ⑭ comparison
    x0 = 1440
    pg.panel(x0, y0, 960, 680, 14, "Choosing an architecture: MLP vs CNN vs RNN vs LSTM vs Transformer")
    pg.table(x0 + 20, y0 + 50, 920, [
        ["", "<b>★ MLP</b>", "<b>CNN</b>", "<b>RNN</b>", "<b>LSTM / GRU</b>", "<b>Transformer</b>"],
        ["Best input", "tabular vector", "grid: image, 1-D signal", "short sequences", "long sequences",
         "long sequences, text"],
        ["Connection", "all-to-all", "local filters", "loop over time", "gated loop", "attention all-to-all"],
        ["Weight sharing", "none", "across space", "across time", "across time", "across positions"],
        ["Inductive bias", "none (most general)", "locality, shift invariance", "order, recency",
         "order, long memory", "pairwise relevance"],
        ["Memory / context", "none", "receptive field", "short (vanishing)", "long (100s of steps)",
         "whole window (quadratic cost)"],
        ["Training", "fast, parallel", "fast, parallel (GPU)", "sequential, slow", "sequential, slower",
         "parallel, data hungry"],
        ["Trading example", "features → up/down", "1-D conv on OHLCV, chart images", "next-bar return",
         "multi-day price + indicators", "news headlines (BERT), long context"],
        ["Key assumption", "i.i.d. rows, fixed-size input", "local patterns matter, stationarity of features",
         "Markov-like recent dependence", "ordered, regularly sampled", "large data, positional encoding"]],
        size=10, h=360)
    pg.bullets(x0 + 20, y0 + 440, 920, [
        "Start with the <b>MLP</b> (or gradient boosting) on engineered lag features → it is the baseline every deeper model must beat",
        "Use <b>CNN</b> when nearby values form patterns (images, local price shapes); <b>RNN/LSTM/GRU</b> when order and memory matter",
        "Use <b>Transformers</b> for text or very long contexts with lots of data; hybrids (CNN-LSTM, LSTM + attention) are common",
        "All of them: scale inputs, use time-ordered / purged validation for time series, early stopping & dropout (panel ⑧)"],
        head="Rules of thumb:", size=11)
    pg.ask("Which network should I try first on tabular data?",
           "the MLP (or gradient boosting) on lag features: the baseline that CNN / LSTM must beat",
           x0 + 650, y0 + 450, x0 + 625, y0 + 468, w=290)

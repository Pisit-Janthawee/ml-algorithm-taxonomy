"""Multi-Layer Perceptron page: neuron, activations, network, forward pass, backprop, training curve."""
import math
from dio import *


def _node(pg, cx, cy, r, label="", fill="#FFFFFF", stroke=BLACK, size=12, sw=1):
    return pg.ellipse(cx - r, cy - r, 2 * r, 2 * r, label, fill=fill, stroke=stroke, size=size, sw=sw, raw=True)


def _edge(pg, a, b, r1, r2, **kw):
    """Line between circle centers a and b, trimmed by radii."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy) or 1
    pg.line([(a[0] + dx / d * r1, a[1] + dy / d * r1), (b[0] - dx / d * r2, b[1] - dy / d * r2)], **kw)


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① perceptron
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 640, 1, "The Neuron (Perceptron)", "Background: weighted sum + activation")
    ins = [(x0 + 90, y0 + 130 + k * 90) for k in range(3)]
    sx, sy = x0 + 330, y0 + 220
    for k, c in enumerate(ins):
        _node(pg, *c, 24, f"\\(x_{k + 1}\\)", size=14)
        _edge(pg, c, (sx, sy), 24, 34, arrow="classic")
        mx, my = (c[0] + sx) / 2, (c[1] + sy) / 2
        pg.math(f"w_{k + 1}", mx - 30, my - 26, 40, 22, 13, color=BLUE)
    _node(pg, x0 + 210, y0 + 410, 18, "1", fill=LIGHTGREY)
    _edge(pg, (x0 + 210, y0 + 410), (sx, sy), 18, 34, arrow="classic", dashed=True)
    pg.math("b", x0 + 268, y0 + 330, 30, 22, 13, color=RED)
    _node(pg, sx, sy, 34, "\\(\\Sigma\\)", size=22, sw=1.5)
    ax = sx + 120
    pg.rect(ax - 40, sy - 30, 80, 60, "\\(g(z)\\)", raw=True, size=16, sw=1.5)
    pg.line([(sx + 34, sy), (ax - 40, sy)], arrow="classic", label="z", size=12)
    pg.line([(ax + 40, sy), (ax + 110, sy)], arrow="classic")
    pg.math("\\hat{y}", ax + 112, sy - 14, 40, 28, 18)
    pg.note("inputs (features)", x0 + 30, y0 + 70, ins[0][0], ins[0][1] - 24, w=110)
    pg.note("weights = importance of each input", x0 + 170, y0 + 90, x0 + 205, y0 + 150, w=200, color=BLUE)
    pg.note("bias shifts the threshold", x0 + 60, y0 + 450, x0 + 205, y0 + 420, w=150, color=RED)
    pg.note("weighted sum", sx - 40, y0 + 140, sx, sy - 34, w=90)
    pg.note("activation (non-linear)", ax - 40, y0 + 140, ax, sy - 30, w=140)
    pg.ask("What does one neuron compute?", "a weighted sum of the inputs + bias, squashed by g",
           x0 + 440, y0 + 42, sx + 24, sy - 26, w=190)
    pg.ask("Why add a non-linear activation g(z)?",
           "stacked linear layers collapse into one line; ReLU / sigmoid let the network bend",
           x0 + 390, y0 + 305, 1080 + 50 + 10 + 165, 240 + 365 + 48, w=240)
    pg.math("z = \\sum_{i=1}^{n} w_i x_i + b = \\mathbf{w}^T\\mathbf{x} + b", x0 + 80, y0 + 470, 480, 40, 18)
    pg.math("\\hat{y} = g(z)", x0 + 80, y0 + 510, 480, 34, 18)
    pg.bullets(x0 + 20, y0 + 550, 600, [
        "Rosenblatt's perceptron (1958) uses a step function: output 1 if z > 0 else 0",
        "With sigmoid g, one neuron = <b>Logistic Regression</b>; stacking neurons in layers = <b>MLP</b>",
        "One neuron draws only a straight boundary — it cannot solve XOR → need hidden layers"],
        head="Notes:")

    # ---------------------------------------------------------------- ② activation functions
    x0 = 1080
    pg.panel(x0, y0, 620, 640, 2, "Activation Functions", "Add non-linearity so the network can learn curves")
    sig = lambda v: 1 / (1 + math.exp(-v))
    acts = [
        ("Sigmoid", sig, (-0.3, 1.3), "\\sigma(z)=\\frac{1}{1+e^{-z}}", "range (0, 1) · output layer for probability"),
        ("Tanh", math.tanh, (-1.3, 1.3), "\\tanh(z)=\\frac{e^z-e^{-z}}{e^z+e^{-z}}", "range (−1, 1) · zero-centred"),
        ("ReLU", lambda v: max(0.0, v), (-1, 4), "ReLU(z)=\\max(0,z)", "default for hidden layers · fast"),
        ("Leaky ReLU", lambda v: v if v > 0 else 0.1 * v, (-1, 4), "f(z)=\\max(0.1z,\\,z)",
         "fixes 'dying ReLU' (small slope for z < 0)"),
    ]
    for k, (name, f, yr, tex, note) in enumerate(acts):
        gx = x0 + 40 + (k % 2) * 300
        gy = y0 + 95 + (k // 2) * 270
        pg.text(name, gx, gy - 30, 240, 20, 13, bold=True)
        p = pg.axes(gx + 10, gy, 220, 120, (-4, 4), yr, "z", "", label_size=11, origin=(0, 0))
        smooth = name in ("Sigmoid", "Tanh")
        p.curve(f, -4, 4, n=48 if smooth else 2 if name == "ReLU" else 2, color=BLUE, width=2, curved=smooth) \
            if smooth else None
        if not smooth:
            p.poly([(-4, f(-4)), (0, 0), (4, f(4))], color=BLUE, width=2)
        pg.math(tex, gx - 10, gy + 128, 260, 36, 13)
        pg.text(note, gx - 10, gy + 166, 270, 30, 10, color="#555555", align="center")
    pg.text("Without activation, many linear layers collapse into ONE linear layer.", x0 + 20, y0 + 600, 580, 20,
            11, italic=True, color=RED, align="center")

    # ---------------------------------------------------------------- ③ network
    x0 = 1740
    pg.panel(x0, y0, 660, 640, 3, "Multi-Layer Perceptron (fully connected)", "Input → Hidden → Output")
    layers = [3, 4, 4, 2]
    names = ["Input layer", "Hidden layer 1", "Hidden layer 2", "Output layer"]
    xs = [x0 + 90 + k * 160 for k in range(4)]
    cy0, R = y0 + 300, 20
    pos = []
    for k, n in enumerate(layers):
        pos.append([(xs[k], cy0 + (j - (n - 1) / 2) * 70) for j in range(n)])
    for k in range(3):
        for a in pos[k]:
            for b in pos[k + 1]:
                _edge(pg, a, b, R, R, color="#888888", width=0.75)
    labels = [["x_1", "x_2", "x_3"], ["h^{(1)}_1", "h^{(1)}_2", "h^{(1)}_3", "h^{(1)}_4"],
              ["h^{(2)}_1", "h^{(2)}_2", "h^{(2)}_3", "h^{(2)}_4"], ["\\hat{y}_1", "\\hat{y}_2"]]
    fills = ["#FFFFFF", LIGHTBLUE, LIGHTBLUE, HL]
    for k in range(4):
        for j, c in enumerate(pos[k]):
            _node(pg, *c, R, f"\\({labels[k][j]}\\)", fill=fills[k], size=10)
        pg.text(names[k], xs[k] - 60, y0 + 120, 120, 20, 12, bold=True, align="center")
    for k, tex in enumerate(["W^{(1)}: 4\\times3", "W^{(2)}: 4\\times4", "W^{(3)}: 2\\times4"]):
        pg.math(tex, (xs[k] + xs[k + 1]) / 2 - 55, y0 + 470, 110, 26, 12, color=BLUE)
    pg.ask("How many numbers does the network learn?", "46 weights & biases: (3·4+4)+(4·4+4)+(4·2+2)",
           x0 + 500, y0 + 365, (xs[2] + xs[3]) / 2 + 10, y0 + 483, w=150)
    pg.line([(xs[0], y0 + 160), (xs[3], y0 + 160)], arrow="classic", color=GREEN, width=1.5)
    pg.text("forward pass", (xs[0] + xs[3]) / 2 - 60, y0 + 140, 120, 18, 11, color=GREEN, align="center")
    pg.bullets(x0 + 20, y0 + 510, 620, [
        "Every neuron connects to every neuron in the next layer (dense / fully-connected)",
        "Parameters: (3·4+4) + (4·4+4) + (4·2+2) = <b>46</b> weights & biases",
        "More layers / neurons → more capacity, but more data needed and higher risk of overfitting",
        "Universal approximation: one hidden layer with enough neurons can approximate any function"],
        head="Notes:")

    # ---------------------------------------------------------------- ④ forward propagation
    x0, y0 = 400, 920
    pg.panel(x0, y0, 640, 700, 4, "Forward Propagation (worked example)", "Background: matrix multiplication")
    pg.math("z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}", x0 + 60, y0 + 70, 520, 40, 20)
    pg.math("a^{(l)} = g\\left(z^{(l)}\\right),\\qquad a^{(0)} = x", x0 + 60, y0 + 110, 520, 40, 18)
    pg.note("weights of layer l", x0 + 70, y0 + 160, x0 + 278, y0 + 95, w=110, color=BLUE)
    pg.note("output of previous layer", x0 + 380, y0 + 160, x0 + 340, y0 + 95, w=150)
    pg.text("Network 2 → 2 (ReLU) → 1 (sigmoid),  input x = [1, 2],  target y = 1", x0 + 20, y0 + 200, 600, 20, 12,
            bold=True)
    steps = [
        ("Hidden", "z^{(1)}=\\begin{bmatrix}0.5 & -1\\\\ 1 & 0.5\\end{bmatrix}\\begin{bmatrix}1\\\\2\\end{bmatrix}"
                   "+\\begin{bmatrix}0\\\\1\\end{bmatrix}=\\begin{bmatrix}-1.5\\\\3\\end{bmatrix}"),
        ("ReLU", "a^{(1)}=\\max(0,z^{(1)})=\\begin{bmatrix}0\\\\3\\end{bmatrix}"),
        ("Output", "z^{(2)}=\\begin{bmatrix}1 & 0.5\\end{bmatrix}\\begin{bmatrix}0\\\\3\\end{bmatrix}+0.2=1.7"),
        ("Sigmoid", "\\hat{y}=\\sigma(1.7)=\\frac{1}{1+e^{-1.7}}=0.846"),
        ("Loss", "L=-[y\\log\\hat{y}+(1-y)\\log(1-\\hat{y})]=-\\log(0.846)=0.167"),
    ]
    yy = y0 + 235
    for k, (lab, tex) in enumerate(steps):
        h = 70 if k in (0, 1, 2) else 46
        pg.text(f"{k + 1}. {lab}", x0 + 20, yy, 90, h, 12, bold=True)
        pg.math(tex, x0 + 110, yy, 510, h, 15, align="left")
        yy += h + 12
    pg.ask("What does the network predict for x = [1, 2]?",
           "ŷ = 0.846 → class 1 (85% sure); small loss 0.167 since y = 1",
           x0 + 420, y0 + 325, x0 + 352, y0 + 508, w=210)
    pg.html(f'<span style="background:{HL}">Prediction ŷ = 0.846 → class 1;  loss = 0.167</span>', x0 + 20, yy + 6,
            600, 26, 13, align="center", valign="middle")

    # ---------------------------------------------------------------- ⑤ backpropagation
    x0 = 1080
    pg.panel(x0, y0, 620, 700, 5, "Backpropagation + Gradient Descent", "Background: chain rule of calculus")
    names = [("\\(W^{(2)}\\)", LIGHTBLUE), ("\\(z^{(2)}\\)", "#FFFFFF"), ("\\(\\hat{y}\\)", "#FFFFFF"),
             ("\\(L\\)", LIGHTRED)]
    bx = [x0 + 40 + k * 145 for k in range(4)]
    by = y0 + 110
    for k, (lab, fill) in enumerate(names):
        pg.rect(bx[k], by, 90, 50, lab, fill=fill, raw=True, size=16)
    for k in range(3):
        pg.line([(bx[k] + 90, by + 18), (bx[k + 1], by + 18)], arrow="classic", color=GREEN, width=1.5)
        pg.line([(bx[k + 1], by + 38), (bx[k] + 90, by + 38)], arrow="classic", color=RED, width=1.5)
    pg.text("forward: compute values", x0 + 40, by - 28, 260, 18, 11, color=GREEN)
    pg.text("backward: pass gradients (error) back", x0 + 300, by + 112, 290, 18, 11, color=RED, align="right")
    for k, tex in enumerate(["\\frac{\\partial z^{(2)}}{\\partial W^{(2)}}", "\\frac{\\partial \\hat{y}}{\\partial z^{(2)}}",
                             "\\frac{\\partial L}{\\partial \\hat{y}}"]):
        pg.math(tex, bx[k] + 85, by + 56, 70, 44, 13, color=RED)
    pg.math("\\frac{\\partial L}{\\partial W^{(2)}} = \\frac{\\partial L}{\\partial \\hat{y}}\\cdot"
            "\\frac{\\partial \\hat{y}}{\\partial z^{(2)}}\\cdot\\frac{\\partial z^{(2)}}{\\partial W^{(2)}}"
            " = (\\hat{y}-y)\\,a^{(1)}", x0 + 20, y0 + 245, 580, 56, 17)
    pg.note("sigmoid + log-loss simplify to ŷ − y", x0 + 360, y0 + 305, x0 + 480, y0 + 288, w=220, color=RED)
    pg.text("Continue the example (α = 0.1):", x0 + 20, y0 + 320, 400, 20, 12, bold=True)
    pg.math("\\hat{y}-y = 0.846-1 = -0.154", x0 + 20, y0 + 345, 580, 34, 15)
    pg.math("\\frac{\\partial L}{\\partial W^{(2)}} = -0.154\\,[0,\\;3] = [0,\\;-0.46]", x0 + 20, y0 + 380, 580, 40,
            15)
    pg.math("W^{(2)} := W^{(2)} - \\alpha\\frac{\\partial L}{\\partial W^{(2)}} = [1,\\;0.5] - 0.1[0,\\;-0.46]",
            x0 + 20, y0 + 425, 580, 40, 15)
    pg.ask("Does neuron 1's weight change?", "No: its ReLU output was 0 → gradient 0",
           x0 + 465, y0 + 340, x0 + 378, y0 + 416, w=150)
    pg.html(f'<span style="background:{HL}">\\(W^{{(2)}} = [1,\\;0.546]\\)</span> → ŷ moves closer to 1',
            x0 + 20, y0 + 470, 580, 30, 14, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 520, 580, [
        "Gradients flow from the loss back to every layer, re-using values from the forward pass",
        "Neuron 1 had ReLU output 0 → its weight gets gradient 0 (it 'did not fire')",
        "Same update rule as Linear Regression's Gradient Descent — only the cost function differs",
        "Optimizers: SGD, Momentum, <b>Adam</b> (see Optimization page)"],
        head="Notes:")

    # ---------------------------------------------------------------- ⑥ training curve
    x0 = 1740
    pg.panel(x0, y0, 660, 700, 6, "Training: Loss vs. Epochs", "Detect over / under-fitting")
    tr = lambda e: 1.1 * math.exp(-e / 12) + 0.08
    va = lambda e: tr(e) + 0.05 + (0.00035 * (e - 25) ** 2 if e > 25 else 0)
    p = pg.axes(x0 + 70, y0 + 100, 500, 300, (0, 80), (0, 1.3), "epoch", "Loss", label_size=12,
                xticks=(0, 20, 40, 60, 80), yticks=(0.5, 1.0))
    p.curve(tr, 0, 80, color=BLUE, width=2)
    p.curve(va, 0, 80, color=ORANGE, width=2)
    p.vline(25, 0, 1.3, color=GREEN)
    p.scatter([(25, va(25))], color=GREEN, r=5)
    pg.note("training loss", x0 + 470, y0 + 380, *p.P(70, tr(70)), w=90, color=BLUE)
    pg.note("validation loss", x0 + 470, y0 + 160, *p.P(72, va(72)), w=100, color=ORANGE)
    pg.note("early stopping: best validation loss", x0 + 230, y0 + 90, *p.P(25, 1.25), w=210, color=GREEN)
    pg.ask("When should training stop?",
           "at epoch 25, where validation loss is lowest; after that the net memorises",
           x0 + 220, y0 + 120, *p.P(25, va(25) + 0.04), w=200)
    p.text("underfitting", 10, 1.15, w=90, size=11, italic=True)
    p.text("overfitting →", 55, 0.75, w=100, size=11, italic=True, color=RED)
    pg.bullets(x0 + 20, y0 + 440, 620, [
        "<b>Epoch</b> = one pass over all training data;  <b>batch</b> = samples per weight update",
        "Gap between validation and training loss growing = <b>overfitting</b> (memorizing)",
        "Fixes: early stopping, <b>Dropout</b> layer, L2 weight decay, more data",
        "Both losses high = <b>underfitting</b> → bigger network / train longer",
        "Scale inputs (standardize) and initialize weights randomly (Xavier / He)"],
        head="Notes:")

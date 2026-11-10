"""Logistic Regression page: sigmoid, from linear regression to probability, decision boundary,
log loss, gradient descent (worked numbers), evaluation."""
import math
from dio import *


def sig(z):
    return 1 / (1 + math.exp(-z))


XS, YS = [1, 2, 3, 4], [0, 0, 1, 1]          # hours studied -> pass?


def step(b0, b1, alpha=0.5):
    ps = [sig(b0 + b1 * x) for x in XS]
    g0 = sum(p - y for p, y in zip(ps, YS)) / len(XS)
    g1 = sum((p - y) * x for p, y, x in zip(ps, YS, XS)) / len(XS)
    return ps, g0, g1, b0 - alpha * g0, b1 - alpha * g1


def loss(b0, b1):
    ps = [sig(b0 + b1 * x) for x in XS]
    return -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(ps, YS)) / len(XS)


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① sigmoid
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 700, 1, "Linear score → Sigmoid → Probability", "Background: sigmoid (logistic) function")
    pg.math("z = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\cdots + \\beta_p x_p", x0 + 40, y0 + 62, 560, 40, 18)
    pg.note("same linear score as Linear Regression", x0 + 400, y0 + 100, x0 + 300, y0 + 92, w=220)
    p = pg.axes(x0 + 80, y0 + 150, 460, 230, (-6, 6), (0, 1), "z", "\\sigma(z)", origin=(0, 0),
                xticks=(-4, -2, 2, 4), yticks=(0.5, 1))
    p.hline(1, color=GREY)
    p.hline(0.5, color=RED)
    p.curve(sig, -6, 6, width=2.5, color=BLUE)
    p.scatter([(0, 0.5)], color=RED, r=4.5)
    pg.note("z ≫ 0  →  p̂ → 1", x0 + 470, y0 + 132, *p.P(4.5, sig(4.5)), w=120)
    pg.note("z ≪ 0  →  p̂ → 0", x0 + 90, y0 + 330, *p.P(-4.5, sig(-4.5)), w=110)
    pg.note("z = 0  →  p̂ = 0.5 (threshold)", x0 + 330, y0 + 300, *p.P(0, 0.5), w=180, color=RED)
    pg.math("\\hat{p} = P(y=1\\mid x) = \\sigma(z) = \\frac{1}{1+e^{-z}}", x0 + 60, y0 + 405, 520, 56, 20)
    pg.bullets(x0 + 20, y0 + 470, 600, [
        "\\(\\sigma\\) squashes any real number \\(z \\in (-\\infty, \\infty)\\) into a probability \\((0, 1)\\)",
        "\\(z\\) is the <b>log-odds</b>:  \\(\\ln\\frac{\\hat{p}}{1-\\hat{p}} = z\\)  →  \\(e^{\\beta_j}\\) = odds ratio of feature j",
        "Prediction:  \\(\\hat{y} = 1\\) if \\(\\hat{p} \\ge 0.5\\) (i.e. \\(z \\ge 0\\)), else \\(\\hat{y} = 0\\)",
        "Derivative is simple:  \\(\\sigma'(z) = \\sigma(z)\\,(1-\\sigma(z))\\) — used in gradient descent ④",
        "Example: z = 2  →  \\(\\hat{p} = 1/(1+e^{-2}) = 0.881\\)  →  class 1"],
        head="Where:")
    pg.ask("How likely is this row to be class 1?",
           f"read σ(z): z = 2 gives p̂ = {sig(2):.3f}; z = 0 is the 50 % boundary; each +1 in z multiplies the odds by e ≈ 2.7",
           x0 + 20, y0 + 605, *p.P(2, sig(2)), w=600)

    # ---------------------------------------------------------------- ② from linear regression to probability
    x0 = 1080
    pg.panel(x0, y0, 620, 700, 2, "From Linear Regression to Classification",
             "Background: linear regression + decision boundary")
    q = pg.axes(x0 + 60, y0 + 100, 220, 170, (0, 10), (-0.3, 1.3), "hours", "y", label_size=11, origin=(0, 0),
                yticks=(0.5, 1))
    zero = [1.0, 1.8, 2.5, 3.2, 3.9, 4.6, 5.6]
    one = [4.2, 5.2, 6.0, 6.7, 7.4, 8.1, 8.9]
    q.scatter([(v, 0) for v in zero], color=RED, r=3.5)
    q.scatter([(v, 1) for v in one], color=BLUE, r=3.5)
    q.hline(1, color=GREY)
    q.curve(lambda v: -0.25 + 0.15 * v, 0, 10, n=2, curved=False, dashed=True, color=GREY, width=1.5)
    q.curve(lambda v: sig(1.5 * (v - 4.9)), 0, 10, width=2, color=BLUE)
    pg.note("linear fit: ŷ < 0 or > 1  ✘", x0 + 150, y0 + 70, *q.P(9.3, 1.15), w=150, color=GREY)
    pg.note("sigmoid fit stays in (0,1) ✔", x0 + 140, y0 + 290, *q.P(6.5, sig(1.5 * 1.6)), w=170, color=BLUE)
    # 2-D decision boundary
    r = seeded(11)
    d = pg.axes(x0 + 350, y0 + 100, 220, 170, (0, 10), (0, 10), "x_1", "x_2", label_size=11)
    c1 = [(r.gauss(6.8, 1.2), r.gauss(6.8, 1.2)) for _ in range(22)]
    c0 = [(r.gauss(3.0, 1.2), r.gauss(3.0, 1.2)) for _ in range(22)]
    c1 = [(a, b) for a, b in c1 if a + b > 10.3 and a < 9.8 and b < 9.8]
    c0 = [(a, b) for a, b in c0 if a + b < 8.8 and a > 0.2 and b > 0.2]
    d.scatter(c1, color=BLUE, r=3)
    d.scatter(c0, color=RED, r=3)
    d.curve(lambda v: 9.5 - v, 0, 9.5, n=2, curved=False, width=2, color=GREEN)
    d.text("p̂ > 0.5 → class 1", 8.2, 9.6, w=110, size=9, color=BLUE)
    d.text("p̂ < 0.5 → class 0", 2.0, 0.8, w=110, size=9, color=RED)
    pg.note("decision boundary  \\(z = 0\\)", x0 + 450, y0 + 290, *d.P(6.5, 3.0), w=150, raw=True, color=GREEN)
    # flow
    labels = ["x₁ … xₚ", "z = Σ βⱼxⱼ", "σ(z)", "p̂", "p̂ ≥ 0.5 ?", "ŷ ∈ {0, 1}"]
    ids, fx = [], x0 + 20
    for k, lab in enumerate(labels):
        ids.append(pg.rect(fx, y0 + 330, 82, 36, lab, size=11, fill=HL if k == 2 else "none"))
        fx += 100
    for a, b in zip(ids, ids[1:]):
        pg.connect(a, b, orth=False)
    pg.bullets(x0 + 20, y0 + 395, 580, [
        "Step 1 — compute the linear score exactly like Linear Regression",
        "Step 2 — pass it through the sigmoid → probability of class 1",
        "Step 3 — threshold the probability → class label",
        "Boundary \\(\\beta_0+\\beta_1x_1+\\beta_2x_2=0\\) is a straight line → add polynomial features for curves",
        "Multi-class: <b>Softmax</b> \\(\\hat{p}_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}\\)  (or One-vs-Rest)",
        "Linear Regression with a 0/1 target fails: outputs outside [0,1] and outliers shift the line"],
        head="How it works:")
    pg.ask("Which class does a new point belong to?",
           "compute z: above the green line z > 0 → p̂ > 0.5 → class 1; below → class 0; on the line the model is 50/50",
           x0 + 20, y0 + 605, *d.P(6.5, 3.0), w=580)

    # ---------------------------------------------------------------- ③ log loss
    x0 = 1740
    pg.panel(x0, y0, 660, 700, 3, "Cost Function: Log Loss (Binary Cross-Entropy)", "Background: maximum likelihood")
    pg.math("J(\\beta) = -\\frac{1}{m}\\sum_{i=1}^{m}\\left[\\,y_i\\log\\hat{p}_i + (1-y_i)\\log(1-\\hat{p}_i)\\right]",
            x0 + 40, y0 + 62, 580, 56, 19)
    pg.note("active when y = 1", x0 + 250, y0 + 135, x0 + 330, y0 + 108, w=110, color=BLUE)
    pg.note("active when y = 0", x0 + 470, y0 + 135, x0 + 470, y0 + 108, w=110, color=RED)
    c = pg.axes(x0 + 80, y0 + 190, 420, 230, (0, 1), (0, 4.6), "\\hat{p}", "Loss", label_size=12,
                xticks=(0, 0.5, 1), yticks=(1, 2, 3, 4))
    c.curve(lambda v: -math.log(v), 0.01, 1, n=60, color=BLUE, width=2)
    c.curve(lambda v: -math.log(1 - v), 0, 0.99, n=60, color=RED, width=2)
    c.text("y = 1:  −log(p̂)", 0.2, 2.6, w=120, size=11, color=BLUE)
    c.text("y = 0:  −log(1 − p̂)", 0.8, 2.6, w=130, size=11, color=RED)
    pg.note("confident & wrong → loss → ∞", x0 + 120, y0 + 160, *c.P(0.03, 3.5), w=170, color=RED)
    pg.note("correct & confident → loss ≈ 0", x0 + 470, y0 + 380, *c.P(0.97, 0.05), w=170)
    c.scatter([(0.9, -math.log(0.9)), (0.1, -math.log(0.1))], color=BLACK, r=4)
    pg.html(f"Example (y = 1):  p̂ = 0.9 → loss = {-math.log(0.9):.3f}  ·  "
            f'<span style="background:{HL}">p̂ = 0.1 → loss = {-math.log(0.1):.3f}</span>',
            x0 + 40, y0 + 440, 580, 22, 12, align="center", valign="middle")
    pg.bullets(x0 + 20, y0 + 475, 620, [
        "Likelihood of one sample (Bernoulli):  \\(P(y\\mid x) = \\hat{p}^{\\,y}(1-\\hat{p})^{1-y}\\);  log + minus sign → log loss",
        "Why not MSE?  Sigmoid + MSE is <b>non-convex</b> (local minima); log loss is <b>convex</b>",
        "Regularized: \\(J + \\frac{\\lambda}{2m}\\sum_j\\beta_j^2\\)  (scikit-learn uses \\(C = 1/\\lambda\\))",
        "Objective: \\(\\min_\\beta J(\\beta)\\) → no closed form → Gradient Descent ④"],
        head="Notes:")
    pg.ask("How much is a confident wrong prediction punished?",
           f"heavily: p̂ = 0.1 for a true 1 costs {-math.log(0.1):.2f} vs {-math.log(0.9):.3f} for p̂ = 0.9 — "
           "log loss rewards honest probabilities",
           x0 + 20, y0 + 605, *c.P(0.1, -math.log(0.1)), w=620)

    # ---------------------------------------------------------------- ④ gradient descent with numbers
    x0, y0 = 400, 980
    pg.panel(x0, y0, 1320, 700, 4, "Training with Gradient Descent (worked example)",
             "Data: hours studied x = [1, 2, 3, 4], passed y = [0, 0, 1, 1],  α = 0.5,  start β₀ = β₁ = 0")
    pg.text("Repeat until convergence:", x0 + 30, y0 + 80, 380, 22, 13, bold=True)
    pg.math("\\beta_j := \\beta_j - \\alpha\\,\\frac{\\partial J}{\\partial\\beta_j}", x0 + 30, y0 + 105, 380, 50, 20)
    pg.math("\\frac{\\partial J}{\\partial\\beta_j} = \\frac{1}{m}\\sum_{i=1}^{m}(\\hat{p}_i - y_i)\\,x_{ij}",
            x0 + 30, y0 + 165, 380, 56, 18)
    pg.html(f'<span style="background:{HL}">Same form as Linear Regression!</span><br>'
            "only \\(\\hat{p} = \\sigma(z)\\) replaces \\(\\hat{y} = z\\)", x0 + 30, y0 + 230, 380, 44, 13,
            align="center", valign="middle")
    pg.note("(prediction − actual) × feature", x0 + 250, y0 + 290, x0 + 300, y0 + 200, w=180)
    pg.rect(x0 + 30, y0 + 330, 380, 150,
            "<b>Pseudo-code</b><br><font face='Courier New'>β = zeros(p+1)<br>for epoch in range(N):<br>"
            "&nbsp;&nbsp;p̂ = sigmoid(X @ β)<br>&nbsp;&nbsp;grad = Xᵀ(p̂ − y) / m<br>"
            "&nbsp;&nbsp;β -= α · grad</font>", fill="#F5F5F5", stroke="#999999", size=12, align="left",
            valign="top", raw=True)
    # iteration table
    b0, b1 = 0.0, 0.0
    ps, g0, g1, nb0, nb1 = step(b0, b1)
    tx = x0 + 450
    pg.text("Iteration 1 (β₀ = 0, β₁ = 0)", tx, y0 + 80, 420, 22, 13, bold=True)
    rows = [["\\(x\\)", "\\(y\\)", "\\(z\\)", "\\(\\hat{p}\\)", "\\(\\hat{p}-y\\)", "\\((\\hat{p}-y)x\\)"]]
    for x, y, p in zip(XS, YS, ps):
        rows.append([x, y, f"{b0 + b1 * x:.2f}", f"{p:.3f}", f"{p - y:+.3f}", f"{(p - y) * x:+.3f}"])
    rows.append(["", "", "", "mean", f"<b>{g0:+.3f}</b>", f"<b>{g1:+.3f}</b>"])
    pg.table(tx, y0 + 108, 420, rows, size=11, highlight_rows=(5,))
    pg.math(f"\\beta_0 = 0 - 0.5({g0:.3f}) = {nb0:.3f}", tx, y0 + 260, 420, 30, 14)
    pg.math(f"\\beta_1 = 0 - 0.5({g1:.3f}) = {nb1:.3f}", tx, y0 + 292, 420, 30, 14)
    # iteration 2
    ps2, g02, g12, nb02, nb12 = step(nb0, nb1)
    pg.text(f"Iteration 2 (β₀ = {nb0:.2f}, β₁ = {nb1:.2f})", tx, y0 + 335, 420, 22, 13, bold=True)
    rows = [["\\(x\\)", "\\(z\\)", "\\(\\hat{p}\\)", "\\(\\hat{p}-y\\)"]]
    for x, y, p in zip(XS, YS, ps2):
        rows.append([x, f"{nb0 + nb1 * x:.2f}", f"{p:.3f}", f"{p - y:+.3f}"])
    pg.table(tx, y0 + 362, 420, rows, size=11)
    pg.math(f"J: {loss(0, 0):.3f} \\rightarrow {loss(nb0, nb1):.3f} \\rightarrow {loss(nb02, nb12):.3f}",
            tx, y0 + 490, 420, 30, 15)
    pg.text("loss goes down every iteration", tx, y0 + 520, 420, 18, 11, italic=True, align="center", color=GREEN)
    # loss vs iteration plot
    hist, bb0, bb1 = [], 0.0, 0.0
    for _ in range(60):
        hist.append(loss(bb0, bb1))
        *_, bb0, bb1 = step(bb0, bb1)
    lx = x0 + 930
    L = pg.axes(lx + 40, y0 + 110, 300, 170, (0, 60), (0, 0.75), "iteration", "J", label_size=11,
                xticks=(0, 20, 40, 60), yticks=(0.25, 0.5, 0.693))
    L.poly([(i, v) for i, v in enumerate(hist)], color=BLUE, width=2, curved=True)
    L.scatter([(0, hist[0])], color=RED, r=4)
    pg.note(f"after 60 iterations J = {hist[-1]:.3f}", lx + 140, y0 + 225, *L.P(59, hist[-1]), w=170)
    pg.text(f"Result: β₀ = {bb0:.2f}, β₁ = {bb1:.2f}", lx + 20, y0 + 320, 340, 20, 12, bold=True)
    S = pg.axes(lx + 40, y0 + 370, 300, 140, (0, 5), (0, 1), "x", "\\hat{p}", label_size=11,
                xticks=(1, 2, 3, 4), yticks=(0.5, 1))
    S.hline(0.5, color=RED)
    S.curve(lambda v: sig(bb0 + bb1 * v), 0, 5, color=BLUE, width=2)
    S.scatter([(x, y) for x, y in zip(XS, YS)], color=BLACK, r=3.5)
    pg.note(f"boundary x = {-bb0 / bb1:.2f} h", lx + 230, y0 + 520, *S.P(-bb0 / bb1, 0.5), w=120, color=RED)
    pg.bullets(x0 + 20, y0 + 580, 1280, [
        "Feature scaling speeds up convergence;  perfectly separable data makes β grow forever → regularization (C) keeps it finite",
        "Solvers in practice: lbfgs / newton-cg (second order), saga / SGD for large data (see Optimization page)"],
        head="Notes:")
    pg.ask("After training, where does 'fail' switch to 'pass'?",
           f"at x = {-bb0 / bb1:.2f} h, where σ(β₀ + β₁x) = 0.5; the loss fell from 0.693 to {hist[-1]:.3f} in 60 steps",
           x0 + 660, y0 + 565, *S.P(-bb0 / bb1, 0.5), w=640)

    # ---------------------------------------------------------------- ⑤ evaluation
    x0 = 1760
    pg.panel(x0, y0, 640, 700, 5, "Evaluation: Confusion Matrix & ROC")
    rows = [["", "Predicted 1", "Predicted 0"], ["<b>Actual 1</b>", "TP = 40", "FN = 10"],
            ["<b>Actual 0</b>", "FP = 5", "TN = 45"]]
    pg.table(x0 + 30, y0 + 70, 290, rows, size=12, h=100)
    mx = x0 + 340
    for k, t in enumerate(["Accuracy = \\frac{TP+TN}{N} = \\frac{85}{100} = 0.85",
                           "Precision = \\frac{TP}{TP+FP} = \\frac{40}{45} = 0.889",
                           "Recall = \\frac{TP}{TP+FN} = \\frac{40}{50} = 0.80",
                           "F_1 = \\frac{2PR}{P+R} = 0.842"]):
        pg.math(t, mx, y0 + 55 + k * 40, 290, 38, 13, align="left")
    R = pg.axes(x0 + 70, y0 + 260, 220, 190, (0, 1), (0, 1), "FPR", "TPR", label_size=11,
                xticks=(0.5, 1), yticks=(0.5, 1))
    R.curve(lambda v: 1 - (1 - v) ** 3, 0, 1, color=BLUE, width=2)
    R.curve(lambda v: v, 0, 1, n=2, curved=False, dashed=True, color=GREY)
    R.text("AUC = 0.75", 0.55, 0.55, w=80, size=11, color=BLUE, dx=-30, dy=-50)
    R.text("random (0.5)", 0.75, 0.6, w=80, size=9, color=GREY, dx=10, dy=20)
    T = pg.axes(x0 + 380, y0 + 260, 200, 190, (0, 1), (0, 1), "threshold", "", label_size=11,
                xticks=(0.5, 1), yticks=(0.5, 1))
    T.curve(lambda t: 0.5 + 0.48 * t ** 0.6, 0, 1, color=GREEN, width=2)
    T.curve(lambda t: 1 - t ** 2.2, 0, 1, color=RED, width=2)
    T.text("precision", 0.8, 0.98, w=70, size=10, color=GREEN, dy=-10)
    T.text("recall", 0.85, 0.25, w=60, size=10, color=RED, dx=-35)
    T.vline(0.5)
    pg.bullets(x0 + 20, y0 + 480, 600, [
        "Default threshold 0.5 — move it to trade <b>precision ↔ recall</b>",
        "ROC: TPR vs FPR for every threshold;  AUC = P(score of a positive > score of a negative)",
        "Imbalanced data → use F1 / PR-AUC and <b>class_weight</b>, not accuracy",
        "Interpretation: \\(e^{\\beta_j}\\) = how the odds multiply when \\(x_j\\) increases by 1"],
        head="Notes:")
    pg.ask("Which probability threshold should I use?",
           "0.5 by default; raise it for fewer false alarms (precision ↑), lower it to catch more positives (recall ↑)",
           x0 + 20, y0 + 600, *T.P(0.5, 0.72), w=600)

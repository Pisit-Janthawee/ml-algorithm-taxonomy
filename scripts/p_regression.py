"""Simple Linear Regression page: model, least squares, cost function, gradient descent, evaluation."""
from dio import *

X5, Y5 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]          # worked example -> b0 = 2.2, b1 = 0.6
B0, B1 = 2.2, 0.6


def J(b1, b0=B0):
    return sum((b0 + b1 * x - y) ** 2 for x, y in zip(X5, Y5)) / (2 * len(X5))


def dJ(b1, b0=B0):
    return sum((b0 + b1 * x - y) * x for x, y in zip(X5, Y5)) / len(X5)


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① the model
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 700, 1, "The Model: fit a straight line", "Univariate (single predictor)")
    p = pg.axes(x0 + 70, y0 + 100, 460, 300, (0, 10), (0, 10), "X", "Y")
    r = seeded(7)
    pts = [(xv, 1.5 + 0.7 * xv + r.gauss(0, 0.75)) for xv in [0.6 + 8.8 * r.random() for _ in range(42)]]
    pts = [(a, b) for a, b in pts if 0.3 < b < 9.7]
    p.scatter(pts)
    f = lambda v: 1.5 + 0.7 * v
    p.curve(f, 0, 10, n=2, curved=False, width=2)
    # residual
    rx, ry = 7.5, 8.6
    p.scatter([(rx, ry)], color=RED, r=4.5)
    pg.ellipse(*[c - 4 for c in p.P(rx, f(rx))], 8, 8, fill="#FFFFFF")
    p.vline(rx, f(rx), ry, color=RED, dashed=True, width=1.5)
    pg.note("Residual  \\(e_i = y_i - \\hat{y}_i\\)", x0 + 470, y0 + 96, *p.P(rx, (ry + f(rx)) / 2), w=160, raw=True,
            color=RED)
    pg.note("Data point \\((x_i, y_i)\\)", x0 + 380, y0 + 66, *p.P(rx, ry), w=120, raw=True)
    pg.note("Best-fit line  \\(\\hat{y}\\)", x0 + 470, y0 + 140, *p.P(9.3, f(9.3)), w=130, raw=True)
    # slope triangle
    a, b = 2, 4
    p.poly([(a, f(a)), (b, f(a)), (b, f(b))], dashed=True, color=GREEN, width=1.5)
    p.math("\\Delta x", (a + b) / 2, f(a), dy=12, size=11, color=GREEN)
    p.math("\\Delta y", b, (f(a) + f(b)) / 2, dx=26, size=11, color=GREEN)
    # intercept
    ox, oy = p.P(0, 0)
    _, iy = p.P(0, 1.5)
    pg.bracket(ox - 16, iy, 10, oy - iy)
    pg.note("\\(\\beta_0\\) Intercept (value of ŷ at x = 0)", x0 + 14, y0 + 410, ox - 16, (iy + oy) / 2, w=200,
            raw=True)
    pg.note("\\(\\beta_1 = \\frac{\\Delta y}{\\Delta x}\\) Slope", x0 + 300, y0 + 420, *p.P(b, f(b) - 0.3), w=130,
            raw=True, color=GREEN)
    pg.math("\\hat{y}_i = \\beta_0 + \\beta_1 x_i + \\epsilon_i", x0 + 120, y0 + 455, 400, 44, 20)
    pg.ask("How much does y change when x grows by one unit?",
           "read the slope β₁ (here ≈ 0.7): +1 in x → +0.7 in ŷ; sign = direction of the relationship",
           x0 + 390, y0 + 250, *p.P(3.0, f(3.0)), w=230)
    pg.bullets(x0 + 20, y0 + 505, 600, [
        "\\(\\hat{y}_i\\) = predicted value,  \\(y_i\\) = actual (observed) value",
        "\\(x_i\\) = input feature (independent variable)",
        "\\(\\beta_0\\) = intercept,  \\(\\beta_1\\) = slope (change of ŷ when x increases by 1)",
        "\\(\\epsilon_i\\) = error term (residual), assumed \\(\\sim N(0, \\sigma^2)\\)",
        "<b>Goal:</b> find \\(\\beta_0, \\beta_1\\) that minimize the error between \\(\\hat{y}_i\\) and \\(y_i\\)"],
        head="Where:")

    # ---------------------------------------------------------------- ② least squares (closed form)
    x0 = 1080
    pg.panel(x0, y0, 620, 700, 2, "Least Squares (closed-form solution)", "Background: mean, variance, covariance")
    pg.math("\\beta_1 = \\frac{\\sum_{i=1}^{n}(x_i-\\bar{x})(y_i-\\bar{y})}{\\sum_{i=1}^{n}(x_i-\\bar{x})^2}"
            " = \\frac{Cov(x,y)}{Var(x)}", x0 + 20, y0 + 70, 580, 60, 17)
    pg.math("\\beta_0 = \\bar{y} - \\beta_1\\bar{x}", x0 + 20, y0 + 130, 580, 36, 17)
    pg.text("Example: 5 houses (x = size, y = price)", x0 + 20, y0 + 175, 400, 20, 12, bold=True)
    rows = [["\\(x\\)", "\\(y\\)", "\\(x-\\bar{x}\\)", "\\(y-\\bar{y}\\)", "\\((x-\\bar{x})(y-\\bar{y})\\)",
             "\\((x-\\bar{x})^2\\)"]]
    for xv, yv in zip(X5, Y5):
        rows.append([xv, yv, xv - 3, yv - 4, (xv - 3) * (yv - 4), (xv - 3) ** 2])
    rows.append(["\\(\\bar{x}=3\\)", "\\(\\bar{y}=4\\)", "", "Σ", "<b>6</b>", "<b>10</b>"])
    pg.table(x0 + 20, y0 + 200, 360, rows, size=11, highlight_rows=(6,))
    pg.math("\\beta_1 = \\frac{6}{10} = 0.6", x0 + 20, y0 + 395, 180, 40, 15)
    pg.math("\\beta_0 = 4 - 0.6(3) = 2.2", x0 + 200, y0 + 395, 200, 40, 15)
    pg.html(f'<span style="background:{HL}">\\(\\hat{{y}} = 2.2 + 0.6x\\)</span>', x0 + 120, y0 + 440, 200, 34,
            16, align="center", valign="middle")
    q = pg.axes(x0 + 420, y0 + 220, 170, 170, (0, 6), (0, 6), "x", "y", label_size=12, xticks=(1, 2, 3, 4, 5),
                yticks=(2, 4))
    q.scatter(list(zip(X5, Y5)), r=4)
    q.curve(lambda v: B0 + B1 * v, 0, 6, n=2, curved=False, width=1.5)
    q.scatter([(3, 4)], color=RED, r=4)
    q.text("(x̄, ȳ)", 3, 4, dy=-14, size=9, color=RED)
    pg.bullets(x0 + 20, y0 + 490, 580, [
        "\\(\\bar{x}=\\frac{1}{n}\\sum x_i\\) (mean) — the line always passes through \\((\\bar{x},\\bar{y})\\)",
        "Works only for small problems; with many features use the Normal Equation \\(\\beta=(X^TX)^{-1}X^Ty\\)",
        "For large data / many features → minimize the <b>cost function</b> with <b>Gradient Descent</b> ③ ④"],
        head="Notes:")
    pg.ask("What is the best straight line for these 5 houses?",
           "ŷ = 2.2 + 0.6x: each extra unit of size adds 0.6 to the price, and the line passes through (x̄, ȳ) = (3, 4)",
           x0 + 20, y0 + 595, x0 + 220, y0 + 476, w=580)

    # ---------------------------------------------------------------- ③ cost function
    x0 = 1740
    pg.panel(x0, y0, 660, 700, 3, "Cost Function (Mean Squared Error)", "How wrong is the line?")
    pg.math("J(\\beta_0,\\beta_1) = \\frac{1}{2m}\\sum_{i=1}^{m}\\left(\\hat{y}_i - y_i\\right)^2", x0 + 80, y0 + 75,
            440, 56, 20)
    pg.note("m = number of training examples", x0 + 20, y0 + 140, x0 + 205, y0 + 115, w=180)
    pg.note("prediction − actual = error", x0 + 230, y0 + 160, x0 + 420, y0 + 118, w=170)
    pg.note("power 2 → always positive, punishes big errors", x0 + 440, y0 + 140, x0 + 505, y0 + 92, w=210)
    # squared residuals drawn as squares (equal axis scale)
    s = 34
    q = pg.axes(x0 + 50, y0 + 205, 6 * s, 6 * s, (0, 6), (0, 6), "x", "y", label_size=12)
    q.curve(lambda v: B0 + B1 * v, 0, 6, n=2, curved=False, width=1.5)
    for xv, yv in zip(X5, Y5):
        yh = B0 + B1 * xv
        side = abs(yv - yh) * s
        px, py = q.P(xv, max(yv, yh))
        pg.rect(px, py, side, side, fill=LIGHTRED, stroke=RED, opacity=70)
        q.vline(xv, yh, yv, color=RED, dashed=False)
    q.scatter(list(zip(X5, Y5)), r=4)
    pg.text("Each square = (error)²  →  J = average area ÷ 2", x0 + 30, y0 + 430, 280, 30, 11, align="center")
    # J(b1) bowl
    c = pg.axes(x0 + 360, y0 + 200, 260, 200, (-0.1, 1.3), (0, 2.2), "\\beta_1", "J(\\beta_1)", label_size=12,
                origin=(-0.1, 0), xticks=(0, 0.6, 1.2))
    c.curve(J, -0.1, 1.3, width=2, color=BLUE)
    c.scatter([(0.6, J(0.6))], color=RED, r=5)
    c.vline(0.6, 0, J(0.6))
    pg.note("minimum cost → best \\(\\beta_1 = 0.6\\)", x0 + 470, y0 + 420, *c.P(0.6, J(0.6)), w=170, raw=True,
            color=RED)
    pg.text("Convex 'bowl' → one global minimum", x0 + 370, y0 + 445, 260, 20, 11, italic=True, align="center")
    pg.bullets(x0 + 20, y0 + 480, 620, [
        "MSE = \\(\\frac{1}{m}\\sum(\\hat{y}-y)^2\\);  the extra \\(\\frac{1}{2}\\) only makes the derivative cleaner",
        "RMSE = \\(\\sqrt{MSE}\\) is in the same unit as y;  MAE = \\(\\frac{1}{m}\\sum|\\hat{y}-y|\\) is robust to outliers",
        "With L2 regularization (Ridge): \\(J + \\frac{\\lambda}{2m}\\sum_{j}\\beta_j^2\\)",
        "Objective: \\(\\min_{\\beta_0,\\beta_1} J(\\beta_0,\\beta_1)\\)  →  solved by Gradient Descent ④"],
        head="Notes:")
    pg.ask("Which slope makes the line least wrong?",
           f"the bottom of the bowl: β₁ = 0.6 with J = {J(0.6):.2f}; every other slope gives a larger average error",
           x0 + 20, y0 + 595, *c.P(0.6, J(0.6)), w=620)

    # ---------------------------------------------------------------- ④ gradient descent
    x0, y0 = 400, 980
    pg.panel(x0, y0, 1320, 700, 4, "Gradient Descent (Back Bone of ML)",
             "Objective: minimize the cost function step by step")
    c = pg.axes(x0 + 70, y0 + 110, 380, 300, (-0.2, 1.4), (0, 3.2), "\\beta_1", "J(\\beta_1)", label_size=13,
                origin=(-0.2, 0))
    c.curve(J, -0.2, 1.4, width=2, color=BLUE)
    b, alpha, path = -0.1, 0.05, []
    for _ in range(7):
        path.append(b)
        b = b - alpha * dJ(b)
    for i, (u, v) in enumerate(zip(path, path[1:])):
        pg.line([c.P(u, J(u)), c.P(v, J(v))], color=RED, arrow="classic", width=1.5)
    c.scatter([(u, J(u)) for u in path], color=RED, r=4)
    c.scatter([(0.6, J(0.6))], color=GREEN, r=5)
    pg.note("start (random β)", x0 + 90, y0 + 95, *c.P(path[0], J(path[0])), w=110, color=RED)
    pg.note("slope (gradient) \\(\\frac{\\partial J}{\\partial\\beta_1}\\) tells the direction", x0 + 250, y0 + 90,
            *c.P(path[1], J(path[1])), w=200, raw=True)
    pg.note("converged: gradient ≈ 0", x0 + 330, y0 + 385, *c.P(0.6, J(0.6)), w=150, color=GREEN)
    # update rules
    mx = x0 + 500
    pg.text("Repeat until convergence:", mx, y0 + 100, 360, 22, 13, bold=True)
    pg.math("\\beta_j := \\beta_j - \\alpha\\,\\frac{\\partial}{\\partial\\beta_j}J(\\beta_0,\\beta_1)", mx,
            y0 + 125, 380, 50, 20)
    pg.note("α = learning rate (step size)", mx + 30, y0 + 185, mx + 175, y0 + 160, w=170, color=RED)
    pg.note("gradient (partial derivative)", mx + 230, y0 + 185, mx + 290, y0 + 160, w=170)
    pg.text("Partial derivatives (chain rule):", mx, y0 + 220, 360, 20, 12, bold=True)
    pg.math("\\frac{\\partial J}{\\partial\\beta_0} = \\frac{1}{m}\\sum_{i=1}^{m}(\\hat{y}_i-y_i)", mx, y0 + 245,
            380, 50, 16)
    pg.math("\\frac{\\partial J}{\\partial\\beta_1} = \\frac{1}{m}\\sum_{i=1}^{m}(\\hat{y}_i-y_i)\\,x_i", mx,
            y0 + 300, 380, 50, 16)
    pg.rect(mx, y0 + 365, 380, 140,
            "<b>Pseudo-code</b><br><font face='Courier New'>β0, β1 = 0, 0<br>for epoch in range(N):<br>"
            "&nbsp;&nbsp;ŷ = β0 + β1·x<br>&nbsp;&nbsp;β0 -= α·mean(ŷ − y)<br>"
            "&nbsp;&nbsp;β1 -= α·mean((ŷ − y)·x)</font>", fill="#F5F5F5", stroke="#999999", size=12, align="left",
            valign="top", raw=True)
    pg.text("Update β0 and β1 simultaneously!", mx, y0 + 510, 380, 20, 11, italic=True, color=RED)
    # learning-rate comparison
    lx = x0 + 920
    pg.text("Choosing the learning rate α", lx, y0 + 95, 380, 22, 13, bold=True)
    cases = [("α too small → slow", 0.012, 6), ("α good → converges", 0.12, 5), ("α too large → diverges", 0.19, 5)]
    for k, (lab, al, n) in enumerate(cases):
        yy = y0 + 125 + k * 150
        cc = pg.axes(lx + 20, yy + 20, 200, 100, (-0.2, 1.4), (0, 3.6), "", "", origin=(-0.2, 0))
        cc.curve(J, -0.2, 1.4, color=BLUE, width=1.5)
        bb, pp = -0.1, []
        for _ in range(n):
            pp.append(bb)
            bb -= al * dJ(bb)
        pp = [min(max(u, -0.2), 1.4) for u in pp]
        for u, v in zip(pp, pp[1:]):
            pg.line([cc.P(u, min(J(u), 3.6)), cc.P(v, min(J(v), 3.6))], color=RED, arrow="classic", width=1.2)
        cc.scatter([(u, min(J(u), 3.6)) for u in pp], color=RED, r=3)
        pg.text(lab, lx + 235, yy + 55, 160, 30, 12, bold=(k == 1), color=GREEN if k == 1 else BLACK)
    pg.bullets(x0 + 20, y0 + 560, 880, [
        "<b>Batch GD</b> uses all m samples per step;  <b>SGD</b> one sample;  <b>Mini-batch</b> a small batch (see Optimization page)",
        "Scale features first (standardize) so the bowl is round and GD converges faster",
        "Same algorithm trains Logistic Regression and Neural Networks — only the cost function changes"],
        head="Notes:")
    pg.ask("Which way and how far should β move at each step?",
           f"downhill (opposite the gradient), by α × slope: steps shrink near the bottom and stop at β₁ ≈ {path[-1]:.2f}",
           x0 + 600, y0 + 545, *c.P(path[2], J(path[2])), w=290)
    pg.ask("What happens if the learning rate α is too large?",
           "each step overshoots the minimum and lands higher up the other side → the cost grows instead of falling",
           lx + 20, y0 + 560, lx + 120, y0 + 125 + 300 + 70, w=375)

    # ---------------------------------------------------------------- ⑤ evaluation & assumptions
    x0 = 1760
    pg.panel(x0, y0, 640, 700, 5, "Evaluation & Assumptions")
    pg.math("R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum(y_i-\\hat{y}_i)^2}{\\sum(y_i-\\bar{y})^2}",
            x0 + 40, y0 + 55, 560, 60, 18)
    pg.text("R² = share of the variance of y explained by the line (0 → 1). Example: R² = 0.6",
            x0 + 40, y0 + 115, 560, 20, 11, align="center")
    pg.text("Assumptions (L.I.N.E.) — check with residual plots", x0 + 20, y0 + 150, 600, 22, 13, bold=True)
    r = seeded(3)
    good = [(xv, r.gauss(0, 0.8)) for xv in [r.uniform(0.5, 9.5) for _ in range(40)]]
    bad = [(xv, r.gauss(0, 0.12 * xv + 0.1)) for xv in [r.uniform(0.5, 9.5) for _ in range(40)]]
    for k, (data, lab) in enumerate([(good, "✔ Random band around 0 (good)"), (bad, "✘ Funnel → heteroscedasticity")]):
        gx = x0 + 40 + k * 300
        g = pg.axes(gx, y0 + 200, 240, 150, (0, 10), (-3, 3), "\\hat{y}", "e", label_size=11, origin=(0, -3))
        g.hline(0, color=RED)
        g.scatter([(a, max(-2.9, min(2.9, bb))) for a, bb in data], r=2.5)
        pg.text(lab, gx - 10, y0 + 365, 270, 20, 11, bold=True, color=GREEN if k == 0 else RED, align="center")
    pg.bullets(x0 + 20, y0 + 400, 600, [
        "<b>L</b>inearity — relationship between x and y is linear",
        "<b>I</b>ndependence — residuals are not correlated (no pattern over time)",
        "<b>N</b>ormality — residuals \\(\\epsilon \\sim N(0,\\sigma^2)\\) (see Distributions page)",
        "<b>E</b>qual variance (homoscedasticity) — constant spread of residuals",
        "Multiple regression adds: no <b>multicollinearity</b> between features (check VIF)",
        "Sensitive to outliers → inspect with Z-score / IQR (Anomaly Detection page)"],
        head="Checklist:", h=190)
    pg.ask("Can I trust the line's p-values and confidence intervals?",
           "only if residuals look like the left plot (random band); a funnel means unequal variance → "
           "transform y or use robust standard errors",
           x0 + 20, y0 + 600, x0 + 465, y0 + 385, w=600)

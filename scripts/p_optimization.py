"""Batch Gradient Descent page: derivative, gradient on contours, batch/SGD/mini-batch, optimizers,
local/global minima & saddle, convergence vs alpha, feature scaling, variant summary."""
import math
from dio import *


def contours(pg, p, a, b, levels, color="#999999", cx=0.0, cy=0.0):
    """Concentric axis-aligned ellipses of J = a*(x-cx)^2 + b*(y-cy)^2 for each level."""
    for c in levels:
        rx, ry = math.sqrt(c / a), math.sqrt(c / b)
        x0, y0 = p.P(cx - rx, cy + ry)
        x1, y1 = p.P(cx + rx, cy - ry)
        pg.ellipse(x0, y0, x1 - x0, y1 - y0, stroke=color, sw=0.75)


def path_line(pg, p, pts, color, width=1.5, dots=True, r=2.5, arrow_last=True):
    page_pts = [p.P(x, y) for x, y in pts]
    pg.line(page_pts, color=color, width=width, arrow="classic" if arrow_last else None)
    if dots:
        for q in page_pts[:-1]:
            pg.dot(*q, r=r, color=color)


def clip_pts(pts, xr, yr):
    return [(min(max(x, xr[0]), xr[1]), min(max(y, yr[0]), yr[1])) for x, y in pts]


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① derivative = slope
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 640, 1, "Derivative = slope of the cost curve", "Background: calculus (rate of change)")
    f = lambda v: 0.5 * v * v
    p = pg.axes(x0 + 80, y0 + 90, 470, 300, (-3.2, 3.2), (0, 5.2), "\\theta", "J(\\theta)", origin=(0, 0))
    p.curve(f, -3.2, 3.2, width=2, color=BLUE)
    for xt, col, lab in [(2.0, RED, "slope > 0 → move LEFT (θ decreases)"),
                         (-1.5, GREEN, "slope < 0 → move RIGHT (θ increases)")]:
        m = xt
        tan = lambda v, m=m, xt=xt: f(xt) + m * (v - xt)
        p.curve(tan, xt - 1.1, xt + 1.1, n=2, curved=False, color=col, width=1.5, dashed=True)
        p.scatter([(xt, f(xt))], color=col, r=5)
        step = -0.6 * m / abs(m)
        pg.line([p.P(xt, f(xt) + 0.35), p.P(xt + step, f(xt) + 0.35)], color=col, arrow="classic", width=2)
    pg.note("slope > 0 → step LEFT (θ decreases)", x0 + 430, y0 + 160, *p.P(2.0, f(2.0)), w=200, color=RED)
    pg.note("slope < 0 → step RIGHT (θ increases)", x0 + 30, y0 + 150, *p.P(-1.5, f(-1.5)), w=200, color=GREEN)
    p.scatter([(0, 0)], color=BLACK, r=4)
    pg.note("slope = 0 → minimum (stop)", x0 + 360, y0 + 400, *p.P(0, 0), w=170)
    pg.math("f'(\\theta) = \\lim_{h\\to 0}\\frac{f(\\theta+h)-f(\\theta)}{h}", x0 + 40, y0 + 430, 560, 50, 18)
    pg.math("\\theta := \\theta - \\alpha\\, f'(\\theta)", x0 + 40, y0 + 480, 560, 40, 18)
    pg.bullets(x0 + 20, y0 + 530, 330, [
        "The derivative is the slope of the tangent line at the current θ",
        "Subtracting \\(\\alpha f'(\\theta)\\) always moves <b>downhill</b>, opposite to the slope",
        "Step size shrinks automatically as the slope flattens near the minimum"], head="Idea:", h=100)
    pg.ask("Which way should θ move, and how far?",
           "opposite to the slope: f′(θ) > 0 → decrease θ; steps shrink as the slope → 0 at the minimum",
           x0 + 360, y0 + 525, x0 + 400, y0 + 508, w=260)

    # ---------------------------------------------------------------- ② gradient vector on contours
    x0 = 1080
    pg.panel(x0, y0, 620, 640, 2, "Partial derivatives → gradient vector",
             "Two parameters: J(θ₀, θ₁) seen from above as contour lines")
    a, b = 1.0, 3.0
    p = pg.axes(x0 + 70, y0 + 90, 400, 280, (-5, 5), (-3.5, 3.5), "\\theta_0", "\\theta_1", origin=(-5, -3.5))
    contours(pg, p, a, b, [0.5, 2, 4.5, 8, 12.5, 18])
    th = (-4.2, 2.6)
    pts = [th]
    for _ in range(12):
        x, y = pts[-1]
        pts.append((x - 0.12 * 2 * a * x, y - 0.12 * 2 * b * y))
    path_line(pg, p, pts, RED)
    gx, gy = 2 * a * th[0], 2 * b * th[1]
    nrm = math.hypot(gx, gy)
    pg.line([p.P(*th), p.P(th[0] + 1.6 * gx / nrm, th[1] + 1.6 * gy / nrm)], color=GREEN, width=2, arrow="classic")
    pg.note("\\(\\nabla J\\) points uphill (steepest ascent)", x0 + 130, y0 + 62, *p.P(th[0] + 1.0 * gx / nrm,
            th[1] + 1.0 * gy / nrm), w=220, raw=True, color=GREEN)
    pg.note("GD steps along \\(-\\nabla J\\), ⟂ to contour lines", x0 + 380, y0 + 60, *p.P(*pts[2]), w=210,
            raw=True, color=RED)
    p.scatter([(0, 0)], color=BLACK, r=4)
    pg.note("minimum", x0 + 480, y0 + 250, *p.P(0, 0), w=70)
    pg.math("\\nabla J(\\theta) = \\begin{bmatrix} \\frac{\\partial J}{\\partial\\theta_0} \\\\ "
            "\\frac{\\partial J}{\\partial\\theta_1} \\end{bmatrix}", x0 + 30, y0 + 395, 260, 80, 18)
    pg.math("\\theta := \\theta - \\alpha\\,\\nabla J(\\theta)", x0 + 300, y0 + 410, 300, 50, 18)
    pg.bullets(x0 + 20, y0 + 490, 330, [
        "Each contour = set of (θ₀, θ₁) with the same cost",
        "Partial derivative: change J along ONE parameter, keep the others fixed",
        "For linear regression: \\(\\frac{\\partial J}{\\partial\\theta_j} = \\frac{1}{m}\\sum(\\hat{y}_i-y_i)x_{ij}\\)",
        "All θⱼ are updated <b>simultaneously</b> with the same α"], head="Where:", h=130)
    pg.ask("Which direction lowers the cost fastest?",
           "−∇J: perpendicular to the contour lines, straight 'downhill'; steps shrink near the minimum",
           x0 + 360, y0 + 490, *p.P(*pts[4]), w=250)

    # ---------------------------------------------------------------- ③ batch vs sgd vs mini-batch
    x0 = 1740
    pg.panel(x0, y0, 660, 640, 3, "Batch vs Stochastic vs Mini-Batch", "How many samples per update?")
    p = pg.axes(x0 + 60, y0 + 90, 380, 260, (-5, 5), (-3.5, 3.5), "\\theta_0", "\\theta_1", origin=(-5, -3.5))
    contours(pg, p, a, b, [0.5, 2, 4.5, 8, 12.5, 18])
    r = seeded(11)
    start = (-4.3, 2.4)
    for col, noise, n, lab in [(RED, 0.0, 10, "Batch"), (GREEN, 0.30, 16, "Mini-batch"), (ORANGE, 0.75, 22, "SGD")]:
        pts = [start]
        for _ in range(n):
            x, y = pts[-1]
            pts.append((x - 0.13 * 2 * a * x + r.gauss(0, noise), y - 0.13 * 2 * b * y + r.gauss(0, noise * 0.6)))
        path_line(pg, p, clip_pts(pts, (-5, 5), (-3.5, 3.5)), col, width=1.2, r=2)
    for k, (col, lab) in enumerate([(RED, "Batch GD — smooth, direct"), (GREEN, "Mini-batch — slightly noisy"),
                                    (ORANGE, "SGD — noisy zig-zag")]):
        pg.line([(x0 + 470, y0 + 140 + k * 34), (x0 + 500, y0 + 140 + k * 34)], color=col, width=2)
        pg.text(lab, x0 + 505, y0 + 130 + k * 34, 150, 20, 11, color=col, bold=True)
    pg.ask("How many samples should each update use?",
           "mini-batch (32–512) is the usual default: nearly as smooth as batch, as fast as SGD on a GPU",
           x0 + 455, y0 + 240, *p.P(0.6, 0.05), w=195)
    pg.table(x0 + 20, y0 + 380, 620, [
        ["", "Batch GD", "Mini-batch GD", "SGD"],
        ["Samples / update", "all m", "b (32–512)", "1"],
        ["Update rule", "\\(\\frac{1}{m}\\sum_{i=1}^{m}\\nabla J_i\\)", "\\(\\frac{1}{b}\\sum_{i\\in B}\\nabla J_i\\)",
         "\\(\\nabla J_i\\)"],
        ["Speed / step", "slow", "fast (GPU)", "very fast"],
        ["Path", "smooth", "small noise", "noisy"],
        ["Memory", "whole dataset", "one batch", "one sample"],
        ["Typical use", "small data, convex", "deep learning default", "online / streaming"]], size=11)
    pg.text("Epoch = one full pass over the training data", x0 + 20, y0 + 600, 620, 20, 11, italic=True,
            align="center")

    # ---------------------------------------------------------------- ④ optimizers
    x0, y0 = 400, 920
    pg.panel(x0, y0, 640, 660, 4, "Momentum · RMSProp · Adam", "Smarter steps on long, narrow valleys")
    a2, b2 = 0.15, 3.0
    p = pg.axes(x0 + 60, y0 + 80, 420, 220, (-6, 6), (-2.6, 2.6), "\\theta_0", "\\theta_1", origin=(-6, -2.6))
    contours(pg, p, a2, b2, [0.15, 0.6, 1.35, 2.4, 3.75, 5.2])
    st = (-5.6, 1.8)
    grad = lambda x, y: (2 * a2 * x, 2 * b2 * y)
    # plain GD (zig-zag)
    pts, (x, y) = [st], st
    for _ in range(14):
        gx, gy = grad(x, y); x, y = x - 0.30 * gx, y - 0.30 * gy; pts.append((x, y))
    path_line(pg, p, clip_pts(pts, (-6, 6), (-2.6, 2.6)), ORANGE, width=1.2, r=2)
    # momentum
    pts, (x, y), vx, vy = [st], st, 0, 0
    for _ in range(14):
        gx, gy = grad(x, y); vx, vy = 0.6 * vx + 0.12 * gx, 0.6 * vy + 0.12 * gy; x, y = x - vx, y - vy
        pts.append((x, y))
    path_line(pg, p, pts, GREEN, width=1.2, r=2)
    # adam
    pts, (x, y) = [st], st
    m = [0, 0]; v = [0, 0]
    for t in range(1, 14):
        g = grad(x, y)
        for i in range(2):
            m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
        mh = [mi / (1 - 0.9 ** t) for mi in m]; vh = [vi / (1 - 0.999 ** t) for vi in v]
        x -= 0.42 * mh[0] / (math.sqrt(vh[0]) + 1e-8); y -= 0.42 * mh[1] / (math.sqrt(vh[1]) + 1e-8)
        pts.append((x, y))
    path_line(pg, p, clip_pts(pts, (-6, 6), (-2.6, 2.6)), RED, width=1.2, r=2)
    adam_end = clip_pts(pts, (-6, 6), (-2.6, 2.6))[-1]
    pg.ask("Why does Adam get down the valley faster?",
           "it sizes each parameter's step separately: less zig-zag across, faster along the valley",
           x0 + 490, y0 + 195, *p.P(*adam_end), w=145, size=9)
    for k, (col, lab) in enumerate([(ORANGE, "Plain GD (zig-zag)"), (GREEN, "Momentum"), (RED, "Adam")]):
        pg.line([(x0 + 500, y0 + 110 + k * 30), (x0 + 530, y0 + 110 + k * 30)], color=col, width=2)
        pg.text(lab, x0 + 535, y0 + 100 + k * 30, 100, 20, 11, color=col, bold=True)
    pg.text("Momentum", x0 + 20, y0 + 320, 200, 20, 12, bold=True, color=GREEN)
    pg.math("v_t = \\beta v_{t-1} + \\alpha\\nabla J,\\quad \\theta := \\theta - v_t", x0 + 20, y0 + 340, 600, 36, 15)
    pg.text("RMSProp", x0 + 20, y0 + 380, 200, 20, 12, bold=True)
    pg.math("s_t = \\beta s_{t-1} + (1-\\beta)(\\nabla J)^2,\\quad \\theta := \\theta - \\frac{\\alpha}{\\sqrt{s_t}"
            "+\\epsilon}\\nabla J", x0 + 20, y0 + 400, 600, 44, 15)
    pg.text("Adam = Momentum + RMSProp (+ bias correction)", x0 + 20, y0 + 450, 400, 20, 12, bold=True, color=RED)
    pg.math("m_t = \\beta_1 m_{t-1} + (1-\\beta_1)\\nabla J,\\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)(\\nabla J)^2",
            x0 + 20, y0 + 470, 600, 36, 14)
    pg.math("\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t},\\ \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t},\\quad "
            "\\theta := \\theta - \\frac{\\alpha\\,\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}", x0 + 20, y0 + 508, 600,
            48, 14)
    pg.bullets(x0 + 20, y0 + 565, 600, [
        "Momentum averages past gradients → damps zig-zag, speeds up along the valley",
        "RMSProp divides by recent gradient size → per-parameter learning rate",
        "Adam defaults: \\(\\alpha=0.001, \\beta_1=0.9, \\beta_2=0.999, \\epsilon=10^{-8}\\)"])

    # ---------------------------------------------------------------- ⑤ local / global minimum, saddle
    x0 = 1080
    pg.panel(x0, y0, 620, 660, 5, "Local minimum · Global minimum · Saddle point", "Non-convex cost functions")
    g = lambda v: v ** 4 / 20 - 0.6 * v * v + 0.4 * v
    p = pg.axes(x0 + 60, y0 + 90, 460, 260, (-3.6, 3.6), (-3.2, 2.6), "\\theta", "J(\\theta)", origin=(-3.6, -3.2))
    p.curve(g, -3.6, 3.6, n=60, width=2, color=BLUE)
    glob, loc, mx = -2.6, 2.29, 0.34
    p.scatter([(glob, g(glob))], color=GREEN, r=5)
    p.scatter([(loc, g(loc))], color=RED, r=5)
    p.scatter([(mx, g(mx))], color=BLACK, r=4)
    pg.note("Global minimum (lowest cost)", x0 + 70, y0 + 380, *p.P(glob, g(glob)), w=170, color=GREEN)
    pg.note("Local minimum — GD can get stuck here", x0 + 360, y0 + 380, *p.P(loc, g(loc)), w=220, color=RED)
    pg.note("Local maximum (gradient = 0 too)", x0 + 300, y0 + 70, *p.P(mx, g(mx)), w=190)
    # start point that ends in the local min
    s = 1.0
    pts = [s]
    for _ in range(6):
        s = s - 0.35 * (s ** 3 / 5 - 1.2 * s + 0.4); pts.append(s)
    for u, w in zip(pts, pts[1:]):
        pg.line([p.P(u, g(u)), p.P(w, g(w))], color=ORANGE, arrow="classic", width=1.2)
    pg.note("start here → rolls into local min", x0 + 400, y0 + 150, *p.P(1.0, g(1.0)), w=200, color=ORANGE)
    pg.ask("Will gradient descent always find the best solution?",
           f"no: started at θ = 1 it stops in the local minimum (J = {g(loc):.2f}); the global one is lower "
           f"(J = {g(glob):.2f})",
           x0 + 75, y0 + 92, *p.P(loc, g(loc)), w=190)
    # saddle
    q = pg.axes(x0 + 60, y0 + 440, 200, 140, (-1.5, 1.5), (-3.4, 3.4), "\\theta", "J", label_size=12,
                origin=(0, 0))
    q.curve(lambda v: v ** 3, -1.5, 1.5, width=2, color=BLUE)
    q.scatter([(0, 0)], color=RED, r=4)
    pg.text("Saddle / flat point: \\(J=\\theta^3\\) at θ = 0 — gradient is 0 but it is not a minimum",
            x0 + 290, y0 + 440, 310, 40, 11, raw=True)
    pg.math("J(x,y)=x^2-y^2", x0 + 290, y0 + 485, 200, 30, 14)
    pg.bullets(x0 + 290, y0 + 520, 320, [
        "Linear / logistic regression costs are <b>convex</b> → one global minimum",
        "Neural networks are non-convex: use momentum / Adam, random restarts, good init"])

    # ---------------------------------------------------------------- ⑥ convergence vs alpha
    x0 = 1740
    pg.panel(x0, y0, 660, 660, 6, "Convergence: cost vs. iterations", "Debugging gradient descent")
    p = pg.axes(x0 + 70, y0 + 90, 420, 280, (0, 40), (0, 10), "iterations", "J(\\theta)", label_size=12,
                xticks=(10, 20, 30, 40))
    p.curve(lambda k: 9 * 0.97 ** k + 0.3, 0, 40, color=ORANGE, width=2)
    p.curve(lambda k: 9 * 0.78 ** k + 0.3, 0, 40, color=GREEN, width=2)
    p.poly([(0, 9), (2, 7.2), (4, 9.3), (6, 7.0), (8, 9.6), (10, 7.4), (12, 9.9)], color=RED, width=2)
    p.hline(0.3, color=GREY)
    pg.text("α too small → slow decrease", x0 + 470, y0 + 230, 180, 20, 11, bold=True, color=ORANGE)
    pg.text("α good → fast & flat", x0 + 470, y0 + 330, 180, 20, 11, bold=True, color=GREEN)
    pg.text("α too large → cost increases / oscillates", x0 + 470, y0 + 100, 180, 34, 11, bold=True, color=RED)
    pg.note("plateau = converged", x0 + 260, y0 + 395, *p.P(30, 0.3), w=130)
    pg.math("\\text{stop when } |J^{(k)} - J^{(k-1)}| \\lt \\varepsilon", x0 + 60, y0 + 420, 520, 40, 16)
    pg.bullets(x0 + 20, y0 + 470, 620, [
        "Plot J every iteration: it must <b>decrease every step</b> (for batch GD)",
        "Try α on a log scale: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1",
        "Learning-rate schedules: step decay, \\(\\alpha_t = \\frac{\\alpha_0}{1+kt}\\), cosine, warm-up",
        "Early stopping: stop when the <b>validation</b> loss starts to rise (overfitting)"], head="Rules of thumb:")
    pg.ask("Is my learning rate α set correctly?",
           "yes if J falls every step and flattens (green); rising / oscillating (red) → lower α; very slow (orange) → raise α",
           x0 + 20, y0 + 568, x0 + 560, y0 + 352, w=620)

    # ---------------------------------------------------------------- ⑦ feature scaling
    x0, y0 = 400, 1620
    pg.panel(x0, y0, 960, 540, 7, "Feature scaling makes descent faster", "Same data, different contour shape")
    a3, b3 = 0.08, 3.0
    p = pg.axes(x0 + 60, y0 + 90, 380, 240, (-8, 8), (-2.5, 2.5), "\\theta_1\\ (size\\ m^2)",
                "\\theta_2\\ (rooms)", label_size=11, origin=(-8, -2.5))
    contours(pg, p, a3, b3, [0.2, 0.8, 1.8, 3.2, 5.0])
    pts, (x, y) = [(-7.3, 1.6)], (-7.3, 1.6)
    for _ in range(16):
        x, y = x - 0.31 * 2 * a3 * x, y - 0.31 * 2 * b3 * y; pts.append((x, y))
    path_line(pg, p, clip_pts(pts, (-8, 8), (-2.5, 2.5)), RED, width=1.2, r=2)
    pg.text("✘ Unscaled: elongated ellipses → zig-zag, many steps", x0 + 40, y0 + 350, 420, 20, 12, bold=True,
            color=RED, align="center")
    p = pg.axes(x0 + 540, y0 + 90, 240, 240, (-3, 3), (-3, 3), "z_1", "z_2", label_size=11, origin=(-3, -3))
    contours(pg, p, 1, 1, [0.25, 1, 2.25, 4, 6.25])
    pts, (x, y) = [(-2.0, 1.7)], (-2.0, 1.7)
    for _ in range(5):
        x, y = x - 0.3 * 2 * x, y - 0.3 * 2 * y; pts.append((x, y))
    path_line(pg, p, pts, GREEN, width=1.5, r=2.5)
    pg.ask("Why scale the features before gradient descent?",
           "round contours → the gradient points at the minimum: 5 steps instead of 16 zig-zags",
           x0 + 800, y0 + 100, *p.P(*pts[1]), w=150)
    pg.text("✔ Scaled: round contours → straight to the minimum", x0 + 500, y0 + 350, 330, 20, 12, bold=True,
            color=GREEN, align="center")
    pg.math("\\text{Standardization: } z = \\frac{x-\\mu}{\\sigma}", x0 + 40, y0 + 390, 420, 40, 16)
    pg.math("\\text{Min-Max: } x' = \\frac{x-x_{min}}{x_{max}-x_{min}}", x0 + 480, y0 + 390, 440, 40, 16)
    pg.bullets(x0 + 20, y0 + 440, 920, [
        "Features on very different scales (e.g. size 50–500 m² vs rooms 1–6) stretch the cost bowl",
        "Fit μ, σ (or min, max) on the <b>training set only</b>, then apply the same values to validation / test",
        "Tree-based models do not need scaling; GD-based models (linear, logistic, SVM, NN, k-NN) do"])

    # ---------------------------------------------------------------- ⑧ variants summary
    x0 = 1400
    pg.panel(x0, y0, 1000, 540, 8, "Optimization family at a glance")
    H = "background:#FFFF00"
    pg.table(x0 + 20, y0 + 50, 960, [
        ["Algorithm", "Update idea", "Pros", "Cons"],
        [f'<span style="{H}"><b>★ Batch GD</b></span>', "\\(\\theta -= \\alpha\\frac{1}{m}\\sum\\nabla J_i\\)",
         "stable, exact gradient", "slow on big data"],
        ["SGD", "\\(\\theta -= \\alpha\\nabla J_i\\) (one sample)", "fast, escapes shallow minima", "noisy"],
        ["Mini-batch GD", "average over a batch b", "GPU friendly, default", "batch size to tune"],
        ["Momentum / Nesterov", "add velocity \\(v_t\\)", "less zig-zag", "extra β"],
        ["AdaGrad", "\\(\\alpha / \\sqrt{\\sum g^2}\\)", "good for sparse features", "α decays to 0"],
        ["RMSProp", "\\(\\alpha / \\sqrt{E[g^2]}\\)", "fixes AdaGrad decay", "needs β"],
        ["Adam", "momentum + RMSProp", "robust default for DL", "may generalize worse"],
        ["Normal Equation", "\\(\\theta = (X^TX)^{-1}X^Ty\\)", "no α, exact", "\\(O(n^3)\\), linear only"],
        ["Newton / L-BFGS", "uses curvature (Hessian)", "few iterations", "expensive per step"],
        ["Genetic / PSO / Bayesian Opt.", "search without gradients", "hyperparameter tuning", "slow, many evaluations"],
    ], size=11)
    pg.ask("Which optimizer should I start with?",
           "Adam (or mini-batch SGD + momentum) for neural nets; Batch GD / Normal Equation for small convex models",
           x0 + 800, y0 + 185, x0 + 788, y0 + 253, w=190)

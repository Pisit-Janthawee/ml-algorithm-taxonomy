"""Shared helpers for the *_extra pages written by fork L (data illustration + hyperparameter search)."""
import math
from dio import *


def solve(A, b):
    """Gaussian elimination with partial pivoting (A: list of lists, b: list)."""
    n = len(A)
    M = [list(map(float, A[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12: M[c][c] = 1e-12
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def fit_ls(X, y, alpha=0.0, pen_from=1):
    """(Ridge) least squares; X rows already include the intercept column. Intercept (col 0) not penalised."""
    p = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (alpha if (i == j and i >= pen_from) else 0) for j in range(p)]
         for i in range(p)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(p)]
    return solve(A, b)


def predict(X, beta):
    return [sum(a * b for a, b in zip(r, beta)) for r in X]


def mse(y, yh):
    return sum((a - b) ** 2 for a, b in zip(y, yh)) / len(y)


def polyfeat(xs, d):
    return [[x ** k for k in range(d + 1)] for x in xs]


def kfold(n, k):
    idx = list(range(n))
    return [idx[i::k] for i in range(k)]


def lerp_color(c0, c1, t):
    t = max(0.0, min(1.0, t))
    a = [int(c0[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def heatmap(pg, x, y, cw, ch, rlabels, clabels, vals, fmt="{:.2f}", best=None, high_good=True, rtitle="", ctitle="",
            size=11):
    lo = min(min(r) for r in vals)
    hi = max(max(r) for r in vals)
    for i, rl in enumerate(rlabels):
        pg.text(str(rl), x - 70, y + i * ch, 64, ch, size, align="right")
        for j, v in enumerate(vals[i]):
            t = (v - lo) / (hi - lo + 1e-12)
            if not high_good: t = 1 - t
            fill = lerp_color("#FFFFFF", "#3333FF", 0.85 * t)
            is_best = best == (i, j)
            pg.rect(x + j * cw, y + i * ch, cw, ch, fmt.format(v), fill=HL if is_best else fill,
                    stroke=RED if is_best else "#999999", sw=3 if is_best else 0.5, size=size, bold=is_best,
                    color="#FFFFFF" if (t > 0.6 and not is_best) else None)
    for j, cl in enumerate(clabels):
        pg.text(str(cl), x + j * cw, y + len(rlabels) * ch + 2, cw, 16, size - 1, align="center")
    if ctitle: pg.text(ctitle, x, y + len(rlabels) * ch + 18, cw * len(clabels), 18, size, bold=True, align="center")
    if rtitle: pg.text(rtitle, x - 70, y - 22, 140, 18, size, bold=True)


def arrow_label(pg, x0, y0, x1, y1, label, color=BLACK):
    pg.line([(x0, y0), (x1, y1)], arrow="block", width=2, color=color)
    pg.text(label, min(x0, x1) - 20, min(y0, y1) - 22, abs(x1 - x0) + 40, 18, 11, bold=True, align="center",
            color=color)

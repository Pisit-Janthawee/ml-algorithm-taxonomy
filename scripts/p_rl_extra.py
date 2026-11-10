"""Q-Learning page extras: trading-episode data illustration + hyperparameter tuning (α, γ, ε schedule)."""
import math
from dio import *

ACTS = ["Buy", "Hold", "Sell"]
STATES = [("down", "flat"), ("down", "long"), ("up", "flat"), ("up", "long")]
COST = 0.05          # % per trade


def price_path(n=120, seed=8):
    """Alternating up / down regimes (≈ 20 bars each) so the trend state carries information."""
    r = seeded(seed)
    p, drift = [100.0], 0.5
    for t in range(1, n):
        if t % 20 == 0: drift = -drift
        p.append(p[-1] * (1 + (drift + r.gauss(0, 0.6)) / 100))
    return p


def trend(p, t, k=5):
    return "up" if t >= k and p[t] > sum(p[t - k:t]) / k else "down"


def env_step(p, t, pos, a):
    """Return (reward %, new position). Buy opens long, Sell closes, Hold keeps."""
    newpos, cost = pos, 0.0
    if a == "Buy" and pos == "flat": newpos, cost = "long", COST
    if a == "Sell" and pos == "long": newpos, cost = "flat", COST
    ret = (p[t + 1] / p[t] - 1) * 100
    return (ret if newpos == "long" else 0.0) - cost, newpos


def run_trading(p, episodes=60, alpha=0.1, gamma=0.9, seed=2, log_last=True):
    r = seeded(seed)
    Q = {s: {a: 0.0 for a in ACTS} for s in STATES}
    log = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (episodes * 0.7))
        pos = "flat"
        for t in range(5, len(p) - 1):
            s = (trend(p, t), pos)
            a = r.choice(ACTS) if r.random() < eps else max(ACTS, key=lambda x: Q[s][x])
            rew, pos2 = env_step(p, t, pos, a)
            s2 = (trend(p, t + 1), pos2)
            target = rew + gamma * max(Q[s2].values())
            before = Q[s][a]
            Q[s][a] += alpha * (target - Q[s][a])
            if ep == episodes - 1 and log_last:
                log.append((t, p[t], s, a, rew, s2, before, target, Q[s][a], eps))
            pos = pos2
    return Q, log


# ---------------------------------------------------------------- grid world for tuning
GW = 5
GOAL, TRAPS, START = (4, 0), {(2, 1), (1, 3)}, (0, 4)
MV = [(0, -1), (0, 1), (-1, 0), (1, 0)]


def gw_episode(Q, r, alpha, gamma, eps, learn=True):
    s, tot = START, 0
    for _ in range(40):
        a = r.randrange(4) if r.random() < eps else max(range(4), key=lambda i: Q[s][i])
        nx, ny = s[0] + MV[a][0], s[1] + MV[a][1]
        s2 = (min(max(nx, 0), GW - 1), min(max(ny, 0), GW - 1))
        rew, done = -1, False
        if s2 == GOAL: rew, done = 10, True
        elif s2 in TRAPS: rew, done = -10, True
        tot += rew
        if learn:
            tgt = rew + (0 if done else gamma * max(Q[s2]))
            Q[s][a] += alpha * (tgt - Q[s][a])
        s = s2
        if done: break
    return tot


def gw_curve(alpha=0.1, gamma=0.9, sched="exp", episodes=300, seeds=6):
    acc = [0.0] * episodes
    for sd in range(seeds):
        r = seeded(100 + sd)
        Q = {(x, y): [0.0] * 4 for x in range(GW) for y in range(GW)}
        for ep in range(episodes):
            eps = {"const": 0.1, "linear": max(0.02, 1 - ep / (0.6 * episodes)),
                   "exp": max(0.02, 0.985 ** ep), "none": 0.0}[sched]
            gw_episode(Q, r, alpha, gamma, eps)
            acc[ep] += gw_episode(Q, r, 0, gamma, 0.0, learn=False)   # greedy evaluation return
    acc = [a / seeds for a in acc]
    sm, out = None, []
    for v in acc:
        sm = v if sm is None else 0.9 * sm + 0.1 * v
        out.append(sm)
    return out


def build(pg: Page, cat):
    p = price_path()
    Q, log = run_trading(p)

    # ---------------------------------------------------------------- ⑦ data illustration
    x0, y0 = 400, pg.max_y() + 60
    pg.panel(x0, y0, 2000, 760, 7, "Data Illustration: one trading episode as Q-learning data",
             f"Simulated {len(p)} bars · state = (trend vs MA5, position) · actions Buy / Hold / Sell · "
             f"reward = position × next return − {COST}% cost · α = 0.1, γ = 0.9 · 60 episodes")
    # price plot with greedy actions
    lo, hi = min(p) * 0.995, max(p) * 1.005
    pp = pg.axes(x0 + 70, y0 + 90, 900, 230, (0, len(p)), (lo, hi), "t", "price", label_size=12,
                 origin=(0, lo), xticks=(0, 30, 60, 90, 120))
    pp.poly([(t, v) for t, v in enumerate(p)], color=BLUE, width=1.5)
    ma = [(t, sum(p[t - 5:t]) / 5) for t in range(5, len(p))]
    pp.poly(ma, color=GREY, dashed=True)
    pos = "flat"
    eq, eqs = 0.0, []
    for t in range(5, len(p) - 1):
        s = (trend(p, t), pos)
        a = max(ACTS, key=lambda x: Q[s][x])
        rew, pos2 = env_step(p, t, pos, a)
        eq += rew; eqs.append((t, eq))
        if pos == "flat" and pos2 == "long": pp.scatter([(t, p[t])], color=GREEN, r=5)
        if pos == "long" and pos2 == "flat": pp.scatter([(t, p[t])], color=RED, r=5)
        pos = pos2
    pg.html('<font color="#009900">●</font> Buy (greedy policy) &nbsp; <font color="#FF0000">●</font> Sell &nbsp; '
            '<font color="#808080">- - MA(5)</font>', x0 + 600, y0 + 66, 380, 20, 11)
    pg.text(f"Greedy policy P&L over the path: {eq:+.2f} %  (buy & hold: {(p[-1] / p[5] - 1) * 100:+.2f} %)",
            x0 + 70, y0 + 345, 900, 20, 12, bold=True, align="center")
    # final Q table
    rows = [["state (trend, position)", "Q(Buy)", "Q(Hold)", "Q(Sell)", "greedy action"]]
    for s in STATES:
        best = max(ACTS, key=lambda x: Q[s][x])
        rows.append([f"({s[0]}, {s[1]})"] + [f"<b>{Q[s][a]:+.2f}</b>" if a == best else f"{Q[s][a]:+.2f}"
                                               for a in ACTS] + [best])
    pg.text("Learned Q-table after 60 episodes", x0 + 1040, y0 + 66, 500, 20, 12, bold=True)
    pg.table(x0 + 1040, y0 + 90, 600, rows, size=12)
    s_uf = ("up", "flat") if ("up", "flat") in Q else STATES[2]
    b_uf = max(ACTS, key=lambda x: Q[s_uf][x])
    pg.ask("What does the trained bot do in an uptrend while flat?",
           f"{b_uf}: Q({b_uf}) = {Q[s_uf][b_uf]:+.2f} is the largest in that row → highest expected future P&L",
           x0 + 1530, y0 + 92, x0 + 1512, y0 + 158, w=440)
    pg.bullets(x0 + 1040, y0 + 230, 920, [
        "Only 4 states × 3 actions → a tiny table; real systems add more features (RSI bucket, volatility…) or use DQN",
        "Actions that do nothing (Buy while long, Sell while flat) behave like Hold",
        "Reward is the next-bar P&L, so Q(s, a) ≈ expected discounted future P&L of taking a in s"], size=11)
    # episode log table
    pg.text("Episode log (last episode, 8 consecutive rows) — each row is one Q update", x0 + 20, y0 + 380, 900,
            20, 12, bold=True)
    rows = [["t", "price", "state s", "action a", "reward r", "next s'", "Q(s,a) before",
             "TD target \\(r+\\gamma\\max Q(s',\\cdot)\\)", "Q(s,a) after"]]
    start = next((i for i, l in enumerate(log) if l[3] != "Hold"), 0)
    start = max(0, min(start - 2, len(log) - 8))
    for l in log[start:start + 8]:
        t, pr, s, a, rew, s2, b, tg, af, eps = l
        rows.append([t, f"{pr:.2f}", f"({s[0]}, {s[1]})", a, f"{rew:+.3f}", f"({s2[0]}, {s2[1]})", f"{b:+.3f}",
                     f"{tg:+.3f}", f"<b>{af:+.3f}</b>"])
    pg.table(x0 + 20, y0 + 406, 1300, rows, size=11, highlight_rows=(1,))
    l = log[start]
    pg.math(f"Q \\leftarrow {l[6]:+.3f} + 0.1\\,({l[7]:+.3f} - ({l[6]:+.3f})) = {l[8]:+.3f}",
            x0 + 20, y0 + 650, 700, 40, 16)
    pg.note("first row worked out (α = 0.1)", x0 + 740, y0 + 660, x0 + 690, y0 + 670, w=200, color=RED)
    pg.ask("How does one row of data change the Q-table?",
           f"Q moves 10 % (α) of the way from {l[6]:+.3f} toward the TD target {l[7]:+.3f} → {l[8]:+.3f}",
           x0 + 975, y0 + 445, x0 + 942, y0 + 430, w=400)
    # equity curve
    q = pg.axes(x0 + 1420, y0 + 420, 500, 230, (5, len(p)), (min(v for _, v in eqs) - 0.5,
                max(v for _, v in eqs) + 0.5), "t", "P\\&L\\ \\%", label_size=11, origin=(5, min(v for _, v in eqs) - 0.5))
    q.poly(eqs, color=GREEN, width=2)
    q.hline(0, color=GREY)
    pg.text("Cumulative P&L of the greedy policy", x0 + 1480, y0 + 372, 480, 20, 12, bold=True)

    # ---------------------------------------------------------------- ⑧ hyperparameters
    y0 = y0 + 800
    pg.panel(x0, y0, 2000, 720, 8, "Finding the optimal hyperparameters: α, γ and the ε schedule",
             "Tabular Q-learning on a 5×5 grid world (start bottom-left, goal +10 top-right, 2 traps −10, −1 per step) · "
             "greedy return after each episode, averaged over 6 seeds")
    groups = [("Learning rate α (γ = 0.9, ε exp-decay)", [("α = 0.01", dict(alpha=0.01)), ("α = 0.1", dict(alpha=0.1)),
                                                          ("α = 0.5", dict(alpha=0.5))]),
              ("Discount γ (α = 0.1, ε exp-decay)", [("γ = 0.5", dict(gamma=0.5)), ("γ = 0.9", dict(gamma=0.9)),
                                                      ("γ = 0.99", dict(gamma=0.99))]),
              ("Exploration schedule (α = 0.1, γ = 0.9)", [("ε = 0 (no exploration)", dict(sched="none")),
                                                           ("ε = 0.1 constant", dict(sched="const")),
                                                           ("ε exp-decay 0.985ᵉᵖ", dict(sched="exp"))])]
    cols = [RED, ORANGE, GREEN]
    summary = []
    for gi, (title, cfgs) in enumerate(groups):
        gx = x0 + 40 + gi * 650
        pg.text(title, gx, y0 + 66, 600, 20, 13, bold=True)
        curves = [(lab, gw_curve(**kw)) for lab, kw in cfgs]
        lo = min(min(c) for _, c in curves) - 1
        g = pg.axes(gx + 50, y0 + 130, 480, 240, (0, 300), (lo, 6), "episode", "return", label_size=11,
                    origin=(0, lo), xticks=(0, 100, 200, 300), yticks=(-30, -20, -10, 0, 5))
        g.hline(3, color=GREY)
        for (lab, c), col in zip(curves, cols):
            g.poly([(i, v) for i, v in enumerate(c)][::3], color=col, width=2)
            summary.append((title.split(" (")[0], lab, sum(c[-50:]) / 50, next((i for i, v in enumerate(c) if v >= 2.5), None)))
        for k, ((lab, c), col) in enumerate(zip(curves, cols)):
            pg.text(lab, gx + 60, y0 + 385 + k * 18, 300, 16, 11, bold=True, color=col)
        g.text("optimal path = +3 (7 steps × −1 + 10)", 300, 3, w=230, align="right", dy=-10, size=9, color="#555555")
    rows = [["Hyperparameter", "Setting", "final return (last 50 ep.)", "episodes until return ≥ 2.5"]]
    hl = ()
    for i, (h, lab, fin, conv) in enumerate(summary):
        rows.append([h, lab, f"{fin:+.2f}", "never" if conv is None else conv])
    for g0 in range(0, 9, 3):                       # highlight the fastest converging setting of each group
        grp = [(summary[g0 + k][3] if summary[g0 + k][3] is not None else 9999, g0 + k) for k in range(3)]
        hl += (min(grp)[1] + 1,)
    pg.table(x0 + 40, y0 + 450, 1100, rows, size=11, h=210, highlight_rows=hl)
    win = ", ".join(summary[i - 1][1] for i in hl)
    pg.ask("Which α, γ and ε schedule should I pick?",
           f"the highlighted rows reach the optimal return fastest in each group: {win}",
           x0 + 1180, y0 + 640, x0 + 828, y0 + 450 + (hl[0] + 0.5) * 21,
           targets=[(x0 + 828, y0 + 450 + (i + 0.5) * 21) for i in hl[1:]], w=780)
    pg.bullets(x0 + 1180, y0 + 450, 800, [
        "<b>α</b>: too small learns slowly; too large is noisy (fine here because rewards are deterministic)",
        "<b>γ</b>: small γ is short-sighted — the distant goal is worth almost nothing, so the agent may not find it",
        "<b>ε</b>: here ε = 0 still works because Q starts at 0, above the true (negative) values → optimistic initialization explores for free; in noisy / trading environments a decaying ε is needed",
        "Tune on <b>learning curves</b> (several seeds!), pick the setting with the best final return and fastest convergence",
        "For trading: evaluate the learned policy on a later, unseen price period (walk-forward)"], head="Reading the curves:",
        size=11)

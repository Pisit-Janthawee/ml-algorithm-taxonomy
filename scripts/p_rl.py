"""Q-Learning page: MDP loop, grid world policy, Q-table + Bellman update, ε-greedy, discount, learning curve."""
import math
from dio import *

# ---- 4x4 grid world solved by value iteration (deterministic moves) ----
GOAL, TRAP, WALL, START = (3, 0), (3, 1), (1, 1), (0, 3)
MOVES = {"↑": (0, -1), "↓": (0, 1), "←": (-1, 0), "→": (1, 0)}
GAMMA = 0.9


def _step(s, a):
    nx, ny = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    if not (0 <= nx < 4 and 0 <= ny < 4) or (nx, ny) == WALL:
        nx, ny = s
    r = 10 if (nx, ny) == GOAL else -10 if (nx, ny) == TRAP else -1
    return (nx, ny), r


def _solve():
    V = {(x, y): 0.0 for x in range(4) for y in range(4)}
    for _ in range(100):
        for s in V:
            if s in (GOAL, TRAP, WALL):
                continue
            V[s] = max(r + GAMMA * V[n] for n, r in (_step(s, a) for a in MOVES))
    pol = {}
    for s in V:
        if s in (GOAL, TRAP, WALL):
            continue
        pol[s] = max(MOVES, key=lambda a: (lambda n, r: r + GAMMA * V[n])(*_step(s, a)))
    return V, pol


def build(pg: Page, cat):
    # ---------------------------------------------------------------- ① agent-environment loop
    x0, y0 = 400, 240
    pg.panel(x0, y0, 640, 640, 1, "Agent ↔ Environment (Markov Decision Process)",
             "Background: states, actions, rewards")
    ag = pg.rect(x0 + 230, y0 + 90, 180, 70, "<b>Agent</b><br><font style='font-size:10px'>(trading bot, policy π)</font>",
                 raw=True, fill=LIGHTBLUE, size=15, rounded=True)
    env = pg.rect(x0 + 230, y0 + 300, 180, 70,
                  "<b>Environment</b><br><font style='font-size:10px'>(market / MT5)</font>", raw=True,
                  fill="#D5E8D4", size=15, rounded=True)
    pg.line([(x0 + 410, y0 + 125), (x0 + 520, y0 + 125), (x0 + 520, y0 + 335), (x0 + 410, y0 + 335)],
            arrow="classic", width=1.5)
    pg.math("a_t", x0 + 528, y0 + 210, 40, 26, 16)
    pg.text("action: Buy / Hold / Sell", x0 + 470, y0 + 236, 160, 18, 10, color="#555555")
    pg.line([(x0 + 230, y0 + 320), (x0 + 120, y0 + 320), (x0 + 120, y0 + 115), (x0 + 230, y0 + 115)],
            arrow="classic", width=1.5, color=BLUE)
    pg.line([(x0 + 230, y0 + 350), (x0 + 80, y0 + 350), (x0 + 80, y0 + 145), (x0 + 230, y0 + 145)],
            arrow="classic", width=1.5, color=GREEN)
    pg.math("s_{t+1}", x0 + 124, y0 + 200, 50, 26, 15, color=BLUE)
    pg.math("r_{t+1}", x0 + 24, y0 + 240, 50, 26, 15, color=GREEN)
    pg.text("new state (trend, RSI…)", x0 + 125, y0 + 228, 110, 30, 10, color=BLUE)
    pg.text("reward (PnL)", x0 + 14, y0 + 266, 70, 30, 10, color=GREEN)
    pg.math("\\text{MDP} = (S,\\ A,\\ P(s'\\mid s,a),\\ R,\\ \\gamma)", x0 + 60, y0 + 395, 520, 40, 18)
    pg.bullets(x0 + 20, y0 + 445, 360, [
        "\\(S\\) states — what the agent observes (e.g. trend + RSI zone + position)",
        "\\(A\\) actions — Buy, Hold, Sell;  \\(R\\) reward — profit / loss after the action",
        "\\(P\\) transition — how the market moves (unknown → Q-Learning is <b>model-free</b>)",
        "\\(\\gamma\\) discount — how much future rewards matter (0 → greedy, 1 → far-sighted)",
        "Markov property: the next state depends only on the current state and action",
        "<b>Goal:</b> learn a policy \\(\\pi(s)\\) that maximizes total discounted reward"],
        head="Where:", h=190)
    pg.ask("What must the agent see, do and get back each step?",
           "state s (trend, RSI, position) → action a (Buy / Hold / Sell) → reward r (PnL); γ = how far it plans",
           x0 + 390, y0 + 452, x0 + 452, y0 + 430, w=230)

    # ---------------------------------------------------------------- ② grid world
    x0 = 1080
    pg.panel(x0, y0, 620, 640, 2, "Grid World: the learned policy", "Reward −1 per step, +10 goal, −10 trap, γ = 0.9")
    V, pol = _solve()
    C, gx, gy = 90, x0 + 60, y0 + 90
    for (cx, cy), v in V.items():
        X, Y = gx + cx * C, gy + cy * C
        if (cx, cy) == WALL:
            pg.rect(X, Y, C, C, "wall", fill=GREY, stroke=BLACK, color="#FFFFFF", size=11)
        elif (cx, cy) == GOAL:
            pg.rect(X, Y, C, C, "<b>GOAL</b><br>+10", fill="#D5E8D4", stroke=BLACK, raw=True, size=13)
        elif (cx, cy) == TRAP:
            pg.rect(X, Y, C, C, "<b>TRAP</b><br>−10", fill=LIGHTRED, stroke=BLACK, raw=True, size=13)
        else:
            pg.rect(X, Y, C, C, fill=HL if (cx, cy) == START else "none")
            pg.text(pol[(cx, cy)], X, Y + 10, C, 44, 30, align="center", color=BLUE, bold=True)
            pg.text(f"V = {v:.2f}", X, Y + C - 26, C, 18, 10, align="center", color="#555555")
            if (cx, cy) == START:
                pg.text("START", X + 2, Y + 2, 50, 14, 9, bold=True)
    # optimal path
    s, path = START, [START]
    while s not in (GOAL, TRAP) and len(path) < 12:
        s, _ = _step(s, pol[s]); path.append(s)
    pg.line([(gx + px * C + C / 2 + 22, gy + py * C + C / 2 + 8) for px, py in path], color=RED, width=2,
            dashed=True, arrow="classic")
    pg.text(f"Optimal path from START: {len(path) - 1} steps (red dashed)", x0 + 60, y0 + 460, 400, 18, 11,
            color=RED)
    cell = (3, 2)
    pg.ask("What should the agent do in each cell?",
           f"follow the arrow (best action): next to the TRAP it goes {pol.get(cell, '←')} to avoid −10; "
           f"from START it needs {len(path) - 1} steps",
           x0 + 432, y0 + 255, gx + 4 * C - 6, gy + 2.5 * C, w=180)
    pg.bullets(x0 + 20, y0 + 490, 580, [
        "Arrow = best action \\(\\pi(s)=\\arg\\max_a Q(s,a)\\);  V = value of the state = \\(\\max_a Q(s,a)\\)",
        "Values grow as cells get closer to the goal; cells next to the trap learn to avoid it",
        "Values computed with the Bellman equation (value iteration) — Q-Learning learns the same by trial & error"],
        head="Notes:")

    # ---------------------------------------------------------------- ③ Q-table + Bellman
    x0 = 1740
    pg.panel(x0, y0, 660, 640, 3, "Q-Table & the Bellman Update", "Background: Bellman equation, temporal difference")
    pg.math("Q(s,a) \\leftarrow Q(s,a) + \\alpha\\Big[\\,r + \\gamma\\max_{a'}Q(s',a') - Q(s,a)\\Big]", x0 + 30,
            y0 + 75, 600, 50, 19)
    pg.note("learning rate", x0 + 160, y0 + 140, x0 + 262, y0 + 105, w=90, color=RED)
    pg.note("reward now", x0 + 250, y0 + 135, x0 + 330, y0 + 108, w=80, color=GREEN)
    pg.note("discount × best future value", x0 + 360, y0 + 150, x0 + 430, y0 + 112, w=170, color=BLUE)
    pg.text("[ … ] = TD error = (TD target) − (old estimate)", x0 + 30, y0 + 178, 600, 18, 11, align="center",
            italic=True)
    rows = [["state \\ action", "Buy", "Hold", "Sell"],
            ["Uptrend", "<b>4.00 → 4.16</b>", "1.20", "−2.10"],
            ["Sideways", "0.30", "0.80", "0.10"],
            ["Downtrend", "−1.80", "0.40", "3.60"]]
    pg.table(x0 + 60, y0 + 205, 540, rows, size=13, highlight_rows=(1,), h=120)
    pg.ask("Which action is best in an Uptrend?",
           "Buy: the highest Q in the row (4.16 after the update) = best guess of future PnL from here",
           x0 + 455, y0 + 195, x0 + 438, y0 + 267, w=190)
    pg.text("Worked update:  s = Uptrend, a = Buy,  r = +2,  s' = Uptrend,  α = 0.1,  γ = 0.9", x0 + 30, y0 + 338, 600,
            20, 12, bold=True)
    pg.math("\\max_{a'}Q(\\text{Uptrend},a') = 4.00", x0 + 30, y0 + 362, 600, 30, 14)
    pg.math("Q \\leftarrow 4.00 + 0.1\\,[\\,2 + 0.9(4.00) - 4.00\\,] = 4.00 + 0.1(1.60) = 4.16", x0 + 30, y0 + 395,
            600, 36, 14)
    pg.bullets(x0 + 20, y0 + 450, 620, [
        "Rows = states, columns = actions; start with all zeros",
        "Policy = take the column with the highest value in the current row (greedy)",
        "<b>Off-policy</b>: target uses the best next action even if the agent explored",
        "SARSA uses the action actually taken instead of max (on-policy)",
        "Too many states for a table → approximate Q with a neural network = <b>DQN</b>"],
        head="Notes:")

    # ---------------------------------------------------------------- ④ epsilon-greedy
    x0, y0 = 400, 920
    pg.panel(x0, y0, 640, 700, 4, "ε-Greedy: Explore vs. Exploit", "Background: exploration strategy")
    d = pg.ellipse(x0 + 230, y0 + 80, 180, 70, "random number<br>u ~ U(0, 1)", raw=True, size=12)
    e1 = pg.rect(x0 + 60, y0 + 210, 200, 70, "<b>Explore</b><br>random action", raw=True, fill="#FFE6CC", size=13)
    e2 = pg.rect(x0 + 380, y0 + 210, 200, 70, "<b>Exploit</b><br>\\(\\arg\\max_a Q(s,a)\\)", raw=True, fill="#D5E8D4",
                 size=13)
    pg.connect(d, e1, orth=False, label="u < ε")
    pg.connect(d, e2, orth=False, label="u ≥ ε")
    pg.text("try new things — may find better trades", x0 + 40, y0 + 285, 240, 30, 10, align="center")
    pg.text("use what the agent already knows", x0 + 360, y0 + 285, 240, 30, 10, align="center")
    pg.text("ε decay over episodes", x0 + 20, y0 + 330, 300, 20, 13, bold=True)
    p = pg.axes(x0 + 80, y0 + 370, 460, 170, (0, 1000), (0, 1.1), "\\text{episode}", "\\varepsilon", label_size=12,
                xticks=(0, 250, 500, 750, 1000), yticks=(0.5, 1.0))
    eps = lambda t: max(0.05, 0.997 ** t)
    p.curve(eps, 0, 1000, n=60, color=BLUE, width=2)
    p.hline(0.05, color=RED)
    pg.math("\\varepsilon_t = \\max(\\varepsilon_{min},\\ \\varepsilon_0\\cdot 0.997^{t})", x0 + 260, y0 + 380, 300,
            36, 13)
    pg.note("mostly exploring", x0 + 130, y0 + 350, *p.P(40, eps(40)), w=110)
    pg.note("ε_min = 0.05: keep a little exploration", x0 + 330, y0 + 480, *p.P(900, 0.05), w=210, color=RED)
    pg.bullets(x0 + 20, y0 + 575, 330, [
        "Early: ε ≈ 1 → mostly random, fills the Q-table;  later: ε small → mostly greedy",
        "Without exploration the agent can get stuck with the first action that looked good"],
        head="Notes:", h=90)
    pg.ask("How much does the agent still explore at episode 500?",
           f"ε = {eps(500):.2f}: about {eps(500) * 100:.0f} % random actions; it reaches the 5 % floor near episode 1000",
           x0 + 365, y0 + 570, *p.P(500, eps(500)), w=255)

    # ---------------------------------------------------------------- ⑤ discount
    x0 = 1080
    pg.panel(x0, y0, 620, 700, 5, "Discount Factor γ & the Return", "Background: value of future rewards")
    pg.math("G_t = r_{t+1} + \\gamma r_{t+2} + \\gamma^2 r_{t+3} + \\dots = \\sum_{k=0}^{\\infty}\\gamma^k r_{t+k+1}",
            x0 + 30, y0 + 70, 560, 50, 17)
    rewards = [1, 1, 1, 1, 10]
    for k, g in enumerate([0.9, 0.5]):
        yy = y0 + 150 + k * 210
        pg.text(f"γ = {g}", x0 + 30, yy, 100, 20, 14, bold=True, color=BLUE if k == 0 else ORANGE)
        p = pg.axes(x0 + 90, yy + 30, 330, 130, (0.3, 5.7), (0, 11), "t", "", label_size=11)
        p.bars([(i + 1, r_) for i, r_ in enumerate(rewards)], 0.5, fill="#DDDDDD", stroke=GREY)
        p.bars([(i + 1, r_ * g ** i) for i, r_ in enumerate(rewards)], 0.5, fill=BLUE if k == 0 else ORANGE,
               opacity=80, labels=True)
        for i in range(5):
            p.math(f"\\gamma^{i}", i + 1, 0, w=40, dy=14, size=10)
        G = sum(r_ * g ** i for i, r_ in enumerate(rewards))
        pg.html(f'<span style="background:{HL}">\\(G = {G:.2f}\\)</span>', x0 + 440, yy + 70, 150, 30, 15,
                valign="middle")
        if k == 0:
            pg.ask("What is the +10 reward at step 5 worth today?",
                   f"γ = 0.9 → {10 * 0.9 ** 4:.2f};  γ = 0.5 → only {10 * 0.5 ** 4:.2f}: higher γ = more patient agent",
                   x0 + 435, yy + 112, *p.P(5, 10 * 0.9 ** 4), w=175)
    pg.text("grey = raw reward, colour = discounted reward (big +10 at step 5)", x0 + 30, y0 + 570, 560, 18, 10,
            italic=True, color="#555555")
    pg.bullets(x0 + 20, y0 + 595, 580, [
        "γ = 0.9: the big future reward still counts (6.56 of 10) → patient strategy",
        "γ = 0.5: future reward almost ignored (0.63) → short-term strategy"],
        head="Notes:")

    # ---------------------------------------------------------------- ⑥ algorithm + learning curve
    x0 = 1740
    pg.panel(x0, y0, 660, 700, 6, "Q-Learning Algorithm & Learning Curve")
    pg.rect(x0 + 30, y0 + 60, 600, 190,
            "<b>Pseudo-code</b><br><font face='Courier New'>Q = zeros(states, actions)<br>"
            "for episode in range(N):<br>&nbsp;&nbsp;s = env.reset()<br>&nbsp;&nbsp;while not done:<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;a = random() if rand() &lt; ε else argmax(Q[s])<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;s2, r, done = env.step(a)<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;Q[s,a] += α·(r + γ·max(Q[s2]) − Q[s,a])<br>"
            "&nbsp;&nbsp;&nbsp;&nbsp;s = s2<br>&nbsp;&nbsp;ε = max(ε_min, ε·decay)</font>",
            fill="#F5F5F5", stroke="#999999", size=12, align="left", valign="top", raw=True)
    r = seeded(5)
    curve = [-8 + 14 * (1 - math.exp(-e / 60)) + r.gauss(0, 2.2) for e in range(0, 300, 5)]
    p = pg.axes(x0 + 80, y0 + 300, 500, 200, (0, 300), (-14, 10), "\\text{episode}", "\\text{total reward}",
                label_size=11, origin=(0, -14), xticks=(0, 100, 200, 300), yticks=(-10, 0, 10))
    p.hline(0, color=GREY)
    p.poly([(e, max(-13.5, min(9.8, v))) for e, v in zip(range(0, 300, 5), curve)], color="#99AAFF", width=1)
    p.curve(lambda e: -8 + 14 * (1 - math.exp(-e / 60)), 0, 300, color=BLUE, width=2.5)
    pg.note("noisy: exploration + random market", x0 + 200, y0 + 515, *p.P(40, curve[8]), w=200, color="#5566CC")
    pg.note("moving average converges", x0 + 420, y0 + 290, *p.P(260, 5.8), w=160, color=BLUE)
    pg.ask("Has the agent finished learning?",
           "yes once the average flattens (≈ episode 200); the leftover noise = exploration + random market",
           x0 + 335, y0 + 398, *p.P(220, -8 + 14 * (1 - math.exp(-220 / 60))), w=240)
    pg.bullets(x0 + 20, y0 + 560, 620, [
        "Tabular Q-Learning converges to the optimal Q* if all state–action pairs are visited often enough",
        "In your EA: state = signal features, action = Buy / Hold / Sell, reward = PnL after spread & slippage",
        "Validate with backtesting on unseen data — RL overfits to the training period easily"],
        head="Notes:")

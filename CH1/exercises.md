## Exercise 1.1 — What happens during self-play?

Both sides share (or each maintain) a value table and both update after every move. The opponent is no longer a fixed environment — it's **also learning and changing**. This means the environment seen by each player is **non-stationary**: as one player improves, the other faces a harder opponent, which changes which states are valuable, which changes the policy, which changes the opponent again...

This is a **moving target problem**. Standard RL convergence guarantees assume a stationary environment. Here neither player has that.


### Would it learn a different policy?

**Yes, almost certainly.** A few things happen:

**It would likely converge toward stronger play.** Against a random opponent, you can win with sloppy strategies — the opponent hands you victories without punishing your mistakes. Against yourself, every weakness gets exploited, forcing both sides to patch their play. The resulting policy is stress-tested in a way random-opponent training never achieves.

**It would discover more balanced, symmetric strategies.** A random opponent doesn't punish you for ignoring certain board positions. Self-play forces both sides to fight over every advantage, so the learned values reflect the true competitive worth of each position.

**The two sides would co-evolve.** Early on both play badly, so they learn from bad games. But as both improve simultaneously, the games get better and the values get refined — this is essentially what made AlphaGo and AlphaZero so powerful.


### The key risk: cycling and local equilibria

Because both policies change simultaneously, you can get **cycles** — player A adapts to beat player B's current strategy, then B adapts back, and you loop without converging to anything stable. There's no guarantee of reaching a true Nash equilibrium this way, especially with a simple TD learning setup.

You might also get **co-adaptation** — both players learn a set of moves that beat *each other specifically*, but would lose against a third party playing differently. The policy is optimized for the co-evolved opponent, not for general play.


## Exercise 1.2 — Symmetries

**How to amend the learning process:** Map all symmetrically equivalent board positions (rotations × 4, reflections × 4 = 8 transforms) to a single canonical state. Before looking up or updating a value, transform the board to its canonical form. This reduces the state space by up to a factor of 8.

**Why it helps:** Faster learning — each real game experience now updates 8 positions' worth of knowledge simultaneously. The value table also shrinks, reducing memory and generalization time.

**But should we, if the opponent doesn't?**
This is the subtle part. If the opponent does *not* treat symmetric positions equivalently, then they may play differently in symmetric positions — meaning those positions are *not* actually equivalent from our agent's perspective. The opponent's policy is part of the environment, and a non-symmetric opponent makes the MDP non-symmetric.

**Conclusion:** Symmetrically equivalent positions should only be forced to share a value if the *entire environment* (including the opponent) respects those symmetries. If the opponent breaks symmetry, conflating those states loses information and can hurt performance. The value of a state reflects expected return under a specific opponent — and that opponent may behave differently in symmetric-looking boards.

---

## Exercise 1.3 — Greedy Play

A fully greedy player (ε = 0, no exploration) would likely learn **worse** long-term, despite possibly winning more early on.

**Problems:**
- It never discovers that moves it initially rated poorly might actually be better — early value estimates are noisy, and without exploration those errors never get corrected.
- It can get trapped in a **local optimum**: a strategy that beats weak early opponents but never develops the moves needed to beat stronger play.
- Many states simply never get visited, so their values remain at the arbitrary initial estimate forever — the agent has a systematically incomplete model of the board.

Exploration is what allows the agent to collect the experience needed to *refine* its value estimates. A greedy player is exploiting a map it never finishes drawing.

---

## Exercise 1.4 — Learning from Exploration

When you make an exploratory move, you are taking an action you don't believe is optimal. Whether you learn from that move's outcome changes what your values converge to:

**Not learning from exploratory moves** → it means that if we DO an EXPLANATORY MOVE than we **DO NOT** PERFORM THE UPDATING RULE of PREVIOUS STATE → similiar to OFF-POLICY ("how often would I win if I always played greedily from here?") → values converge to the probability of winning from each state *under the optimal (greedy) policy*. The exploratory moves happen but their outcomes don't corrupt the value estimates — you're estimating what would happen if you always played your best.

**Learning from exploratory moves** → it means that if we DO an EXPLANATORY MOVE than we **DO** PERFORM THE UPDATING RULE of PREVIOUS STATE → similiar to ON-POLICY ("how often do I win given that I keep exploring randomly sometimes?" ) → values converge to the probability of winning from each state *under the mixed policy* (greedy + random exploration). The values are "diluted" by the randomness injected by exploration.

**Which is better?**
Learning from exploratory moves gives you values that reflect your *actual behavior*, including the exploration. But since you're continuing to explore, those values are tied to a suboptimal policy. The first approach — not learning from exploratory moves — gives values that reflect the *optimal* policy, which is a cleaner target and leads to **more wins**, because you're learning what the best play is worth, not what your noisy practice behavior is worth.

This distinction is essentially the difference between **on-policy** (learn from what you actually do) and a form of **off-policy** learning (learn about the greedy policy while behaving with exploration).

---

## Exercise 1.5 — Other Improvements

Several natural extensions:

**Better value initialization:** Instead of 0.5 everywhere, initialize with domain knowledge — e.g. give winning positions 1, losing 0, and use a heuristic (center/corner preference) for others. This dramatically speeds up early learning.

**Opponent modeling:** Rather than treating the opponent as a fixed environment, maintain a model of their tendencies and adapt the policy to exploit their weaknesses specifically.

**Eligibility traces / TD(λ):** Instead of updating only the immediately preceding state after each move, propagate credit back through the full trajectory. This speeds up learning in long games where the reward (win/loss) is only observed at the end.

**Adaptive ε:** Start with high exploration and decay it over time as value estimates stabilize — rather than keeping exploration constant forever.

# REL301m — Reinforcement Learning · Study Notes

This folder contains **39 study-note markdown files** that cover the entire `REL301m: Reinforcement Learning` course (the same PowerPoint decks as the `.pptx` / `.ppt` files in this folder). Each `.md` is a revision-friendly summary of one lecture: key terms in **bold**, formulas in `code blocks`, light analogies/jokes, and a `## Key Takeaways` section at the end.

## How to use

- Pick the lecture you want to revise → open the matching `.md`.
- Reading order = the numbered order below (Bandits → MDPs → DP → MC → TD → FA → Policy Gradient → Review).
- For quick recall before an exam, just read the `Key Takeaways` section of each file.
- The **last file** (`6. Review course.md`) is the full-course cheat-sheet recap.

---

## Chapter 1 — Foundations

| #   | Lecture                                             | Note                                                                                                                                 |
| --- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 0   | Course Introduction                                 | [0. Course Introduction.md](./0.%20Course%20Introduction.md)                                                                         |
| 1   | 1.1 The K-Armed Bandit Problem                      | [1.1. The K-Armed Bandit Problem.md](./1.1.%20The%20K-Armed%20Bandit%20Problem.md)                                                   |
| 2   | 1.2 Estimating Action Values                        | [1.2. Estimating Action Values.md](./1.2.%20Estimating%20Action%20Values.md)                                                         |
| 3   | 1.3 Exploration vs. Exploitation Tradeoff           | [1.3 Exploration vs. Exploitation Tradeoff.md](./1.3%20Exploration%20vs.%20Exploitation%20Tradeoff.md)                               |
| 4   | 1.4 Introduction to Markov Decision Processes       | [1.4 Introduction to Markov Decision Processes.md](./1.4%20Introduction%20to%20Markov%20Decision%20Processes.md)                     |
| 5   | 1.5 Goal of Reinforcement Learning                  | [1.5 Goal of Reinforcement Learning.md](./1.5%20Goal%20of%20Reinforcement%20Learning.md)                                             |
| 6   | 1.6 Continuing Tasks                                | [1.6 Continuing Tasks.md](./1.6%20Continuing%20Tasks.md)                                                                             |
| 7   | 1.7 Policies and Value Functions                    | [1.7 Policies and Value Functions.md](./1.7%20Policies%20and%20Value%20Functions.md)                                                 |
| 8   | 1.8 Bellman Equations                               | [1.8 Bellman Equations.md](./1.8%20Bellman%20Equations.md)                                                                           |
| 9   | 1.9 Optimality (Optimal Policies & Value Functions) | [1.9 Optimality (Optimal Policies & Value Functions).md](./1.9%20Optimality%20%28Optimal%20Policies%20%26%20Value%20Functions%29.md) |
| 10  | 1.10 Policy Evaluation (Prediction)                 | [1.10 Policy Evaluation (Prediction).md](./1.10%20Policy%20Evaluation%20%28Prediction%29.md)                                         |
| 11  | 1.11 Policy Iteration (Control)                     | [1.11 Policy Iteration (Control) .md](./1.11%20Policy%20Iteration%20%28Control%29%20.md)                                             |
| 12  | 1.12 Generalized Policy Iteration                   | [1.12 Generalized Policy Iteration.md](./1.12%20Generalized%20Policy%20Iteration.md)                                                 |

## Chapter 2 — Model-Free Prediction & Control (MC + TD)

| # | Lecture | Note |
|---|---|---|
| 13 | 2.1 Introduction to Monte-Carlo Methods | [2.1 Introduction to Monte-Carlo Methods.md](./2.1%20Introduction%20to%20Monte-Carlo%20Methods.md) |
| 14 | 2.2 Monte-Carlo for Control | [2.2 Monte-Carlo for Control.md](./2.2%20Monte-Carlo%20for%20Control.md) |
| 15 | 2.3 Exploration Methods for Monte-Carlo | [2.3 Exploration Methods for Monte-Carlo.md](./2.3%20Exploration%20Methods%20for%20Monte-Carlo.md) |
| 16 | 2.4 Off-policy learning for prediction | [2.4 Off-policy learning for prediction.md](./2.4%20Off-policy%20learning%20for%20prediction.md) |
| 17 | 2.5 Introduction to Temporal Difference Learning | [2.5 Introduction to Temporal Difference Learning.md](./2.5%20Introduction%20to%20Temporal%20Difference%20Learning.md) |
| 18 | 2.6 Advantages of Temporal Difference | [2.6 Advantages of Temporal Difference.md](./2.6%20Advantages%20of%20Temporal%20Difference.md) |
| 19 | 2.7 Temporal Difference for Control (SARSA) | [2.7 Temporal Difference for Control.md](./2.7%20Temporal%20Difference%20for%20Control.md) |
| 20 | 2.8 Off-policy TD Control — Q-learning | [2.8 Off-policy Temporal Difference Control Q-learning.md](./2.8%20Off-policy%20Temporal%20Difference%20Control%20Q-learning.md) |
| 21 | 2.9 Expected Sarsa | [2.9 Expected Sarsa.md](./2.9%20Expected%20Sarsa.md) |
| 22 | 2.10 Define model in Reinforcement Learning | [2.10 Define model in Reinforcement Learning.md](./2.10%20Define%20model%20in%20Reinforcement%20Learning.md) |
| 23 | 2.11 Define Planning in Reinforcement Learning | [2.11 Define Planning in Reinforcement Learning.md](./2.11%20Define%20Planning%20in%20Reinforcement%20Learning.md) |
| 24 | 2.12 Dyna as a formalism for planning | [2.12 Dyna as a formalism for planning.md](./2.12%20Dyna%20as%20a%20formalism%20for%20planning.md) |
| 25 | 2.13 Dealing with inaccurate models | [2.13 Dealing with inaccurate models.md](./2.13%20Dealing%20with%20inaccurate%20models.md) |

## Chapter 3 — Function Approximation & Policy Gradient

| # | Lecture | Note |
|---|---|---|
| 26 | 3.1 Estimating Value Functions as Supervised Learning | [3.1 Estimating Value Functions as Supervised Learning.md](./3.1%20Estimating%20Value%20Functions%20as%20Supervised%20Learning.md) |
| 27 | 3.2 The Objective for On-policy Prediction | [3.2 The Objective for On-policy Prediction.md](./3.2%20The%20Objective%20for%20On-policy%20Prediction.md) |
| 28 | 3.3 The Objective for Temporal Difference | [3.3 The Objective for Temporal Difference.md](./3.3%20The%20Objective%20for%20Temporal%20Difference.md) |
| 29 | 3.4 Linear Temporal Difference | [3.4 Linear Temporal Difference.md](./3.4%20Linear%20Temporal%20Difference.md) |
| 30 | 3.5 Feature Construction for Linear Methods | [3.5 Feature Construction for Linear Methods.md](./3.5%20Feature%20Construction%20for%20Linear%20Methods.md) |
| 31 | 3.6 Episodic Sarsa with Function Approximation | [3.6 Episodic Sarsa with Function Approximation.md](./3.6%20Episodic%20Sarsa%20with%20Function%20Approximation.md) |
| 32 | 3.7 Exploration under Function Approximation | [3.7 Exploration under Function Approximation.md](./3.7%20Exploration%20under%20Function%20Approximation.md) |
| 33 | 3.8 Understand Average Reward | [3.8 Understand Average Reward.md](./3.8%20Understand%20Average%20Reward.md) |
| 34 | 3.9 Learning Parameterized Policies | [3.9 Learning Parameterized Policies.md](./3.9%20Learning%20Parameterized%20Policies.md) |
| 35 | 3.10 Policy Gradient for Continuing Tasks | [3.10 Policy Gradient for Continuing Tasks.md](./3.10%20Policy%20Gradient%20for%20Continuing%20Tasks.md) |
| 36 | 3.11 Actor-Critic for Continuing Tasks | [3.11 Actor-Critic for Continuing Tasks.md](./3.11%20Actor-Critic%20for%20Continuing%20Tasks.md) |
| 37 | 3.12 Policy Parameterizations | [3.12 Policy Parameterizations.md](./3.12%20Policy%20Parameterizations.md) |

## Course Review

| # | Lecture | Note |
|---|---|---|
| 38 | Course Review | [6. Review course.md](./6.%20Review%20course.md) |

---

## Cross-cutting "exam cheat sheet" formulas

If you only have 30 minutes, internalize these from the notes:

### Bandits
- `q*(a) = E[R_t | A_t = a]`
- Sample average: `Q_n = (1/n) Σ R_i`
- Incremental: `Q_{n+1} = Q_n + (1/n)(R_n − Q_n)`
- ε-greedy / UCB1 / softmax

### MDP & DP
- Tuple `(S, A, P, R, γ)`
- Bellman expectation: `v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γ v_π(s')]`
- Bellman optimality: `v*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γ v*(s')]`
- Iterative PE / VI / PI

### Monte Carlo
- `V(s) = avg returns from visits to s` (first-visit or every-visit)
- Importance sampling ratio: `ρ_{t:T-1} = Π π(A_k|S_k) / μ(A_k|S_k)`

### Temporal Difference
- TD(0): `V(S_t) ← V(S_t) + α[R_{t+1} + γ V(S_{t+1}) − V(S_t)]`
- **SARSA** (on-policy): `Q(S,A) ← Q(S,A) + α[R + γ Q(S',A') − Q(S,A)]`
- **Q-learning** (off-policy): `Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) − Q(S,A)]`
- Expected SARSA: replace max with expectation over next action distribution

### Function Approximation
- Linear: `V̂(s,w) = x(s)ᵀ w`
- Semi-gradient TD: `w ← w + α·δ_t·∇V̂(s,w)`
- "Deadly triad": function approximation + bootstrapping + off-policy → instability

### Average Reward (continuing tasks)
- `r(π) = lim E[(1/T) Σ R_t]`
- Differential return: `G_t = Σ_k (R_{t+k+1} − r(π))`

### Policy Gradient & Actor-Critic
- REINFORCE: `Δθ ∝ G·∇ ln π(A|S,θ)`
- Actor update (with baseline): `Δθ ∝ (G − b(s))·∇ ln π`
- Actor-Critic: `Δθ ∝ δ_t·∇ ln π(A|S,θ)`, with `δ_t` from TD error

# Non-delusional Q-learning and value iteration

## Abstract

- We identify a fundamental source of error cost by function approximation;
- **Delustional bias** arises when the approximation architecture is poor;
- We introduce a new notion of **policy consistency** and define a local backup process that ensures global consistency through the use of **information sets**;
- **information sets** record constraints on policies consistent with backed-up Q-values;

## Contributions

- Identification and precise definition of delusional bias, and a demonstration of its detrimental consequences; (More expressive approximators, larger training sets and increased computaion do not resolve the issue.)
- A new policy-consistent backup operator that fully resolves the problem of delusion; 
- Several heuristic methods for imposing policy consistency.

## Preliminaries

- Value function approximators: $\mathcal{F} = \{f_\theta : S \times A \rightarrow \mathbb R | \theta \in \Theta\}$;
- Admissible greedy policies: $G(\Theta) = \{\pi_\theta | \pi_\theta(s) = \arg\max_{a in A} f_\theta(s, a), \theta \in \Theta\}$

## Delusional bias and its consequences

- There will always be a set of $d+1$ state-action choices that are jointly infeasible given a function approximation architecture with VC-dimension $d < \infty$;
- Example in 3.1 shows how delusional bias prevents Q-learning from reaching a reasonable fixed-point;

## Non-delusional Q-learning and dynamic programming

- The source of the problem: the potential inconsistency of the set of Q-values;
- <font color=red>Since backed-up values might be designated inconsistent when new dependencies are added, this policy-consistent backup must maintain alternative information sets and their corresponding Q-values, allowing the backtracking of prior decisions. </font>
- Policy-consistent backup can be viewed as unifying both value- and policy-based RL methods.
- Policy-class value iteration
  - The methods below require an **oracle** or **witness** to check whether a policy $\pi_\theta$ is consistent with a set of state-to-action constraints: i.e., given $\{(s,a)\} \subseteq S\times A$, whether there exists a $\theta \in \Theta$ such that $\pi_\theta(s) = a$ for all pairs;
  - Define $[s \mapsto a] = \{\theta \in \Theta | \pi_\theta(s) = a\}$;
  - 

## Appendix

- Example 1 shows that q-learning converges to the second best policy;
- Example 2 shows that delusional bias can actually lead to divergence;
- Example 4 shows that delusion causes cyclic behavior;
- Example 5: the discounting paradox, $\gamma = 0$ is better than $\gamma = 1$;
- A.5 Comparisons to double Q-learning:
  - If the action space is large, and there are not enough transition examples to confidently estimate the true $Q(s', a')$, then the variance of each $\hat Q(s,a)$ for any a is large.
  - double Q-learning does not solve the delusion.
- A.6 Concepts and proofs for PCVI and PCQL
  - Definition 4. $P(\mathcal{X}) = \{X_1, \ldots, X_k\}$ such that $X_1 \cup \ldots \cup X_k = \mathcal{X}$ and $X_i \cap X_j = \emptyset$. We call any $X_i \in P$ a cell. $P'$ is a refinement of P if for all $X' \in P'$ there exists a $X \in P$ such that $X' \subseteq X$. Let $\mathcal{P}(\mathcal{X}) = \{P(\mathcal{X})\}$;
  - Definition 5.  $\mathcal{H} = \{h : P \rightarrow \mathbb{R} | P \in \mathcal{P}(\mathcal{X})\}$. And we define $h_1\oplus h_2(X_1 \cap X_2) = h_1(X_1) + h_2(X_2), X_1 \in dom(h_1) and X_2 \in dom(h_2)$;
  - Proposition 6.
  - Assumption 7.
  - 


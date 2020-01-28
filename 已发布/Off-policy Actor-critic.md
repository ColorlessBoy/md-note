# Off-policy Actor-critic

## Introduction

- Off-policy algorithms' advantage. The paper seems to miss the key advantage of off-policy.
- Related works:
  - Q-learning algorithm may diverge when using linear function approximation;
  - LSTD and LSPI are sound with linear function approximation but computationally expansive (quadratically with number of features and weights);
  - Gradient TD methods, such as Greedy-GQ, are of linear complexity and convergent off-policy training with function approximation.
- Action-value based algorithm suffer from three important limitations:
  - Target policy is deterministic;
  - Finding greedy action w.r.t. action-value function becomes problematical for large action space;
  - A small change in action-value function can cause large changes in policy, which creates difficulties for convergence proof and real-time applications.
- Contributions:
  - The first actor-critic method that can be applied off-policy, we call $Off-PAC$;
  - Off-policy gradient theorem and convergence proof;
  - Experience: $Q(\lambda)$, $Greedy-GQ$, $Softmax-GQ$, $Off-PAC$.

## 1. Notation and Problem set

- $V^{\pi, \gamma}(s) = \mathbb{E}[r_{t+1} + \ldots + r_{t+T} | s_t = s], \forall s \in S$;
- $Q^{\pi, \gamma}(s,a) = \sum_{s' \in S} P(s' | s, a) [ R(s, a, s') + \gamma(s') V^{\pi, \gamma}(s')]$;

- Objective: $J_{\gamma}(u) = \sum_{s} d^b(x) V^{\pi_u, \gamma}(s)$, where b is behavior policy. (I think the target is problematical). According to some papers, if we exactly correct the probability, the variance will be too high to apply.

## 2. The Off-PAC Algorithms

This section has three section:

	- Gradient-TD method used in critic;
	- Off-policy policy-gradient theorem;
	- Mechanistic algorithm.

### 2.1 The Critic: Policy Evaluation

- GTD methods minimize the $\lambda$-weighted mean-squared projected Bellman error: $MSPBE(v) = \Arrowvert \hat V - \Pi T^{\lambda, \gamma}_\pi \hat V \Arrowvert^2_D$, where $\hat V = X v$, and for linear representation, $\Pi = X(X^T D X)^{-1} X^T D$;

### 2.2 Off-policy Policy-gradient Theorem

- Off-PAC updates: $u_{t+1} - u_t \approx \alpha_{u,t} \nabla_u J_{\gamma} (u_t)$;

- The problem: $\nabla_u J_\gamma(u) = \nabla_u\left[\sum_{s\in S} d^b(s) \sum_{a \in A} \pi(a | s) Q^{\pi,\gamma} (s,a)\right]$
  $$
  \begin{align*}
  \nabla_u J_\gamma(u) =& \nabla_u\left[\sum_{s\in S} d^b(s) \sum_{a \in A} \pi(a | s) Q^{\pi,\gamma} (s,a)\right] \\
  =& \sum_{s \in S} d^b(s) \sum_{a \in A}
  \left[\nabla_u \pi(a | s) Q^{\pi,\gamma}(s, a) 
  	+ \pi(a|s) \nabla_u Q^{\pi, \gamma}(s,a) \right]
  \end{align*}
  $$
  We use $\nabla_u J_\gamma(u) \approx g(u) = \sum_{s \in S} d^b(s) \sum_{a \in A}\left[\nabla_u \pi(a | s) Q^{\pi,\gamma}(s, a)\right]$;

- **Theorem 1** (Policy Improvement). Given any policy parameter u, let $u' = u + \alpha g(u)$. Then, there exists an $\epsilon > 0$ such that, for all positive $\alpha < \epsilon$, $J_\gamma(u') \ge J_\gamma(u)$. Further, if $\pi$ has a tabular representation, then $V^{\pi_{u'}, \gamma}(s) \ge V^{\pi_u, \gamma}(s)$ for all $s \in S$.

  **proof**: The key point is
  $$
  J_\gamma(u) \le \sum_{s\in S} d^b(s) \sum_{a \in A} \pi_{u'}(a | s) Q^{\pi_u, \gamma}(s, a) \\ 
  \le \sum_{s\in S} d^b(s) \sum_{a \in A} \pi_{u'}(a | s) Q^{\pi_u', \gamma}(s, a) \le J_\gamma(u')
  $$

  The first inequation comes from the definition of  partial gradient, and the second inequation has problem.

  In tabular representation, we have $\sum_{a \in A} \pi_{u}(a | s) Q^{\pi_u, \gamma}(s, a) \le \sum_{a \in A}\pi_{u'}(a | s) Q^{\pi_u, \gamma}(s, a)$, which can get $V^{\pi_u, \gamma} \le V^{\pi_u', \gamma}(s)$.

  **Errata**:

  We have
  $$
  \sum_{s\in S} d^b(s) \sum_{a \in A} \pi_{u}(a | s) 
  \sum_{s,a,s_{t+1}} P(s,a,s_{t+1})[R(s, a, s_{t+1}) + \gamma_{t+1} V^{\pi_u, \gamma}(s_{t+1})] \\
  \le \sum_{s\in S} d^b(s) \sum_{a\in A} \pi_{u'}(a|s) \sum_{s,a,s_{t+1}} P(s,a,s_{t+1})[R(s, a, s_{t+1}) + \gamma_{t+1} V^{\pi_u, \gamma}(s_{t+1})]
  $$
  But if we take a further step, it might not hold:
  $$
  \sum_a \pi_{u'}(a|s) \sum_{s, a, s_{t+1}}P(s, a, s_{t+1})\sum_{a_{t+1}} \pi_u(a_{t+1} | s_{t+1}) \cdot\\
  \sum_{s_{t+2}} P(s_{t+1}, a_{t+1}, s_{t+2})[R(s_{t+1}, a_{t+1}, s_{t+2}) + \gamma_{t+2} V^{\pi_u, \gamma}(s_{t+2})]\\
  \le
  \sum_a \pi_{u'}(a|s) \sum_{s, a, s_{t+1}}P(s, a, s_{t+1})\sum_{a_{t+1}} \pi_{u'}(a_{t+1} | s_{t+1}) \cdot\\
  \sum_{s_{t+2}} P(s_{t+1}, a_{t+1}, s_{t+2})[R(s_{t+1}, a_{t+1}, s_{t+2}) + \gamma_{t+2} V^{\pi_u, \gamma}(s_{t+2})]
  $$
  
- **Theorem 2** (Off-Policy Policy-Gradient Theorem).
  $$
  \tilde Z = \{u \in \mathcal{U} | g(u) = 0\}\\
  Z = \{u \in \mathcal{U} | \nabla_u J_{\gamma}(u) = 0 \}
  $$
  In some function, we can guarantee $Z \subset \tilde Z$. Moreover, if we use a tabular representation, then $Z = \tilde Z$.

  **proof**: 

  Assume there exists $u^* \in Z$ such that $u^* \notin \tilde Z$. Then $\exists \alpha_{u,t}$, $J_\gamma(u^* + \alpha_{u,t} g(u^*)) > J_\gamma(u)$, which is contradict to theorem1.

  In tabular representation, we let u with tabular index $i_{s}, j$,(where $1 \le j \le m$) then
$$
  \sum_{s' \in S} d^b(s') \sum_{a \in A} \frac{\partial}{\partial u_{i_s, j}} \pi_u(a | s') Q^{\pi_u, \gamma} (s', a) \\= d^b(s) \sum_{a \in A} \frac{\partial}{\partial u_{i_s, j}} \pi_u(a|s) Q^{\pi_u, \gamma}(s, a) := g_1(u_{i_s, j})
$$
  Similarly, we denote
$$
  g_2(u_{i_s,j}) = \sum_{s'\in S} d^b(s') \sum_{a \in A} \pi_u (a | s') \frac{\partial}{\partial u_{i_s, k}} Q^{\pi_u, \gamma} (s', a) 
\\= d^b(s) \sum_{a \in  A} \pi_u(a | s) \frac{\partial}{\partial u_{i_s, k}} Q^{\pi_u, \gamma} (s, a)
$$
If $g_2(u_{i_s, j}) \ne 0$, we can get $u'$ that satisfy $Q^{\pi_{u'}, \gamma}(s,a) > Q^{\pi_{u},\gamma}(s,a)$, which means that $\sum^m_{j=1} \sum_{a \in A} \frac{\partial}{\partial u_{i_s, j}} \pi_u(a|s) Q^{\pi_u, \gamma} (s,a) \ne 0 \Rightarrow \exists j, g_1(u_{i_s, j}) \ne 0$.

Therefore, in the tabular case, we have $\tilde Z \subset Z$. We already have $Z \subset \tilde Z$, we can get $Z = \tilde Z$.

- Our optimization problem is $\max_{u} J_\gamma(u)$. From an optimization perspective, $\forall u \in \tilde Z \backslash Z$, $J_\gamma(u) < \min_{u' \in Z} J_\gamma(u')$.

### 2.3 The Actor: Incremental Update Algorithm with Eligibility Traces

- The expectation of the gradient:
    $$
    \begin{align*}
    g(u) =& \mathbb{E} \left[ \sum_{a \in A} \nabla_u \pi(a | s) Q^{\pi,\gamma} (s, a) \Bigg| s \sim d^b \right] \\
    =& \mathbb{E} \left[ \sum_{a \in A}b(a|s) \frac{\pi(a|s)}{b(a|s)} \frac{\nabla_u \pi(a|s)}{\pi(a|s)} Q^{\pi,\gamma}(s,a) \Bigg| s \sim d^b \right] \\
    =& \mathbb{E} \left[\frac{\pi(a|s)}{b(a|s)} \frac{\nabla \pi(a|s)}{\pi(a|s)} Q^{\pi,\gamma}(s,a) \Bigg| s \sim d^b, a \sim b(\cdot|s) \right] \\
    =& \mathbb{E}_b \left[\frac{\pi(a|s)}{b(a|s)} \frac{\nabla \pi(a|s)}{\pi(a|s)} Q^{\pi,\gamma}(s,a) \right] \\
    =& \mathbb{E}_b \left[\frac{\pi(a|s)}{b(a|s)} \frac{\nabla \pi(a|s)}{\pi(a|s)} (Q^{\pi,\gamma}(s,a) - \hat V(s_t)) \right]
    \end{align*}
    $$

- Here is a further approximation:
  $$
  g(u) \approx \hat g(u) = \mathbb{E}_b [\rho(s_t, a_t) \psi(s_t, a_t) (R^\lambda_t - \hat V(s_t))]
  $$
  where $R^\lambda_t = r_{t+1} + (1 - \lambda) \gamma(s_{t+1}) \hat V(s_{t+1}) + \lambda \gamma(s_{t+1})\rho(s_{t+1}, a_{t+1}) R^\lambda_{t+1}$.

- 
  $$
  \begin{align*}
  \delta^\lambda_t =& R^\lambda_t - \hat V(s_t) \\
  =& r_{t+1} + (1-\lambda)\gamma_{t+1} \hat V(s_{t+1})
  	+ \lambda \gamma_{t+1}\rho_{t+1} R^\lambda_{t+1} - \hat V(s_t) \\
  =& r_{t+1} + \gamma_{t+1}\hat V(s_{t+1}) - \hat V(s_t) + \lambda \gamma_{t+1} (\rho_{t+1} R^\lambda_{t+1} - \hat V(s_t))\\
  =& \delta_t + \lambda \gamma_{t+1}(\rho_{t+1} R^\lambda_{t+1} - \rho_{t+1}\hat V(s_{t+1}) - (1 - \rho_{t+1})\hat V(s_{t+1}))\\
  =& \delta_t + \lambda \gamma_{t+1} (\rho_{t+1} \delta^\lambda_{t+1} - (1-\rho_{t+1})\hat V(s_{t+1}))
  \end{align*}
  $$

- ![](.\Off-policy-Actor-critic\equation1.png)

- ![equation2](.\Off-policy-Actor-critic\equation2.png)

- Algorithm 1 The Off-PAC algorithm

  <img src=".\Off-policy-Actor-critic\Algorithm1.png" alt="Algorithm1" style="zoom: 67%;" />




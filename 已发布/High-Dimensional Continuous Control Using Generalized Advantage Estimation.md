## High-Dimensional Continuous Control Using Generalized Advantage Estimation

## 1. Contributions[^1]

- Generalized advantage estimation: an effective variance reduction scheme  for policy gradients;
- A trust region optimization method for value evaluation;
- An empirically effective algorithm combining preceding features.

## 2. Preliminaries

- Policy gradient $g := \nabla_\theta \mathbb{E}[\sum^\infty_{t=0} r_t] = \mathbb{E}\left\{\sum^{\infty}_{t=0} \Psi_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right\}$. $\Psi_t$ has several forms: 

  - Total reward of the trajectory: $\sum^\infty_{t=0} r_t$;
  - State-action value function: $Q^\pi(s_t, a_t)$;
  - Reward following action $a_t$: $\sum^\infty_{t' = t} r_{t'}$;
  - Advantage function: $A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$;
  - TD residual: $r_t + V^\pi(s_{t+1}) - V^\pi(s_t)$.

- The choice $\Psi_t = A^\pi(s_t, a_t)$ yields almost the lowest possible variance, though in practice, the advantage function is not known and must be estimated. (See Greensmith et al. 2004 for more rigorous analysis of the variance of policy gradient estimators and the effect of using a baseline)

- We treat $\gamma$ as a variance reduction parameter in an undiscounted problem.

  - $V^{\pi, \gamma}(s_t):= \mathbb{E}\left[\sum^\infty_{l=0} \gamma^l r_{t+l} | s_{t+1:\infty}\sim P^{\pi}, a_{t:\infty} \sim \pi\right]$;
  - $Q^{\pi,\gamma}(s_t, a_t) := \mathbb{E}[\sum^\infty_{l=0}\gamma^l r_{t+l} | s_{t+1:\infty}\sim P^\pi, a_{t+1:\infty}\sim \pi]$;
  - $A^{\pi,\gamma}(s_t, a_t) = Q^{\pi, \gamma}(s_t, a_t) - V^{\pi, \gamma}(s_t)$.

- $g^\gamma := \mathbb{E}_{s_{0:\infty}, a_{0:\infty}}\left[\sum^\infty_{t=0} A^{\pi,\gamma} \nabla_\theta\log\pi_\theta(a_t | s_t)\right]$;

- **Definition 1**: The estimator $\hat A_t$ is $\gamma$-just if
  $$
  \mathbb{E}_{s_{0:\infty}, a_{0:\infty}}[\hat A(s_{0:\infty}, a_{0:\infty})\nabla_\theta \log \pi_\theta(a_t | s_t)]\\
  = \mathbb{E}_{s_{0:\infty}, a_{0:\infty}}[A^{\pi,\gamma}(s_t, a_t) \nabla_\theta \log \pi_\theta(a_t | s_t)]
  $$

- **Proposition 1**: If $\hat A_t(s_{0:\infty}, a_{0:\infty}) = Q_t(s_{t:\infty}, a_{t:\infty}) - b_t(s_{0:t}, a_{0:t-1})$ such that $\forall (s_t, a_t)$, $\mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty}|s_t, a_t}[Q_t(s_{t:\infty}, a_{t:\infty})] = Q^{\pi, \gamma}(s_t, a_t)$. Then $\hat A$ is $\gamma$-just.

  **proof**: 
  $$
  \begin{align*}
  &\mathbb{E}_{s_{0:\infty}, a_{0:\infty}}[\nabla_\theta \log \pi_\theta(a_t | s_t) b_t(s_{0:t}, a_{0:t-1})]\\
  =&\mathbb{E}_{s_{0:t}, a_{0:t-1}}[\mathbb{E}_{s_{t+1:\infty}, a_{t:\infty}}[\nabla_\theta\log\pi_\theta(a_t|s_t)] b_t(s_{0:t}, a_{0:t-1})]\\
  =&\mathbb{E}_{s_{0:t, a_{0:t-1}}}[0\cdot b_t(s_{0:t}, a_{0:t-1})] = 0
  \end{align*}
  $$

  $$
  \begin{align*}
  &\mathbb{E}_{s_{0:\infty}, a_{0:\infty}}[\nabla_\theta\log\pi_\theta(a_t | s_t) Q_t(s_{0:\infty}, a_{0:\infty})]\\
  =& \mathbb{E}_{s_{0:t}, a_{0:t}}[\nabla_\theta\log\pi_\theta(a_t|s_t)\mathbb{E}_{s_{t+1:\infty}, a_{t+1:\infty}}Q_t(s_{0:\infty}, a_{0:\infty})]\\
  =& \mathbb{E}_{s_{0:t}, a_{0:t-1}}[\nabla_\theta\log\pi_\theta(a_t | s_t) Q^{\pi,\gamma}(s_t, a_t)]
  \end{align*}
  $$

## 3. Advantage Function Estimation

- Normal estimation: $A^{\pi,\gamma}(s_t, a_t) = \mathbb E_{s_{t+1}}[\delta^{V^{\pi,\gamma}}_{t}] = \mathbb{E}_{s_{t+1}}[r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t)]$;

- $\hat A_t^{(1)} = \delta^V_t \Rightarrow \hat A_t^{(k)} \approx \sum^{k-1}_{l=0}\gamma^l \delta^{V_{\theta'}}_{t+l} \\= -V_{\theta'}(s_t) + \sum^{t-1}_{i=0}\gamma^i r_{t+i} + \gamma^k V_{\theta'}(s_{t+k}) := \tilde A^{(k)}_t.$

  Because $V_{\theta'}(s_{t+k})$ is a estimate of $V^{\pi,\gamma}(s_{t+k})$ and $V_{\theta'}(s_t)$ is just a baseline,  the policy-gradient bias caused by $\tilde A^{(k)}_t$ decreases as k increase.

- **Definition 2** (Generalized advantage estimator).
  $$
  \begin{align*}
  &\hat A^{GAE(\gamma, \lambda)}_t
  := (1-\lambda)(\sum^{\infty}_{k=1} \lambda^{k-1}\hat A^{(k)}_t)
  = (1-\lambda)\sum^\infty_{k=1} \lambda^{k-1} \sum^{k-1}_{l=0}\gamma^l\delta^V_{t+l}\\
  =& (1-\lambda)\sum^{\infty}_{k=0}\lambda^k\sum^k_{l=0}\gamma^l\delta^V_{t+l} 
  = (1-\lambda)\sum^\infty_{l=0}\delta^V_{t+l} \gamma^l \sum^{\infty}_{k=l}\lambda^k
  = \sum^\infty_{l=0}(\gamma \lambda)^l \delta^V_{t+l}
  \end{align*}
  $$

  - $GAE(\gamma, 1)$: zero bias but high variance;
  - $GAE(\gamma, 0)$: high bias but low variance;
  - $\gamma$ is most importantly determines the scale of the value function $V^{\pi,\gamma}$, which does not depend on $\lambda$;
  - $\lambda$ introduces bias only when the value function is inaccurate.

- Using the generalized advantage estimator, we get
  $$
  g^\gamma \approx \mathbb{E}\left[\sum^\infty_{t=0} \nabla_\theta \log \pi_\theta(a_t | s_t) \hat A^{GAE(\gamma, \lambda)}_t\right]\\
  =\mathbb{E}\left[\sum^\infty_{t=0} \nabla_\theta \log\pi_\theta(a_t|s_t) \sum^\infty_{l=0} (\gamma\lambda)^l \delta^V_{t+l}\right],
  $$
  where equality holds when $\lambda=1$.

## 4. Interpretation as Reward Shaping

- **Definition 3** (Transformed MDP) Let $\Phi : S \rightarrow \mathbb{R}$ be an arbitrary scalar-valued function on state space, then
  $$
  Transform[MDP(S, A, P(s, a, s'), r(s, a, s'), \gamma)] \\
  = MDP(S, A, P(s, a, s'), \tilde r(s, a, s') = r(s, a, s')+\gamma\Phi(s') - \phi(s), \gamma)
  $$

  - $\sum^\infty_{l=0} \gamma^l\tilde r(s_{t+l}, a_{t+1}, s_{t+l+1}) = \sum^\infty_{l=0} \gamma^l r(s_{t+l}, a_{t+l}, s_{t+l+1}) - \Phi(s_t)$.
  - $\tilde Q^{\pi, \gamma}(s,a) = Q^{\pi, \gamma}(s,a) - \Phi(s)$;
  - $\tilde V^{\pi,\gamma}(s,a) = V^{\pi, \gamma}(s) - \Phi(s)$;
  - $\tilde A^{\pi, \gamma}(s,a) = \tilde Q^{\pi,\gamma}(s,a)-\tilde V^{\pi, \gamma} (s,a) = A^{\pi,\gamma}(s,a)$;
  - If $\Phi(s) = V^{\pi,\gamma}(s)$, then $\forall s, \tilde V^{\pi,\gamma}(s) = 0$;
  - Note that (Ng et al., 1999) showed that the reward shaping transformation leaves the policy gradient and optimal policy unchanged when objective is $\sum^\infty_{t=0} \gamma^t r(s_t, a_t, s_{t+1})$. (**without verification**)

- This paper is concerned with maximizing the undiscounted sum of rewards, where the discount $\gamma$ is used as a variance-reduction parameter.

- If $\Phi = V_{\theta'}$, then $\sum^\infty_{l=0}(\gamma \lambda)^l \tilde r(s_{t+l}, a_{t+l}, s_{t+l+1}) = \sum^{\infty}_{l=0}(\gamma \lambda)^l \delta^{V_\theta}_{t+l} = \hat A^{GAE(\gamma, \lambda)}_{t}$. 

## 5. Value Function Estimation

Optimization problem with trust region (seeing TRPO):
$$
\min_{\theta'} \sum^{N}_{n=1}\Vert V_{\theta'}(s_n) - \hat V_n\Vert^2, \\
s.t. \frac{1}{N}\sum^{N}_{n=1}\frac{\Vert V_{\theta'}(s_n) - V_{\theta'_{old}}(s_n)\Vert^2}{2\sigma^2} \le \epsilon,\\
\sigma^2 = \frac{1}{N}\sum^{N}_{n=1}\Vert V_{\theta'_{old}(s_n)} - \hat V_n\Vert^2.
$$

## 6. Algorithm

![Generalized_Advantage_Estimation_Alg1](.\pic\Generalized_Advantage_Estimation_Alg1.png)

## 7. Discussion

- We have provided an intuitive but informal analysis of the problem of advantage function estimation, and  justified the generalized advantage estimator, which has two parameters $\gamma$, $\lambda$ which adjust the bias-variance tradeoff.
- One question that merits future investigation is the relationship between value function estimation error and policy gradient estimation error.

## A question

- Why don't you just use a Q-function?
  - First, state-value function has lower-dimensional input;
  - Second, the method of this paper allows us to smoothly interpolate between the high-bias estimator ($\lambda = 0$) and the low-bias estimator ($\lambda=1$).

[^1]: J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” arXiv:1506.02438 [cs], Oct. 2018.
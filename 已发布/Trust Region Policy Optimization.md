# Trust Region Policy Optimization

## 1. Introduction

- Policy optimization can be classified into three broad categories:
  - Policy iteration;
  - Policy gradient method;
  - Derivative-free optimization methods: such as cross-entropy method and covariance matrix adaption.
- Trust region policy optimization:
  - The single-path method: model-free;
  - The vine method:  requires the system to be restored to particular states,
    which is typically only possible in simulation.

## 2. Preliminaries

- MDP: $\{S, A, p(s_{t+1} \vert s_t, a_t), r(s_t, a_t, s_{t+1}), p_0(s_0), \pi(a_t\vert s_t), \gamma\}$;

- The distribution of initial state $s_0$ is $\rho_0$;

- $$\Tau^\pi = \{\tau = (s_0, a_0, s_1, \ldots) \vert s_0 \sim \rho_0, a_t \sim\pi(\cdot \vert s_t), s_{t+1} \sim p(\cdot \vert s_t, a_t)\}$$;

- Total reward: $$\eta(\pi) = \mathbb{E}[\sum^\infty_{t=0} \gamma^t r(s_t, a_t, s_{t+1}) \vert s_0 \sim \rho_0, a_t \sim \pi(\cdot \vert s_t), s_{t+1}\sim p(\cdot \vert s_t, a_t)]$$;

- State-action-value function: 

  $$Q^\pi(s_t, a_t) = \mathbb{E}[\sum^{\infty}_{l=0} \gamma^l r(s_{t+l}, a_{t+l}, s_{t+1+l} ) \vert s_{t+1+l} \sim p(\cdot \vert s_{t+l}, a_{t+l}),  a_{t+1+l} \sim \pi(\cdot \vert s_{t+1+l})]$$;

- State-value function: 

  $$V^\pi(s_t) = \mathbb{E}[\sum^{\infty}_{l=0} \gamma^l r(s_{t+l}, a_{t+l}, s_{t+1+l}) \vert a_{t+l} \sim \pi(\cdot \vert s_{t+l}), s_{t+1+l} \sim p(\cdot \vert s_{t+l}, a_{t+l})]$$;

- Advantage function: $$A^\pi(s_t, a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$$;

- $$\eta(\tilde\pi) = \eta(\pi) + \mathbb{E}_{\tau\vert\tilde\pi} [\sum^\infty_{t=0} \gamma^t A^\pi(s_t, a_t)]$$;

  **proof**:
  
  $$
  \begin{align*}
  &\mathbb{E}_{\tau|\tilde\pi}\left[\sum^{\infty}_{t=0} \gamma^t A^\pi(s_t, a_t)\right]\\
  =& \mathbb{E}_{\tau|\tilde\pi}\left[\sum^{\infty}_{t=0} \gamma^t (Q^\pi(s_t, a_t) - V^\pi(s_t))\right]\\
  =&\mathbb{E}_{\tau|\tilde\pi}\left[\sum^{\infty}_{t=0} \gamma^t (r(s_t, a_t, s_{t+1})+\gamma V^\pi(s_{t+1}) - V^\pi(s_t))\right]\\
  =& \mathbb{E}_{\tau|\tilde\pi}\left[-V^\pi(s_0) + \sum^{\infty}_{t=1} \gamma^t r(s_t, a_t, s_{t+1})\right]\\
  =& -\eta(\pi) + \eta(\tilde\pi).
  \end{align*}
  $$
  

Let $$ \rho_\pi(s) = P(s_0=s)+\gamma P(s_1=s) + \gamma^2 P(s_2 = s)+\cdots $$, then

$$
\begin{align*}
&\mathbb{E}_{\tau\sim\tilde\pi}\left[ \sum^{\infty}_{t=0} \gamma^t A^{\pi}(s_t,a_t) \right]\\
=& \sum^{\infty}_{t=0} \sum^{}_{s} P(s_t=s \vert \tilde\pi) \sum^{}_{a} \tilde\pi(a|s) \gamma^t A^{\pi}(s,a)\\
=& \sum^{}_{s} \sum^{\infty}_{t=0} \gamma^t P(s_t = s \vert \tilde \pi) \sum^{}_{a} \tilde\pi(a \vert s) A^{\pi}(s,a)\\
=& \sum^{}_{s} \rho_{\tilde\pi}(s) \sum^{}_{a} \tilde \pi(a \vert s) A^\pi(s,a).
\end{align*}
$$

We get $$\eta(\tilde\pi) = \eta(\pi) + \sum^{}_{s} \rho_{\tilde\pi}(s) \sum^{}_{a} \tilde\pi(a|s) A^{\pi}(s,a)$$, which implies a method of policy optimization

$$
\pi_{t+1} = \arg\max_{\pi} \sum_s \rho_{\pi}(s) \sum_a\pi(a|s)A^{\pi_t}(s,a).
$$

So, we consider a local approximation

$$
L_\pi(\tilde\pi) = \eta(\pi) + \sum_s \rho_{\pi}(s) \sum_a \tilde\pi(a|s) A^\pi(s,a),
$$

which means the policy optimization step becomes

$$
\pi_{t+1} = \arg\max_{\pi} \sum_s \rho_{\pi_t}(s) \sum_a\pi(a|s)A^{\pi_t}(s,a).
$$

If we parameterized policy $$\pi_\theta$$, where $$\pi_\theta(a | s)$$ is differentiable function with parameter vector $$\theta$$, then $$L_\pi$$ matches $$\eta$$ to first order.
Firstly, it's easy to verify that $L_\pi(\pi) = \eta(\pi)$.
Secondly,  we have $$\nabla_\theta \eta(\pi_\theta)\vert_{\theta = \theta_0}
= \nabla_\theta L_{\pi_{\theta_0}}(\pi_\theta) \vert_{\theta=\theta_0}$$. 
$$
\nabla_{\theta}L_{\pi}(\pi_\theta) = \sum^{}_{s} \rho_{\pi}(s) \sum^{}_{a} A^{\pi}(s,a) \nabla_{\theta}\pi_{\theta}(a\vert s),\\

\nabla_{\theta}L_{\pi_{\theta_0}}(\pi_\theta) \vert _{\theta = \theta_0} = \sum^{}_{s} \rho_{\pi_{\theta_0}}(s) \sum^{}_{a} A^{\pi_{\theta_0}}(s,a) \nabla_{\theta}\pi_{\theta}(a|s)\vert_{\theta=\theta_0}.
$$

From policy gradient theorem, we have

$$
\nabla_\theta \eta(\pi_\theta)\vert_{\theta = \theta_0} = \sum_s\rho_{\pi_\theta}(s) \sum_a A^{\pi_\theta}(s,a) \nabla_\theta\pi_\theta(a|s)\vert_{\theta=\theta_0}
= \nabla_\theta L_{\pi_{\theta_0}}(\pi_\theta) \vert_{\theta=\theta_0}
$$

### 2.1 Conservative policy iteration[^1]

Firstly, we denote  $$\bar A(s) = \mathbb{E}_{a \sim \tilde \pi(\cdot | s)}\left[ A_\pi(s,a) \right]$$, then

$$
\eta(\tilde \pi) = \eta(\pi) + \mathbb{E}_{\tau \sim \tilde \pi}\left[\sum^\infty_{t=0} \gamma^t \bar A(s_t) \right],\\
L(\tilde \pi) = \eta(\pi) + \mathbb{E}_{\tau \sim \pi}\left[\sum^\infty_{t=0} \gamma^t \bar A(s_t) \right].
$$

We are interested in

$$
\vert\eta(\tilde \pi) - L_{\pi}(\tilde \pi)\vert = \sum^{\infty}_{t=0} \gamma^{t} \vert \mathbb{E}_{\tau\sim\tilde\pi}\left[ \bar A(s_t) \right] - \mathbb{E}_{\tau\sim\pi}[\bar A(s_t) ]\vert.
$$

 Let $$n_t$$ be the number of times that $$a_i \ne \tilde a_i$$ for all $$i < t$$,

$$
\begin{cases}
\mathbb{E}_{s_t\sim\tilde\pi}[\bar A(s_t)] = P(n_t=0)\mathbb{E}_{s_t \sim \tilde\pi | n_t = 0}[\bar A(s_t)] + P(n_t>0)\mathbb{E}_{s_t \sim \tilde\pi | n_t > 0}[\bar A(s_t)],\\
\mathbb{E}_{s_t\sim\pi}[\bar A(s_t)] = P(n_t=0)\mathbb{E}_{s_t \sim \pi | n_t = 0}[\bar A(s_t)] + P(n_t>0)\mathbb{E}_{s_t \sim \pi | n_t > 0}[\bar A(s_t)],\\
\mathbb{E}_{s_t \sim \tilde\pi | n_t = 0}[\bar A(s_t)] =\mathbb{E}_{s_t \sim \pi | n_t = 0}[\bar A(s_t)].
\end{cases}
$$

>**Definition 1**:  $$(\pi, \tilde \pi)$$ is $$\alpha$$-coupled policy pair if  it defines a joint distribution $$(a, \tilde a) \vert s$$, such that
>$$
>P(a \ne \tilde a \vert (a, \tilde a) \sim (\pi, \tilde \pi); s) \le \alpha.
>$$
>whose marginal distributions are $$\pi$$ and $$\tilde \pi$$.

Note that: we can let $$\pi' = \arg\max_{\pi'} L_{\pi_{old}}(\pi')$$ and $$\pi_{new}(a|s) = (1 - \alpha)\pi_{old}(a|s) + \alpha \pi'(a|s) $$, then $$ \pi_{new} $$ and $$ \pi_{old} $$ are $$\alpha$$-coupled.[^1]

>**Lemma 2**: If $$(\pi, \tilde \pi)$$ is $$\alpha$$-coupled, then for all s, $$\vert \bar A(s,a)\vert \le 2 \alpha \max \vert A^\pi(s,a) \vert$$.
>
>**proof**:
>$$
>\begin{align*}
>&\bar A(s) = \mathbb{E}[A^\pi(s, \tilde a) \vert \tilde a \sim \tilde \pi(\cdot \vert s)]\\
>=& \mathbb{E}[A^\pi(s,\tilde a) - A^\pi(s,a) \vert (a, \tilde a) \sim (\pi(\cdot \vert s), \tilde \pi(\cdot \vert s))]\\
>=& P(a \ne \tilde a | s) \mathbb{E}[A^\pi(s,\tilde a) - A^\pi(s,a) \vert (a, \tilde a) \sim (\pi(\cdot \vert s), \tilde \pi(\cdot \vert s)), a \ne \tilde a]\\
>\Rightarrow&
>\vert \bar A(s) \vert \le 2\alpha \max_{s,a} \vert A^\pi(s,a) \vert
>\end{align*}
>\\
>$$

>**Lemma 3**: If $$(\pi, \tilde \pi)$$ is $$\alpha$$-coupled, then $$\vert \eta(\tilde \pi) - L(\tilde \pi)\vert \le \frac{4\gamma\alpha^2}{(1-\gamma)^2}\max_{s,a \sim \tilde \pi} \vert A^\pi(s,a) \vert$$.
>
>**proof**:
>$$
>\begin{align*}
>&\left\vert\mathbb{E}_{s_t\sim\tilde\pi}[\bar A(s_t)] - \mathbb{E}_{s_t\sim\pi}[\bar A(s_t)] \right\vert\\
>=&\left\vert P(n_t>0) \{\mathbb{E}_{s_t \sim \tilde\pi \vert n_t > 0}[\bar A(s_t)] - \mathbb{E}_{s_t \sim \pi | n_t > 0}[\bar A(s_t)]\}\right\vert\\
>\le& \left[ 1 - {(1-\alpha)}^{t} \right] \vert\{\mathbb{E}_{s_t \sim \tilde\pi \vert n_t > 0}[\bar A(s_t)] - \mathbb{E}_{s_t \sim \pi | n_t > 0}[\bar A(s_t)]\}\vert\\
>\le& [1 - (1-\alpha)^t]\cdot 4\alpha \max_{s,a}\vert A^\pi(s,a) \vert
>\end{align*}
>$$
>
>$$
>\begin{align*}
>&\vert \eta(\tilde \pi) - L(\tilde \pi)\vert\\
>=& \sum^{\infty}_{t=0} \gamma^t \left\vert\mathbb{E}_{s_t\sim\tilde\pi}[\bar A(s_t)] - \mathbb{E}_{s_t\sim\pi}[\bar A(s_t)] \right\vert\\
>\le& \sum^\infty_{t=0} \gamma^t [1 - (1-\alpha)^t]\cdot 4\alpha \max_{s,a}\vert A^\pi(s,a) \vert \\
>\le& \left[ \frac{1}{1-\gamma} - \frac{1}{1 - \gamma(1-\alpha)}\right]\cdot 4\alpha \max_{s,a}\vert A^\pi(s,a) \vert\\
>=& \frac{4\gamma\alpha^2}{(1-\gamma)(1-\gamma(1-\alpha))} \max_{s,a} \vert A^\pi(s,a) \vert\\
>\le& \frac{4\gamma\alpha^2}{(1-\gamma)^2}\max_{s,a} \vert A^\pi(s,a) \vert
>\end{align*}
>$$

## 3. Monotonic Improvement Guarantee for General Stochastic Policy

### 3.1 Total Variance Divergence

> **Definition 2** (Total variance divergence). 
>
> $$
> D_{TV}(p \Arrowvert q) = \max_{A' \subseteq A} \vert p(A') - q(A')\vert.
> $$

> **Proposition 1**: $$D_{TV}(p \Arrowvert q) = \max_{A' \subseteq A} \vert p(A') - q(A')\vert = \frac{1}{2} \sum_a \vert p(a) - q(a) \vert$$;
>
> **proof**: We let $$A_1 = \{a : p(a) \ge q(a)\}$$ and $$A_2 = \{a: p(a) < q(a)\}$$. $$\forall A_3 \subset A$$，
> $$
> p(A_3) - q(A_3) \le p(A_3 \cap A_1) - q(A_3 \cap A_1) \le p(A_1) - q(A_1)\\
> p(A_3) - q(A_3) \le q(A_3 \cap A_2) - p(A_3 \cap A_2) \le q(A_2) - q(A_2)
> $$
> So, we have
> $$
> \begin{align*}
> D_{TV}(p \Arrowvert q) =& \max_{A' \subseteq A} \vert p(a) - q(a)\vert\\
> =& p(A_1) - q(A_1) = q(A_2) - p(A_2)\\
> =& \frac{1}{2}\sum_a \vert p(a) - q(a) \vert
> \end{align*}
> $$

> **Proposition 2**: $$D_{TV}(p \Arrowvert q) = \inf_\mathcal{D} \{P_\mathcal{D}(a \ne a') : (a, a') \sim \mathcal{D}\}$$, where $$\mathcal{D}$$ is any distribution whose marginal distributions are $p$ and $q$.
>
> **proof**: $$\forall A_1 \subseteq A$$ , we have
> $$
> \begin{align*}
> &p(A_1) - q(A_1)\\
> =& P(a \in A_1) - P(a' \in A_2)\\
> \le& P(a \in A_1, a' \notin A_2)\\
> \le& P(a \ne a')
> \end{align*}
> $$
> We let $$A_1 = \{s : p(a) \ge q(a)\}$$ and $$A_2 = \{s: p(a) < q(a)\}$$, then
> $$
> D_{TV}(p\Arrowvert q) = \frac{1}{2}[p(A_1)-q(A_1) + q(A_2) - p(A_2)] \le P(a \ne a'\vert (a, a') \sim \mathcal{D})
> $$
>
> Then, we will construct a special distribution $$\mathcal{D}$$ to make inequality into equality.
>
> - Denote $$ \beta = \sum_a \min\{p(a), q(a)\} = 1 - D_{TV}(p \Arrowvert q)$$. 
>
> - With probability $$\beta$$, we pick the point $$(a, a)$$, where $$a \sim \frac{1}{\beta} \min\{p(a), q(a)\}$$; 
>
> - With probability $$1-\beta $$, we pick the point $$(a, b)$$, where $$a \sim \frac{1}{1-\beta} \max\{p(a)-q(a), 0\}$$ and $$b \sim \frac{1}{1-\beta} \max\{q(b) - p(b), 0\}$$.
>
> This distribution satisfies $P(a \ne a') = D_{TV}(p \Arrowvert q)$.

### 3.2 Improvement Guarantee

> **Definition 3**:
> $$
> D^{\max}_{TV} (\pi, \tilde \pi) = \max_s D_{TV}(\pi(\cdot \vert s) \Arrowvert \tilde\pi(\cdot\vert s)).
> $$

> **Theorem 1**:
> $$
> \eta(\pi_{new}) \ge L_{\pi_{old}}(\pi_{new}) - \frac{4\gamma}{(1-\gamma)^2} (D^{max}_{TV}(\pi_{new}, \pi_{old}))^2 \cdot \max_{s,a\sim\pi_{new}} \vert A^{\pi_{old}}(s,a) \vert.
> $$
> **proof**:
>
> We can define a joint distribution $$(\pi_{new}, \pi_{old})$$ that is $$D^\max_{TV}(\pi_{new}, \pi_{old})$$-coupled.

> **Theorem 2**: Pinsker's inequality:
> $$
> D^2_{TV}(p\Arrowvert q) \le 2 D_{KL}(p \Arrowvert q)
> $$

> **Algorithm 1**:
> $$
> \pi_{t+1} = \arg\min_{pi} L_{\pi_t}(\pi)-\frac{8\gamma}{(1-\gamma)^2} D^{\max}_{KL}(\pi_{t}, \pi) \cdot \max_{s,a\sim \pi} \vert A^{\pi_t}(s,a) \vert
> $$
> **proof**:
> $$
> \begin{align*}
> \eta(\pi_{t+1}) \ge& L_{\pi_{t}}(\pi_{t+1}) - \frac{8\gamma}{(1-\gamma)^2} D^{\max}_{KL}(\pi_{t}, \pi_{t+1}) \cdot \max_{s,a\sim\pi_{t+1}} \vert A^{\pi_t}(s,a) \vert\\
> \ge& L_{\pi_{t}}(\pi_{t}) - \frac{8\gamma}{(1-\gamma)^2} D^{\max}_{KL}(\pi_{t}, \pi_{t}) \cdot \max_{s,a\sim\pi_{t}} \vert A^{\pi_t}(s,a) \vert = \eta(\pi_t)
> \end{align*}
> $$

## 4. Optimization of Parameterized Policy

In practice, if we use the penalty coefficient C recommended by the theory above, the step size would be very small. One way to take a larger step in a robust way is to use a constraint on the KL divergence between the old policy and the new policy, i.e., a trust region constraint:
$$
\max_\theta L_{\theta_{old}}(\theta), s.t. D_{KL}(\theta_{old} \Arrowvert \theta) \le \delta.
$$
But the constraint needs to be bounded on every point in the state space, which is impractical. So here is a heuristic approximation:
$$
\bar D^\rho_{KL}(\theta_1, \theta_2) = \mathbb{E}_{s \sim \rho}[ D_{KL}(\pi_{\theta_1}(\cdot\vert s) \Arrowvert \pi_{\theta_2}(\cdot \vert s))].
$$
So the optimization problem becomes
$$
\max_\theta L_{\theta_{old}}(\theta) = \mu(\theta_{old}) + \sum_s \rho_{\theta_{old}}(s) \sum_a \pi_\theta(a|s) A^{\pi_{old}}(s,a),\\
s.t. \bar D^{\rho_{\theta_{old}}}_{KL}(\theta_{old} \Arrowvert \theta) \le \delta.
$$
We are now closed to the TRPO optimization. We first replace $$\sum_s \rho_{\theta_{old}}$$ with $$\frac{1}{1-\gamma}\mathbb{E}_{s \sim \rho_{\theta_{old}}}$$ and use off-policy strategy. Finally we will get
$$
\max_{\theta} \mathbb{E}_{s \sim \rho_{\theta_{old}}, a \sim q}\left[\frac{\pi(a\vert s)}{q(a\vert s)} Q^{\theta_{old}}(s, a)\right],\\
s.t. \mathbb{E}_{s \sim \rho^{\theta_{old}}} [D_{KL}(\theta_{old} \Arrowvert \theta)] \le \delta.
$$

## 5. Sample-Based Estimation of the Objective and Constraint

- Single path: Monte Carlo method, samples some trajectories by simulating $$\pi_{old}$$;

- Vine path: 

  - Generate trajectories by simulating $$\pi_{old}$$;

  - Choose subset of N states $$S_{sub} = \{s_1, s_2, \ldots, s_N\}$$;

  - For each $$s_n \in S_{sub}$$, we choose K actions according to $$a_{n,k}\sim q(\cdot \vert s_n)$$; ($$q = \pi_{old}$$ works well on continuous problems or even works better. )

  - Estimate $\hat Q(s_n, a_{n,k})$ by performing a rollout. (Same K random numbers, i.e., common random numbers)

  - $$L_n(\theta) = \sum^K_{k=1} \pi_\theta(a_{n,k} \vert s_n) \hat Q(s_n, a_{n,k})$$;

  - Self-normalized estimator:
    $$
    L_n(\theta) = \frac{\sum^K_{k=1} \frac{\pi_\theta(a_{n,k}\vert s_n)}{\pi_{\theta_{old}}(a_{n,k}\vert s_n)} \hat Q(s_n, a_{n,k})}{\sum^K_{k=1} \frac{\pi_\theta(a_{n,k}\vert s_n)}{\pi_{\theta_{old}}(a_{n,k}\vert s_n)}}
    $$

- The benefit of vine path method over the single path method is that $\hat Q$ value function has much lower variance.
- Vine path method needs the system can be reset to any arbitrary state and needs much more simulation steps. And single path can be directly implemented on a physical system.

### 5.1 Efficiently Solving the Trust-Region Constraint Optimization[^2]

For 

$$
L_{\theta_k}(\theta) \approx L_{\theta_k}(\theta_k) + \nabla_\theta L_{\theta_k}(\theta) \vert_{\theta = \theta_k} \cdot (\theta - \theta_{k}):= g\cdot (\theta-\theta_k)
$$

and
$$
\begin{align*}
D_{KL}(\theta_k \Arrowvert \theta) 
\approx& D_{KL}(\theta_k \Arrowvert \theta_k)
+ \nabla_\theta D_{KL}(\theta_k \Arrowvert \theta)\vert_{\theta = \theta_k} \cdot (\theta - \theta_k)\\
&+ \frac{1}{2}(\theta - \theta_k)^T \nabla^2_\theta D_{KL}(\theta_k \Arrowvert \theta)\vert_{\theta = \theta_k} (\theta - \theta_k).
\end{align*}
$$
Because $$D_{KL}(\theta_k \Arrowvert \theta_k) = 0$$ and $$\nabla_\theta D_{KL}(\theta_k \Arrowvert \theta) \vert_{\theta=\theta_k} = 0$$ (for $$\theta_k = \arg\min_\theta D_{KL}(\theta_k \Arrowvert \theta)$$), we have
$$
D_{KL}(\theta_k \Arrowvert \theta) \approx\frac{1}{2}(\theta - \theta_k)^T \nabla^2_\theta D_{KL}(\theta_k \Arrowvert \theta)\vert_{\theta = \theta_k} (\theta - \theta_k) := \frac{1}{2}(\theta - \theta_k)^T H (\theta - \theta_k).
$$
Now we a approximate optimization problem:
$$
\theta_{k+1} = \arg\max_{\theta} g\cdot(\theta-\theta_k),
s.t. \frac{1}{2}(\theta-\theta_k)^T H (\theta - \theta_k)\le \delta.
$$
The  expression of Lagrangian is given by 
$$
\max_{\theta} \min_{\lambda \ge 0} g\cdot (\theta - \theta_k) - \lambda\left[ \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) - \delta\right],
$$
whose duality problem is  (**I need to check the strong duality.**)
$$
\min_{\lambda \ge 0} \max_{\theta} g\cdot (\theta - \theta_k) - \lambda\left[ \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) - \delta\right].
$$
Firstly, we solve the inner max problem by take deviation w.r.t. $$\theta$$:
$$
g - \lambda H (\theta - \theta_k) = 0
\Rightarrow \theta - \theta_k = \frac{1}{\lambda}H^{-1} g
$$
Then a new problem comes as
$$
\min_{\lambda \ge 0} \frac{1}{\lambda}g^TH^{-1}g 
- \lambda\left[\frac{1}{2\lambda^2}g^T H^{-1} g - \delta\right] 
= \delta\lambda + \frac{1}{2\lambda} g^T H^{-1}g\\
\Rightarrow \lambda_{\min} = \sqrt{\frac{g^T H^{-1} g}{2\delta}}, 
\theta - \theta_k = \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g.
$$

We notice that we are interested in $$H^{-1}g$$  not the $$H^{-1}$$ itself. We can use conjugate gradient method to avoid matrix inversion.

## 6 Practical Algorithm

![Pseudocode TRPO](https://spinningup.openai.com/en/latest/_images/math/5808864ea60ebc3702704717d9f4c3773c90540d.svg)

## 6.1 Code Tricks





[^1]: Approximately optimal approximate reinforcement learning.
[^2]: http://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/ (code implementation)
[^3]: https://spinningup.openai.com/en/latest/algorithms/trpo.html (nice and neat blog with beautiful pseudocode)



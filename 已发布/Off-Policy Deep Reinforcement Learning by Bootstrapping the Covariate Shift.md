# Off-Policy Deep Reinforcement Learning by Bootstrapping the Covariate Shift

## Abstract

> Under this method, online updates to the value function are reweighted to avoid divergence issues typical of off-policy learning.   

Problems of COP-TD:

- It's not known to be a contraction mapping, and hence, may be more unstable in practice;
- Projection step onto the probability simplex.

Solution:

- A discount factor into COP-TD;
- A soft normalization penalty.

## Introduction

> Central to reinforcement learning is the idea that an agent should learn from experience.

> More recently, value divergence was perhaps the most significant issue dealt with in the design of the DQN agent, and remains a source of concern in deep reinforcement learning.

The TD-algorithm with linear function approximation is semi-gradient update rule for TD learning:
$$
\theta = \theta + \alpha[r_\pi(s) + \gamma \phi(s')^T \theta - \phi(s)^T \theta].
$$
The update rule converges to the stationary point of **projected Bellman equation**:
$$
\hat V^\pi = \Pi_d T_\pi \hat V^{\pi}.
$$

If it uses on-policy statistics, $$d = d_\pi$$, this update rules converges to this fixed point provided $$\alpha$$ is taken to satisfy the Robbins-Monro conditions and other mild assumptions. If $d \ne d_\pi$, however, may lead to divergence of the weight vector.

> A sign of the importance of this issue can be seen in Sutton and Barto’s choice to dub “deadly triad” the combination of off-policy learning, function approximation, and bootstrapping.  

> **Theorem 1**. $$\forall d \in \Delta (S)$$ be some arbitrary distribution. Suppose that $$\Vert \Pi_d P_{d_\pi}\Vert_{d_\pi} < 1/\gamma$$ and there is a fixed point $$\hat V^\pi_d$$ to the projected Bellman equation $$V := \Pi_d T_\pi V$$. Then its approximation error in $$d_\pi-weighted$$ norm is at most
> $$
> \Vert \hat V^\pi_d - V^\pi \Vert_{d_\pi}
> \le \frac{\Vert \Pi_d V^\pi - V^\pi \Vert_{d_\pi}}{1 - \gamma \Vert \Pi_d P_\pi \Vert_{d_\pi}}
> $$
> Furthermore, this error is minimized when $$d = d_\pi$$.
>
> **proof**:
> $$
> \begin{align*}
> &\Pi_d V^\pi - V^\pi\\
> =& \Pi_d V^\pi - \hat V^\pi_d + \hat V^\pi_d - V^\pi\\
> =& \gamma \Pi_d P_\pi V^\pi - \gamma \Pi_d P_\pi \hat V^\pi_d + \hat V^\pi_d - V^\pi\\
> =&(I - \gamma \Pi_d P_\pi) (\hat V^\pi_d - V^\pi)
> \end{align*}
> $$
>
> $$
> \begin{align*}
> &\Vert \hat V^\pi_d - V^\pi \Vert_{d_\pi}\\
> =& \Vert (I - \gamma \Pi_d P_\pi)^{-1} (\Pi_d V^\pi - V^\pi) \Vert_{d_\pi}\\
> =& \Vert \sum^\infty_{t=0} (\gamma \Pi_d P_\pi)^t (\Pi_d V^\pi - V^\pi) \Vert_{d_\pi}\\
> \le& \sum^\infty_{t=0} \Vert(\gamma \Pi_d P_\pi)^t\Vert_{d_\pi} \Vert \Pi_d V^\pi - V^\pi \Vert_{d_\pi}\\
> =& \frac{\Vert \Pi_d V^\pi - V^\pi \Vert_{d_\pi}}{1 - \gamma\Vert \Pi_d P_\pi\Vert_{d_\pi}}
> \end{align*}
> $$

## COP-TD

$$
\theta = \theta + \alpha \frac{d_\pi(s)}{d_\mu(s)} \frac{\pi(a\vert s)}{\mu(a\vert s)} [r(s,a) + \gamma \phi^T(s') \theta - \phi^T(s)\theta] \phi^T(s).
$$

Question: Learn the ratio $$\frac{d_\pi}{d_\mu}$$.

COP-TD performs the following update: $$(s, a, r, s') \sim \mu$$ 
$$
c(s') = c(s') + \alpha \left[\frac{\pi(a\vert s)}{\mu(a \vert s)}c(s) - c(s')\right].
$$
We define a transformation: $$c_{t+1} = Yc_t$$
$$
(Yc)(s') := \mathbb{E}_{s \sim d_\mu, a \sim \mu, s' \sim p(s' \vert s, a)}\left[\frac{\pi(a\vert s)}{\mu(a \vert s)} c(s) \Big\vert s'\right].
$$
In vector notation, this operator is:
$$
\begin{align*}
&(Yc)(s') \\
=& \frac{1}{d_\mu(s')}\left[\sum_s d_\mu(s) \sum_a \mu(a\vert s) \frac{\pi(a \vert s)}{\mu(a\vert s)} c(s) p(s, a, s')\right]\\
=& \frac{1}{d_\mu(s')} \sum_{s} d_\mu(s) c(s) \sum_a \pi(a \vert s)p(s, a, s')\\
=& D^{-1}_{d_\mu} P^T_{\pi} D_{d_\mu} c(s').
\end{align*}
$$
Any multiple of $$\frac{d_\pi}{d_\mu}=(\frac{d_\pi(s)}{d_\mu(s)})$$ is a fixed point of Y: $$Y\beta \frac{d_\pi}{d_\mu} = \beta \frac{d_\pi}{d_\mu}$$.(hint: $$\beta \frac{d_\pi}{d_\mu} = \beta D^{-1}_{\mu} d_\pi$$). Or I prefer call it fixed line.

So we define the normalized COP operator:
$$
(\bar Y c) (s') := \frac{(Yc)(s')}{\sum_s (Yc)(s)}.
$$

> **Theorem 2**: Suppose that $$P_\pi$$ defines an ergodic Markov chain on the state space. Then the process $$c^{k+1} = Y c^k$$ converges to $$C \frac{d_\pi}{d_\mu}$$.
>
> **proof**:
> $$
> \begin{align*}
> c^{k} =& Y^k c^0 = (D^{-1}_{d_\mu} P^T_{\pi}D_{d_\mu})^k c^0\\
> =& D^{-1}_{d_\mu}(P^T_\pi)^k D_{d_\mu} c^0\\
> =& D^{-1}_{d_\mu} [d_\pi + \epsilon_k e]C\\
> =& C \frac{d_\pi}{d_\mu} + C \epsilon_k D^{-1}_{d_\mu}e\\
> \rightarrow & C\frac{d_\pi}{d_\mu}
> \end{align*}
> $$
> **Corollary 1**：$$c^{k+1} = (\bar Y c)(s') c^k$$ converges to $$\frac{d_\pi}{d_\mu}$$. 

### COP-TD with Linear Function Approximation

We use linear approximation $$\hat c(s) = \phi(s)^T w$$, then
$$
\tilde w = w + \alpha \left[\frac{\pi(a\vert s)}{\mu(a \vert s)}\phi(s)^T w - \phi(s')^T w\right] \phi(s'),
$$
and it is followed by a projection step on the $$d_\mu-weighted$$ simplex $$\Delta_{\Phi, d_\mu}$$ defined by the set $$W_{\Phi, d_\mu} := \{u \in \mathbb{R}^k : \sum_{s \in S} d_{\mu}(s) \phi(s)^T u = 1, \phi(s)^T u \ge 0\}$$($$d^T_\mu \Phi u = 1$$):
$$
w = \arg\min_{u \in W_{\Phi, d_\mu}}\Vert u - \tilde w \Vert.
$$
The combined process is summarized by the normalized COP operator:
$$
\hat c^{k+1} := \Pi_{\Delta_{\Phi, d_\mu}}\Pi_d Y \hat c^k.
$$

> **Lemma 1**. Let Y be a symmetric COP-TD operator and $$\Pi$$ be the projection onto $$\Phi$$ in $$L_2$$ norm. If $$\frac{d_\pi}{d_\mu}$$ is not in the span of $$\Phi$$, then $$c = 0$$ is the only solution to
> $$
> \Pi Y c = c.
> $$
> **proof**:
>
> - Step 1: If $$\frac{d_\pi}{d_\mu}$$ are not in the span of $$\Phi$$, $$\alpha \frac{d_\pi}{d_\mu}$$ can't be a fixe point of $$\Pi Y$$. 
>   $$
>   \left\Vert \alpha \frac{d_\pi}{d_\mu}\right\Vert 
>   = \left\Vert \Pi Y \alpha \frac{d_\pi}{d_\mu}\right\Vert 
>   = \left\Vert \Pi\alpha \frac{d_\pi}{d_\mu}\right\Vert
>   < \left\Vert \alpha \frac{d_\pi}{d_\mu}\right\Vert.
>   $$
>
> - Step 2: If $$c \ne 0$$ and $$c \ne \alpha\frac{d_\pi}{d_\mu}$$, c can't be a fixed point.
>
>   Because Y is similar to $$P_\pi$$,  whose largest eigenvalue is 1, then $$\Vert Yc \Vert \le \Vert c \Vert$$,
>   $$
>   \Vert c \Vert = \Vert \Pi Y c \Vert = \Vert \Pi \Vert \Vert Yc \Vert < \Vert c \Vert.
>   $$

## A Practical COP-TD

The limitations to COP-TD:

- **Lack of contraction factor**: 
  - Y is not in general a contraction mapping；
  - converge at a slow rate;
  - greater variations;
  - unstable with function approximation.
- **Hard-to-satisfy projection step**:

## The Discounted COP Learning Rule

> **Definition 1**: $$\hat \gamma$$-discounted COP-TD learning rule is
> $$
> c(s') = c(s') + \alpha\left[\hat\gamma \frac{\pi(a \vert s)}{\mu(a \vert s)} c(s) + (1 - \hat \gamma) - c(s')\right].
> $$
> The corresponding operator is
> $$
> Y_{\hat \gamma}c := \hat \gamma Y c + (1 - \hat \gamma)e.
> $$

discounted COP-TD learning rule has some desirable properties.

> **Definition 2**. For a given $$\hat \gamma \in[0, 1]$$, we define the discounted rest transition function $$\hat P_\pi$$ as:
> $$
> \hat P_{\pi} := \hat \gamma P_\pi + (1 - \hat \gamma)e d^T_\mu.
> $$
> The corresponding stationary distribution is $$\hat d_\pi$$.

> The discounted reset transition function can be understood as a process which either transitions as usual with probability $$\hat \gamma$$, or resets to the stationary distribution $$d_\mu$$ with the remainder probability.   

> **Proposition 1**:
> $$
> \hat d_\pi = (1-\hat\gamma)(I - \hat\gamma P^T_\pi)^{-1} d_\mu,
> $$
>
> **proof**:
>
> Because $$(I - \hat \gamma P^T_\pi)^{-1} = \sum^{\infty}_{i=0}(\hat \gamma P^T_\pi)^{i}$$, then
> $$
> e^T (1-\hat\gamma)(I - \hat\gamma P^T_\pi)^{-1} d_\mu = (1-\hat\gamma)\sum^{\infty}_{i=0}\gamma^i e^T d_\mu = 1,
> $$
> and
> $$
> \begin{align*}
> &\hat P^T_\pi (1-\hat\gamma)(I - \hat\gamma P^T_\pi)^{-1} d_\mu\\
> =& [\hat \gamma P_\pi + (1 - \hat \gamma)e d^T_\mu](1-\hat\gamma)(I - \hat\gamma P^T_\pi)^{-1} d_\mu\\
> =& \hat \gamma P_\pi(1-\hat\gamma)(I - \hat\gamma P^T_\pi)^{-1} d_\mu\\
> &+ (1 - \hat \gamma)^2 d_\mu e^T(I - \hat\gamma P^T_\pi)^{-1} d_\mu\\
> =& \hat \gamma P_\pi(1-\hat\gamma)\sum^{\infty}_{i=0}(\hat\gamma P^T_\pi)^i d_\mu + (1 - \hat \gamma) d_\mu\\
> =& (1 - \hat\gamma)\sum^{\infty}_{i=0}(\hat\gamma P^T_\pi)^i d_\mu\\
> =& (1-\hat\gamma)(I - \hat\gamma P^T_\pi)^{-1} d_\mu
> \end{align*}
> $$

> **Lemma 2**. For $$\hat \gamma < 1$$, the ratio $$\frac{\hat d_\pi}{d_\mu}$$ is a  unique fixed point of $$Y_{\hat \gamma}$$.
>
> **proof**:
>
> - Step 1, fixed point proof: $$ Y c_1 - Y c_2 < \hat \gamma (c_1 - c2) $$ 
>   $$
>   \begin{align*}
>   Y_{\hat \gamma} \frac{\hat d_\pi}{d_\mu}
>   =& \hat \gamma D^{-1}_{d_\mu} P^T_\pi D_{d_\mu} \frac{\hat d_\pi}{d_\mu} + (1 - \hat\gamma) e\\
>   =& \hat \gamma D^{-1}_{d_\mu} P^T_\pi \hat d_\pi + (1 - \hat\gamma) D^{-1}_{d_\mu} d_\mu\\
>   =& D^{-1}_{d_\mu}(\hat \gamma P^T_\pi \hat d_\pi + (1 - \hat\gamma) d_\mu)\\
>   =& D^{-1}_{d_\mu}(\hat \gamma P^T_\pi \hat d_\pi + (1 - \hat\gamma) d_\mu e^T \hat d_\mu)\\
>   =& D^{-1}_{d_\mu}(\hat P^T_\pi \hat d_\mu)
>   = \frac{\hat d_\pi}{d_\mu}
>   \end{align*}
>   $$
>
> - Step 2, uniqueness proof: $$Y^n_{\hat \gamma}$$ is a contraction mapping and Banach's fixed point theorem.

> **Theorem 3**. The process $$c^{k+1} := Y_{\hat \gamma} c^k$$ converges to $$\frac{\hat d_\pi}{d_\mu}$$.
>
> **proof**.
> $$
> \begin{align*}
> Y^k_{\hat \gamma}c^0 =& \sum^{k-1}_{i=0}(\hat\gamma Y)^i (1-\hat\gamma)e + \hat\gamma^k Y^kc^0\\
> =& \sum^{k-1}_{i=0}(\hat\gamma D^{-1}_{d_\mu} P^T_\pi D_{d_\mu})^i (1-\hat\gamma)e + \hat\gamma^k Y^kc^0\\
> =& (1 - \hat\gamma)D^{-1}_{d_\mu}\sum^{k-1}_{i=0}(\hat\gamma P^T_\pi)^i d_\mu + \hat\gamma^k Y^k c^0\\
> \rightarrow & (1 - \hat\gamma) D^{-1}_{d_\mu} (I - \hat\gamma P^T_{\pi})^{-1} d_\mu\\
> =& D^{-1}_{d_\mu} \hat d_\pi
> \end{align*}
> $$
> 

## Discounted COP with Linear Function Approximation

> **Lemma 3**. 
>
> **proof**.
>
> From definition:
> $$
> (Y^nc)(s') = \sum_{s}\frac{d_\mu(s)}{d_\mu(s')} P^n_\pi(s'\vert s) c(s).
> $$
> Let $$z(s') = \sum_s \frac{d_\mu(s)}{d_\mu(s')}P^n_\pi(s'\vert s) $$ . Therefore
> $$
> \begin{align*}
> &\Vert Y x\Vert^2_{d_\mu}\\
> =& \sum_{s'} d_\mu(s')\left(\sum_s \frac{d_\mu(s)}{d_\mu(s')} P^n_\pi(s'\vert s) x(s)\right)^2\\
> =& \sum_{s'} d_\mu(s') z^2(s') \left(\sum_s \frac{1}{z(s')} \frac{d_\mu(s)}{d_\mu(s')} P^n_\pi(s'\vert s) x(s)\right)^2\\
> \le& \sum_{s'} d_\mu(s') z^2(s') \sum_s \frac{1}{z(s')} \frac{d_\mu(s)}{d_\mu(s')} P^n_\pi(s'\vert s) x^2(s)\\
> =& \sum_{s'} z(s') \sum_{s} d_\mu(s) P^n_\pi(s'\vert s) x^2(s)\\
> \le& \sup_{s'\in S} z(s')\sum_{s'} \sum_{s}d_\mu(s)P^n_\pi(s'\vert s) x^2(s)\\
> :=& K_{\pi,\mu,n}\sum_{s} d_\mu(s) x^2(s) \sum_{s'} P^n_{\pi}(s'\vert s)\\
> =& K_{\pi,\mu,n}\sum_{s} d_\mu(s) x^2(s) = K_{\pi,\mu,n}\Vert x\Vert^2_{d_\mu}
> \end{align*}
> $$
>
> $$
> \begin{align*}
> K_{\pi, \mu, n} =& \sup_{s' \in S} \sum_{s'\in S} \sum_s \frac{d_\mu(s)}{d_\mu(s')}P^n_\pi(s'\vert s) \\
> =& \sup_{s' \in S}\sum_{s'\in S} \sum_s \frac{d_\mu(s)}{d_\pi(s)} \frac{d_\pi(s)}{d_\mu(s')}P^n_\pi(s'\vert s) \\
> \le& \left\Vert\frac{d_\mu(s)}{d_\pi(s)} \right\Vert_\infty 
> \sup_{s' \in S} \sum_s \frac{d_\pi(s)}{d_\mu(s')}P^n_\pi(s'\vert s) \\
> =& \left\Vert\frac{d_\mu(s)}{d_\pi(s)} \right\Vert_\infty 
> \sup_{s' \in S} \sum_s \frac{d_\pi(s')}{d_\mu(s')}\\
> =& \left\Vert\frac{d_\mu(s)}{d_\pi(s)} \right\Vert_\infty
> \left\Vert\frac{d_\mu(s')}{d_\mu(s')} \right\Vert_\infty := K_{\pi, \mu}.
> \end{align*}
> $$

> **Theorem 4**. Consider the n-step discounted COP operator $$Y^n_{\hat\gamma}$$. Then for any $$c \in \mathbb{R}^n$$,
> $$
> \left\Vert Y^n_{\hat\gamma} c - \frac{\hat d_\pi}{d_\mu}\right\Vert_{d_\mu}
> \le \hat\gamma^n \sqrt{K_{\pi, \mu, n}}
> \left\Vert c - \frac{\hat d_\pi}{d_\mu}\right\Vert_{d_\mu}
> $$
> **proof**:
> $$
> \begin{align*}
> &\left\Vert Y^n_{\hat\gamma} c - \frac{\hat d_\pi}{d_\mu}\right\Vert_{d_\mu} \\
> =& \left\Vert Y^n_{\hat\gamma} c - Y^n_{\hat\gamma} \frac{\hat d_\pi}{d_\mu}\right\Vert_{d_\mu} \\
> =& \left\Vert \hat \gamma^n Y^n c - \hat\gamma^n Y^n \frac{\hat d_\pi}{d_\mu}\right\Vert_{d_\mu} \\
> =& \hat\gamma^n \sqrt{K_{\pi, \mu, n}} \left\Vert c - \frac{\hat d_\pi}{d_\mu}\right\Vert_{d_\mu}
> \end{align*}
> $$

## Soft Ratio Normalization

Suppose we are given a approximation function $$ c: S \rightarrow \mathbb{R}$$, we want
$$
\sum_{s\in S} d_\mu(s) c(s) = 1.
$$
The normalization loss is
$$
L(c) = \frac{1}{2}\left(\sum_{s} d_\mu(s) c(s) - 1\right)^2,
$$

$$
\nabla L(c) = \left(\sum_s d_\mu(s) c(s) - 1\right) \sum_{s} d_\mu(s) \nabla c(s).
$$

> **Theorem 5**. The unbiased estimate of $$\nabla L(c)$$ is
> $$
> \frac{1}{m} \sum^{m}_{i=1} (\frac{1}{m-1}\sum_{j\ne i} c(s_j) - 1) \nabla c(s_i).
> $$
>
> **proof**:
> $$
> \begin{align*}
> &\mathbb{E}_{d_\mu}\left[\frac{1}{m} \sum^{m}_{i=1} (\frac{1}{m-1}\sum_{j\ne i} c(s_j) - 1) \nabla c(s_i)\right] \\
> =& \frac{1}{m} \sum^{m}_{i=1}\mathbb{E}_{d_\mu} \left[(\frac{1}{m-1}\sum_{j\ne i} c(s_j) - 1) \nabla c(s_i)\right] \\
> =& \frac{1}{m} \sum^{m}_{i=1}\mathbb{E}_{d_\mu}[c(s) - 1] \mathbb{E}_{d_\mu}[\nabla c(s)] = \nabla L(c)
> \end{align*}
> $$

> In our experimental section we will see that the normalization loss plays an important role in making COP-TD practical.

## Experimental Results

> - Instead, to reweight sample transitions we use a prioritized replay memory (Schaul et al. 2016) where priorities correspond to the approximate ratios of our model, which in expectation recovers the reweighting. 
> - These adjusted sampling priorities result in large portions of the dataset being mostly ignored (i.e. those unlikely under policy π); hence, the effective size of the data set is reduced and we risk overfitting. 
> - In our experiment we mitigated this effect by taking a larger replay memory size (10 million frames) than usual.

> Preliminary experiments showed that learning the ratio with prioritized sampling led to stability issues, hence we train the ratio model by sampling transitions uniformly from the replay memory.  Each training step samples two independent transition batches, prioritized and uniform for the value function and covariate shift respectively.  
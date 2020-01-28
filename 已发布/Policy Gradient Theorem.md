# Policy Gradient Theorem

I write this essay because I just read a very good blog[^1], which encourages me to take notes about everything including policy gradient theorem. I have token some notes by latex mostly. I find it's more convenient by markdown.

## 1. Preliminaries

- $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_{T-1}, a_{T-1}, r_{T-1}, s_T)$;
- State initial distribution $s_0 \sim \mu$;
- $R(\tau) = \mathbb{E}_{tau}[\sum^{T-1}_{t=0} \gamma^t r_t]$.


## 2. Computing the Raw Gradient

- Two tricks before the proof:
    - Log-derivative trick:
      $$
      \begin{align*}
      \nabla_{\theta} \mathbb{E}_{p_\theta(x)}[f(x)] 
      =& \nabla_\theta \int p_\theta(x) f(x) dx
      = \int \nabla_\theta p_\theta(x) f(x) dx\\
      =& \int p_\theta(x) \nabla_\theta \log p_\theta(x) f(x) dx
      = \mathbb{E}_{p_\theta(x)}[f(x) \nabla_\theta \log p_\theta(x)]
      \end{align*}
      $$

    - Determining log probability:

    $$
    \begin{align*}
    \nabla_\theta \log p_\theta(\tau)
    =& \nabla_\theta \log \left( \mu(s_0) \prod^{T-1}_{t=0}\pi_\theta(a_t|s_t) P(s_{t+1} | s_t, a_t) \right)\\
    =& \nabla_\theta \sum^{T-1}_{t=0} \log \pi_\theta(a_t | s_t)
    \end{align*}
    $$


- In trajectory space, we have:
    $$
    \begin{align*}
    &\nabla_\theta \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
    = \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau) \cdot \nabla_\theta \log p(\tau)]
    = \mathbb{E}_{\tau\sim\pi_\theta} \left[R(\tau) \cdot \nabla_\theta \left(\sum^{T-1}_{t=0} \log \pi_\theta(a_t | s_t)\right)\right]\\
    =& \mathbb{E}_{\tau\sim\pi_\theta}\left[\left( \sum^{T-1}_{t=0} r_t\right) \cdot \nabla_\theta\left(\sum^{T-1}_{t=0} \log \pi_\theta(a_t | s_t)\right)\right]
    \end{align*}
    $$

- In state-action-reward-state space, we have:
  $$
  \begin{align*}
  &\nabla_\theta \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
  = \nabla_\theta \mathbb{E}_{\tau\sim\pi_\theta}\left[\left( \sum^{T-1}_{t=0} r_t\right) \right]
  = \sum^{T-1}_{t=0} \nabla_\theta\mathbb{E}_{\tau\sim\pi_\theta}[r_t]
  = \sum^{T-1}_{t=0} \nabla_\theta\mathbb{E}_{\tau^t}[r_t]\\
  =& \sum^{T-1}_{t=0} \mathbb{E}_{\tau} \left[{r^t \cdot \sum^{t}_{t'=0} \nabla_\theta \log \pi_\theta(a_{t'}|s_{t'})}\right]
  = \mathbb{E}_\tau \left[ \sum^{T-1}_{t=0} r_t \sum^t_{t'=0} \nabla_\theta\log\pi_\theta(a_{t'} | s_{t‘})\right] \\
  =& \mathbb{E}_\tau \left[ \sum^{T-1}_{t'=0} \nabla_\theta\log\pi_\theta(a_{t'}|s_{t'}) \sum^{T-1}_{t=t'} r_{t}  \right]
  \end{align*}
  $$
  

## 3. Understanding the Baseline

- Gradient trick2:
  $$
  \mathbb{E}_{a_t}[\nabla_\theta \log \pi_\theta(a_t | s_t)]
  = \int \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_\theta(a_t | s_t)}\pi_\theta(a_t | s_t) d a_t
  = \nabla_\theta \int \pi_\theta(a_t|s_t) d a_t = \nabla_\theta 1 = 0
  $$

- Baseline function's properties:
  $$
  \begin{align*}
  &\mathbb{E}_{\tau\sim\pi_\theta}[\nabla_\theta \log \pi_\theta(a_t | s_t) b(s_t)]
  =\mathbb{E}_{s_{0:t}, a_{0:t-1}}[b_t \cdot \mathbb{E}_{s_{t+1:T}, a_{t:T-1}}[\nabla_\theta \log \pi_\theta(a_t | s_t)]]\\
  =& \mathbb{E}_{s_{0:t}, a_{0:t-1}}[b_t \cdot \mathbb{E}_{a_t}[\nabla_\theta\log\pi_\theta(a_t|s_t)]] = 0 
  \end{align*}
  $$

- Policy gradient with baseline:
  $$
  \nabla_\theta \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] = 
  \mathbb{E}_\tau \left[ \sum^{T-1}_{t'=0} \nabla_\theta\log\pi_\theta(a_{t'}|s_{t'}) \left(\sum^{T-1}_{t=t'} r_{t} - b(s_t)\right)  \right]
  $$

## 4. Infinite-horizon and Discounted Situation

In infinite-horizon and discounted situation, the policy gradient is
$$
\begin{align*}
\nabla_\theta \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] 
=& \mathbb{E}_\tau \left[ \sum^{\infty}_{t'=0} \nabla_\theta\log\pi_\theta(a_{t'}|s_{t'}) \left(\sum^{\infty}_{t=t'} \gamma^{t-t'} r_{t} - b(s_t)\right)  \right]\\
=& \mathbb{E}_\tau \left[ \sum^{\infty}_{t'=0} \nabla_\theta\log\pi_\theta(a_{t'}|s_{t'}) \left(Q_\gamma(s_{t'}, a_{t'}) - b(s_t)\right)  \right].
\end{align*}
$$

Let $d^\pi$ be the stationary distribution, then
$$
\nabla_\theta \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
= \sum_{s\in S} d^{\pi_\theta}(s) \sum_{a \in A}\pi_\theta(a|s)
\nabla_\theta \log \pi_\theta(a|s) (Q^{\pi_\theta}_\gamma(s, a) - b(s)).
$$

## 5. Policy Gradient with Function Approximation

If we use $f_w(s, a)$ to approximate $Q^{\pi_\theta}_\lambda(s,a)$, and we optimize $w$ according to
$$
\min_w J(w) = \min_w \sum_{s \in S} d^\pi(s) \sum_{a \in A} \pi(a|s) [Q^\pi(s,a) - f_w(s,a)]^2.
$$
Then, The local minimum satisfies
$$
\sum_{s \in S} d^\pi(s) \sum_{a \in A} \pi(a|s) [Q^\pi(s,a) - f_w(s,a)]\nabla_w(f_w) = 0
$$
We construct that $\nabla_w (f_w) = \nabla_\theta\log\pi_\theta(a|s)$, which means $f_w(s, a) = w^T \cdot \log\pi_\theta(a|s) $. Then
$$
\sum_{s \in S} d^{\pi_\theta}(s) \sum_{a \in A} \pi_\theta(a|s) [Q^{\pi_\theta}(s,a) - f_w(s,a)]\nabla_\theta\log\pi_\theta(a|s) = 0\\
\Rightarrow
\nabla_\theta\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)] 
= \sum_{s \in S} d^{\pi_\theta}(s) \sum_{a \in A} \pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s) f_w(s,a)
$$










[^1]:  https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/ 
# Temperate-difference Algorithm



## 1. Approximate Policy Evaluation Using $$TD(1)$$

The key question is that we want use $$V(s, \theta)$$ to approximate true state-value function $$V^\pi(s)$$, 

For random trajectory $$\tau=(s_0, a_0, r_0, \ldots, s_N)$$ generated from policy $$\pi$$, we want to minimize the true error:
$$
J(\pi) = \mathbb{E}_{\tau\sim\pi}\left\{\frac{1}{2} \sum^{N-1}_{t=0}\left[V(s_t, \theta) - \sum^{N-1}_{k=t} \gamma^{k-t} r_t(s_t, \pi(s_{t}), s_{t+1})\right]^2 \right\}.
$$
For $$m$$th trajectory $$\tau^m=(s^m_0, a^m_0, r^m_0, s^m_1, \ldots, s^m_N)$$ and $$V(s^m_N, \theta) = 0$$, we update $$w$$ by Monte Carlo method and SGD(mini_batch = 1):
$$
\begin{align*}
\Delta\theta_{m} =& \nabla_\theta\left\{ \frac{1}{2}\sum^{N-1}_{t=0}\left[V(s^m_t,\theta_{m}) - \sum^{N-1}_{k=t}\gamma^{k-t} r^m_k(s^m_k, \pi(s^m_k), s^m_{k+1})\right]^2 \right\}\\
=& \sum^{N-1}_{t=0}  \nabla_{\theta}V(s^m_t, \theta_m) \left[V(s^m_t,\theta_{m}) - \sum^{N-1}_{k=t}\gamma^{k-t} r^m_k(s^m_k, \pi(s^m_k), s^m_{k+1})\right]\\
=& \sum^{N-1}_{t=0}\nabla_{\theta}V(s^m_t, \theta_m)\left[\sum^{N-1}_{k=t} [\gamma^{k-t} V(s^m_{k},\theta_m) - \gamma^{k+1-t}V(s^m_{k+1}, \theta_m)] - \sum^{N-1}_{k=t} \gamma^{k-t} r^m_k (s^m_k, \pi(s^m_k), s^m_{k+1}) \right]\\
=& \sum^{N-1}_{t=0}\nabla_{\theta}V(s^m_t, \theta_m)\sum^{N-1}_{k=t}\left[\gamma^{k-t} V(s^m_k, \theta_m) - \gamma^{k+1-t} V(s^m_{k+1}, \theta_m) - \gamma^{k-t}r^m_k(s^m_k, \pi(s^m_k), s^m_{k+1})\right]\\
=& \sum^{N-1}_{k=0} \left[V(s^m_k, \theta_m) - \gamma V(s^m_{k+1}, \theta_m) - r^m_k(s^m_k, \pi(s^m_k), s^m_{k+1})\right] \sum^{k}_{t=0} \gamma^{k-t} \nabla_{\theta}V(s^m_t, \theta_m) \\
:=& -\sum^{N-1}_{k=0} d^m_k \sum^{k}_{t=0} \gamma^{k-t} \nabla_{\theta}V(s^m_t, \theta_m).
\end{align*}
$$

## 2. $$TD(\lambda)$$ for General $$\lambda$$

$$
\begin{align*}
\Delta \theta_m =& \sum^{N-1}_{t=0}\nabla_{\theta}V(s^m_t, \theta_m)\sum^{N-1}_{k=t} \lambda^{k-t} \left[\gamma^{k-t} V(s^m_k, \theta_m) - \gamma^{k+1-t} V(s^m_{k+1}, \theta_m) - \gamma^{k-t}r^m_k(s^m_k, \pi(s^m_k), s^m_{k+1})\right]\\
=& -\sum^{N-1}_{t=0}\nabla_{\theta}V(s^m_t, \theta_m)\sum^{N-1}_{k=t} (\lambda\gamma)^{k-t} d^m_k\\
=& -\sum^{N-1}_{k=0} d^m_k\sum^{k}_{t=0} (\lambda\gamma)^{k-t}\nabla_{\theta}V(s^m_t, \theta_m).
\end{align*}
$$

## 3. $$TD(\lambda)$$ with Linear Function Approximation

For a linear function approximation is

$$
V(s, \theta) = \phi^T(s)\theta.
$$

If we denote $$ \Phi = [\phi(s_1), \ldots, \phi(s_n)]^T, for\ S=\{s_1, \ldots, s_n\}$$, then

$$
V(\cdot, \theta) = \Phi \theta.
$$

Now, we are interested in on-line update:

$$
\begin{align*}
w^m_{t+1} =& w^m_{t} + \alpha_t d_t \sum^{t}_{k=0}(\lambda \gamma)^{t-k} \phi(s_k)\\
:=& w^m_{t} + \alpha_t d_t z_t,\\
w.r.t. z_{t+1} =&  \lambda\gamma z_t + \phi(s_{t+1}).
\end{align*}
$$

We look into 

$$
\begin{align*}
d_t z_t =& (r(s_t, a_t, s_{t+1}) + \gamma \phi^T(s_{t+1})w_t - \phi^T(s_t)w_t)z_t\\
=& z_t(\gamma \phi^T(s_{t+1}) - \phi^T(s_t))w_t + z_t r(s_t, a_t, s_{t+1})\\
=& A(X_t) w_t + b(X_t), where\ X_t = \{s_0, \ldots, s_{t+1}\}.
\end{align*}
$$

The state-transition function is

$$
w_{t+1} = (I + \alpha_t A(X_t)) w_t + b(X_t).
$$

We denote $$p_{\infty}$$ to be the distribution $$p^T_\infty P^\pi = p^T_{\infty}$$, and $$ D = diag(p_\infty)$$:
- $$ \mathbb{E}_{p_{\infty}} \left[ V^T(s_0) V(s_m) \right] = V^T D P^{m} V $$;

- $$ \mathbb{E}_{p_{\infty}} \left[ \phi(s_0) \phi^T(s_m) \right] = \Phi^T D P^m \Phi$$;

- We take expectation for $$A(X_t)$$:
  
  $$
  \begin{align*}
  &\lim_{t \to \infty} \mathbb{E}_{p_\infty} \left[ A(X_t) \right] \\
  =& \lim_{t \to \infty} \mathbb{E}_{p_{\infty}} \left[\sum^{t}_{k=0} {(\gamma\lambda)}^{t-k }\phi(s_k)(\gamma\phi^T (s_{t+1}) - \phi^T(s_t))\right]\\
  =& \lim_{t \to \infty} \sum^{t}_{k=0} {(\gamma \lambda)}^{t-k}\Phi^{T} D \left[ \gamma P^{t-k+1} - P^{t-k} \right] \Phi\\
  =& \lim_{t \to \infty} \sum^{t}_{m = 0} {(\gamma\lambda)}^{m} \Phi^T D \left[ \gamma P^{m+1} - P^{m} \right] \Phi\\
  =& \Phi^T D \left( (1 - \lambda) \sum^{\infty}_{m=0} \lambda^m{(\gamma P)}^{m+1} - I \right) \Phi\\
  :=& \Phi^T D (M - I) \Phi
  \end{align*}
  $$
  
- We take expectation for $$b(X_t)$$:
  
  $$
  \begin{align*}
   &\lim_{t \to \infty} \mathbb{E}_{p_{\infty}}\left[ b(X_t) \right]\\
   =& \lim_{t \to \infty} \mathbb{E}_{p_{\infty}}\left[ \sum^{t}_{k=0} {(\gamma\lambda)}^{t-k} \phi(s_k) r(s_t,s_{t+1}) \right]\\
   =& \lim_{t \to \infty} \sum^{t}_{k=0} {(\gamma\lambda)}^{t-k} \mathbb{E}_{p_{\infty}} \left[ \phi(s_k) r(s_t, s_{t+1}) \right]\\
   =& \lim_{t \to \infty} \sum^{t}_{k=0} {(\gamma\lambda)}^{t-k} \Phi^T D P^{t-k} \sum^{}_{s' \in S} r(s, s')\\
   =& \lim_{t \to \infty} \sum^{t}_{m=0} {(\gamma\lambda)}^{m} \Phi^{T} D P^{m} \sum^{}_{s' \in S} r(s,s')\\
   =& \Phi^T D \sum^{\infty}_{m=0} {(\gamma\lambda P)}^m \bar r, \quad \left( where\ \bar r = \sum^{}_{s' \in S} p(s, s') r(s, s') \right)
   \end{align*}
  $$

> **Lemma 1**: $$\forall V$$, $$\Arrowvert PV \Arrowvert^2_D \le \Arrowvert V \Arrowvert^2_{D}$$.

**proof**:
$$
\begin{align*}
&\Arrowvert PV \Arrowvert^2_D = V^T P^T D P V \\
=& \sum^{n}_{i=1} p_{\infty}(i) {\left( \sum^{n}_{j=1} p_{ij} V(j) \right)}^2 \\
\le& \sum^{n}_{i=1} p_\infty(i) \sum^{n}_{j=1} p_{ij}V^2(j)\\
\le& \sum^{n}_{j=1} p_\infty(j) V^2(j) = V^T D V = \Arrowvert V \Arrowvert^2_{D},
\end{align*}
$$

> **Lemma 2**:
> 
> $$
> \begin{align*}
> \Arrowvert MV \Arrowvert_D \le& (1 - \lambda) \sum^{\infty}_{m = 0}  \lambda^m \gamma^{m+1} \Arrowvert P^{m+1} V \Arrowvert_D \le \frac{\gamma (1 - \lambda)}{1 - \gamma \lambda} \Arrowvert V \Arrowvert_D.
> \end{align*}
> $$

> **Lemma 3**: $$ \mathbb{E}_{p_\infty}[A(X_t)] = A \prec 0$$.

**proof**: 
$$
\begin{align*}
&V^T D M V \\
=& V^T D^{1/2} D^{1/2} MV \\
\le& {\Arrowvert D^{1/2} V \Arrowvert}_2 \cdot {\Arrowvert D^{1/2} M V \Arrowvert}_2\\
=& \Arrowvert V \Arrowvert_D \Arrowvert MV \Arrowvert_D \\
\le& \frac{\gamma(1 - \lambda)}{1 - \gamma\lambda} \Arrowvert V \Arrowvert_D \cdot \Arrowvert V \Arrowvert_D \\
=& \frac{\gamma(1 - \lambda)}{1 - \gamma\lambda} V^T D V \\
\le& \gamma V^T D V.
\end{align*}
$$
So
$$
\forall \theta, \mathbb{E}_{p_\infty}[\theta^T A(X_t)\theta]\le (\gamma-1) V^T D V < 0.
$$

> **Lemma 4**: $$\exists \rho < 1$$ and $$ C > 0 $$ such that for all $$ t \ge 0 $$ and every X, we have
> $$
> \Arrowvert \mathbb{E} \left[ A(X_t) | X_0 = X \right] - A \Arrowvert \le C \rho^t,\\
> \Arrowvert \mathbb{E}\left[ b(X_t) | X_0 = X \right] - b \Arrowvert \le C \rho^t
> $$

From Martingales theorems we have

> **Theorem 1**: $$TD(\lambda)$$'s convergence $$\theta^*$$ satisfies that
> $$
> A \theta^* + b = 0
> $$

$$
\Phi^T D \left( (1 - \lambda) \sum^{\infty}_{m = 0} \lambda^m{(\gamma P)}^{m+1} - I \right) \Phi \cdot \theta^*
+ \Phi^{T} D \sum^{\infty}_{m = 0} {(\gamma \lambda P)}^{m} \bar r = 0\\
\Phi^T D \left( (1 - \lambda) \sum^{\infty}_{m = 0} \lambda^m{(\gamma P)}^{m+1}  \Phi \cdot \theta^*
+ \sum^{\infty}_{m = 0} {(\gamma \lambda P)}^{m} \bar r\right) = \Phi^T D \Phi \theta^*
$$

> **Definition 1**:
> $$
> \begin{align*}
> T^{(\lambda)}_\pi V =& (1-\lambda) \sum^\infty_{m=0}\lambda^m(\gamma P)^{m+1}V +  \sum^{\infty}_{m = 0} {(\gamma \lambda P)}^{m} \bar r = MV + q\\
> =& (1-\lambda) \sum^\infty_{m=0}\lambda^m\mathbb{E}_{\pi}
> \left[\sum^m_{t=0} \gamma^t r(s_t, s_{t+1}) + \gamma^{m+1} V(s_{m+1}) \Big\vert s_0 = s\right]
> \end{align*}
> $$

Then
$$
\begin{align*}
&\Phi^T D T^{(\lambda)}_\pi(\Phi \theta^*) = \Phi^T D \Phi \theta^*\\
\Rightarrow& \Phi \theta^* = \Phi(\Phi^T D \Phi)^{-1}\Phi^T D T^{(\lambda)}_\pi(\Phi \theta^*).
\end{align*}
$$

> **Lemma 5**: If $$ \Pi = \Phi(\Phi^T D \Phi)^{-1} \Phi^T D$$ satisfies
> $$
> \Arrowvert \Phi V - V \Arrowvert_{D} = \min_{\theta}\Arrowvert \Phi \theta - V \Arrowvert_{D}.
> $$

Then
$$
\Phi\theta^* = \Pi T^{(\lambda)}_\pi(\Phi \theta^*).
$$
We now get that $$\Phi \theta^*$$ is the fixed point of $$\Pi T^{(\lambda)}_{\pi}$$, and it's easy to verify that $$V^{\pi}$$ is the fixed point of $$T^{(\lambda)}_{\pi}$$.

> **Lemma 6**:
> $$
> \Arrowvert \Phi \theta^{*} - V^\pi\Arrowvert_D \le \frac{1-\gamma\lambda}{1-\gamma}\Arrowvert \Pi V^\pi - V^\pi \Arrowvert_D
> $$

**proof**:
$$
\begin{align*}
&\Arrowvert \Phi\theta^* - V^{\pi} \Arrowvert_D \\
\le& \Arrowvert \Phi\theta^* - \Pi J^{\pi} \Arrowvert_D + \Arrowvert \Pi J^{\pi} - J^{\pi} \Arrowvert_D\\
\le& \Arrowvert \Pi T^{(\lambda)}_{\pi} \Phi\theta^* - \Pi T^{(\lambda)}_{\pi} J^{\pi} \Arrowvert_{D} +\Arrowvert \Pi J^{\pi} - J^{\pi} \Arrowvert_D\\
\le& \frac{\gamma(1 - \lambda)}{1 - \gamma\lambda} \Arrowvert \Phi\theta^* - J^{\pi} \Arrowvert_D + \Arrowvert \Pi J^{\pi} - J^{\pi} \Arrowvert_D\\
\end{align*}
$$

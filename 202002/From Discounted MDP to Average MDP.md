# From Discounted MDP to Average MDP

## 1. Discounted MDP

$$
\begin{aligned}
&\pi 
\xrightarrow{P(s' \vert s, a)}
P_\pi(s' \vert s) = \sum_a \pi(a \vert s) P(s' \vert s, a) \\
&\xrightarrow{p_0}
MC_\pi = \left\{ \tau=(s_0, a_0, s_1, a_1, s_2, a_2, \ldots): s_0 \sim p_0, a_t = \pi(\cdot \vert s_t), s_{t+1} \sim P(\cdot \vert s_t, a_t) \right\} \\
&\xrightarrow{\gamma, r}
\rho_\gamma(\pi) = \mathbb{E}_{\tau\sim MC_\pi} \left[ (1 - \gamma) \sum^{\infty}_{t=0} \gamma^t r(s_t)\right] = \sum_{s} p_\gamma(s;\pi) r(s), \\
&where\ p_\gamma(s;\pi) = (1 - \gamma) \sum^{\infty}_{t=0} \gamma^t p_t(s; \pi),
\ p_t(s; \pi) = Pr(s_t = s; \pi).
\end{aligned}
$$

### Matrix Form

We denote that $$\mathcal{S} = \{ s1, s2, \ldots, sN \}$$, then
$$
P_\pi = 
\begin{bmatrix}
P_\pi(s1 \vert s1) & P_\pi(s2 \vert s1) & \cdots & P_\pi(sN \vert s1)\\
P_\pi(s1 \vert s2) & P_\pi(s2 \vert s2) & \cdots & P_\pi(sN \vert s2) \\
\vdots & \vdots & \ddots & \vdots \\
P_\pi(s1 \vert sN) & P_\pi(s2 \vert sN) & \cdots & P_\pi(sN \vert sN)
\end{bmatrix}
$$
and
$$
p^\pi_1 = P^T_\pi p_0, \quad p^\pi_2 = (P^T_\pi)^2p_0, \quad \ldots,\quad p^\pi_t = (P^T_\pi)^t p_0.
$$
Now
$$
p^\pi_\gamma = (1 - \gamma) \sum^\infty_{t=0} \gamma^t (P^T_\pi)^t p_0.
$$

### Policy gradient theorem

$$
\theta_\pi \rightarrow \pi \rightarrow \cdots \rightarrow \rho_{\gamma}(\pi) \rightarrow \rho_{\gamma}(\theta_\pi).
$$

$$
\frac{\operatorname{d} \rho_{\gamma} (\theta_\pi)}{\operatorname{d} \theta_\pi} = \sum_{s} p^\pi_\gamma(s) \sum_a \pi(a \vert s)\nabla_\theta\log\pi(a \vert s) Q^\pi_\gamma(s, a)
$$

$$
Q^\pi_\gamma(s, a) = \mathbb{E}_{\tau \sim MC_{\pi}}\left[ \sum^{\infty}_{t=0} \gamma^t r_t \big\vert s_0 = s, a_0 = a \right].
$$

### Off-policy Settings

We only follows behavior policy to sample from environment
$$
MC_\mu = \left\{ \tau=(s_0, a_0, r_0,  s_1, a_1, r_1, s_2, a_2, r_2, \ldots): s_0 \sim p_0, a_t = \mu(\cdot \vert s_t), s_{t+1} \sim P(\cdot \vert s_t, a_t) \right\}.
$$
Here  are three problems:
$$
\frac{\operatorname{d} \rho(\theta_\pi)}{\operatorname{d} \theta_\pi} 
=

\mathbb{E}_{s \sim p^\pi_\gamma, a \sim \pi(s)}\{
\nabla_\theta\log\pi(a \vert s) 
Q_\gamma^\pi(s, a)\}
$$


- From $$p^\mu_\gamma = (1 - \gamma) \sum^\infty_{t=0} \gamma^t (P^T_\mu)^t p_0$$ to $$p^\pi_\gamma = (1 - \gamma) \sum^\infty_{t=0} \gamma^t (P^T_\pi)^t p_0$$;
- From $$\mu(a \vert s)$$ to $$\pi(a \vert s)$$;
- From $$Q_\gamma^\pi(s, a)$$ to $$Q_\gamma^\mu(s, a)$$.

$$
\frac{\operatorname{d} \rho(\theta_\pi)}{\operatorname{d} \theta_\pi} 
=

\mathbb{E}_{s \sim p^\mu_\gamma, a \sim \mu(s)}\left\{
\frac{p^\pi_\gamma(s)}{p^\mu_\gamma(s)} \frac{\pi(a \vert s)}{\mu(a \vert s)}
\nabla_\theta\log\pi(a \vert s) 
Q_\gamma^\pi(s, a)\right\}.
$$

## 2. From Discounted MDP to Average MDP

$$
\begin{aligned}
&\pi 
\xrightarrow{P(s' \vert s, a)}
P_\pi(s' \vert s) = \sum_a \pi(a \vert s) P(s' \vert s, a) \\
&\xrightarrow{p_0}
MC_\pi = \left\{ \tau=(s_0, a_0, s_1, a_1, s_2, a_2, \ldots): s_0 \sim p_0, a_t = \pi(\cdot \vert s_t), s_{t+1} \sim P(\cdot \vert s_t, a_t) \right\} \\
&\xrightarrow{\gamma, r}
\rho_\gamma(\pi) = \mathbb{E}_{\tau\sim MC_\pi} \left[ (1 - \gamma) \sum^{\infty}_{t=0} \gamma^t r(s_t)\right] = \sum_{s} p_\gamma(s;\pi) r(s), \\
&where\ p_\gamma(s;\pi) = (1 - \gamma) \sum^{\infty}_{t=0} \gamma^t p_t(s; \pi),
\ p_t(s; \pi) = Pr(s_t = s; \pi).
\end{aligned}
$$

$$
\begin{aligned}
&\pi 
\xrightarrow{P(s' \vert s, a), p_0(s), \gamma}
P_{\pi, \gamma}(s' \vert s) 
= \gamma \sum_a \pi(a \vert s) P(s' \vert s, a) + (1 - \gamma) p_0(s') \\
&\xrightarrow{p_0}
MC_{\pi,\gamma} = \left\{ \tau=(s_0, a_0, s_1, a_1, s_2, a_2, \ldots): s_0 \sim p_0, a_t = \pi(\cdot \vert s_t), s_{t+1} \sim \gamma P(\cdot \vert s_t, a_t) + (1- \gamma) p_0 \right\} \\
&\xrightarrow{r}
\rho_{stationary} (\pi) = \mathbb{E}_{s \sim d_{\pi,\gamma}} \left[ r(s)\right] = \sum_{s} d_{\pi,\gamma}(s) r(s), \\

&\text{where } d_{\pi,\gamma} \text{is stationary distribution that satisfies } d_{\pi,\gamma} = P^T_{\pi,\gamma} d_{\pi,\gamma}.
\end{aligned}
$$

### Matrix Form

The state transition matrix is
$$
P_{\pi, \gamma} = \gamma P_\pi + (1 - \gamma)e p_0^T.
$$

> **Lemma 1**.
> $$
> d_{\pi,\gamma} = p^\pi_\gamma = (1 - \gamma) \sum^\infty_{t=0} \gamma^t (P^T_\pi)^t p_0.
> $$
> This theorem also means $$\rho_{stationary}(\pi) = \rho_\gamma(\pi)$$.
>
> **proof**:
> $$
> \begin{aligned}
> d_{\pi, \gamma} 
> =& P^T_{\pi, \gamma} d_{\pi,\gamma} \\
> =& [\gamma P^T_\pi + (1 - \gamma)p_0 e^T] d_{\pi,\gamma}\\
> 
> (I - \gamma P^T_\pi) d_{\pi,\gamma}
> =& (1 - \gamma) p_0 \\
> 
> d_{\pi, \gamma} 
> =& (1 - \gamma) (I - \gamma P^T_\pi) ^{-1} p_0 \\
> =& (1 - \gamma) \sum^{\infty} \gamma^t (P^T_\pi)^t p_0
> \end{aligned}
> $$

### Average MDP

$$
\rho_{avg} (\pi) = \lim_{T \rightarrow \infty} \frac{1}{T} \sum^{T-1}_{t=0} \mathbb{E}_{\tau \sim MC_{\pi,\gamma}}[r_t].
$$

$$
\rho_{stationary}(\pi) = \mathbb{E}_{s \sim d_{\pi,\gamma}} \left[ r(s)\right] = \sum_{s} d_{\pi,\gamma}(s) r(s)
$$

> **Lemma 2**.
> $$
> \rho_{avg}(\pi) = \rho_{stationary}(\pi).
> $$

### Policy Gradient Theorem

$$
\frac{\operatorname{d} \rho_{avg} (\theta_\pi)}{\operatorname{d} \theta_\pi} = \sum_{s} d_{\pi,\gamma} (s) \sum_a \pi(a \vert s)\nabla_\theta\log\pi(a \vert s) Q^\pi_{avg}(s, a)
$$

$$
Q^\pi_{avg}(s, a) = \mathbb{E}_{\tau \sim MC_{\pi, \gamma}}\left[ \sum^{\infty}_{t=0} (r_t - \rho(\pi)) \big\vert s_0 = s, a_0 = a \right]
$$

###  Off-policy Settings

We only follows behavior policy to sample from environment

- $$MC_{\mu,\gamma} = \left\{ \tau=(s_0, a_0, r_0, s_1, a_1, r_1,  s_2, a_2, r_2, \ldots): \\ 
  s_0 \sim p_0, a_t = \mu(\cdot \vert s_t), s_{t+1} \sim \gamma P(\cdot \vert s_t, a_t) + (1- \gamma) p_0 \right\};$$
- $$MC2_{\mu,\gamma} = \left\{ m = (s, a, r, s'):
  s \sim d_{\mu,\gamma}, a = \mu(\cdot \vert s), s' \sim \gamma P(\cdot \vert s, a) + (1- \gamma) p_0 \right\}.$$

## 3. Off-policy Algorithms

### 3.1 COP-TD

The algorithm's key target is to get $$c(s) = \frac{d_{\pi, \gamma} (s)}{d_{\mu}(s)}$$ that satisfies
$$
\begin{cases}
d_{\pi, \gamma} = P^T_{\pi,\gamma} d_{\pi,\gamma} \\
d_{\pi, \gamma} = D_{\mu} c\\
D_{\mu} = diag(d_{\mu})
\end{cases}
\Rightarrow 
D_{\mu} c = P^T_{\pi,\gamma} D_{\mu} c.
$$
The loss of COP-TD algorithm is
$$
\begin{cases}
L(c) = \frac{1}{2} \Vert  c - D^{-1}_{\mu} P^T_{\pi,\gamma} D_{\mu} c_{target} \Vert^2, \\
L(c_{target}) = \frac{1}{2} \Vert c_{target} - c \Vert^2.
\end{cases}
$$

$$
\begin{aligned}
d_{\pi, \gamma} =& P^T_{\pi,\gamma} d_{\pi,\gamma} \\ 
d_{\pi, \gamma}(s') 
=& \int \int [\gamma P(s' \vert s, a) \pi(a \vert s) + (1 - \gamma) p_0(s')] d_{\pi,\gamma}(s) ds da \\
=& \gamma \int \int P(s' \vert s, a) \pi(a \vert s) d_{\pi,\gamma}(s) ds da  + (1 - \gamma) p_0(s')

\end{aligned}
$$

> **Algorithm 1**. (Discounted COP-TD algorithm)
> $$
> c(s') = c(s') + \alpha \left[ \gamma \frac{\pi(a \vert s)}{ \mu(a \vert s)} c(s) + (1 - \gamma) - c(s') \right].
> $$
>
> - Target space：$$MC_{\pi,\gamma} = \left\{ \tau=(s_0, a_0, s_1, a_1, s_2, a_2, \ldots): s_0 \sim d_\mu, a_t = \pi(\cdot \vert s_t), s_{t+1} \sim \gamma P(\cdot \vert s_t, a_t) + (1- \gamma) d_\mu \right\}$$;
>
> - Sample space: $$ MC2_{\mu} = \{ m = (s, a, r, s'): s \sim d_\mu, a \sim \mu(s), s' \sim P(s' \vert s, a)\}$$.

## 3.2 GenDICE

$$
\frac{\operatorname{d} \rho(\theta_\pi)}{\operatorname{d} \theta_\pi} 
=

\mathbb{E}_{s \sim p^\mu_\gamma, a \sim \mu(s)}\left\{
\frac{p^\pi_\gamma(s)}{p^\mu_\gamma(s)} \frac{\pi(a \vert s)}{\mu(a \vert s)}
\nabla_\theta\log\pi(a \vert s) 
Q_\gamma^\pi(s, a)\right\}.
$$



 A new target is finding the ratio function
$$
r(s, a) = \frac{p^{\pi}_{\gamma}(s, a)}{p^{\pi}_{\gamma}(s,a)} 
= \frac{p^\pi_\gamma(s) \pi(a \vert s)}{p^\mu_\gamma(s) \mu(a \vert s)}
= \frac{d_{\pi,\gamma}(s) \pi(a \vert s)}{d_{\mu,\gamma}(s) \mu(a \vert s)}
$$
We need to find a new target equation: 
$$
\begin{aligned}
 d_{\pi, \gamma} =& P^T_{\pi,\gamma} d_{\pi,\gamma} \\
d_{\pi, \gamma}(s') =& \int \int [\gamma P(s' \vert s, a) \pi(a \vert s) + (1 - \gamma) p_0(s')] d_{\pi,\gamma}(s) ds da \\
=& \gamma \int \int P(s' \vert s, a) \pi(a \vert s) d_{\pi,\gamma}(s) ds da  + (1 - \gamma) p_0(s')\\
\pi(a' \vert s') d_{\pi,\gamma}(s')
=& \gamma \int \int \pi(a' \vert s') P(s' \vert s, a) \pi(a \vert s) d_{\pi,\gamma}(s) ds da  + (1 - \gamma) \pi(a' \vert s') p_0(s')\\
d_{\pi,\gamma}(s', a')
=& \gamma \int \int \pi(a' \vert s') P(s' \vert s, a) d_{\pi, \gamma}(s' ,a') ds da  + (1 - \gamma) \pi(a' \vert s') p_0(s') \\
=& \int \int \pi(a' \vert s') [\gamma P(s' \vert s, a) + (1 - \gamma) p_0(s')] d_{\pi, \gamma}(s' ,a') ds da \\
=& \int \int P_{\pi,\gamma}(s', a' \vert s, a) d_{\pi,\gamma}(s', a') ds da \\
d_{\mu, \gamma}(s', a') r(s', a') 
=& \int \int P_{\pi,\gamma}(s', a' \vert s, a) d_{\mu,\gamma}(s, a) r(s, a) ds da\\
D_{\mu, \gamma} r =& P_{\pi, \gamma} D_{\mu, \gamma} r.
\end{aligned}
$$

The loss of GenDICE is
$$
\min_{r \succeq 0}  D_{\phi} (P_{\pi, \gamma}D_{\mu, \gamma} r \Vert D_{\mu, \gamma} r), \quad s.t. \mathbb{E}_{d_{\mu, \gamma}}[r] = 1.
$$

> Definition (f-divergence) For $\phi: \mathbb{R}_{+} \rightarrow \mathbb{R}$ is convex function, lower-semicontinuous function with $\phi(1)=0$
> $$
> D_{\phi}(p \| q)=\int q(x) \phi\left(\frac{p(x)}{q(x)}\right) d x
> $$

$$
\begin{aligned}
& \min_r D_{\phi} (P_{\pi, \gamma}D_{\mu, \gamma} r \Vert D_{\mu, \gamma} r) \\
=&  \min_r \int \int D_{\mu, \gamma}r(s, a) \phi \left( \frac{P_{\pi,\gamma} D_{\mu, \gamma} r(s, a) }{D_{\mu, \gamma} r(s, a) } \right) ds da \\
=& \min_r \int \int D_{\mu, \gamma}r(s, a) \max_{f(s, a)} \left( \frac{P_{\pi,\gamma} D_{\mu, \gamma} r(s, a) }{D_{\mu, \gamma} r(s, a) } f(s, a) - \phi^*(f(s, a)) \right) ds da \\
=& \min_r \max_f \int \int P_{\pi,\gamma} D_{\mu,\gamma} r(s, a) f(s, a) - D_{\mu, \gamma} r(s, a) \phi^*(f(s, a)) ds da
\end{aligned}
$$

## 3.3 GradientDICE

$$
\min_{r \succeq 0}  \frac{1}{2} \Vert P_{\pi, \gamma}D_{\mu, \gamma} r - D_{\mu, \gamma} r\Vert^2_{D^{-1}_{\mu,\gamma}}, \quad s.t. \mathbb{E}_{d_{\mu, \gamma}}[r] = 1.
$$

$$
\min_{r \succeq 0}  \frac{1}{2} \Vert D^{-1}_{\mu, \gamma} P_{\pi, \gamma}D_{\mu, \gamma} r - r\Vert^2_{D_{\mu,\gamma}}, \quad s.t. \mathbb{E}_{d_{\mu, \gamma}}[r] = 1.
$$


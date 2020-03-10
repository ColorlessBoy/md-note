$$
\begin{cases}
\rho_{\gamma}(\pi) = \mathbb{E}_{\tau\sim MC_\pi} \left[ (1 - \gamma) \sum^{\infty}_{t=0} \gamma^t r(s_t)\right]\\

\frac{\operatorname{d} \rho(\theta_\pi)}{\operatorname{d} \theta_\pi} 
=

\mathbb{E}_{s \sim p^\pi_\gamma, a \sim \pi(s)}\{
\nabla_\theta\log\pi(a \vert s) 
Q_\gamma^\pi(s, a)\}\\

MC_\pi = \left\{ \tau=(s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, \ldots): s_0 \sim p_0, a_t \sim \pi(\cdot \vert s_t), s_{t+1} \sim P(\cdot \vert s_t, a_t) \right\} \\

MC2_\pi = \left\{ m=(s, a, r, s'): s \sim d_\pi, a\sim \pi(\cdot \vert s), s' \sim P(\cdot, \vert s, a) \right\}
\end{cases}
$$

$$
\begin{cases}
\rho_{stationary} (\pi) = \mathbb{E}_{s \sim d_{\pi,\gamma}} \left[ r(s)\right]\\
\frac{\operatorname{d} \rho(\theta_\pi)}{\operatorname{d} \theta_\pi} 
=

\mathbb{E}_{s \sim p^\pi_{stationary}, a \sim \pi(s)}\{
\nabla_\theta\log\pi(a \vert s) 
Q_{avg}^\pi(s, a)\}\\

MC_{\pi,\gamma} = \left\{ \tau=(s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, \ldots): \\ \quad \quad s_0 \sim p_0, a_t = \pi(\cdot \vert s_t), s_{t+1} \sim \gamma P(\cdot \vert s_t, a_t) + (1- \gamma) p_0 \right\} \\

MC2_{\pi,\gamma} = \left\{ m=(s, a, r, s'): s \sim d_{\pi,\gamma}, a\sim \pi(\cdot \vert s), s' \sim \gamma P(\cdot, \vert s, a) + (1 - \gamma) p_0 \right\}
\end{cases}
$$

# 1. COP-TD

- Samples: $$MC2_{\mu} = \left\{ m=(s, a, r, s'): s \sim d_\mu, a\sim \mu(\cdot \vert s), s' \sim P(\cdot, \vert s, a) \right\} $$;

- Target: $$c(s) = \frac{p^{\pi}_{\gamma} (s)}{ d_\mu(s) } = \frac{d_{\pi, \gamma} (s)}{ d_\mu(s) }$$;

- Using the equation: $$d_{\pi, \gamma} = P^T_{\pi,\gamma} d_{\pi,\gamma}$$, where $$P_{\pi, \gamma} = \gamma P_\pi + (1 - \gamma)e p_0^T$$.

  Because $$ D_\mu c = d_{\pi,\gamma} $$, therefore
  $$
  D_\mu c = P^T_{\pi, \gamma} D_\mu c = \gamma P^T_\pi D_\mu c + (1 - \gamma) p_0.
  $$

The loss of COP-TD algorithm is
$$
\begin{cases}
L(c) = \frac{1}{2} \Vert  c - D^{-1}_{\mu} P^T_{\pi,\gamma} D_{\mu} c_{target} \Vert^2, \\
L(c_{target}) = \frac{1}{2} \Vert c_{target} - c \Vert^2.
\end{cases}
$$
**Algorithm 1**. (Discounted COP-TD algorithm)
$$
c(s') = c(s') + \alpha \left[ \gamma \frac{\pi(a \vert s)}{ \mu(a \vert s)} c(s) + (1 - \gamma) - c(s') \right].
$$

- Sample space: $$ MC2_{\mu} = \{ m = (s, a, r, s'): s \sim d_\mu, a \sim \mu(s), s' \sim P(s' \vert s, a)\}$$；
- Target space: $$ MC2_{\pi} = \{ m = (s, a, r, s'): s \sim p^\pi_{\gamma}, a \sim \pi(s), s' \sim P(s' \vert s, a)\}$$;
- $$p_0$$  替换成 $$d_{\mu}$$. 

# 2. GenDICE

$$
\frac{\operatorname{d} \rho(\theta_\pi)}{\operatorname{d} \theta_\pi} 
=

\mathbb{E}_{(s, a) \sim \mathcal{D}}\left\{
\frac{p^\pi_\gamma(s) \pi(a \vert s)}{Pr_{\mathcal{D}}(s, a)} 
\nabla_\theta\log\pi(a \vert s) 
Q_\gamma^\pi(s, a)\right\}.
$$

The new target is to get
$$
r(s, a) = \frac{p^\pi_\gamma(s) \pi(a \vert s)}{Pr_{\mathcal{D}}(s, a)} 
$$
We need to find a new equation
$$
\begin{aligned}
 d_{\pi, \gamma} =& P^T_{\pi,\gamma} d_{\pi,\gamma} \\
d_{\pi, \gamma}(s') =& \int \int [\gamma P(s' \vert s, a) \pi(a \vert s) + (1 - \gamma) p_0(s')] d_{\pi,\gamma}(s) ds da \\
=& \gamma \int \int P(s' \vert s, a) \pi(a \vert s) d_{\pi,\gamma}(s) ds da  + (1 - \gamma) p_0(s')\\
\pi(a' \vert s') d_{\pi,\gamma}(s')
=& \gamma \int \int \pi(a' \vert s') P(s' \vert s, a) \pi(a \vert s) d_{\pi,\gamma}(s) ds da  + (1 - \gamma) \pi(a' \vert s') p_0(s')\\
d_{\pi,\gamma}(s', a')
=& \gamma \int \int \pi(a' \vert s') P(s' \vert s, a) d_{\pi, \gamma}(s ,a) ds da  + (1 - \gamma) \pi(a' \vert s') p_0(s') \\
=& \int \int \pi(a' \vert s') [\gamma P(s' \vert s, a) + (1 - \gamma) p_0(s')] d_{\pi, \gamma}(s' ,a') ds da \\
=& \int \int P_{\pi,\gamma}(s', a' \vert s, a) d_{\pi,\gamma}(s', a') ds da \\
Pr_{\mathcal{D}}(s', a') r(s', a') 
=& \int \int P_{\pi,\gamma}(s', a' \vert s, a) Pr_{\mathcal{D}}(s, a) r(s, a) ds da\\
D_{\mathcal{D}} r =& P^T_{\pi, \gamma} D_{\mathcal{D}} r.
\end{aligned}
$$
The loss of GenDICE is  
$$
\min_{r \succeq 0}  D_{\phi} (P_{\pi, \gamma}D_{\mathcal{D}} r \Vert D_{\mathcal{D}} r), \quad s.t. \mathbb{E}_{d_{\mu, \gamma}}[r] = 1.
$$

> Definition (f-divergence) For $\phi: \mathbb{R}_{+} \rightarrow \mathbb{R}$ is convex function, lower-semicontinuous function with $\phi(1)=0$
> $$
> D_{\phi}(p \| q)=\int q(x) \phi\left(\frac{p(x)}{q(x)}\right) d x
> $$

$$
\begin{aligned}
& \min_r D_{\phi} (P_{\pi, \gamma} D_{\mathcal{D}} r \Vert D_{\mathcal{D}} r) \\
=&  \min_r \int \int D_{\mathcal{D}}r(s, a) \phi \left( \frac{P_{\pi,\gamma} D_{\mathcal{D}} r(s, a) }{D_{\mathcal{D}} r(s, a) } \right) ds da \\
=& \min_r \int \int D_{\mathcal{D}}r(s, a) \max_{f(s, a)} \left( \frac{P_{\pi,\gamma} D_{\mathcal{D}} r(s, a) }{D_{\mathcal{D}} r(s, a) } f(s, a) - \phi^*(f(s, a)) \right) ds da \\
=& \min_r \max_f \int \int P_{\pi,\gamma} D_{\mathcal{D}} r(s, a) f(s, a) - D_{\mathcal{D}} r(s, a) \phi^*(f(s, a)) ds da
\end{aligned}
$$

- Sample space: $$ MC2_{\mu} = \{ m = (s, a, r, s'): s \sim d_\mu, a \sim \mu(s), s' \sim P(s' \vert s, a)\}$$；

## 3.3 GradientDICE

$$
\min_{r \succeq 0}  \frac{1}{2} \Vert P_{\pi, \gamma}D_{\mu, \gamma} r - D_{\mu, \gamma} r\Vert^2_{D^{-1}_{\mu,\gamma}}, \quad s.t. \mathbb{E}_{d_{\mu, \gamma}}[r] = 1.
$$

$$
\min_{r \succeq 0}  \frac{1}{2} \Vert D^{-1}_{\mu, \gamma} P_{\pi, \gamma}D_{\mu, \gamma} r - r\Vert^2_{D_{\mu,\gamma}}, \quad s.t. \mathbb{E}_{d_{\mu, \gamma}}[r] = 1.
$$


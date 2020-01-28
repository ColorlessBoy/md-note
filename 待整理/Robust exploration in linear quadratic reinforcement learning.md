## Robust exploration in linear quadratic reinforcement learning

- Related work
  - Safe RL: it seeks to respect certain safety constraints during exploration and/or policy optimization;
  - Risk-sensitive RL: it search for a policy also considers the variance of the reward.

## 2 Problem statement

- $\otimes$ denotes the Kronecker product;

- **Dynamics and cost function**: $x_{t+1} = A x_t + B u_t + w_t, w_t \sim \mathcal{N}(0, \sigma^2_w I_{n_x}), x_0 = 0$;
  - State $x_t \in \mathbb{R}^{n_x}$, action $u_t \in \mathbb{R}^{n_u}$ and noise$w_t \in \mathbb{R}^n$;
  - $u_t = \phi(\{x_{1:t}, u_{1:t-1}\})$;
  - cost function $\sum^T_{t=i} c(x_t, u_t)$, where $c(x_t, u_t) = x^T_tQx_t + u^T_t R u_t$;
  - The parameters of the true system $A_{tr}, B_{tr}$;

- **Modeling and data**: 

  - Data: $\mathcal{D}_n = \{x_t, u_t\}^n_{t=1}$:

  - $\sigma^2_w$ is known, or has been estimated;

  - The parameter uncertainty can be quantified as:

    ---

    **Proposition 2.1.** 
    $$
    \Theta_e(\mathcal{D}_n) := \{\theta: (\theta - \mu_\theta)^T \Sigma^{-1}_{\theta} (\theta - \mu_\theta) \le c_\delta\},
    $$

    where $c_\delta = \chi^2_{n^2_x + n_x n_u}(\delta)$ for $ 0 < \delta < 1$. Then, $\theta_{tr} = vec([A_{tr}, B_{tr}]) \in \Theta_e(\mathcal{D}_n)$ w.p. $1-\delta$.
    
    **proof**:
    $$
    \begin{align*}x_{t+1} = A x_t + B u_t + w_t =& x^T_t \otimes I_{n_x} vec(A) + u^T_t \otimes I_{n_x} vec(B) + w_t\\=& [x^T_t,u^T_t] \otimes I_{n_x} vec([A, B]) + w_t\\=& \phi_t \theta + w_t\end{align*}
    $$
    $$
    \begin{align*}
    &p(\mathcal{D}_n | \theta) = \prod^{n-1}_{t=1} p(x_{t+1} | x_t, u_t)
    \propto \exp \left( -\frac{1}{2\sigma^2_w} \sum^{n-1}_{t=1} \Arrowvert x_{t+1} - \phi_t \theta\Arrowvert^2_2 \right)\\
    =& \exp\left(-\frac{1}{2\sigma^2_w} \sum^{n-1}_{t=1} (x^T_{t+1}x_{t+1} - 2x^T_{t+1} \phi_{t}\theta + \theta^T \phi^T_t \phi_t \theta)\right)
    \propto \exp(-\frac{1}{2} \Arrowvert \theta - \mu_\theta \Arrowvert^2_{\Sigma^{-1}_{\theta}})
    \end{align*}
    $$
    
    where $\mu_\theta = \left(\sum^{n-1}_{t=1}\phi^T_t\phi_t\right)^{-1}\sum^{n-1}_{t=1} (\phi^T_t x_{t+1}) = \Sigma \cdot \sum^{n-1}_{t=1} (\phi^T_t x_{t+1})$ and $\Sigma^{-1}_{\theta} = \frac{1}{\sigma^2_w}\sum^{n-1}_{t=1} \phi^T_t\phi_t = \frac{1}{\sigma^2_w}\sum^{n-1}_{t=1}\begin{bmatrix}x_t\\u_t\end{bmatrix}[x^T_t, u^T_t] \otimes I_{n_x}$. 
    
    $\mu_\theta = vec([\hat A, \hat B]) = \arg\min_{\theta\in\mathbb{R}^{n^2_x + n_x n_u}} \sum^{n-1}_{t=1}\Arrowvert x_{t+1} - \phi_t \theta \Arrowvert^2_2$.
    
    ---

- **Policies**:

  - $u_t = K x_t + \Sigma^{1/2} e_t$, where $e_t \sim \mathcal{N}(0, I)$, $K \in \mathbb{R}^{p \times n}$ and $\Sigma \in \mathbb{S}^{n_u}_{+}$;
  - let $\{t_i\}^N_{i=0}$ with $0 = t_0 \le t_1 \le \ldots \le t_N = T$, and $T_i := t_i - t_{i-1}$.
  - $\mathcal{K} = \{K, \Sigma\}$; denote N policies, $\{\mathcal{K}_i\}^N_{i=1}$, such that $\mathcal{K}_i$ is deployed during the ith interval, $t \in [t_{i-1}, t_i )$;
  - denote $\mathcal{I}(t) := \arg\min_{i\in\mathcal{N}}\{i : t_i \ge t\}$;
  - denote $u_t = \mathcal{K}(x_t) = K x_t + \Sigma^{1/2} e_t$.

- **Worst-case dynamics**:

  - $\min_{\{\mathcal{K}_i\}^N_{i=1}} \mathbb{E} \left[ \sum^T_{t=0} \sup_{\{A_t, B_t\} \in \Theta_e(\mathcal{D}_t)} c(x_t, u_t)\right], s.t. x_{t+1} = A_t x_t + B_t u_t + w_t, u_t = \mathcal{K}_{\mathcal{I}(t)}(x_t)$

  - $w_t \sim \mathcal{N}(0, \sigma^2_w I_{n_x})$ and $e_t \sim \mathcal{N}(0, I_{n_u})$;

## 3 Modeling uncertainty for robust control

- $\mathcal{M}(\mathcal{D}) := \{\hat A, \hat B, D\}$, where $vec([\hat A, \hat B]) = \mu_\theta$ and $D \in \mathbb{S}^{n_x + n_u}$:
  $$
  \Theta_m(\mathcal{M}) := \{A, B: X^T D X \preceq I, X = [\hat A - A, \hat B - B]^T\}.
  $$

- ---
	**Lemma 3.1.** Given data $\mathcal{D}_n$, and $0 < \delta < 1$, let $D = \frac{1}{\sigma^2_w c_\delta} \sum^{n-1}_{t=1} \begin{bmatrix}x_t \\ u_t\end{bmatrix} \begin{bmatrix} x^T_t & u^T_t \end{bmatrix}$, with $c_\delta = \chi^2_{n^2_x + n_x n_u}(\delta)$. Then $[A_{tr}, B_{tr}] \in \Theta_m(\mathcal{M})$ w.p. $1-\delta$.

	**proof**:  Because $vec(CEF) = (F\otimes C)vec(E)$ and $tr(A^TB)=vec(A)^T vec(B)$we have
  $$
  \begin{align*}
  1 \ge& vec(X^T)^T \left( D \otimes I_{n_x}\right) vec(X^T)\\
  =& vec(X^T)^T vec(I_{n_x} X^T D) = tr(X X^T D) = tr(X^T D X) \ge \lambda_{max}(X^T D X)
  \end{align*}
  $$
  
  ---
  

## 4 Convex approximation to robust reinforcement learning problem

- The main contribution of this paper: a convex approximation to the 'robust reinforcement learning' problem.

- Approximation function: $\sum^{N}_{i=1} \sup_{\{A, B\}\in \Theta(\mathcal{D}_{t_i}) } \mathbb{E} \left[ \sum^{t_i}_{t=t_{i-1}} c(x_t, u_t) \right], s.t. x_{t+1} = A x_t + B u_t + w_t, u_t = \mathcal{K}_{\mathcal{I}(t)}(x_t)$;

- $\Theta_e \subseteq \Theta_m$;

- $J_{T_i}(x_{t_i}, \mathcal{K}, \Theta_m(\mathcal{M})):= \sup_{\{A, B\} \in \Theta_m(\mathcal{M})} \sum^{t_i+T_s}_{t=t_i} c(x_t, u_t)$,

  s.t. $x_{t+1} = A x_t + B u_t + w_t, u_t = \mathcal{K}(x_t)$;

- $J_{T_i}(x_{t_i}, \mathcal{K}_i, \Theta_m(\mathcal{M}(\mathcal{D}_{t_{i-1}}))) \approx T_i \times \{J_{\infty} (\mathcal{K}_i, \Theta_m(\mathcal{M(\mathcal{D}_{t_{i-1}})})) := \lim_{\tau\rightarrow \infty} \frac{1}{\tau} J_\tau(0, \mathcal{K}_i, \Theta_m(\mathcal{M}(\mathcal{D}_{t_{i-1}})))\}$;

- We seek to minimize  $\mathbb{E} \left[\sum^N_{i=1} T_i \times J_\infty (\mathcal{K}_i, \Theta_m(\mathcal{M}(\mathcal{D}_{t_{i-1}})))\right]$;

- $\lim_{\tau\rightarrow \infty} \frac{1}{\tau} \mathbb{E}\left[\sum^{\tau}_{t=1} x^T_t Q x_t + u^T_t R u_t\right] = tr \left( \begin{bmatrix}Q & 0 \\ 0 & R\end{bmatrix} \lim_{\tau \rightarrow \infty} \frac{1}{\tau} \sum^\tau_{t=1} \mathbb{E}\left[\begin{pmatrix}x_t \\ u_t\end{pmatrix} \begin{pmatrix} x^T_t & u^T_t\end{pmatrix}\right]\right)$;

  $\mathbb{E}\left[\lim_{\tau \rightarrow \infty} \frac{1}{\tau} \sum^\tau_{t=1} \begin{pmatrix}x_t \\ Kx_t + \Sigma^{1/2}e_t\end{pmatrix} \begin{pmatrix} x^T_t & x^T_t K^T + e^T_t \Sigma^{1/2}\end{pmatrix}\right] = \begin{bmatrix} W & WK^T \\ KW & KWK^T + \Sigma\end{bmatrix}$;

  where $W = \mathbb{E}[x_t x^T_t]$.

- If A and B are known, according by Lyapunov inequality, we have

  $W \succeq (A+BK)W(A+BK)^T+B\Sigma BT + \sigma^2_w I_{n_x}$.

- 


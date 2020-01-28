# Gradient Temporal-Different Learning Algorithms

## Chapter1 Introduction

- TD learning is a bootstrapping method; that is the current prediction is updated based on the next prediction.
- The advantages of off-policy algorithms:
  - exploration;
  - training data can come from uncorrelated controllers, such as manual human control;
  - multiple target policies (parallel learning).
- The stability problem arises when we seek the following four algorithmic features:
  - TD learning;
  - function approximation;
  - off-policy learning;
  - linear complexity both in terms of memories and per-time-step computation.
- Related works:
  - Importance sampling idea: high variance;
  - Residual gradient method: double-sampling, not reliable when using function approximation;
  - LSTD, LSPI: $O(d^2)$ complexity.
- Two good pictures present in the paper that describe the contributions of this paper.
  - ![1_1](C:\Users\pengl\Documents\md-notes\TD\1_1.png)
  - ![1_2](C:\Users\pengl\Documents\md-notes\TD\1_2.png)

## Chapter2 Background

### 2.3 Temporal-difference Learning

- Sarsa: The theorems, which have been used for the convergence of TD(0), also can be used for the convergence of Sarsa. 
- Q-learning: tabular Q-learning is guaranteed to converge to optimal action-values if states are visited infinitely (Sutton & Barto, 1998).

### 2.4 TD learning with eligibility traces

Several important properties of eligibility traces are as follows:

- They bridge the temporal gaps in cause and effect when experience is processed at a temporally fine resolution;
- They make classical TD methods more like efficient incremental Monte-Carlo algorithms;
- They are particularly of interest when reward is delayed by many steps, thus, by adjusting λ function we may get faster and efficient learning.  

### 2.6 Derivation of TD(0) with function approximation

Many TD-learning methods based on gradient-descent, are not true gradient-descent methods;

- Target objection: $MSE(\theta) = \mathbb{E} [(V^\pi(S_t) - V_\theta(S_t))^2]$. Because we can't get $V^\pi(S_t)$, we use $R_t + \gamma V_{\theta}(S_t)$ instead;
- Residual gradient method:
  - $J(\theta) = \Arrowvert T^\pi V_\theta - V_\theta \Arrowvert^2_{\mu} = \mathbb{E}_{S_{t}\sim \mu}\{[\mathbb{E}_{S_{t+1}}[\delta_t | S_t]]^2 \}$;
  
    - This objective is not the exact objective that RG methods uses.
  
  - $-\frac{1}{2} \nabla_\theta J(\theta) = E_{S_{t}\sim \mu}\{\mathbb{E}_{S_{t+1}}[\delta_t \cdot (\nabla_\theta V_\theta(S_t) - \gamma\nabla_\theta V_\theta(S_{t+1}))|S_t]\}$; 
  
  - But RG uses single sample, $\Delta \theta = \alpha_t \delta_t(\nabla_\theta(S_t) - \nabla_\theta V_\theta(S_{t+1}))$, which is considered to inferior to TD-solution. Properly speaking, $J(\theta) = \mathbb{E}_{S_t} [\delta^2_t(\theta)]$.
  
  - Here is a clear prove:
    $$
    \begin{align*}
    & J(\theta) := \Arrowvert T^\pi V_\theta - V_\theta \Arrowvert^2_{\mu}\\
    =& \sum_{s}\mu(s)\left(\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[r(s,a) + \gamma V_\theta(s') - V_\theta(s)]\right)^2\\
    &\nabla_\theta J(\theta)= \sum_{s}\mu(s)\left(\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[r(s,a) + \gamma V_\theta(s') - V_\theta(s)]\right)\\
    &\cdot \left(\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[\gamma \nabla V_\theta(s') - \nabla V_\theta(s)]\right)\\
    =& -\sum_{s}\mu(s)\left(\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[r(s,a) + \gamma V_\theta(s') - V_\theta(s)]\right)\nabla V_{\theta}(s) +\\
    &\sum_{s}\mu(s)\left\{\left(\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[r(s,a) + \gamma V_\theta(s') - V_\theta(s)]\right)\\
    \cdot\left(\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[\gamma \nabla V_\theta(s')]\right) \right\} \\
    =& -\mathbb{E}\left\{[r_t + \gamma V_\theta(s') - V_\theta(s)]\cdot\nabla_\theta V_\theta(s) \Big\vert s\sim\mu, a\sim\pi(\cdot\vert s), s' \sim p(\cdot\vert s, a)\right\}\\
    &+ \gamma\mathbb{E}_{s\sim \mu}\left\{\mathbb{E}\left[r_t+\gamma V_\theta(s') - V_\theta(s) \Big\vert s, a\sim\pi(\cdot\vert s), s'\sim p(\cdot\vert s, a)\right] \\ \cdot \mathbb{E}\left[\nabla_\theta V_\theta(s) \Big\vert s, a\sim\pi(\cdot\vert s), s' \sim p(\cdot\vert s, a)\right]\right\}
    \end{align*}
    $$

## Chapter3 Objective Function for Temporal-Different Learning

- Possible objective functions:
  - Mean-square Error: $MSE(\theta) = \Arrowvert V_\theta(s) - V^\pi(s)\Arrowvert^2_\mu$;
  
  - Mean-square Bellman Error: $MSBE(\theta) = \Arrowvert V_\theta - T^\pi V_\theta \Arrowvert^2_\mu$;
    - $MSBE(\theta)$ result could be inferior to the TD-solution;
    - double sampling;
    - the space of states is too big;
    - Most of TD algorithm does not converge to the minimum of MSBE objective.
    
  - Mean-square Temporal-difference Error: $MSTDE(\theta) = \mathbb{E}_{S_t}[\delta^2_t(\theta)]$; such as RG;
    
    - The major problem with this objective function is its inferior results.   
    
    - I like a more clear expressions:
      $$
      MSTDE(\theta)= \sum_{s}\mu(s)\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)\left([r(s,a) + \gamma V_\theta(s') - V_\theta(s)]\right)^2
      $$
    
  - Mean-square Projected Temporal-difference Error: $MSPTDE(\theta) = \Arrowvert V_\theta - \Pi T^\pi V_\theta \Arrowvert_\mu$; such as LSTD and GTD;
  
    - In paper's example, it's the only objective to get the true result.
  
  - **The norm of the expected TD update **: $NEU(\theta) = \mathbb{E}_{S_t}[\delta(\theta)\phi]^T \mathbb{E}_{S_t}[\delta(\theta)\phi]$.
  
    - Result in TD(0) solution too.
- We provide  feature-based $MSBE$ objective function: $J(\theta) = E_{S_t}\{ [E_{S_{t+1}}( \delta_t | \phi(S_t) )]^2\}$.
  - $-\frac{1}{2} \nabla_\theta J(\theta) = \mathbb E_{S_{t}}\{\mathbb{E}_{S_{t+1}}[\delta_t \cdot (\nabla_\theta V_\theta(S_t) - \nabla_\theta V_\theta(S_{t+1}))|\phi(S_t)]\}$;
  - If $V_\theta(S) = \theta^T \phi(S)$, $-\frac{1}{2} \nabla_\theta J(\theta) = \mathbb E_{S_{t}}\{\mathbb{E}_{S_{t+1}}[\delta_t \cdot (\phi(S_t) - \gamma\phi(S_{t+1}))|\phi(S_t)]\}$;

- I have a lot of questions for the rest of this chapter.

## Chapter4 Off-policy Formulation of Temporal-difference Learning

- Target policy $\pi(s)$, behavior policy $\pi_b(s)$, and stationary state distribution $\mu P = \mu$;
- This section has serious problem, too;
- $\mathbb{E}^{\pi}[\delta\phi] = \sum_{s,a,s'} \mu_\pi \frac{\pi(a|s)}{\pi_{b}(a|s)} \pi_b(a|s) P(s' | s, a) \delta(s, a, s' | \theta) \phi(s)$;
- $\mathbb{E}^{\pi_b}[ \frac{\pi(a|s)}{\pi_b{(a|s)}} \delta\phi] = \sum_{s,a,s'} \mu_b \frac{\pi(a|s)}{\pi_b(a|s)} \pi_b(a|s) P(s'|s,a) \delta(s, a, s' | \theta) \phi(s)$;

## Chapter5 Gradient Temporal-difference Learning with Linear function approximation

- GTD algorithm:
  - $NEU(\theta) = \mathbb{E}_{S_t}[\delta \phi]^T \mathbb{E}_{S_t}[\delta\phi]$; (The norm of the expected TD update).
    $$
    \begin{align*}
    NEU(\theta) =& \left\Vert\sum_{s}\mu(s)\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[r(s,a) + \gamma V_\theta(s') - V_\theta(s)]\nabla V_\theta(s)\right\Vert_2^2 \\
    =&\left\Vert\sum_{s}\mu(s)\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[r(s,a) + \gamma V_\theta(s') - V_\theta(s)]\phi(s)\right\Vert_2^2 \\
    \nabla_\theta NEU(\theta) 
    =& \left(\sum_{s}\mu(s)\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[\gamma\phi(s') - \phi(s)]\phi^T(s)\right)\\
    & \cdot \left(\sum_{s}\mu(s)\sum_{a}\pi(a \vert s)\sum_{s'}p(s'\vert s, a)[r(s,a) + \gamma V_\theta(s') - V_\theta(s)]\phi(s)\right)\\
    =& \mathbb{E}_{s, a, s'}[(\gamma\phi(s') - \phi(s))\phi^T(s)]\cdot \mathbb{E}_{s,a,s'}[\delta \phi(s)]
    \end{align*}
    $$
  
  - $-\frac{1}{2}\nabla_{\theta} NEU(\theta) = \mathbb{E}_{S_t}[(\phi - \gamma \phi')\phi^T] \cdot \mathbb{E}_{S_t}[\delta(\theta) \phi]$;
  
  - $\theta_{k+1} = \theta_k + \alpha_k(\phi_k - \gamma \phi'_k)\phi^T_k u_k$;
  
  - $u_{k+1} = u_k + \beta_k(\delta_k \phi_k - u_k)$;
  
  - If we take exactly steps instead of stochastic steps, we can get
    $$
    -\frac{1}{2} \nabla_\theta NEU(\theta) = A \cdot (-A^T \theta + b),
    $$
    where $$A$$ and $$b$$ in exactly the same way as appearing in TD algorithm.
  
    > From concepts of numerical analysis, the condition number of A$A is always worse than A—notice -A is the underlying matrix for the expected TD(0) update. As such, GTD’s asymptotic rate of convergence is usually much worse than TD(0) on problems where TD(0) converges.  
- GTD2 algorithm:
  - $J(\theta) = \Arrowvert V_\theta - \Pi T^\pi V_\theta\Arrowvert^2_\mu$, where $\Pi = \Phi (\Phi^T D \Phi)^{-1}\Phi^T D$, then $J(\theta) = (\phi^T D (TV_\theta - V_\theta))^T (\phi^T D \phi)^{-1} \phi^T D(TV_\theta - V_\theta) = \mathbb E_{\mu}[\delta(\theta) \phi]^T (\mathbb{E}_\mu[\phi \phi^T])^{-1} \mathbb{E}_\mu[\delta(\theta) \phi]$;
  - $-\frac{1}{2}\nabla J(\theta) = \mathbb{E}_{\mu}[(\phi - \gamma \phi') \phi^T]\cdot (\mathbb{E}_{\mu}[\phi\phi^T])^{-1}\cdot\mathbb E_{\mu}[\delta(\theta)\phi]$;
  - $\theta_{k+1} = \theta_k + \alpha_k(\phi_k - \gamma \phi'_k)\phi^T_k w_k$;
  - $w_{k+1} = w_k + \beta_k ( \delta_k - \phi^T_k w_k)\phi_k$;
    - $$w = \mathbb{E}[\phi \phi^T]^{-1} \mathbb{E}[\delta(\theta) \phi]$$ because the convergence point $$w^*$$ satisfies $$\mathbb{E}_\mu[\delta \phi] = \mathbb{E}_\mu[\phi \phi^T] w^* $$;
- TDC algorithm (C for correction):
  - We change a lit bit of  preceding objective:
    $$
    \begin{align*}
    &-\frac{1}{2} \nabla J(\theta) \\
    =& \mathbb{E}_{\mu}[(\phi - \gamma \phi') \phi^T]\cdot (\mathbb{E}_{\mu}[\phi\phi^T])^{-1}\cdot\mathbb E_{\mu}[\delta(\theta)\phi]\\
    =&\mathbb{E}_\mu[\delta(\theta) \phi] - \gamma\mathbb{E}_\mu[\phi'\phi^T] \mathbb{E}_\mu[\phi\phi^T]^{-1} \mathbb{E}_\mu[\delta(\theta) \phi]\\
    =& \mathbb{E}_\mu[\delta(\theta)\phi] - \gamma\mathbb{E}_\mu[\phi'\phi^T]w(\theta)
    \end{align*}
    $$
  
  - $\theta_{k+1} = \theta_k + \alpha_k(\delta_k \phi_k - \gamma \phi'_k \phi^T_k w_k)$;
  
  - $w_{k+1} = w_k + \beta_k ( \delta_k - \phi^T_k w_k)\phi_k$;
  
  - >The first term is exactly TD(0) with linear function approximation;
    >
    >The second term is essentially an adjustment or correction of the TD update so that it follows the gradient of the MSPBE objective function. If the second parameter vector is initialized to w0 = 0, and βk is small, then this algorithm will start out making almost the same updates as conventional linear TD.

- Off-policy TDC algorithm:

  - The objective function becomes:
    $$
    \begin{align*}
    J(\theta) =& \Vert V_\theta - \Pi T^\pi V_\theta\Vert^2_{\mu_b}\\
    =& \mathbb E_{\mu_b}\left[\frac{\pi(a\vert s)}{\pi_b(a\vert s)}\delta(\theta) \phi\right]^T (\mathbb{E}_{\mu_b}[\phi \phi^T])^{-1} \mathbb{E}_{\mu_b}\left[\frac{\pi(a\vert s)}{\pi_b(a\vert s)}\delta(\theta) \phi\right],
    \end{align*}
    $$
    Then the deviation becomes:
    $$
    \begin{align*}
    &-\frac{1}{2}\nabla J(\theta) \\
    =& \mathbb E_{\mu_b}\left[\frac{\pi(a\vert s)}{\pi_b(a\vert s)}(\phi - \gamma \phi')\phi^T\right] (\mathbb{E}_{\mu_b}[\phi \phi^T])^{-1} \mathbb{E}_{\mu_b}\left[\frac{\pi(a\vert s)}{\pi_b(a\vert s)}\delta(\theta) \phi\right]\\
    =& \mathbb{E}_{\mu_b}\left[\frac{\pi(a\vert s)}{\pi_b(a\vert s)}\delta(\theta) \phi\right]- \gamma \mathbb{E}_{\mu_b}\left[\frac{\pi(a\vert s)}{\pi_b(a\vert s)} \phi' \phi^T\right](\mathbb{E}_{\mu_b}[\phi \phi^T])^{-1} \mathbb{E}_{\mu_b}\left[\frac{\pi(a\vert s)}{\pi_b(a\vert s)}\delta(\theta) \phi\right].
    \end{align*}
    $$

  - We denote $$\rho(a \vert s) = \frac{\pi(a \vert s)}{\pi_b(a \vert s)}$$ and $$\rho_t(a_t \vert s_t) = \frac{\pi(a_t \vert s_t)}{\pi_b(a_t \vert s_t)}$$, then we have the algorithm:

    - $$\theta_{t+1} = \theta_t + \alpha_t \rho_t(\delta_t \phi_t - \gamma \phi'_t\phi^T w_t) $$;
    - $$w_{t+1} = w_t + \beta_t(\rho_t \delta_t - \phi^T_t w_t)\phi_t$$;

- Similarly, off-policy GTD algorithm:
  - $\theta_{k+1} = \theta_k + \alpha_k\rho_t(\phi_k - \gamma \phi'_k)\phi^T_k w_k$;
  - $w_{k+1} = w_k + \beta_k ( \rho_t\delta_k - \phi^T_k w_k)\phi_k$;

### 5.3 The proof of convergence

All of TD algorithms are in the class of recursive stochastic algorithms; they are in the form of:
$$
x_{k+1} = x_k + \alpha_k (h(x_k) + M_{k+1}).
$$
The convergence proof uses the ordinary-differential-equation(ODE) approach. According this approach, it says that preceding equation is a noisy discretization of
$$
\dot{x}(t) = h(x(t)).
$$
Consider the following conditions:

- $$\alpha_k, \forall k$$, and are deterministic;
- $$\sum_k \alpha_k = +\infty, \sum_k \alpha^2_k < \infty$$;
- The function h is Lipschitz and $$h_\infty(x) = \lim_{c\rightarrow \infty} h(cx)/c$$ is well-defined for every $$x \in \mathbb{R}^d$$;
- The sequence $$(M_k)_{k\ge 0}$$ is a martingale difference sequence w.r.t. increasing family of $$\sigma-fields$$, $$\mathcal{F}_k = \sigma(x_0, M_1, \ldots, M_k)$$; that is , $$ \mathbb{E}[M_{k+1}\vert\mathcal{F}_k]=0$$;
- $$\exists K > 0$$, $$\forall k \ge 0$$, $$\mathbb{E}[\Vert M_{k+1} \Vert^2 \vert \mathcal{F}_k] \le K (1 + \Vert x_k \Vert^2)$$, holds for any $$k \ge 0$$ almost surely;
- $$\dot{x} = h_\infty(x) $$ has the origin as its unique globally asymptotically stable equilibrium.

> **Lemma**(The ODE Lemma) Consider the preceding iteration, and assume that the conditions hold. Then, as $$k \rightarrow \infty$$, the $$x_k$$ converges with probability one to the unique globally asymptotically stable equilibrium of the ODE, $$\dot{x}(t) = h(x(t))$$.

#### 5.3.1   Convergence analysis for GTD  

**(Quite a lot of assumptions, which including )**

- Four assumptions:
  - $$S_k \sim \mu, R_k = R(S_k, A_k, S'_k)$$ and $$(R_k, S'_k) \sim P(\cdot, \cdot, \vert S_k, A_k)$$;
  - $$(S_k, R_k, S'_k)_{\ge 0}$$ is an i.i.d. sequence;
  - $$(\phi_k, \phi'_k)_{k\ge 0}$$ is uniformly bounded second moments;
  - $$\exists \hat R_{max} s.t. Var[R_k \vert S_k] \le \hat R_{max}$$ holds almost surely.
- Two assumptions on features:
  - $$\Vert \phi_k \Vert_{\infty} < + \infty$$, and $$\Vert \phi'_k\Vert_{\infty} < + \infty$$;
  - $$C = \mathbb{E} [\phi_k \phi^T_k]$$ and $$A = \mathbb{E} [\phi_k (\phi_k - \gamma \phi'_k)^T]$$ are non-singular and uniformly bounded.
- Two step-size conditions:
  - $$\alpha_k, \beta_k > 0$$ and are deterministic;
  - $$\sum_k \alpha_k = \sum_k \beta_k = +\infty, \sum_k \alpha_k^2 < \infty$$ and $$\sum_k \beta^2_k < \infty$$.

> **Theorem** (Convergence of GTD). If the above assumptions are satisfied, and $$\beta_k = \eta \alpha_k$$, then $$\theta_k$$ converges to the TD-solution with probability one.

We recall the GTD method:

- $$\theta_{k+1} = \theta_k + \alpha_k (\phi_k - \gamma \phi'_k) \phi^T_k u_k$$;
- $$u_{k+1} = u_k + \beta_k(\delta_k \phi_k - u_k)$$;

The assumptions include  $$\beta_k = \eta \alpha_k$$， then
$$
\begin{bmatrix}
\frac{u_{k+1}}{\sqrt{\eta}}\\
\theta_{k+1}
\end{bmatrix}
= 
\begin{bmatrix}
\frac{u_{k}}{\sqrt{\eta}}\\
\theta_{k}
\end{bmatrix}
+ \alpha_k \sqrt{\eta} \left\{
\begin{bmatrix}
-\sqrt{\eta}I & \phi_k(\gamma \phi'_k - \phi_k)^T \\
-(\gamma \phi'_k - \phi_k)\phi^T_k & 0
\end{bmatrix}
\begin{bmatrix}
\frac{u_{k}}{\sqrt{\eta}}\\
\theta_{k}
\end{bmatrix}
+
\begin{bmatrix}
R_k \phi_k\\
0
\end{bmatrix}
\right\}.
$$
Let $$x^T = [v^T, \theta]$$ and $$v = u/\sqrt{\eta}$$, we have
$$
x_{k+1} = x_k + \alpha_k \sqrt{\eta} (G_k x_k + g_k),
$$
where $$G_k$$ and $$g_k$$ are corresponding partition of preceding equation.

Furthermore let
$$
G = \mathbb{E}[G_k] = 
\begin{bmatrix}
-\sqrt{\eta}I & -A\\
A^T & 0
\end{bmatrix}
$$
and
$$
g = \mathbb{E}[g_k] =
\begin{bmatrix}
b \\
0
\end{bmatrix}
$$
so
$$
\begin{align*}
x_{k+1} =& x_k + \alpha_k \sqrt{\eta}[G x_k + g + (G_k - G)x_k + (g_k - g)]\\
=& x_k + \alpha'_k[ h(x_k) + M_{k+1}]
\end{align*}
$$
We need to verify the following conditions:

- $$h$$ is Lipschitz and $$h_{\infty}(x) = \lim_{c\rightarrow \infty}h(cx)/c$$ is well-defined for every $$x \in \mathbb{R}^{2d}$$;

  - $$h(x)$$ is $$\Vert G\Vert$$-Lipschitz and $$h_\infty(x) = Gx$$;

- The sequence $$(M_k, \mathcal{F}_k)$$ is a martingale difference sequence, and

- for some $$c_0 > 0$$, $$\mathbb{E}[\Vert M_{k+1}\Vert \vert \mathcal{F}_k] \le c_0(1 + \Vert x_k \Vert^2)$$ holds for any initial parameter vector $$x_1$$;
  $$
  \begin{align*}
  &\mathbb{E}[\Vert M_{k+1}\Vert \vert \mathcal{F}_k]\\
  =& \mathbb{E}[\Vert  (G_k - G)x_k + (g_k - g)\Vert \vert \mathcal{F}_k]\\
  \le&  \mathbb{E}[\Vert  (G_k - G) \Vert \cdot \Vert x_k\Vert + \Vert(g_k - g)\Vert \vert \mathcal{F}_k]
  \end{align*}
  $$
  
- The sequence $$\alpha'_k$$ satisfies, $$ \sum^\infty_{k=1}\alpha'_k = \infty$$, and $$\sum^\infty_{k=1}(a'_k)^2 < + \infty$$;

- The ODE $$\dot{x} = h_{\infty}(x)$$ has the origin as a global asymptotically stable equilibrium;

  Firstly, we show $$G$$ is non-singular: $$det(G) = det(A^T A) = det^2(A) > 0$$.

  Then we denote the eigenvector $$ x^T = (x^T_1, x^T_2)$$, s.t.$$x^*x = 1$$, and
  $$
  \lambda = x^* G x = -\sqrt{\eta}\Vert x_1\Vert^2 - x^*_1 A x_2 + x^*_2 A^T x_1 = -\sqrt{\eta} \Vert x_1 \Vert^2 < 0.
  $$
  Because G is non-singular, so $$x_1 \ne 0$$.

  Then we need to know **Lyaponov function**.

- The ODE $$\dot{x} = h(x)$$ has a unique globally asymptotically stable equilibrium.

#### 5.2 Convergence analysis for GTD2

We recall GTD2 method:

- $$\theta_{k+1} = \theta_k + \alpha_k(\phi_k - \gamma \phi'_k)\phi^T_k w_k$$;
- $w_{k+1} = w_k + \beta_k ( \delta_k - \phi^T_k w_k)\phi_k$;

$$
\begin{bmatrix}
\frac{u_{k+1}}{\sqrt{\eta}}\\
\theta_{k+1}
\end{bmatrix}
= 
\begin{bmatrix}
\frac{u_{k}}{\sqrt{\eta}}\\
\theta_{k}
\end{bmatrix}
+ \alpha_k \sqrt{\eta} \left\{
\begin{bmatrix}
-\sqrt{\eta} \phi_k \phi^T_k & \phi_k(\gamma \phi'_k - \phi_k)^T \\
-(\gamma \phi'_k - \phi_k)\phi^T_k & 0
\end{bmatrix}
\begin{bmatrix}
\frac{u_{k}}{\sqrt{\eta}}\\
\theta_{k}
\end{bmatrix}
+
\begin{bmatrix}
R_k \phi_k\\
0
\end{bmatrix}
\right\}.
$$

Furthermore let
$$
G = \mathbb{E}[G_k] = \begin{bmatrix}-\sqrt{\eta}C & -A\\A^T & 0\end{bmatrix}
$$
and
$$
g = \mathbb{E}[g_k] =\begin{bmatrix}b \\0\end{bmatrix}
$$
so
$$
\begin{align*}x_{k+1} =& x_k + \alpha_k \sqrt{\eta}[G x_k + g + (G_k - G)x_k + (g_k - g)]\\=& x_k + \alpha'_k[ h(x_k) + M_{k+1}]\end{align*}
$$
The paper only talks about G:

- G is non-singular;
- $$x^* G x = -\sqrt{\eta}\Vert x_1 \Vert^2_{C} < 0$$;

#### 5.3.3 Convergence analysis for TDC

- $\theta_{k+1} = \theta_k + \alpha_k(\delta_k \phi_k - \gamma \phi'_k \phi_k^T w_k)$;
- $w_{k+1} = w_k + \beta_k ( \delta_k - \phi^T_k w_k)\phi_k$;

$$
\begin{bmatrix}
u_{k+1}\\
\theta_{k+1}
\end{bmatrix}
= 
\begin{bmatrix}
u_{k}\\
\theta_{k}
\end{bmatrix}
+ \alpha_k \left\{
\begin{bmatrix}
-\eta \phi_k \phi^T_k & \eta\phi_k(\gamma \phi'_k - \phi_k)^T \\
-\gamma \phi'_k \phi^T_k & \phi_k(\gamma \phi'_k - \phi_k)^T
\end{bmatrix}
\begin{bmatrix}
u_{k}\\
\theta_{k}
\end{bmatrix}
+
\begin{bmatrix}
\eta R_k \phi_k\\
R_k \phi_k
\end{bmatrix}
\right\}.
$$

Furthermore let
$$
G = \mathbb{E}[G_k]
=
\begin{bmatrix}
-\eta C & -\eta A\\
(A-C)^T & -A
\end{bmatrix}
$$
and
$$
g =
\begin{bmatrix}
\eta b \\
b
\end{bmatrix}.
$$

$$
\begin{align*}x_{k+1} =& x_k + \alpha_k [G x_k + g + (G_k - G)x_k + (g_k - g)]\\=& x_k + \alpha_k[ h(x_k) + M_{k+1}]\end{align*}
$$

> **Theorem** (Convergence of TDC). Besides the above assumptions, we also need a sufficient condition:
> $$
> \eta > \max\{0, -\lambda_{min}(C^{-1}H(A))\}, H(A) = \frac{A+A^T}{2},
> $$
> to guarantee the convergence of TDC.

**proof**

We calculate the eigenvalue of matrix G:
$$
\begin{align*}
&det(G - \lambda I)\\ 
=&
det\left(
\begin{bmatrix}
-\eta C - \lambda I & -\eta A\\
(A-C)^T & -A - \lambda I
\end{bmatrix}\right)\\
=& (-1)^{2d}
det\left(
\begin{bmatrix}
\eta C + \lambda I & \eta A\\
C - A^T & A + \lambda I
\end{bmatrix}\right)\\
=& det(\eta C + \lambda I)
det(A + \lambda I - (C - A^T) (\eta C + \lambda I)^{-1} \eta A)\\
=& det(\eta C + \lambda) 
det(A^{-1}[\lambda (\eta C + \lambda I) + \eta A A^T + \lambda A](\eta C + \lambda I)^{-1} A)\\
=& det(\lambda(\eta C + \lambda I) + \eta AA^T + \lambda A)
\end{align*}.
$$
If $$det(G - \lambda) = 0$$, then $$\exists x$$ satisfies:
$$
x^* \left(\lambda(\eta C + \lambda I) + \eta AA^T + \lambda A\right)x = 0,
$$

$$
\Rightarrow \Vert x \Vert^2 \lambda^2 
+ (\eta x^* C x 
+ x^* A x) \lambda
+ \eta\Vert Ax \Vert^2 = 0.
$$

Therefore the eigenvalue has only two solutions, $$\lambda_1$$ and $$\lambda_2$$, and
$$
\lambda_1 \lambda_2 = \frac{\eta \Vert A x \Vert^2}{\Vert x \Vert^2}>0,\\
\lambda_1 + \lambda_2 = \frac{-(\eta x^* C x + x^* A x)}{\Vert x \Vert^2}
$$
so, $$\lambda_2 = k \lambda^*_1$$, where $$k > 0$$. If we have $$ Re(\lambda_1 + \lambda_2) = (1+k)Re(\lambda_1) < 0$$,  then we can get$$ Re(\lambda_1) < 0$$ and $$Re(\lambda_2) < 0$$, which leads that $$G$$ is negative definite matrix.

Because
$$
\begin{align*}
&Re(\lambda_1 + \lambda_2) \\
=& \frac{Re(-\eta x^* C x - x^* A x)}{\Vert x \Vert^2} \\
=& \frac{-2\eta x^* C x + x^* (A + A^T) x}{2\Vert x \Vert^2} < 0,
\end{align*}
$$
therefore we only need
$$
2\eta x^* C x + x^* (A + A^T) x > 0.
$$
The above inequality holds if
$$
\eta > \max_{z \ne 0, z \in \mathbb{R}^d} \frac{- z^T H(A)z}{z^T C z}, H(A) = \frac{A+A^T}{2}.
$$
We let $$ y = C^{1/2}z$$, and $$\Vert y \Vert^2 = 1$$, then
$$
\eta > \max_{\Vert y \Vert^2 = 1} y^T (-C^{-1/2} H(A) C(-1/2))y,
$$
which is equivalent to $$\eta > -\lambda_{min}(C^{-1/2} H(A) C^{-1/2}) = -\lambda_{min}(C^{-1}H(A))$$.

Finally we get the condition of $$G$$ to be negative definite matrix:
$$
\eta > \max\{0, -\lambda_{min}(C^{-1} H(A))\}.
$$



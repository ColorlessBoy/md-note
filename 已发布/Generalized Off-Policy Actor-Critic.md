# Generalized Off-Policy Actor-Critic

## Abstract

- A new objective called **counterfactual objective** is proposed instead of **excursion objective**;

  (What do they mean?)

- Generalized Off-Policy Policy Gradient Theorem.

  (What is off-policy policy gradient theorem?)

## 1. Introduction

Objectives:

- **Alternative life objective**: the stationary distribution of the target policy (infeasible):
$$
  J_\pi = \sum_s d_\pi(s) i(s) V_\pi(s);
$$

- **Excursion objective**: the stationary distribution of the behavior policy (can misleading):
  $$
  J_\mu = \sum_s d_\mu(s) i(s) V_\pi(s);
  $$
  
- **Counterfactual objective**: approximates the alternative life objective:
  $$
  J_{\hat\gamma} = \sum_s d_{\hat\gamma}(s) \hat i(s) V_\pi(s).
  $$

Theorem: generalized off-policy actor-critic.

Algorithm: Counterfactual objective + emphatic approach(compute an unbiased sample for this policy gradient.)

## 2. Background

### Preliminaries

- $$\mathcal{S}$$: finite state set; 
- $$\mathcal{A}$$: finite action set;
- $$r:\mathcal{S}\times\mathcal{A} \rightarrow \mathbb{R}$$: bounded reward, $$\mathbb{E}[R_{t+1}\vert S_t, A_t] = r(S_t, A_t)$$;
- $$p: \mathcal{S}\times\mathcal{A}\times\mathcal{S} \rightarrow \mathbb{R}$$:transition kernel;$$\mathbf{P}_\pi(s, s') = \sum_a\pi(a \vert s) p(s' \vert s, a)$$;
- $$\gamma:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow\mathbb{R}$$:transition based discount function; $$ \gamma_t = \gamma(S_t, A_t, S_{t+1})$$;
- $$\pi:\mathcal{A}\times\mathcal{S}\rightarrow[0,1]$$: target policy; $$\mu:\mathcal{A}\times\mathcal{S}\rightarrow[0,1]$$:behavior policy;
- $$\mathbf{d}_\pi$$, $$\mathbf{D}_\pi = diag(\mathbf{d}_\pi)$$, $$\mathbf{d}_\pi$$ and $$\mathbf{D}_\pi = diag(\mathbf{d}_\pi)$$: stationary distribution;
- $$\rho(s, a) = \frac{\pi(a\vert s)}{\mu(a \vert s)}$$; $$\rho_t = \rho(S_t, A_t)$$;
- $$G_t := \sum^{\infty}_{i=0} \Gamma^{i-1}_t R_{t + 1 + i}$$, where $$\Gamma^{i-1}_t = \prod^{i-1}_{j=0}\gamma(S_{t+j}, A_{t+j}, S_{t+j+1})$$ and $$\Gamma^{-1}_t = 1$$;
- $$V_{\pi}(s) := \mathbb{E}_\pi[G_t \vert S_t = s]$$;
- $$Q_\pi(s, a) := \mathbb{E}_\pi[G_t \vert S_t = s, A_t = a]$$;
- **Assumption 1**: that the Markov chain induced by $$\pi$$ is ergodic;
- **Assumption 2**: Behavior policy $$\mu$$ satisfies that $$\forall (s, a), \pi(a|s) > 0 \Rightarrow \mu(a \vert s)$$.

### Related papers

Gelada and Bellemare (2019) define a new transition matrix
$$
P_{\hat \gamma} = \hat \gamma P_\pi + (1 - \hat \gamma) 1 d^T_\mu,
$$
The corresponding stationary distribution is
$$
d_{\hat \gamma} = (1-\hat \gamma)(I - \hat \gamma P^T_\pi)^{-1} d_\mu.
$$

We denote $$c(s) = \frac{d_{\hat\gamma}(s)}{d_{\mu}(s)}$$, then
$$
c = \hat\gamma D^{-1}_\mu P^T_{\pi}D_{\mu}c + (1-\hat\gamma)1.
$$
**proof**:
$$
d_{\hat \gamma} = (1-\hat \gamma)(I - \hat \gamma P^T_\pi)^{-1} d_\mu\\
(I-\hat\gamma P^T_{\pi}) d_{\hat\gamma} = (1-\hat\gamma)d_\mu\\
d_{\hat\gamma} = \hat\gamma P^T_{\pi}d_{\hat\gamma} + (1-\hat\gamma)d_\mu\\
D^{-1}_{\mu}d_{\hat\gamma} = \hat \gamma D^{-1}_{\mu} P^T_{\pi}d_{\hat\gamma} + (1-\hat\gamma)D^{-1}_{\mu}d_\mu\\
c = \hat\gamma D^{-1}_\mu P^T_{\pi}D_\mu c +(1-\hat\gamma)1.
$$
The algorithm is
$$
c(s') = c(s') + \alpha[\hat\gamma \rho_t c(s') + (1-\hat\gamma)-c(s)].
$$


## 3. The Counterfactual Objective 

- Counterfactual objective
  $$
  J_{\hat\gamma} = \sum_s d_{\hat\gamma}(s) \hat i(s) V_\pi(s).
  $$

- $$d_{\hat \gamma} = (1-\hat \gamma)(I - \hat \gamma P^T_\pi)^{-1} d_\mu$$.

## 4. Generalized Off-Policy Policy Gradient

For off-policy policy gradient:
$$
J_{\mu}(\theta) = \sum_s d_{\mu}(s) i(s) V_{\pi_\theta}(s).
$$

$$
\begin{align*}
\nabla_\theta J_\mu(\theta) =& \sum_{s} d_\mu(s) i(s)\nabla_\theta V_{\pi_\theta}(s)\\
=& \sum_{s} m(s) \sum_a q_\pi(s,a) \nabla_\theta \pi(a \vert s)
\end{align*}
$$

where $$m^T = i^T D_\mu(I - P_{\pi,\gamma})^{-1}$$, $$P_{\pi,\gamma}(s,s') = \sum_a \pi(a\vert s) p(s'\vert s, a) \gamma(s, a, s')$$.

> **Theorem 1**(Generalized Off-Policy Policy Gradient Theorem)
> $$
> \nabla J_{\hat\gamma}(\theta) = \sum_s m(s) \sum_a q_\pi(s,a) \nabla_\theta\pi(a\vert s) + \sum_{s} d_\mu(s) \hat i(s) v_\pi(s) g(s),
> $$
> where $$g = \hat\gamma D^{-1}_\mu (I - \hat\gamma P^T_\pi)^{-1}b$$ and $$b = \nabla_\theta P^T_\pi D_\mu c$$.
>
> **proof**.
> $$
> \begin{align*}
> \nabla_\theta J_{\hat\gamma} =& \sum_{s} d_{\hat\gamma} \hat i(s) \nabla_\theta V_{\pi_\theta}(s) + \sum_s \nabla_\theta d_{\hat\gamma}(s)\hat i(s) V_{\pi_\theta}(s)\\
> =& \sum_{s} d_\mu(s) c(s) \hat i(s)\nabla_\theta V_{\pi_\theta}(s)
> + \sum_s d_\mu(s) \nabla_\theta c(s) \hat i(s) V_{\pi_\theta}(s)\\
> =& \sum_{s} m(s) \sum_a q_\pi(s,a) \nabla_\theta \pi(a \vert s)+ \sum_s d_\mu(s) \nabla_\theta c(s) \hat i(s) V_{\pi_\theta}(s)
> \end{align*}
> $$
>
> $$
> c = \hat\gamma D^{-1}_\mu P^T_{\pi}D_\mu c +(1-\hat\gamma)1\\
> \begin{align*}
> \nabla c =& \hat\gamma D^{-1}_\mu P^T_{\pi}D_\mu \nabla c + \hat\gamma D^{-1}_\mu \nabla P^T_{\pi}D_\mu c\\
> =&(I - \hat\gamma D^{-1}_{\mu} P^T_\pi D_\mu)^{-1}\hat\gamma D^{-1}_{\mu} \nabla P^T_{\pi} D_{\mu} c\\
> =&(D^{-1}_{\mu} (I - \hat\gamma P^T_\pi)^{-1}D_\mu)^{-1}\hat\gamma D^{-1}_\mu \nabla P^T_{\pi}D_\mu c\\
> =& \hat\gamma(D^{-1}_{\mu} (I - \hat\gamma P^T_\pi)^{-1})^{-1} \nabla P^T_{\pi}D_\mu c
> \end{align*}
> $$



[1] S. Zhang, W. Boehmer, and S. Whiteson, “Generalized Off-Policy Actor-Critic,” *arXiv:1903.11329 [cs, stat]*, Oct. 2019.
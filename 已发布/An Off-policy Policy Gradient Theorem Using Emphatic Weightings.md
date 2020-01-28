# An Off-policy Policy Gradient Theorem Using Emphatic Weightings

## 2. Preliminaries

- $$ \mathbb{E}[R_{t+1} \vert S_t, A_t, S_{t+1}] = r(S_t, A_t, S_{t+1})$$;
- $$G_t = R_{t+1} + \gamma_{t+1} R_{t+2} + \gamma_{t+1}\gamma_{t+2}R_{t+3}+\cdots = R_{t+1} + \gamma_{t+1}G_{t+1}$$;
- $$v_\pi(s) = \mathbb{E}_{\pi}[G_t \vert S_t = s]$$;
- $$\mathbb{E}[R_{t+1} \vert S_t, A_t, S_{t+1}] = r(S_t, A_t, S_{t+1})$$;
- $$v_\pi(s) = \sum_{a \in A}\pi(s, a) \sum_{s'\in S} P(s, a, s')[r(s, a, s') + \gamma(s, a, s') v_\pi(s')]$$;
- $$d_\pi(s) = \lim_{t\rightarrow \infty} P(S_t = s \vert s_0, \pi)$$
- $$J_\mu(\theta) = \sum_{s \in S} d_\mu(s) i(s) v_{\pi_\theta}(s)$$, $$\mu$$ is behavior policy;

> $$\gamma:S \times A \times S \rightarrow [0,1]$$ unifies continuing and episodic tasks.

> If $$ i(s) = 1 \forall s$$, the objective reduces to the standard off-policy objective. Otherwise, it naturally encompasses other settings, such as the start state formulation by setting $$i(s) = 0$$ for all states but the start state.

## 3. Off-Policy Policy Gradient Theorem using Emphatic Weightings

Assumptions:

- $$P(s,a,s'), r(s,a,s'),\gamma(s,a,s'),\pi(s;\theta)$$ and their derivatives are continuous in all variables $$s, a, s', \theta$$;
- S is a compact set of $$\mathbb{R}^d$$;
- The policy $$\pi$$ and discount $$\gamma$$ are such that for $$P_{\pi,\gamma}(s,s') = \int_A \pi(s,a) \gamma(s,a,s') P(s,a,s') da$$, the inverse kernel of $$\delta(s, s') - P_{\pi, \gamma}(s, s')$$ exists.

Note that: the assumption 1 and 2 allow us to switch the order of integration and differentiation, and order of multiple integrations later in the proof.

> **Theorem 1** (Off-policy Policy Gradient Theorem).
> $$
> \frac{\partial J_\mu(\theta)}{\partial \theta}
> = \sum_s m(s) \sum_a \frac{\partial \pi(s, a; \theta)}{\partial \theta}q_\pi(s,a),
> $$
> where $$m^T = i^T(I - P_{\pi, \gamma})^{-1}$$, $$i(s)  = d_\mu(s) i(s)$$ and
> $$
> P_{\pi,\gamma}(s, s') = \sum_{a} \pi(s, a; \theta) P(s, a, s') \gamma(s, a, s').
> $$

**proof**:
$$
\begin{align*}
&\frac{\partial v_\pi(s)}{\partial \theta}
= \frac{\partial}{\partial \theta} \sum_a \pi(s, a; \theta) q_\pi(s,a)\\
=& \sum_a \frac{\partial \pi(s,a;\theta)}{\partial \theta} q_\pi(s,a)
+ \sum_a \pi(s,a;\theta) \frac{\partial q_\pi(s,a)}{\partial \theta}\\
=& g(s) + \sum_a\pi(s,a;\theta) \frac{\partial}{\partial \theta}
[\sum_{s'} P(s,a,s')(r(s,a,s')+\gamma(s,a,s') v_\pi(s'))]\\
=& g(s) + \sum_{a} \pi(s,a,s')\sum_{s'}P(s,a,s')\gamma(s,a,s')\frac{\partial  v_\pi(s')}{\partial \theta}
\end{align*}
$$

$$
\dot v_\pi = G + P_{\pi,\gamma} \dot v_\pi \rightarrow \dot v_\pi = (I-P_{\pi,\gamma})^{-1} G
$$

$$
\begin{align*}
& \frac{\partial J_\mu(\theta)}{\partial \theta}\\
=& \frac{\partial \sum_s i(s) v_\pi(s)}{\partial \theta}\\
=& \sum_s i(s) \frac{v_\pi(s)}{\partial \theta}\\
=& \sum_s i(s) (I - P_{\pi, \gamma})^{-1} G\\
=& \sum_s m(s) \sum_{a} \frac{\partial \pi(s,a;\theta)}{\partial \theta} q_\pi(s,a)
\end{align*}
$$

> **Theorem 2** (Deterministic Off-policy Policy Gradient Theorem).
> $$
> \nabla_\theta J_\mu(\theta)
> = \int_s m(s) \nabla_\theta \pi(s; \theta) \nabla q_\pi(s,a)\Big\vert_{a = \pi(s;\theta)} ds
> $$
>
> $$
> m(s') = d_\mu(s) i(s) + \int_s P(s, \pi(s;\theta), s')\gamma(s, \pi(s;\theta), s')m(s) ds
> $$

**proof**:
$$
\begin{align*}
&\frac{\partial v_\pi(s)}{\partial \theta} = \frac{\partial q_\pi(s, \pi(s;\theta))}{\partial \theta} \\
=& \frac{\partial}{\partial\theta} \int_{s' \in S} P(s, \pi(s; \theta), s') (r(s, \pi(s;\theta),s')+\gamma(s, \pi(s;\theta), s')v_\pi(s'))ds'\\
=& \int_{s' \in S} \frac{\partial\pi(s;\theta)}{\partial\theta} \frac{\partial P(s, a, s')}{\partial a}(r + \gamma v_\pi(s'))\\
&+ P\left[\frac{\partial\pi(s;\theta)}{\partial\theta} \frac{\partial r(s, a, s')}{\partial a} + \frac{\partial\pi(s;\theta)}{\partial\theta} \frac{\partial \gamma(s, a, s')}{\partial a} v_\pi(s') + \gamma\frac{\partial v_\pi(s')}{\partial\theta}\right]\\
=& \int_{s' \in S} \frac{\partial\pi(s;\theta)}{\partial\theta} \frac{\partial}{\partial a}\left[P(s, a, s') + r(s, a, s') + \gamma(s, a, s') v_\pi(s')\right]\Big\vert_{a = \pi(s;\theta)} + P \gamma \frac{\partial v_\pi(s')}{\partial\theta} ds'\\
=& \frac{\partial \pi(s; \theta)}{\partial\theta} \frac{\partial q_\pi(s,a)}{\partial a}\Big\vert_{a = \pi(s;\theta)} + \int_{s' \in S} P(s, \pi(s;\theta), s')\gamma(s, \pi(s;\theta), s') \frac{\partial v_\pi(s')}{\partial\theta} ds'\\
:=& g(s) + \int_{s' \in S} P_{\pi, \gamma}(s, s') \frac{\partial v_\pi(s')}{\partial\theta} ds'
\end{align*}
$$

$$
\begin{align*}
&\int_{s' \in S} \delta(s,s')\frac{\partial v_\pi(s')}{\partial \theta} ds' = v_\pi(s)  = g(s) + \int_{s' \in S} P_{\pi, \gamma}(s, s') \frac{\partial v_\pi(s')}{\partial\theta} ds'\\
\Rightarrow& \int_{s' \in S} \left(\delta(s,s') - P_{\pi,\gamma}(s, s')\right)\frac{\partial v_\pi(s')}{\partial \theta} ds' = g(s)\\
\Rightarrow& \frac{\partial v_\pi(s)}{\partial \theta} = \int_{s’ \in S} k(s,s') g(s') ds'
\end{align*}
$$

$$
\begin{align*}
&\frac{\partial J_\mu(\theta)}{\partial \theta} 
= \frac{\partial}{\partial \theta} \int_{s \in S} d_\mu(s) i(s) v_\pi(s) ds\\
=& \int_{s \in S} \frac{\partial v_\pi(s')}{\partial \theta} d_\mu(s) i(s) ds\\
=& \int_{s \in S} \int_{s' \in S} k(s,s') g(s') ds' \cdot d_\mu(s) i(s) ds\\
=& \int_{s' \in S} \int_{s \in S} k(s,s') d_\mu(s) i(s) ds \cdot g(s') ds'
\end{align*}
$$

We define
$$
m(s') = d_\mu(s') i(s') + \int_{s' \in S} P_{\pi,\gamma}(s, s') m(s) ds
$$

> **Lemma**
> $$
> m(s') = \int_{s\in S} k(s, s') d_\mu(s) i(s) d s
> $$
> **proof**:
> $$
> \int_{s \in S} \delta(s, s') m(s) ds = m(s') 
> = d_\mu(s')i(s') + \int_{s\in S} P_{\pi,\gamma}(s, s') m(s) ds
> $$

$$
\frac{\partial J_\mu(\theta)}{\partial \theta} = \int_{s' \in S} m(s') g(s') ds'
= \int_{s \in S} m(s) \frac{\partial \pi(s;\theta)}{\partial \theta} \frac{q_\pi(s, a)}{\partial a}\Big\vert_{a = \pi(s;\theta)} ds
$$

## 4. Actor-Critic with Emphatic Weightings

### Part 1:

$$
\begin{align*}
&\sum_a \frac{\partial \pi(s, a;\theta)}{\partial \theta} q_\pi(s, a)\\
=& \sum_a \mu(s,a) \frac{\pi(s, a;\theta)}{\mu(s,a)} \nabla_\theta ln(\pi(s,a;\theta)) q_\pi(s,a)\\
=& \mathbb{E}_{a \sim \mu(\cdot\vert s)}\left[\frac{\pi(s, a;\theta)}{\mu(s,a)} \nabla_\theta ln(\pi(s,a;\theta)) q_\pi(s,a)\right]
\end{align*}
$$

- Method 1:
  $$
  \theta \leftarrow \theta + \alpha_t \rho_t M_t \frac{\partial \ln\pi(s, a;\theta)}{\partial \theta} q_\pi(s,a).
  $$
  The first approach, though, is necessary when only a value function is estimated.

- Method2: 
  $$
  \theta \leftarrow \theta+ \alpha M_t \sum_{b \in A} \pi(s, b;\theta) \frac{\ln \pi(s, b;\theta)}{\partial \theta} q_\pi(s, a).
  $$
  It avoids potentially high-variance importance sampling ratio.

### Part 2:

We define
$$
\begin{align*}
m^T_{\lambda_a} =& i^T (I - P_{\pi,\gamma})^{-1}(I - (1 - \lambda_a) P_{\pi, \gamma})\\
=& \lambda_a i^T (I - P_{\pi,\gamma})^{-1} + (1-\lambda_a) i^T
\end{align*}
$$

- For $$\lambda_a  = 1$$, we get $$m_{\lambda_a} = m$$ and so get an unbiased emphatic weighting;
- For $$\lambda_a = 0$$, we get regular off-policy actor critic update.

We estimate $$M_t$$ by two steps:

- $$M_t = (1 - \lambda_a) i(S_t) + \lambda_a F_t$$;
- $$ F_t = \gamma_t \rho_{t-1} F_{t-1} + i(S_t)$$;

> **Proposition 1**. For a fixed policy $$\pi$$, and with the conditions on the MDP from paper, *On convergence of emphatic temporal-difference learning*, we have
> $$
> \mathbb{E}_\mu[\rho_t M_t \delta_t \nabla_\theta \ln \pi(S_t, A_t;\theta)]
> = \sum_{s} m(s) \sum_{a} \frac{\partial \pi(s, a;\theta)}{\partial \theta} q_\pi(s,a).
> $$

I have question in proof.

### Pseudocode

<img src="C:\Users\pengl\Documents\md-notes\pic\Emphaic_Actor_Critic.png" alt="Emphaic_Actor_Critic" style="zoom:80%;" />



[1]E. Imani, E. Graves, and M. White, “An Off-policy Policy Gradient Theorem Using Emphatic Weightings,” *arXiv:1811.09013 [cs, stat]*, Jun. 2019.
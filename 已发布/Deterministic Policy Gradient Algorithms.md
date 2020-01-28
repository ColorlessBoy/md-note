# Deterministic Policy Gradient Algorithms 

## 1. Deterministic Policy Gradient Theorem

- Conditions
  - $$ p(s' | s, a), \nabla_a p(s' | s, a), \mu_\theta (s), \nabla_\theta \mu_\theta(s), r(s, a), \nabla_a r(s,a), p_0(s)$$ are continuous in all parameters and variables $ s, a, s'$ and $ \theta $;
  - $$ \exists b, L $$, $$ \sup_s p_0(s) < b $$, $$ \sup_{a, s, s'} p(s' | s,a) < b $$, $$ \sup_{a,s} r(s,a) < b $$, $$ \sup_{a,s,s'} \Arrowvert \nabla_a p(s'|s, a) \Arrowvert < L $$, and $$ \sup_{a,s} \Arrowvert \nabla_a r(s,a) \Arrowvert < L $$.

> **Definition 1**(Deterministic Policy).
> $$
> \mu_\theta(s) = \arg\max_a Q^{\mu_\theta}(s, a).
> $$

> **Definition 2** (Deterministic discounted policy value).
> $$
> J(\mu_\theta) = \int_S \rho^{\mu_\theta}(s) r(s, \mu_\theta(s)) ds
>         = \mathbb{E}_{s\sim\rho^{\mu_\theta}} [r(s, \mu_\theta(s))]\\
> Q^{\mu_\theta} (s, a)
>         = \mathbb{E} \left\{ \sum^{\infty}_{k=1} \gamma^{k-1} r_{t+k} | s_t = s, a_t 		 = a, \pi \right\}\\
> \rho ^{\mu_\theta}(s_0) = \int_S \sum^{\infty}_{t=0} \gamma^t p_0(s_0) p(s_0 \rightarrow s, t, \mu_\theta) d s_0
> $$

> **Theorem 1** (Deterministic Policy Gradient Theorem). If preceding conditions are satisfied, and $$ \nabla_\theta \mu_\theta(s) $$ and $$ \nabla_a Q^{\mu_\theta}(s, a) $$ exist, and that the deterministic policy gradient exists. Then,
> $$
> \begin{align*}
>             \nabla_\theta J(\mu_\theta) 
>             =& \int_S \rho^{\mu_\theta}_\theta(s) \nabla_\theta \mu_\theta(s) \nabla_a Q ^{\mu_\theta} (s, a)|_{a=\mu_\theta} ds \\
>             =& \mathbb{E}_{s \sim \rho^{\mu_\theta}}
>             [ \nabla_\theta \mu_\theta(s) \nabla_a Q^{\mu_\theta} (s, a) |_{a=\mu_\theta} ]
> \end{align*}
> $$

**proof**:
$$
\begin{align*}
            \nabla_\theta V ^{\mu_\theta}(s)
            =& \nabla_\theta Q ^{\mu_\theta}(s, \mu_\theta(s)) \\
            =& \nabla_\theta \left( r(s, \mu_\theta(s)) + \int_S \gamma p(s'|s, \mu_\theta(s)) V^{\mu_\theta}(s') ds' \right) \\
            =& \nabla_\theta \mu_\theta(s) \nabla_a r(s,a) |_{a=\mu_\theta(s)}
                + \nabla_\theta \int_S \gamma p(s'|s, \mu_\theta(s)) V^{\mu_\theta}(s') ds'\\
            =& \nabla_\theta \mu_\theta(s) \nabla_a r(s,a)|_{a=\mu_\theta(s)} \\
             &+ \int_S \gamma \left(
                p(s'|s,\mu_\theta(s)) \nabla_\theta V^{\mu_\theta}(s')
                + \nabla_\theta \mu_\theta(s) \nabla_a p(s'|s, a)|_{a=\mu_\theta(s)} V^{\mu_\theta}(s')
             \right) ds'\\
            =& \nabla_\theta \mu_\theta(s) \nabla_a \left( r(s,a) + \int_S \gamma p(s'|s,a) V^{\mu_\theta} (s') ds' \right)|_{a=\mu_\theta(s)} \\
             &+ \int_S \gamma p(s'|s,\mu_\theta(s)) \nabla_\theta V^{\mu_\theta} (s') ds' \\
            =& \nabla_\theta \mu_\theta(s) \nabla_a Q^{\mu_\theta} (s, a) |_{a=\mu_\theta (s)}
            + \int_S \gamma p(s\rightarrow s', 1, \mu_\theta) \nabla_\theta V^{\mu_\theta} (s') ds' \\
             &\vdots\\
            =& \int_S \sum^{\infty}_{t=0} \gamma^t p(s \rightarrow s', t, \mu_\theta) \nabla_\theta \mu_\theta(s') \nabla_a Q^{\mu_\theta} (s', a) |_{a=\mu_\theta(s')} ds'\\
            \nabla_\theta J(\mu_\theta)
            =& \nabla_\theta \int_S p_0(s) V ^{\mu_\theta}(s) ds \\
            =& \int_S p_0(s) \nabla_\theta V ^{\mu_\theta}(s) ds \\
            =& \int_S \int_S \sum^{\infty}_{t=0} \gamma^t p_0(s) p(s \rightarrow s', t, \mu_\theta) \nabla_\theta \mu_\theta(s') \nabla_a Q^{\mu_\theta} (s', a) |_{a=\mu_\theta(s')} ds' ds\\
            =& \int_S \int_S \sum^{\infty}_{t=0} \gamma^t p_0(s_0) p(s_0 \rightarrow s, t, \mu_\theta) ds_0 \nabla_\theta \mu_\theta(s)\nabla_a Q^{\mu_\theta}(s,a)|_{a=\mu_\theta(s)} ds\\
            =& \int_S \rho ^{\mu_\theta}(s) \nabla_\theta \mu_\theta(s) \nabla_a Q^{\mu_\theta}(s, a) |_{a=\mu_\theta(s)} ds
        \end{align*}
$$

## 2. Trajectory Space Proof

$$
\begin{align*}
&\nabla_\theta\mathbb{E}_{\tau}[R(\tau)] = \nabla_\theta\mathbb{E}_\tau\left[\sum^{T}_{t=0}\gamma^t r(s_t, \mu_\theta(s_t))\right] = \sum^T_{t=0} \lambda^t \nabla_\theta \mathbb{E}_{\tau_t}[r(s_t, \mu_\theta(s_t))]\\
=& \sum^T_{t=0} \lambda^t \left\{\mathbb{E}_{\tau_t}[\nabla_\theta \mu_\theta(s_t) \nabla_a r(s_t, a)\vert_{a=\mu_\theta(s_t)}] + \mathbb{E}_{\tau_t}[r(s_t, \mu_\theta(s_t) ) \nabla_\theta \ln p(\tau_t)]\right\}\\
=& \sum^T_{t=0} \lambda^t \left\{\mathbb{E}_{\tau_t}[\nabla_\theta \mu_\theta(s_t) \nabla_a r(s_t, a)\vert_{a=\mu_\theta(s_t)}]\right\} \\
&+ \sum^{T}_{t=1}\lambda^t\mathbb{E}_{\tau_t}\left[r(s_t, \mu_\theta(s_t) ) \sum^t_{t'=1}\nabla_\theta \mu_{\theta}(s_{t'-1})\nabla_a \ln p(s_{t'}\vert s_{t'-1}, a)\vert_{a=\mu_\theta(s_{t'-1})} \right]\\
=& \mathbb{E}_{\tau}\left\{\sum^T_{t=0} \lambda^t \nabla_\theta \mu_\theta(s_t) \nabla_a r(s_t, a)\vert_{a=\mu_\theta(s_t)}\right\} \\
&+ \mathbb{E}_{\tau}\left\{\sum^{T}_{t=1}\lambda^tr(s_t, \mu_\theta(s_t) ) \sum^t_{t'=1}\nabla_\theta \mu_{\theta}(s_{t'-1})\nabla_a \ln p(s_{t'}\vert s_{t'-1}, a)\vert_{a=\mu_\theta(s_{t'-1})} \right\}\\
=& \mathbb{E}_{\tau}\left\{\sum^T_{t=0} \lambda^t \nabla_\theta \mu_\theta(s_t) \nabla_a r(s_t, a)\vert_{a=\mu_\theta(s_t)}\right\} \\
&+ \mathbb{E}_{\tau}\left\{ \sum^T_{t=1}\nabla_\theta \mu_{\theta}(s_{t-1})\nabla_a \ln p(s_{t}\vert s_{t-1}, a)\vert_{a=\mu_\theta(s_{t-1})}\sum^{T}_{t'=t}\lambda^{t'}r(s_{t'}, \mu_\theta(s_{t'}) ) \right\}\\
=& \mathbb{E}_{\tau}\left\{\sum^T_{t=0} \lambda^t \nabla_\theta \mu_\theta(s_t) \nabla_a r(s_t, a)\vert_{a=\mu_\theta(s_t)}\right\} \\
&+ \mathbb{E}_{\tau}\left\{ \sum^{T-1}_{t=0}\nabla_\theta \mu_{\theta}(s_{t})\nabla_a \ln p(s_{t+1}\vert s_{t}, a)\vert_{a=\mu_\theta(s_{t})}\lambda^{t+1} p(s_{t+1} \vert s_{t}, \mu_\theta(s_{t}))Q(s_{t+1}, \mu_\theta(s_{t+1})) \right\}\\
&(\text{I am not sure, here.})\\
=&\mathbb{E}_{\tau}\left\{\sum^{T}_{t=0}\lambda^t\nabla_\theta \mu_{\theta}(s_{t})\nabla_a \left( r(s_t, a) + \lambda p(s_{t+1}\vert s_t, a) Q(s_{t+1}, \mu_\theta(s_{t+1}))\right)\vert_{a = \mu_\theta(s_t)}\right\}, \quad(Q(s_{T+1}, \cdot) = 0)\\
=& \mathbb{E}_\tau\left\{ \sum^{T}_{t=0} \lambda^t \nabla_\theta \mu_\theta(s_t) \nabla_a Q(s_t, a) \vert_{a=\mu_\theta(s_t)}\right\}
\end{align*}
$$


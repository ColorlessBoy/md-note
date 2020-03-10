# Off-Policy Policy Gradient by Bootstrapping the Covariate Shift

## Off-Policy Policy Gradient

The cost of policy $$\mu$$ is
$$
J_\mu(\theta_\pi) = \sum_s d_\mu(s) V(s; \theta_\pi).
$$
Firstly we deviate $$V(s; \theta_\pi)$$:
$$
\begin{align*}
\nabla_{\theta_\pi} V(s; \theta_\pi) 
=& \nabla_{\theta_\pi} \sum_a \pi(a \vert s; \theta_\pi) Q(s, a; \theta_\pi) \\
=& \sum_a \nabla_{\theta_\pi} \pi(a\vert s; \theta_\pi) Q(s, a; \theta_\pi) 
+ \sum_a \pi(a \vert s; \theta_\pi) \nabla_{\theta_\pi} Q(s, a; \theta_\pi)\\
=& \sum_a \nabla_{\theta_\pi} \pi(a\vert s; \theta_\pi) Q(s, a; \theta_\pi) 
+ \sum_a \pi(a \vert s; \theta_\pi) \nabla_{\theta_\pi} \sum_{s'} p(s'\vert s, a)(r(s, a, s') + \gamma V(s'; \theta_\pi))\\
=& \sum_a \nabla_{\theta_\pi} \pi(a\vert s; \theta_\pi) Q(s, a; \theta_\pi) 
+ \gamma \sum_a \pi(a \vert s; \theta_\pi)\sum_{s'} p(s'\vert s, a) \nabla_{\theta_\pi} V(s'; \theta_\pi)
\end{align*}
$$
We denote
$$
\nabla_\theta V(\theta_\pi) = 
\begin{bmatrix}
\left(\frac{d V(s1; \theta_\pi)}{d\theta_\pi}\right) & \left(\frac{d V(s2; \theta_\pi)}{d\theta_\pi}\right) & \cdots \left(\frac{d V(sN; \theta_\pi)}{d\theta_\pi}\right)
\end{bmatrix}^T,
$$

$$
G(\theta_\pi) = 
\begin{bmatrix}
\sum_a \frac{d \pi(a\vert s1; \theta_\pi)}{d\theta_\pi} Q(s1, a; \theta_\pi)& \sum_a \frac{d \pi(a\vert s1; \theta_\pi)}{d\theta_\pi} Q(s2, a; \theta_\pi) & \cdots & \sum_a \frac{d \pi(a\vert s1; \theta_\pi)}{d\theta_\pi} Q(sN, a; \theta_\pi)
\end{bmatrix}^T
$$

and
$$
P_{\theta_\pi} =
\begin{bmatrix}
p(s' \vert s;\theta_\pi) = \sum_a \pi(a \vert s; \theta_\pi)\sum_{s'} p(s'\vert s, a)
\end{bmatrix}.
$$
Then we have
$$
\nabla_{\theta_\pi} V(\theta_\pi) = G(\theta_\pi) + \gamma P \nabla_{\theta_\pi} V(\theta_\pi)
\Rightarrow \nabla_{\theta_\pi} V(\theta_\pi) = (I - \gamma P_{\theta_\pi})^{-1} G(\theta_\pi)
$$
and

$$
\begin{align*}
\nabla_{\theta_\pi} J_\mu(\theta_\pi) =& [\nabla_{\theta_\pi} V(\theta_\pi)]^T d_\mu = G^T(\theta_\pi) (I-\gamma P_{\theta_\pi}^T)^{-1} d_\mu\\
=& \sum_s m(s) \sum_a \nabla_{\theta_\pi} \pi(a\vert s; \theta_\pi) Q(s, a; \theta_\pi)\\
& m(s) = [(I-\gamma P_{\theta_\pi}^T)^{-1} d_\mu](s).

\end{align*}
$$

$$
d^T_\pi P_\pi = d^T_\pi
$$

$$
d_\pi^T (I - \gamma P_\pi)^{-1} = d^T_\pi (I + \gamma P_\pi + (\gamma P_\pi)^2 + \cdots) = (1 + \gamma + \gamma^2 + \cdots) d^T_\pi = \frac{1}{1-\gamma} d^T_\pi
$$

$$
\nabla_{\theta_\pi} J_\mu(\theta_\pi) = \frac{1}{1-\gamma} \sum_s d_{\pi}(s) \sum_a \nabla_{\theta_\pi} \pi(a\vert s; \theta_\pi) Q(s, a; \theta_\pi)
$$

---

Stationary distribution assumption is bad. So
$$
\nabla_{\theta_\pi} J(\theta_\pi) = p_0^T (I - \gamma P_{\theta_\pi})^{-1} G(\theta_\pi)
$$
We only can get samples following  $$p^T_0(I - \gamma P_\mu)^{-1}$$, we need get diagonal matrix $$R$$ that satisfies
$$
p^T_0(I - \gamma P_\mu)^{-1} R = p^T_0 (I - \gamma P_\pi)^{-1}
$$




## Covariate Shift $$\nabla_\theta J_\pi(\theta)$$  Emphatic weightings $$\nabla_\theta J_\mu(\theta)$$

> **Definition** (covariate operator)
> $$
> Yc = D^{-1}_{d_\mu} P^T_\pi D_{d_\mu} c.
> $$
> **Definition** ($$\hat \gamma$$-discounted covariate shift operator)
> $$
> Y_{\hat \gamma} c := \hat \gamma Yc + (1 - \hat \gamma) e.
> $$
> **Definition** ($$\hat \gamma$$-discounted transition function)
> $$
> \hat P_\pi = \hat \gamma P_\pi + (1- \hat\gamma) e d^T_\mu,
> $$
> and the corresponding stationary distribution is
> $$
> \hat d_\pi = (1 - \hat \gamma) (I - \hat \gamma P^T_\pi)^{-1} d_\mu.
> $$
> **Lemma** discounted covariate operator has unique fixed point: $$\frac{\hat d_\pi}{d_\mu}$$.

$$
Y_{\hat \gamma} c (s') = \mathbb{E}_{s\sim d_\mu, a\sim \mu, s' \sim p(s' \vert s, a)}\left[\frac{\pi(a \vert s; \theta)}{\mu(a \vert s)} c(s) \Big\vert s'\right]
$$

We want
$$
\min_{c} \sum_s d_\mu(s) \left[ Y_{\hat \gamma} c(s) - c(s) \right]^2,
$$

$$
or\quad \min_{\theta_c} \sum_s d_\mu(s) \left[ Y_{\hat \gamma} c(s;\theta_c) - c(s; \theta_c) \right]^2.
$$

Again, we also can use target-c-net to avoid double sampling:
$$
\min_{\theta_c} \sum_s d_\mu(s) \left[ Y_{\hat \gamma} c(s;\theta_{c, target}) - c(s; \theta_c) \right]^2.
$$


## When $$\hat \gamma = \gamma$$

$$
(1 - \gamma) \nabla_\theta J_\mu(\theta) = \mathbb{E}_{s \sim d_\mu(s), a\sim \mu(\cdot \vert s)}\left[\frac{\hat d_\pi(s)}{d_\mu(s)} \frac{\pi(a \vert s; \theta)}{\mu(a \vert s)} \nabla_\theta\log(\pi(a \vert s; \theta)) Q(s, a; \theta) \right]
$$


# SBEED: Convergent Reinforcement Learning with Nonlinear Function Approximation

## The Objective of RL

For optimal Bellman equation:
$$
TV(s) = \max_{\pi(\cdot \vert s)} \sum_{a}\pi(a \vert s) \sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right],
$$

$$
\min_V \sum_s p(s) 
\left\{
TV(s) - V(s)
\right\}^2.
$$

$$
\begin{align*}
&\min_{V} \sum_s p(s) \left\{\max_{\pi(\cdot \vert s)} \sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - V(s)\right\}^2 \\
=& \min_{Q} \sum_s p(s) \left\{\max_{\pi} \sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \vert s') Q(s', a') \right] - \sum_a\pi(a \vert s) Q(s, a)\right\}^2\\
?=& \min_{\theta_Q} \sum_s p(s) \left\{\max_{\theta_\pi} \sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \vert s'; \theta_\pi) Q(s', a';\theta_Q) \right] \\ - \sum_a\pi(a \vert s; \theta_\pi) Q(s, a; \theta_Q)\right\}^2
\end{align*}
$$

$$
L(V) = \sum_s p(s) \left\{\max_{a} \sum_{s'} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - V(s)\right\}^2 \\
\nabla_{V} L(V) = 2 \sum_s p(s) \left\{\max_a \sum_{s'} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - V(s)\right\} (impractiable)
$$

$$
\begin{align*}
L(Q) =& \sum_s p(s) \left\{\max_{\pi} \sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \vert s') Q(s', a') \right] - \sum_a\pi(a \vert s) Q(s, a)\right\}^2\\
=& \sum_s p(s) \left\{\max_{\pi}  \sum_a \pi(a \vert s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \vert s') Q(s', a') \right] - Q(s, a) \right\}\right\}^2 \\
=& \sum_s p(s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \max_{a'} Q(s', a') \right] - Q(s, a) \right\}^2, \quad a = \arg\max_a Q(s,a)
\end{align*}
$$

$$
\begin{align*}
\nabla_Q L(Q) =& 2 \sum_s p(s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \max_{a'} Q(s', a') \right] - Q(s, a) \right\}, \quad a = \arg\max_a Q(s,a)
\end{align*}
$$

$$
\begin{align*}
L(\theta) =& \sum_s p(s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \max_{a'} Q(s', a'; \theta) \right] - Q(s, a; \theta)\right\}^2,\\
\nabla_\theta L(\theta) =& \sum_s p(s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \max_{a'} Q(s', a'; \theta) \right] - Q(s, a; \theta)\right\}\\
&\cdot \left\{\gamma \sum_{s'} p(s' \vert s, a) \nabla_\theta \max_{a'} Q(s', a'; \theta)  - \nabla_{\theta} Q(s, a; \theta)\right\}
\end{align*}
$$

(Hint: from smoothed Bellman equation, we have $$\pi(a \vert s) = \lim_{\lambda \rightarrow 0} \pi_{\lambda}(a \vert s)$$.)

For smoothed Bellman equation:

Smoothed Bellman equation:
$$
T_{\lambda}V(s) = \max_{\pi(\cdot \vert s)}\sum_{a}\pi(a \vert s) \left(\sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]\right)+ \lambda H(\pi(\cdot \vert s)).
$$
If $$H(\pi(\cdot \vert s)) = - \sum_{a} \pi(a \vert s) \log(\pi(a \vert s))$$, then
$$
\pi_{\lambda}(a \vert s) 
= \frac{\exp\{\frac{1}{\lambda} \sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]\}}{\sum_{a'} \exp\{\frac{1}{\lambda} \sum_{s'}p(s' \vert s, a') \left[r(s, a', s') + \gamma V(s') \right]\}}
= \frac{\exp\{\frac{1}{\lambda} Q(s,a)\}}{\sum_{a'} \exp\{\frac{1}{\lambda} Q(s,a')\}}
$$

$$
T_{\lambda} V(s) = \lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]\right\}
$$
The objective becomes:
$$
\begin{align*}
&\min_{V} \sum_s p(s) \left\{T_\lambda V(s) - V(s)\right\}^2\\
=& \min_{V} \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]\right\} - V(s)
\right\}^2\\
=& \min_{V} \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}Q(s,a)\right]\right\} - V(s)
\right\}^2,
\quad s.t. V(s) = {\frac{ \sum_a  Q(s,a) \exp\left\{\frac{1}{\lambda}Q(s, a)\right\} }{ \sum_a \exp\left\{\frac{1}{\lambda}Q(s, a)\right\}}} \\
=& \min_{Q(s,a)} \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}Q(s,a)\right]\right\} - {\frac{ \sum_a Q(s,a) \exp\left\{\frac{1}{\lambda}Q(s, a)\right\} }{ \sum_a \exp\left\{\frac{1}{\lambda}Q(s,a)\right\}}} 
\right\}^2 \\
=&\min_{\hat Q(s,a)}\sum_{s} p(s)\left\{
\log \left\{\sum_{a} \exp\left[\hat Q(s,a)\right]\right\} - {\frac{ \sum_a \hat Q(s,a) \exp\left\{\hat Q(s, a)\right\} }{ \sum_a \exp\left\{\hat Q(s,a)\right\}}} 
\right\}^2, 
\quad \hat Q(s,a) = \frac{1}{\lambda}Q(s,a) \\
=& \min_{\tilde Q(s,a) > 0} \sum_{s} p(s)\left\{
\log \left\{\sum_{a} \tilde Q(s,a)\right\} - {\frac{ \sum_a  \tilde Q(s,a) \log \tilde Q(s,a) }{ \sum_a \tilde Q(s, a)}}
\right\}^2,
\quad \tilde Q(s,a) = \exp\left\{\hat Q(s, a)\right\} \\
=& \min_{\tilde Q(s,a) > 0} \sum_{s} p(s) \left\{\sum_{a} \frac{\tilde Q(s,a)}{\sum_a \tilde Q(s, a)} \cdot \log\left( \frac{\tilde Q(s,a)}{\sum_a \tilde Q(s, a)} \right)\right\}^2
\end{align*}
$$

$$
L(V) =  \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]\right\} - V(s)
\right\}^2,\\
where\quad
\pi_{\lambda}(a \vert s) 
= \frac{\exp\{\frac{1}{\lambda} \sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]\}}{\sum_{a'} \exp\{\frac{1}{\lambda} \sum_{s'}p(s' \vert s, a') \left[r(s, a', s') + \gamma V(s') \right]\}}.
$$

$$
L(\theta) =  \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'; \theta))\right]\right\} - V(s; \theta)
\right\}^2
$$

$$
\begin{align*}
\frac{d L(\theta)}{d \theta} 
= &
\sum_s p(s) \cdot  2\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'; \theta))\right]\right\} - V(s; \theta)
\right\}\\
&\cdot \left\{ \sum_{a} \frac{\exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'; \theta))\right]}{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'; \theta))\right]} \sum_{s'} p(s' \vert s, a) \gamma \frac{d V(s'; \theta)}{d \theta}  - \frac{d V(s; \theta)}{d \theta} \right\} \\
=&\sum_s p(s) \cdot  2\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'; \theta))\right]\right\} - V(s; \theta)
\right\}\\
&\cdot \left\{ \sum_{a} \frac{\exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'; \theta))\right]}{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'; \theta))\right]} \sum_{s'} p(s' \vert s, a) \gamma \frac{d V(s'; \theta)}{d \theta}  - \frac{d V(s; \theta)}{d \theta} \right\}
\end{align*}
$$

$$
L(Q) = \min_{Q} \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma {\frac{ \sum_{a'}  Q(s',a') \exp\left\{\frac{1}{\lambda}Q(s', a')\right\} }{ \sum_{a'} \exp\left\{\frac{1}{\lambda}Q(s', a')\right\}}} )\right]\right\}\\
- {\frac{ \sum_a  Q(s,a) \exp\left\{\frac{1}{\lambda}Q(s, a)\right\} }{ \sum_a \exp\left\{\frac{1}{\lambda}Q(s, a)\right\}}} 
\right\}^2
$$

## The Paper‘s Objective

$$
\begin{align*}
&\min_{V} \sum_s p(s) \left\{T_\lambda V(s) - V(s)\right\}^2\\
=& \min_{V} \sum_{s} p(s)\left\{\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]\right\} - V(s)\right\}^2\\
?&\min_{V} \sum_{s} p(s) \sum_a \pi_\lambda(a \vert s)\left\{\lambda \log \left\{\frac{\exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]}{\pi_\lambda(a \vert s)}\right\} - V(s)\right\}^2\\
=& \min_{V} \sum_{s} p(s) \sum_a \pi_\lambda(a \vert s)\left\{\sum_{s'} p(s'\vert s, a) (r(s, a, s')+ \gamma V(s')) - \lambda\log\pi_{\lambda}(a \vert s) - V(s))\right\}^2
\end{align*}
$$

$$
\min_{V, \pi} l(V, \pi) = \mathbb{E}_{s,a}\left\{ \mathbb{E}_{s' \vert s, a} [r(s,a, s') + \gamma V(s')]  - \lambda \log(\pi(a \vert s)) - V(s) \right\}^2
$$

## Net Structure and Loss Function

$$
\pi_\lambda(a \vert s; \theta) =  \frac{\exp\{\frac{1}{\lambda} Q(s,a; \theta)\}}{\sum_{a'} \exp\{\frac{1}{\lambda} Q(s,a'; \theta)\}}
$$

$$
V(s; \theta) = \sum_{a} Q(s, a; \theta) \pi_{\lambda}(a \vert s; \theta)
$$

$$
L(\theta) = \mathbb{E}_{s,a} \left\{\mathbb{E}_{s' \vert s, a} [r(s, a, s') + \gamma V(s'; \theta)]  - \lambda \log(\pi(a \vert s; \theta)) - V(s;\theta) \right\}^2
$$

### The Objective with Function Approximation

$$
\begin{align*}
\min_{\theta} L(\theta) =& \min_{\theta} \mathbb{E}_{s,a} \left\{\mathbb{E}_{s' \vert s, a} [r(s, a, s') + \gamma V(s'; \theta)]  - \lambda \log(\pi(a \vert s; \theta)) - V(s;\theta) \right\}^2\\
=& \min_{\theta} \mathbb{E}_{s,a} \left\{\mathbb{E}_{s' \vert s, a} [\delta(s, a, s';\theta) - V(s; \theta)] \right\}^2, \\ &\delta(s, a, s';\theta) = r(s, a, s') + \gamma V(s';\theta) - \lambda\log(\pi(a \vert s; \theta))
\end{align*}
$$

$$
\begin{align*}
\min_\theta L(\theta) =& \min_\theta \max_{w} 2 \mathbb{E}_{s, a, s'}\{\nu(s, a; w) [\delta(s, a, s';\theta) - V(s;\theta)]\} - \mathbb{E}_{s, a, s'}\{\nu^2(s, a; w)\}\\
=& \min_\theta \max_{w} \mathbb{E}_{s, a, s'}\{\delta(s, a, s';\theta) - V(s, a; \theta)\}^2
- \mathbb{E}_{s, a, s'}\{\delta(s, a, s'; \theta) - \nu(s,a; w)\}^2
\end{align*}
$$

$$
\nabla_\theta L(\theta) = 2\nu(s, a; w) [\nabla_\theta V(s'; \theta) - \lambda \nabla_\theta \log(\pi(a \vert s; \theta)) - \nabla_\theta V(s; \theta)]
$$

## Appendix: Soft Optimal Bellman equation

Smoothed Bellman equation:
$$
T_{\lambda}V(s) = \max_{\pi(\cdot \vert s)}\sum_{a}\pi(a \vert s) 
\left(
\sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]
\right)+ \lambda H(\pi(\cdot \vert s)).
$$
If $$H(\pi(\cdot \vert s)) = - \sum_{a} \pi(a \vert s) \log(\pi(a \vert s))$$, then
$$
T_{\lambda} V(s) = \lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]\right\}
$$
**proof**:
$$
\max_{\pi(\cdot \vert s)\in \Pi(s)}\sum_{a}\pi(a \vert s) 
\left(
\sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]
- \lambda \log \pi(a \vert s))
\right),\\
s.t. \sum_a \pi(a \vert s) = 1.
$$

$$
\begin{align*}
&\max_{\pi(\cdot \vert s) \succeq0} \min_{k_s \ne 0}
\sum_{a}\pi(a \vert s) 
\left(
Q(s,a) - \lambda \log \pi(a \vert s))
\right) + k_s\left(1 - \sum_{a} \pi(a \vert s)\right)
\\
\le& \min_{k_s \ne 0} \max_{\pi(\cdot \vert s)\succeq0}
\sum_{a}\pi(a \vert s) 
\left(
Q(s, a) - \lambda \log \pi(a \vert s))
\right) + k_s\left(1 - \sum_{a} \pi(a \vert s)\right)
\end{align*}
$$

We solve the dual problem:
$$
\begin{align*}
&Q(s, a) - \lambda(1 + \log \pi(a\vert s)) - k_s = 0\\
\Rightarrow& \pi(a\vert s)exp(1 + k_s/\lambda) = \exp\left\{\frac{1}{\lambda}Q(s, a)\right\}\\
\Rightarrow& exp(1 + k_s/\lambda) = \sum_a \exp\left\{\frac{1}{\lambda}Q(s, a)\right\} \\
\Rightarrow& 1 + k_s/\lambda = \log\left\{\sum_a \exp\left[\frac{1}{\lambda} Q(s, a)\right]\right\}\\
\Rightarrow& \pi(a\vert s) = {\frac{ \exp\left\{\frac{1}{\lambda}Q(s, a)\right\} }{ \sum_a \exp\left\{\frac{1}{\lambda}Q(s, a)\right\}}}
\end{align*}
$$

$$
\sum_a \pi(a\vert s) \left[Q(s, a) - \lambda(1 + \log \pi(a\vert s)) - k_s\right] = 0\\
\Rightarrow
\min_{k_s \ne 0} \max_{\pi(\cdot \vert s)\succeq0}
\sum_{a}\pi(a \vert s) 
\left(
Q(s, a) - \lambda \log \pi(a \vert s))
\right) + k_s\left(1 - \sum_{a} \pi(a \vert s)\right)\\
= k_s + \lambda\sum_{a} \pi(a \vert s)
= k_s + \lambda = \lambda \log\left\{\sum_a \exp\left[\frac{1}{\lambda} Q(s, a)\right]\right\}
$$


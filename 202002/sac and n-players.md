# Soft Actor Critic

## 1.0 Soft Optimal Bellman Equation and Objective

- **Definition**: Optimal Bellman Equation:
  $$
  \forall V \in \mathbb{R}^{\vert S \vert}, \quad 
  TV(s) := \max_{\pi \in \Pi} \sum_{a \in \mathcal{A}} \pi(a \vert s) \sum_{s'\in\mathcal{S}} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')]
  $$

- **Definition**: Soft Optimal Bellman Equation:
  $$
  \forall V \in \mathbb{R}^{\vert S \vert}, \quad 
  T_{\alpha} V(s) := \max_{\pi \in \Pi} \sum_{a \in \mathcal{A}} \pi(a \vert s) \sum_{s'\in\mathcal{S}} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')]
  -\alpha \sum_{a\in \mathcal{A}}\pi(a \vert s) \ln \pi(a \vert s)
  $$

- **Lemma**:
  $$
  T_\alpha V(s) = \sum_{a \in \mathcal{A}} \pi_V(a \vert s) \left\{\sum_{s'\in\mathcal{S}} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - \alpha \ln \pi_V(a \vert s) \right\}
  $$
  where
  $$
  \pi_V := {\frac{ \exp\left\{\frac{1}{\alpha}\sum_{s'\in\mathcal{S}} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')]\right\} }{ \sum_{a'} \exp\left\{\frac{1}{\alpha}\sum_{s'\in\mathcal{S}} p(s' \vert s, a') [r(s, a', s') + \gamma V(s')]\right\}}}
  $$

- **For simplicity**:

  **Definition**: $$Q_V(s, a) = \sum_{s'\in\mathcal{S}} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')]$$.

  Then, we have
  $$
  \pi_V = {\frac{ \exp\left\{\frac{1}{\alpha}Q_V(s, a)\right\} }{ \sum_{a'} \exp\left\{\frac{1}{\alpha}Q_V(s, {a'})\right\}}}
  $$
  and
  $$
  T_\alpha V(s) = \sum_{a \in \mathcal{A}} \pi_V(a \vert s) [Q_V(s, a) - \alpha\ln \pi_V(a \vert s)]
  $$

- We can construct the objective the sac using:
  $$
  J(V) = \frac{1}{2} \sum_s p(s) \left(T_\alpha V(s) - V(s)\right)^2,
  $$
  where $$p$$ can be any distribution. 

  In another form,
  $$
  J(V) = \frac{1}{2} \sum_s p(s) \left(\sum_{a \in \mathcal{A}} \pi_V(a \vert s) [Q_V(s, a) - \alpha\ln \pi_V(a \vert s)] - V(s)\right)^2,
  $$
  where
  $$
  Q_V(s, a) = \sum_{s'\in\mathcal{S}} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')], \quad
  \pi_V = {\frac{ \exp\left\{\frac{1}{\alpha}Q_V(s, a)\right\} }{ \sum_{a'} \exp\left\{\frac{1}{\alpha}Q_V(s, a')\right\}}}
  $$

## 1.1 True Objectives

$$
\begin{cases}
J_1(\theta_Q) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left\{\frac{1}{2} (\sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V(s';\theta_{V_{target}})\right) - Q(s, a; \theta_Q) )^2 \right\}\\

J_2(\theta_\pi) = \mathbb{E}_{s \sim \mathcal{D}} D_{KL}\left(\pi(\cdot\vert s; \theta_\pi) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s; \theta_Q))}{Z(s;\theta_Q)}\right)  \\= \mathbb{E}_{s \sim \mathcal{D}, a\sim\pi(a\vert s;\theta_\pi)} \left\{\log \pi(a\vert s; \theta_\pi) - \frac{1}{\alpha} Q(s, a; \theta_Q) + \log Z(s;\theta_Q)\right\} \\

J_3(\theta_V) = \mathbb{E}_{s \sim \mathcal{D}} \left\{\frac{1}{2} \left(V(s;\theta_V) - \mathbb{E}_{a\sim\pi(a \vert s; \theta_\pi)} \left[Q(s, a; \theta_Q)  - \alpha \log(\pi(a \vert s; \theta_\pi))  \right]\right)^2 \right\}\\

J_4(\theta_{V_{target}}) = \frac{1}{2}\Vert \theta_{V} - \theta_{V_{target}}\Vert^2
\end{cases}
$$

## 1.2 SAC Objectives

$$
\begin{cases}
J_1(\theta_Q) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left\{\frac{1}{2} (\sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V(s';\theta_{V_{target}})\right) - Q(s, a; \theta_Q) )^2 \right\}\\

J_2(\theta_\pi) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon\sim \mathcal{N}(0, 1)} \left\{
[\log \pi(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi})\vert s; \theta_\pi)
- \frac{1}{\alpha} Q(s, \tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}); \theta_Q)]\right\} \\

J_3(\theta_V) = \mathbb{E}_{s \sim \mathcal{D}} \left\{\frac{1}{2} \left(V(s;\theta_V) - \mathbb{E}_{\epsilon \sim \mathcal{N}(0, 1)} \left[Q(s, \tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}); \theta_Q)  - \alpha \log(\pi(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}) \vert s; \theta_\pi))  \right]\right)^2 \right\}\\

J_4(\theta_{V_{target}}) = \frac{1}{2}\Vert \theta_{V} - \theta_{V_{target}}\Vert^2
\end{cases}
$$

**Proof**:

SAC uses this to approximate KL-divergence:
$$
\int_\epsilon p(\epsilon) \left(\log \pi(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi})\vert s; \theta_\pi)
- \frac{1}{\alpha} Q(s, \tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}); \theta_Q)\right) d\epsilon.
$$
The original kl-divergence is
$$
\int_a \pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s; \theta_Q)\right) da.
$$
We replace the variable $$a$$ with $$\epsilon$$:
$$
\begin{align*}
&D_{KL}\left(\pi(\cdot\vert s; \theta_\pi) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s; \theta_Q))}{Z(s;\theta_Q)}\right) \\
=&\int_a \pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s; \theta_Q)\right) da\\
=& \int_\epsilon \pi(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}) \vert s; \theta_\pi) \left(\ln \pi(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}) \vert s;\theta_\pi)\\
- \frac{1}{\alpha} Q(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}) \vert s; \theta_Q) + \ln Z(s; \theta_Q)\right) d\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi})\\
=& \int_\epsilon p(\epsilon) \left(\ln \pi(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}) \vert s; \theta_\pi) 
- \frac{1}{\alpha} Q(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}) \vert s; \theta_Q) + \ln Z(s; \theta_Q)\right) d\epsilon,
\end{align*}
$$
where
$$
\pi(\tanh(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi}) \vert s; \theta_\pi) 
= \frac{p(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi} \vert s; \theta_\pi) }{1 - \tanh^2(\sigma_{\theta_\pi} \epsilon + \mu_{\theta_\pi})}.
$$

$$
\ln p(a \vert s) - \sum^D_{i=1}\ln (1 - a_i^2).
$$

We use $$a \approx \tanh a$$.
$$
\begin{align*}
&2(\ln 2 - x - \ln(1 + e^{-2x}))\\
=& 2\ln\left(\frac{2}{e^x + e^{-x}}\right)\\
=& \ln \left\{1 - \tanh^2 x\right\}
\end{align*}
$$

## 1.3 Double Q-Net Objective

$$
\begin{cases}J_1(\theta_{Q_1}) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left\{\frac{1}{2} (\sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V(s';\theta_{V_{target}})\right) - Q_1(s, a; \theta_{Q_1}) )^2 \right\}\\J_5(\theta_{Q_2}) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left\{\frac{1}{2} (\sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V(s';\theta_{V_{target}})\right) - Q_2(s, a; \theta_{Q_2}) )^2 \right\}\\J_2(\theta_\pi) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}(0, 1)} \left\{\log(\pi(f(s; \epsilon, \theta_\pi) \vert s)) - \frac{1}{\alpha} \min\{Q_1(s, f(s; \epsilon, \theta_\pi);\theta_{Q_1}), Q_2(s, f(s;\epsilon, \theta_\pi);\theta_{Q_2}) \}\right\}\\J_3(\theta_V) = \mathbb{E}_{s \sim \mathcal{D}} \left\{\frac{1}{2} \left(V(s;\theta_V) - \mathbb{E}_{\epsilon \sim \mathcal{N}(0, 1)} \left[Q(s, f(s; \epsilon, \theta_\pi); \theta_Q)  - \alpha \log(\pi(f(s; \epsilon, \theta_\pi) \vert s; \theta_\pi))  \right]\right)^2 \right\}\\J_4(\theta_{V_{target}}) = \frac{1}{2}\Vert \theta_{V} - \theta_{V_{target}}\Vert^2\end{cases}
$$

## 1.4 Dynamic matrix

$$
\xi =
\begin{bmatrix}
\nabla_{\theta_Q} J_1(\theta_Q)\\
\nabla_{\theta_\pi} J_2(\theta_\pi)\\
\nabla_{\theta_V} J_3(\theta_V) \\
\nabla_{\theta_{V_{target}}} J_4(\theta_{V_{target}})
\end{bmatrix},\quad 
H=
\begin{bmatrix}
\nabla^2_{\theta_Q} J_1(\theta_Q) & 0 & 0 & \nabla^2_{\theta_Q, \theta_{V_{target}}} J_1(\theta_Q)\\
\nabla^2_{\theta_\pi, \theta_{Q}} J_2(\theta_\pi) & \nabla^2_{\theta_\pi} J_2(\theta_\pi) & 0 & 0\\
\nabla^2_{\theta_V,\theta_Q} J_3(\theta_V) & \nabla^2_{\theta_V, \theta_{\pi}} J_3(\theta_V) & \nabla^2_{\theta_V} J_3(\theta_V) & 0 \\
\nabla^2_{\theta_{V_{target}}, \theta_Q} J_4(\theta_{V_{target}}) & 0 & 0 & \nabla^2_{\theta_{V_{target}}} J_4(\theta_{V_{target}})
\end{bmatrix}
$$

$$
H^T=
\begin{bmatrix}
\nabla^2_{\theta_Q} J_1(\theta_Q) & \nabla^2_{\theta_{Q}, \theta_\pi} J_2(\theta_\pi) & \nabla^2_{\theta_Q, \theta_V} J_3(\theta_V) & \nabla^2_{\theta_Q, \theta_{V_{target}}} J_4(\theta_{V_{target}})\\

0 & \nabla^2_{\theta_\pi} J_2(\theta_\pi) & \nabla^2_{\theta_{\pi}, \theta_V} J_3(\theta_V) & 0\\

0 & 0 & \nabla^2_{\theta_V} J_3(\theta_V) & 0 \\

\nabla^2_{\theta_{V_{target}}, \theta_Q} J_1(\theta_Q) & 0 & 0 & \nabla^2_{\theta_{V_{target}}} J_4(\theta_{V_{target}})
\end{bmatrix}
$$

$$
\xi = \xi + H^T \xi
$$

If the transition $$(s, a) \rightarrow s'$$ is deterministic, then $$H^T \xi$$ doesn't need double sampling.
$$
A = \frac{1}{2}(H - H^T),\quad 
\xi = \xi + \lambda A^T \xi = \xi + \frac{\lambda}{2}(H - H^T)^T \xi = \xi + \frac{\lambda}{2}(H^T - H) \xi
$$

### 1.4.1 Problem

问题：$$\xi$$ 与 $$H\xi$$、$$A\xi$$ 不在一个数量级上。

解决方式一： $$H\xi$$ 与 $$A\xi$$ 乘以某个系数。

解决方式二：所有梯度都归一化。

## 1.5 Deviation of General KL

$$
\begin{align*}
&D_{KL}\left(\pi(\cdot\vert s) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s))}{Z(s)}\right)\\
=& \int_a \pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s)\right) da
\end{align*}
$$

$$
\begin{align*}
&\nabla_{\theta_\pi} D_{KL}\left(\pi(\cdot\vert s; \theta_\pi) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s))}{Z(s)}\right)\\
=& \int_a \nabla_{\theta_\pi}\pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s)\right) +
\pi(a \vert s; \theta_\pi) \nabla_{\theta_\pi} \ln \pi(a \vert s; \theta_\pi) da\\
=& \int_a \nabla_{\theta_\pi}\pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s) + 1\right)da\\
=& \int_a \nabla_{\theta_\pi}\pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q)\right)da\\
=& \int_a \pi(a \vert s; \theta_\pi)\nabla_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q)\right)da
\end{align*}
$$

$$
\begin{align*}
&\nabla^2_\theta D_{KL}\left(\pi(\cdot\vert s) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s))}{Z(s)}\right)\\
=& \int_a \pi(a \vert s; \theta_\pi)\nabla_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q)\right)da\\
=& \int_a \nabla_{\theta_\pi} \pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q)\right)\nabla^T_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi)\\
& +\pi(a \vert s; \theta_\pi) \nabla_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi) \nabla^T_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi)\\
& +\pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q)\right) \nabla^2_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi) da\\
=& \int_a \pi(a \vert s; \theta_\pi)
\left[\left(1 + \ln \pi(a \vert s; \theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q) \right)\nabla_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi) \nabla^T_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi) \\
+\left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q)\right) \nabla^2_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi) \right] da
\end{align*}
$$

## 1.6 Deviation of Gaussian KL

$$
\begin{align*}
&D_{KL}\left(\pi(\cdot\vert s) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s))}{Z(s)}\right)\\
=& \int_a \pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s)\right) da\\
=& \int_a \pi(a \vert s; \theta_\pi) \ln \pi(a \vert s;\theta_\pi) da + \int_a \pi(a \vert s; \theta_\pi) \left(- \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s)\right) da\\
=& \int_a \pi(a \vert s; \theta_\pi)\left(-\frac{(a - \mu_{\theta_\pi})^2}{2 \sigma_{\theta_\pi}^2} - \ln(\sqrt{2\pi}\sigma_{\theta_\pi})\right) da + \int_a \pi(a \vert s; \theta_\pi) \left(- \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s)\right) da\\
=&-\frac{1}{2} -\ln(\sqrt{2\pi}\sigma_{\theta_\pi}) + \int_a \pi(a \vert s; \theta_\pi) \left(- \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s)\right) da
\end{align*}
$$

$$
\begin{align*}
&\nabla_{\theta_\pi} D_{KL}\left(\pi(\cdot\vert s) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s))}{Z(s)}\right)\\
=& -\nabla_{\theta_\pi} \ln\sigma_{\theta_\pi} 
- \frac{1}{\alpha} \int_a \pi(a\vert s; \theta_\pi) \nabla_{\theta_\pi}\ln(\pi(a \vert s; \theta_\pi) Q(a \vert s; \theta_Q)) da
\end{align*}
$$

$$
\begin{align*}
&\nabla^2_{\theta_\pi} D_{KL}\left(\pi(\cdot\vert s) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s))}{Z(s)}\right)\\
=& -\nabla^2_{\theta_\pi} \ln \sigma_{\theta_\pi} 
- \frac{1}{\alpha}\int_a \left(\nabla_{\theta_\pi}\pi(a \vert s; \theta_\pi) \nabla^T_{\theta_\pi}\ln(\pi(a \vert s; \theta_\pi) Q(a \vert s; \theta_Q))\\
+ \pi(a\vert s; \theta_\pi) \nabla^2_{\theta_\pi}\ln(\pi(a \vert s; \theta_\pi) Q(a \vert s; \theta_Q)) \right) da \\
=& -\nabla^2_{\theta_\pi} \ln \sigma_{\theta_\pi} 
- \frac{1}{\alpha}\int_a \pi(a \vert s; \theta_\pi) \left(\nabla_{\theta_\pi}\ln\pi(a \vert s; \theta_\pi) \nabla^T_{\theta_\pi}\ln(\pi(a \vert s; \theta_\pi)\\
+ \nabla^2_{\theta_\pi} \ln(\pi(a \vert s; \theta_\pi)\right) Q(a \vert s; \theta_Q) da
\end{align*}
$$

## 1.7 Deviation of Tanh-Gaussian KL

$$
\begin{align*}
&D_{KL}\left(\pi(\cdot\vert s) \Big\Vert \frac{\exp(\frac{1}{\alpha} Q(\cdot\vert s))}{Z(s)}\right)\\
=& \int_a \pi(a \vert s; \theta_\pi) \left(\ln \pi(a \vert s;\theta_\pi) - \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s)\right) da\\
=& \int_a \pi(a \vert s; \theta_\pi) \ln \pi(a \vert s;\theta_\pi) da + \int_a \pi(a \vert s; \theta_\pi) \left(- \frac{1}{\alpha} Q(a \vert s; \theta_Q) + \ln Z(s)\right) da
\end{align*}
$$

$$
\begin{align*}
& \int^1_{-1} \pi(a \vert s; \theta_\pi) \ln \pi(a \vert s;\theta_\pi) da\\
=& \int^{\infty}_{-\infty} \frac{p(y\vert s; \theta_\pi)}{1-\tanh^2 y}
\left[\ln p(y  \vert s; \theta_\pi) - \ln(1-\tanh^2 y)\right] d \tanh(y)\\
=& \int^\infty_{-\infty}p(y \vert s; \theta_\pi)[\ln p(y \vert s; \theta_\pi) - \ln(1 - tanh^2 y)] d y \\
=& -\frac{1}{2} - \ln(\sqrt{2\pi} \sigma_{\theta_\pi}) 
- \int^{\infty}_{-\infty} p(y \vert s; \theta_\pi) \ln(1 - tanh^2 y) dy \\
=& -\frac{1}{2} - \ln(\sqrt{2\pi} \sigma_{\theta_\pi}) - \int^{\infty}_{-\infty} p(y \vert s; \theta_\pi) \ln\frac{4}{(e^y + e^{-y})^2} dy \\
=& -\frac{1}{2} - \ln(\sqrt{2\pi} \sigma_{\theta_\pi}) - \ln 4 
+ 2 \int^{\infty}_{-\infty}p(y \vert s; \theta_\pi)\ln(e^y + e^{-y})dy
\end{align*}
$$


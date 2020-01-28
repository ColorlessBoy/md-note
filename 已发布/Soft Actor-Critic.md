# Soft Actor-Critic

## 1. Preliminaries

> **Definition**:
> $$
> J(\pi) =  \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(a_0 \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(a_1 \vert s_1)) \\
> + \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
> $$
>
> $$
> Q^\pi_{soft}(s_0, a_0) = \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(a_0 \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(a_1 \vert s_1)) \\
> + \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
> $$
>
> $$
> V^\pi_{soft}(s_0) = \sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(a_0 \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(a_1 \vert s_1)) \\
> + \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
> $$

### 1.1 V-Based

> **Definition**: (Soft Bellman Equation) $$\forall V \in \mathbb{R}^{\vert S \vert}$$:
> $$
> Q_V(s,a) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V(s') \right)
> $$
>
> $$
> T^\pi V(s) = \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V(s') \right) = \langle\pi(\cdot\vert s_0), Q_V(s_0, \cdot)\rangle
> $$
>
> $$
> \begin{align*}
> T^\pi_{soft}V(s) =& \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \mathcal{H}(\pi(a \vert s)) + \gamma V(s') \right) \\
> =& T^\pi V(s) + \sum_a \pi(a \vert s)\mathcal{H}(\pi(a \vert s))
> \end{align*}
> $$
>
> $$
> Q_{V, soft} (s, a) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \mathcal{H}(\pi(a \vert s)) + \gamma V(s') \right) = Q_V(s, a) + \mathcal{H}(\pi(a \vert s))
> $$

> **Definition**: (Soft Optimal Bellman Equation) $$\forall V \in \mathbb{R}^{\vert S \vert}$$,
> $$
> \begin{align*}
> T_{soft} V(s) =& \max_{\pi} T^\pi_{soft} V = \langle\pi(\cdot\vert s_0), Q_V(s_0, \cdot)\rangle + \sum_a \pi(a\vert s) \mathcal{H}(\pi(a \vert s)) \\
> =& \langle\pi(\cdot\vert s_0), Q_V(s_0, \cdot)\rangle - \alpha \sum_{a} \pi(a \vert s) \log(\pi(a\vert s)) \\
> =& \alpha \log \left\{\sum_{a} \exp\left[\frac{1}{\alpha}Q_V(s, a)\right]\right\}
> \quad Or\quad 
> \alpha \log \left\{\int_a \exp\left[\frac{1}{\alpha}Q_V(s, a)\right] da\right\}
> \end{align*}
> $$
> The corresponding optimal policy is:
> $$
> \pi^*_{V, soft}(a \vert s) = {\frac{ \exp\left\{\frac{1}{\alpha}Q_V(s, a)\right\} }{ \sum_a \exp\left\{\frac{1}{\alpha}Q_V(s, a)\right\}}}
> \quad Or \quad 
> {\frac{ \exp\left\{\frac{1}{\alpha}Q_V(s, a)\right\} }{ \int_a \exp\left\{\frac{1}{\alpha}Q_V(s, a)\right\} da}}
> $$
> **Lemma**:
> $$
> \log \pi^*_{V, soft}(a \vert s) = \frac{1}{\alpha} [Q_V(s, a) - T_{soft} V(s)]
> $$

### 1.2 Q-Based

> **Definition**: (Soft Bellman Equation) $$\forall Q \in \mathbb{R}^{\vert S \vert \times \vert A \vert}$$:
> $$
> \pi_Q(a \vert s) = {\frac{ \exp\left\{\frac{1}{\alpha}Q(s, a)\right\} }{ \sum_a \exp\left\{\frac{1}{\alpha}Q(s, a)\right\}}}
> \quad Or \quad
> {\frac{ \exp\left\{\frac{1}{\alpha}Q(s, a)\right\} }{ \int_a \exp\left\{\frac{1}{\alpha}Q(s, a)\right\} da}}
> $$
>
> $$
> V_{Q}(s) = \sum_a \pi_Q(a\vert s) Q(s, a)
> \quad Or \quad
> \int_a \pi_Q(a \vert s) Q(s, a) da
> $$
>
> $$
> \begin{align*}
> T^\pi V_{Q}(s) =& \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V_{Q}(s') \right)\\
> =& \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma \sum_{a'} \pi_Q(a' \vert s') Q(s', a') \right)
> \end{align*}
> $$
>
> $$
> \begin{align*}
> T^\pi_{soft}V_{Q}(s) =& \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \mathcal{H}(\pi(a \vert s)) + \gamma V_{Q}(s') \right) \\
> =& T^\pi V_{Q}(s) + \sum_a \pi(a \vert s)\mathcal{H}(\pi(a \vert s))
> \end{align*}
> $$
>

> **Definition**: (Soft Optimal Bellman Equation) $$\forall Q \in \mathbb{R}^{\vert S \vert \times \vert A\vert}$$,
> $$
> \begin{align*}
> T_{soft} V_{Q}(s) =& \max_{\pi} T^\pi_{soft} V_{Q}(s)\\
> =& \max_{\pi} \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V_Q(s') \right)
> - \alpha\sum_a \pi(a \vert s) \log(\pi(a\vert s))\\
> =& \alpha \log \left\{\sum_{a} \exp\left[\frac{1}{\alpha}Q_{V_Q}(s, a)\right]\right\}
> \quad Or\quad 
> \alpha \log \left\{\int_a \exp\left[\frac{1}{\alpha}Q_{V_Q}(s, a)\right] da\right\}
> \end{align*}
> $$
> The corresponding optimal policy is:
> $$
> \pi^*_{Q, soft}(a \vert s) = {\frac{ \exp\left\{\frac{1}{\alpha}Q_{V_{Q}}(V_s, a)\right\} }{ \sum_a \exp\left\{\frac{1}{\alpha}Q_{V_{Q}}(s, a)\right\}}}
> \quad Or \quad 
> {\frac{ \exp\left\{\frac{1}{\alpha}Q_{V_{Q}}(s, a)\right\} }{ \int_a \exp\left\{\frac{1}{\alpha}Q_{V_{Q}}(s, a)\right\} da}}
> $$
> where
> $$
> Q_{V_{Q}}(s) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma \sum_{a'} \pi_Q(a' \vert s') Q(s', a') \right)
> $$
> **Lemma**:
> $$
> \log \pi^*_{Q, soft}(a \vert s) = \frac{1}{\alpha} [Q_{V_{Q}}(s, a) - T_{soft} V_{Q}(s)]
> $$

### 1.3 Another Bellman Equation

$$
T^\pi Q(s, a) = \mathbb{E}_{a \sim \pi(\cdot \vert s), s' \sim p(s' \vert s, a)}\left\{r(s, a, s') + \gamma \mathbb{E}_{a' \sim \pi(\cdot \vert s')} \left[Q(s', a')  - \alpha \log(\pi(a' \vert s'))  \right] \right\}
$$

## 2. Algorithm

### 2.1 Soft Policy Iteration

$$
\pi_{t+1} = \arg\min_{\pi} D_{KL}(\pi \Vert \pi^*_{Q_t, soft})
$$

### 2.2 Soft Actor-Critic

The algorithm is a V-based method.

- First we have three net: $$V(s; \theta_V)$$, $$Q(s, a; \theta_Q)$$ and $$\pi(a \vert s; \theta_\pi)$$.

- We want $$Q(s, a; \theta_Q) = Q_V = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V(s';\theta_V) \right)$$;
  $$
  J(\theta_Q) = \mathbb{E}_{(s, a) \sim \mathcal{D}} \left\{\frac{1}{2} (\sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \gamma V(s';\theta_V)\right) - Q(s, a; \theta_Q) )^2 \right\};
  $$

- We want $$\pi(a \vert s; \theta_\pi) = \pi^*_{Q, soft}(a \vert s) = \frac{exp(\frac{1}{\alpha}Q(s, a; \theta_Q))}{\sum_{a'} \exp(\frac{1}{\alpha}Q(s, a'; \theta_Q))}$$;
  $$
  \begin{align*}
  J(\theta_\pi) =& \mathbb{E}_{s \sim \mathcal{D}}\left\{D_{KL} \left(\pi(\cdot \vert s; \theta_\pi) \Vert \pi^*_{Q, soft}(s, \cdot) \right)\right\} \\
  =& \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi(\cdot \vert s;\theta_\pi)} \left\{\log(\pi(a \vert s;\theta_\pi)) - log(\pi^*_{Q, soft}(a \vert s))\right\}\\
  =& \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi(\cdot \vert s;\theta_\pi)} \left\{\log(\pi(a \vert s;\theta_\pi)) - \frac{1}{\alpha} Q(s, a;\theta_Q) + \log\left(\sum_a \exp\left\{\frac{1}{\alpha} Q(s, a; \theta_Q)\right\}\right)\right\}
  \end{align*}
  $$
  If we use Gaussian distribution in continuous action space:
  $$
  J(\theta_\pi) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}(0, 1)} \left\{\log(\pi(f(s; \epsilon, \theta_\pi) \vert s)) - \frac{1}{\alpha} Q(s, f(s; \epsilon, \theta_\pi);\theta_Q) + \log\left(\sum_a \exp\left\{\frac{1}{\alpha} Q(s, a; \theta_Q)\right\}\right)\right\}
  $$

  $$
  \nabla_{\theta_\pi} J(\theta_\pi) = \nabla_{\theta_\pi}\log(\pi(f(s; \epsilon, \theta_\pi) \vert s)) - \frac{1}{\alpha} \nabla_{\theta_\pi} f(s; \epsilon, \theta_\pi) \nabla_a Q(s,a; \theta_{Q})\vert_{a = f(s; \epsilon, \theta_\pi)}
  $$

  **Need more thinking.**

- We want $$V(s; \theta_V) = T^\pi_{soft} V(s; \theta_V) = \mathbb{E}_{a \sim \pi(\cdot \vert s; \theta_\pi)} \left[Q(s, a; \theta_Q)  - \alpha \log(\pi(a \vert s; \theta_\pi))  \right]$$;

$$
\begin{align*}
J(\theta_V) =& \mathbb{E}_{s \sim \mathcal{D}} \left\{\frac{1}{2} \left(V(s;\theta_V) - \mathbb{E}_{a \sim \pi(\cdot \vert s; \theta_\pi)} \left[Q(s, a; \theta_Q)  - \alpha \log(\pi(a \vert s; \theta_\pi))  \right]\right)^2 \right\}
\end{align*}
$$

### 2.3 Another Soft Actor-Critic

It's based on Q-based method.

- Critic: We want $$Q(s, a; \theta_Q) = \mathbb{E}_{s, a, s'\sim \mathcal{D}}[r(s, a, s') + \gamma\mathbb{E}_{a' \sim \pi(\cdot \vert s'; \theta_\pi)} ( Q(s', a'; \theta_Q) - \alpha \log\pi(a' \vert s'; \theta_\pi))]$$;
- Actor: And $$\pi(a \vert s; \theta_\pi) = \frac{Q(s, a; \theta_Q)}{\sum_{a'} Q(s, a'; \theta_Q)}$$.

## 3. Network Trick

- $$mean, std = Net(state)$$;
- $$normal = Normal(mean, std)$$;
- $$y \sim normal$$;
- $$a = \tanh(y) \Rightarrow y=\tanh^{-1} (a) = \frac{1}{2} \ln\frac{1+a}{1-a}$$;
- $$p_{a}(a) = p_y(\tanh^{-1}(a)) * \frac{\mathrm {d} \tanh^{-1}(a)}{\mathrm{d} a} = p_y(y) \frac{1}{1-\tanh^2 y}$$.

### 3.1 KL-Divergence

$$
\begin{align*}
&D_{KL} \{ p_a(a) \Vert q_a(a)\} \\
=& \int_{-1}^{+1} p_a(a;\mu_p, \sigma_p) [\ln(p_a(a; \mu_p, \sigma_p)) - \ln(p_a(a;\mu_q, \sigma_q)) ]da\\
=&\int_{-\infty}^{\infty} \frac{p_y(y)}{1-\tanh^2 y}
\left[\ln (p_y(y;\mu_p, \sigma_p)) - \ln(p_y(y;\mu_q, \sigma_q))\right] d \tanh(y) \\
=& \int_{-\infty}^{\infty} p_y(y) [\ln(p_y(y;\mu_p, \sigma_p)) - \ln(p_y(y; \mu_q, \sigma_q))] dy\\
=& D_{KL}\{p_y(y) \Vert q_y(y)\}
\end{align*}
$$



[1] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,” *arXiv:1801.01290 [cs, stat]*, Aug. 2018.


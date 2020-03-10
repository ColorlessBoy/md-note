# Soft Q-Learning

- Continuous action spaces;
- Infinite-horizon MDP.

## 1. Objective

### 1.1 Normal Reinforcement Learning Objective

$$
\max_{\pi} \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) \\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \left(r(s_1, a_1, s_2)+ \\
+ \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2)(\cdots)\right)\right)
$$

### 1.2 Entropy Based Reinforcement Learning Objective

$$
\max_{\pi} \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(\cdot \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(\cdot \vert s_1)) \\
+ \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
$$

$$
\max_{\pi} \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(a_0 \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(a_1 \vert s_1)) \\
+ \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
$$

## 2. Preliminaries

### 2.1 V Based

> **Definition**:
> $$
> J(\pi) =  \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(\cdot \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(\cdot \vert s_1)) \\
> + \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
> $$
>
> $$
> Q^\pi_{soft}(s_0, a_0) = \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(\cdot \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(\cdot \vert s_1)) \\
> + \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
> $$
>
> $$
> V^\pi_{soft}(s_0) = \sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(\cdot \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(\cdot \vert s_1)) \\
> + \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
> $$

> **Lemma**:
> $$
> J(\pi) = \sum_{s_0} p_0(s_0) V^\pi_{soft}(s_0) = \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) Q^\pi_{soft}(s_0, a_0)
> $$
>
> $$
> V^{\pi}_{soft}(s_0) =  \sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(\cdot \vert s_0)) + V^\pi_{soft} (s_1) \right)
> $$

> **Definition**: (Soft Bellman Equation) $$\forall V \in \mathbb{R}^{\vert S \vert}$$:
>$$
> Q_V(s,a) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V(s') \right)
> $$
> 
> $$
> T^\pi V(s) = \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V(s') \right) = \langle\pi(\cdot\vert s_0), Q_V(s_0, \cdot)\rangle
> $$
> 
>$$
> T^\pi_{soft}V(s) = \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \mathcal{H}(\pi(\cdot \vert s)) + V(s') \right) = T^\pi V(s) + \mathcal{H}(\pi(\cdot \vert s))
> $$
> 
>$$
> Q_{V, soft} (s, a) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \mathcal{H}(\pi(\cdot \vert s)) + V(s') \right) = Q_V(s, a) + \mathcal{H}(\pi(\cdot \vert s))
> $$

> **Definition**: (Soft Optimal Bellman Equation) $$\forall V \in \mathbb{R}^{\vert S \vert}$$
> $$
> \begin{align*}
> T_{soft} V(s) =& \max_{\pi} T^\pi_{soft} V = \langle\pi(\cdot\vert s_0), Q_V(s_0, \cdot)\rangle + \mathcal{H}(\pi(\cdot \vert s)) \\
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

### I am trying another view: harmless but more convincing.

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

> **Definition**: (Soft Bellman Equation) $$\forall V \in \mathbb{R}^{\vert S \vert}$$:
> $$
> Q_V(s,a) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V(s') \right)
> $$
>
> $$
> T^\pi V(s) = \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V(s') \right) = \langle\pi(\cdot\vert s_0), Q_V(s_0, \cdot)\rangle
> $$
>
> $$
> \begin{align*}
> T^\pi_{soft}V(s) =& \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \mathcal{H}(\pi(a \vert s)) + V(s') \right) \\
> =& T^\pi V(s) + \sum_a \pi(a \vert s)\mathcal{H}(\pi(a \vert s))
> \end{align*}
> $$
>
> $$
> Q_{V, soft} (s, a) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \mathcal{H}(\pi(a \vert s)) + V(s') \right) = Q_V(s, a) + \mathcal{H}(\pi(a \vert s))
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

### 3.2 Q-Based

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
> T^\pi V_{Q}(s) =& \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V_{Q}(s') \right)\\
> =& \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \sum_{a'} \pi_Q(a' \vert s') Q(s', a') \right)
> \end{align*}
> $$
>
> $$
> \begin{align*}
> T^\pi_{soft}V_{Q}(s) =& \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \mathcal{H}(\pi(a \vert s)) + V_{Q}(s') \right) \\
> =& T^\pi V_{Q}(s) + \sum_a \pi(a \vert s)\mathcal{H}(\pi(a \vert s))
> \end{align*}
> $$
>

> **Definition**: (Soft Optimal Bellman Equation) $$\forall Q \in \mathbb{R}^{\vert S \vert \times \vert A\vert}$$,
> $$
> \begin{align*}
> T_{soft} V_{Q}(s) =& \max_{\pi} T^\pi_{soft} V_{Q}(s)\\
> =& \max_{\pi} \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V_Q(s') \right)
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
> Q_{V_{Q}}(s) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \sum_{a'} \pi_Q(a' \vert s') Q(s', a') \right)
> $$
> **Lemma**:
> $$
> \log \pi^*_{Q, soft}(a \vert s) = \frac{1}{\alpha} [Q_{V_{Q}}(s, a) - T_{soft} V_{Q}(s)]
> $$

### Another Q-Based Soft Optimal Bellman Equation

$$
\begin{align*}
T_{soft} Q(s', a')
=& \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \sum_{a'} \pi (a' \vert s') Q(s', a') \right) - \alpha\log(\pi(a\vert s))
\end{align*}
$$

$$
\begin{align*}
T_{soft} Q(s', a')
=& \max_{\pi} \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + \sum_{a'} \pi (a' \vert s') Q(s', a') \right)\\
&- \alpha\sum_a \pi(a \vert s) \log(\pi(a\vert s))
\end{align*}
$$

Too hard.

## 3. Soft Q-Learning

### 3.1 Soft Q-Iteration

$$
V_{t+1} = T_{soft} V_t
$$

Or
$$
\begin{cases}
Q_{V_t}(s, a; \theta) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V_t(s';\theta) \right), &\forall s, a\\
V_{t+1}(s;\theta) = \alpha \log \left\{\int_a \exp\left[\frac{1}{\alpha}Q_V(s, a;\theta)\right] da\right\}, &\forall s
\end{cases}
$$

### 3.2 Soft Q-Learning

The key idea from that we want to use q-net $$Q(s, a; \theta)$$ instead of $$V(s; \theta)$$. 

**Discrete action space**
$$
\begin{cases}
\pi_Q(a \vert s; \theta) = {\frac{ \exp\left\{\frac{1}{\alpha}Q(s, a; \theta)\right\} }{ \sum_a \exp\left\{\frac{1}{\alpha}Q(s, a; \theta)\right\}}} \\
V_{Q}(s;\theta) = \sum_a \pi_Q(a\vert s; \theta) Q(s, a;\theta)\\
Q_{V_Q}(s,a;\theta) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V_Q(s';\theta) \right), &\forall s, a \\
T_{soft} V_{Q} = \alpha \log \left\{\sum_a \exp\left[\frac{1}{\alpha}Q_{V_Q}(s, a;\theta)\right] da\right\}, &\forall s\\
Q_{T_{soft}V_Q} = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + T_{soft}V_Q(s';\theta) \right)
\end{cases}
$$

$$
J(\theta) = \sum_s p(s) \left(\alpha \log \left\{\sum_a \exp\left[\frac{1}{\alpha}Q_{V_Q}(s, a;\theta)\right]\right\} - \sum_a \pi_Q(a \vert s;\theta) Q(s, a;\theta)\right)^2
$$

$$
J(\theta) = \sum_s p(s) \sum_a \mu(a \vert s) \left(Q_{T_{soft}V_Q}(s, a; \theta) - Q(s, a; \theta)\right)^2
$$

**Continuous action space**
$$
\begin{cases}
\pi_Q(a \vert s; \theta) = {\frac{ \exp\left\{\frac{1}{\alpha}Q(s, a; \theta)\right\} }{ \sum_a \exp\left\{\frac{1}{\alpha}Q(s, a; \theta)\right\}}} \\
V_{Q}(s;\theta) = \int_a \pi_Q(a \vert s;\theta) Q(s, a;\theta) da\\
Q_{V_Q}(s,a;\theta) = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + V_Q(s';\theta) \right), &\forall s, a \\
T_{soft} V_{Q} = \alpha \log \left\{\int_a \exp\left[\frac{1}{\alpha}Q_{V_Q}(s, a;\theta)\right] da\right\}, &\forall s\\
Q_{T_{soft}V_Q} = \sum_{s'} p(s' \vert s, a)\left(r(s, a, s') + T_{soft}V_Q(s';\theta) \right)
\end{cases}
$$

$$
J(\theta) = \sum_s p(s) \left(T_{soft} V_Q(s) - V_Q(s)\right)^2
$$

$$
J(\theta) = \sum_s p(s) \int_a \mu(a \vert s) \left(Q_{T_{soft}V_Q}(s, a; \theta) - Q(s, a; \theta)\right)^2 da
$$

**Soft q-learning objective**
$$
J(\theta) = \sum_s p(s) \int_a \mu(a \vert s) \left(Q_{T_{soft}V_Q}(s, a; \theta^{old}) - Q(s, a; \theta)\right)^2 da
$$
**How to get soft optimal Bellman equation?**
$$
T_{soft} V_{Q} = \alpha \log \left\{\int_a \exp\left[\frac{1}{\alpha}Q_{V_Q}(s, a;\theta)\right] da\right\} = \alpha \log\mathbb{E}_{a \sim q} \left\{\frac{1}{q(a)}\exp\left[\frac{1}{\alpha}Q_{V_Q}(s, a;\theta)\right]\right\}
$$

## 4. Approximate Sampling and Stein Variational Gradient Descent
$$
J(\theta_\pi) = \sum_s p(s) D_{KL}\left(\pi(\cdot \vert s; \theta_\pi) \Big\Vert \exp\left(\frac{1}{\alpha} Q_{V_Q}(s, \cdot) - T_{soft} V_Q\right)\right)
$$



[1] T. Haarnoja, H. Tang, P. Abbeel, and S. Levine, “Reinforcement Learning with Deep Energy-Based Policies,” *arXiv:1702.08165 [cs]*, Jul. 2017.
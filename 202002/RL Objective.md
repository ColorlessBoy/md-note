# RL Objective

## 1. BELLMAN EQUATION

Because Bellman equation is
$$
T_\pi V(s) = \sum_{a} \pi(a \vert s) \sum_{s'} p(s' \vert s, a)[r(s, a, s') + \gamma V(s')]
$$
and
$$
V_\pi = T_\pi V_\pi.
$$
The target of reinforcement learning by using Bellman equation is
$$
\begin{cases}
    \max_{\pi} \sum_s p_1(s)V(s)\\
    \min_{V}\sum_{s} p_2(s) \left\{\sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - V(s)\right\}^2\\
    V(s) = \sum_a \pi(a \vert s) Q(s, a)
    \end{cases}
$$

### 1.1 Tabular Algorithm

$$
\begin{cases}
l_1(\pi, V) = -\sum_s p_1(s) V(s) \\
l_2(\pi, V) =  \sum_{s} p_2(s) \left\{\sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - V(s)\right\}^2
\end{cases}
$$

Or
$$
\begin{cases}
l_1(\pi, Q) &= -\sum_s p_1(s) \sum_a \pi(a \vert s) Q(s, a) \\
l_2(\pi, Q) &=  \sum_{s} p_2(s) \left\{\sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) [r(s, a, s') + \sum_{a'} \pi(a' \vert s') Q(s', a')] - \sum_{a} \pi(a \vert s) Q(s, a)\right\}^2 \\
&= \sum_{s} p_2(s) \left\{\sum_a \pi(a \vert s) \left[\sum_{s'} p(s' \vert s, a) [r(s, a, s') + \sum_{a'} \pi(a' \vert s') Q(s', a')] - Q(s, a) \right]\right\}^2
\end{cases}
$$

$$
\begin{align*}
\frac{\partial}{\partial \pi(s,a)}l_1(\pi, Q) =& -p_1(s) Q(s, a) = -\mathbb{E}_{p_1, \pi}\left[\frac{Q(s, a)}{\pi(a \vert s)}\right] \\
\frac{\partial}{\partial Q(s,a)}l_1(\pi, Q) =& -p_1(s) \pi(s, a) \\
\frac{\partial}{\partial \pi(a''\vert s'')} l_2(\pi, Q) 
=&
2\sum_{s} p_2(s) \left\{\sum_a \pi(a \vert s) \left[\sum_{s'} p(s' \vert s, a) [r(s, a, s') + \sum_{a'} \pi(a' \vert s') Q(s', a')] - Q(s, a) \right]\right\}\\
&\cdot \left\{\sum_a \pi(a \vert s) p(s'' \vert s, a) Q(s'', a'') \\
- 1\{s=s''\}\left[\sum_{s'} p(s' \vert s'', a'') [r(s'', a'', s') + \sum_{a'} \pi(a' \vert s') Q(s', a')] - Q(s'', a'') \right]\right\}
\\
\frac{\partial}{\partial Q(s'',a'')} l_2(\pi, Q) =&
2 \sum_{s} p_2(s)\left\{\sum_{a} \pi(a \vert s) \left[\sum_{s} p(s' \vert s, a) [r(s, a, s') + \sum_{a'} \pi(a' \vert s') Q(s', a')] - Q(s, a) \right]\right\}\\
&\cdot \left\{\sum_{a} \pi(a \vert s) p(s'' \vert s, a) \pi(a'' \vert s'') - 1\{s =s''\} \pi(a''\vert s'') \right\}
\end{align*}
$$

### 1.2 Approximation Algorithm

$$
\begin{cases}
\theta_\pi = \arg\max_{\theta_\pi} \sum_{s} p_1(s) \sum_a \pi(a \vert s; \theta_\pi) Q(s, a \vert \theta_Q)\\
\theta_Q = \arg\min_{\theta_Q} \sum_s p_2(s)
\left\{
\sum_a \pi(a \vert s; \theta_\pi) \sum_{s'}p(s' \vert s, a) [r(s, a, s') + \gamma V(s'; \theta_\pi, \theta_Q)] - V(s; \theta_\pi, \theta_Q)
\right\}^2\\
V(s; \theta_\pi, \theta_Q) = \sum_a \pi(a \vert s; \theta_\pi) Q(s, a \vert \theta_Q)
\end{cases}
$$

$$
\begin{cases}
l_1 = -\sum_s p_1(s) \sum_a \pi(a \vert s; \theta_\pi) Q(s, a \vert \theta_Q)\\
l_2 = \sum_s p_2(s)
\left\{
\sum_a \pi(a \vert s; \theta_\pi) \sum_{s'}p(s' \vert s, a) [r(s, a, s') + \gamma V(s'; \theta_\pi, \theta_Q)] - V(s; \theta_\pi, \theta_Q)
\right\}^2\\
V(s; \theta_\pi, \theta_Q) = \sum_a \pi(a \vert s; \theta_\pi) Q(s, a \vert \theta_Q)
\end{cases}
$$

## 2. OPTIMAL BELLMAN EQUATION

Because optimal Bellman equation is 
$$
TV(s) = \max_{\pi(\cdot \vert s)}\sum_a \pi(a \vert s) \sum_{s'}p(s' \vert s, a) [r(s, a, s') + \gamma V(s')], \forall V \in \mathbb{R}^{\vert S \vert}
$$
therefore the target becomes
$$
\min_V \sum_s p_2(s) \left\{TV(s) - V(s)\right\}^2.
$$
The target of reinforcement learning by using Optimal equation is 
$$
\min_{V}\sum_{s} p_2(s) \left\{\max_{\pi(\cdot \vert s)}\sum_a \pi(a \vert s) \sum_{s'}p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - V(s)\right\}^2
$$

We make further exploration:
$$
\begin{align*}
&\min_{V} \sum_s p(s) \left\{\max_{\pi(\cdot \vert s)} \sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - V(s)\right\}^2 \\
=& \min_{Q} \sum_s p(s) \left\{\max_{\pi} \sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \vert s') Q(s', a') \right]\\
- \sum_a\pi(a \vert s) Q(s, a)\right\}^2\\
=& \min_{\theta_Q} \sum_s p(s) \left\{\max_{\theta_\pi} \sum_a \pi(a \vert s;\theta_\pi) \sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \vert s'; \theta_\pi) Q(s', a';\theta_Q) \right] \\ - \sum_a\pi(a \vert s; \theta_\pi) Q(s, a; \theta_Q)\right\}^2
\end{align*}
$$

### 2.1 V-Based-Loss Function

$$
L(V) = \sum_s p(s) \left\{\max_{a} \sum_{s'} p(s' \vert s, a) [r(s, a, s') + \gamma V(s')] - V(s)\right\}^2
$$

### 2.2 Q-Based-Loss Function

#### 2.2.1 On-policy

Let $$\pi_Q(a \vert s) = 1\{a = \arg\max_{a'} Q(s, a')\}$$:
$$
\begin{align*}
L(Q) =& \sum_s p(s) \left\{\max_{\pi} \sum_a \pi(a \vert s) \sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \vert s') Q(s', a') \right] - \sum_a\pi(a \vert s) Q(s, a)\right\}^2\\
=& \sum_s p(s) \left\{\max_{\pi}  \sum_a \pi(a \vert s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi(a' \vert s') Q(s', a') \right] - Q(s, a) \right\}\right\}^2 \\
=& \sum_s p(s) \left\{\sum_{a} \pi_Q(a \vert s)\left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi_Q(a' \vert s') Q(s', a') \right] - Q(s, a) \right\}\right\}^2 \\
=& \sum_s p(s) \sum_{a} \pi_Q(a \vert s)\left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi_Q(a' \vert s') Q(s', a') \right] - Q(s, a) \right\}^2
\\&(\text{The property of } \pi_Q)
\end{align*}
$$

(Hint: from smoothed Bellman equation, we have $$\pi(a \vert s) = \lim_{\lambda \rightarrow 0} \pi_{\lambda}(a \vert s)$$.)

#### 2.2.2 Q-Learning

$$
\pi_{Q, \epsilon} = (1 - \epsilon)\pi_Q + \epsilon \pi_{uniform}
$$

$$
L(Q) = \sum_s p(s) \sum_{a} \pi_{Q, \epsilon}(a \vert s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi_Q(a' \vert s') Q(s', a') \right] - Q(s, a) \right\}^2
$$

#### 2.2.3 SARSA

$$
L(Q) = \sum_s p(s) \sum_{a} \pi_{Q, \epsilon}(a \vert s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi_{Q,\epsilon}(a' \vert s') Q(s', a') \right] - Q(s, a) \right\}^2
$$

#### 2.2.3 Q-Learning with Replay Buffer

$$
L(Q) = \sum_s p(s) \sum_{a} \pi_{replay}(a \vert s) \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi_Q(a' \vert s') Q(s', a') \right] - Q(s, a) \right\}^2
$$

### 2.3 Q-Loss with Function Approximation

#### 2.3.1 On-policy with function approximation

Let $$\pi_Q(a \vert s; \theta_Q) = 1\{a = \arg\max_{a'} Q(s, a'; \theta_Q)\}$$
$$
\begin{align*}
L(\theta_Q) =& \sum_s p(s) \sum_{a} \pi_Q(a \vert s;\theta_Q) \\
&\cdot \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_a \pi_Q(a' \vert s'; \theta_Q) Q(s', a'; \theta_Q) \right] - Q(s, a; \theta_Q)\right\}^2,\\
\nabla_{\theta_Q} L(\theta_Q) =& \sum_s p(s) \sum_a \pi_Q(a \vert s; \theta_Q) 
\\&\cdot \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_a \pi_Q(a' \vert s'; \theta_Q) Q(s', a'; \theta_Q) \right] - Q(s, a; \theta_Q)\right\}\\
&\cdot \left\{\gamma \sum_{s'} p(s' \vert s, a) \nabla_{\theta_Q} \sum_a \pi_Q(a' \vert s'; \theta_Q) Q(s', a'; \theta_Q)  - \nabla_{\theta_Q} Q(s, a; \theta_Q)\right\}
\end{align*}
$$

#### 2.3.2 Q-learning with function approximation

$$
\begin{align*}
L(\theta_Q) =& \sum_s p(s) \sum_{a} \pi_{Q, \epsilon} (a \vert s; \theta_Q)\\ 
&\cdot \left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_a \pi_{Q}(a \vert s; \theta_Q) Q(s', a'; \theta_Q) \right] - Q(s, a; \theta_Q)\right\}^2 \\
\nabla_{\theta_Q} L(\theta_Q) =& \sum_s p(s) \sum_a \pi_{Q,\epsilon}(a \vert s; \theta_Q) 
\\&\cdot\left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_a \pi_Q(a' \vert s'; \theta_Q) Q(s', a'; \theta_Q) \right] - Q(s, a; \theta_Q)\right\}\\
&\cdot \left\{\gamma \sum_{s'} p(s' \vert s, a) \nabla_{\theta_Q} \sum_a \pi_Q(a' \vert s'; \theta_Q) Q(s', a'; \theta_Q)  - \nabla_{\theta_Q} Q(s, a; \theta_Q)\right\}
\end{align*}
$$

#### 2.3.3 SARSA with function approximation

$$
\begin{align*}
L(\theta_Q) =& \sum_s p(s) \sum_{a} \pi_{Q, \epsilon} (a \vert s; \theta_Q) \\
&\cdot\left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_{a'} \pi_{Q, \epsilon}(a' \vert s'; \theta_Q)Q(s', a'; \theta_Q) \right] - Q(s, a; \theta_Q)\right\}^2\\
\nabla_{\theta_Q} L(\theta_Q) =& \sum_s p(s) \sum_a \pi_{Q,\epsilon}(a \vert s; \theta_Q) 
\\&\cdot\left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \sum_a \pi_{Q, \epsilon}(a' \vert s'; \theta_Q) Q(s', a'; \theta_Q) \right] - Q(s, a; \theta_Q)\right\}\\
&\cdot \left\{\gamma \sum_{s'} p(s' \vert s, a) \nabla_{\theta_Q} \sum_a \pi_{Q,\epsilon}(a' \vert s'; \theta_Q) Q(s', a'; \theta_Q)  - \nabla_{\theta_Q} Q(s, a; \theta_Q)\right\}
\end{align*}
$$

#### 2.3.4 DQN

**The loss of DQN**:
$$
\begin{align*}
L_Q(\theta_Q, \theta_{Q_{target}}) =& \frac{1}{2}\sum_s p(s) \sum_{a} \pi_{replay} (a \vert s) \\ &\left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \max_{a'} Q(s', a'; \theta_{Q_{target}}) \right] - Q(s, a; \theta_Q)\right\}^2\\
L_{Q_{target}}(\theta_Q, \theta_{Q_{target}}) =& \frac{1}{2} \Vert \theta_{Q_{target}} - \theta_Q \Vert^2_2
\end{align*}
$$

**The derivative of the loss**:
$$
\begin{align*}
\nabla_{\theta_Q} L_{Q}(\theta_Q, \theta_{Q_{target}})
=& - \sum_s p(s) \sum_{a} \pi_{replay} (a \vert s) \\ &\left\{\sum_{s'} p(s' \vert s, a) \left[r(s, a, s') + \gamma \max_{a'} Q(s', a'; \theta_{Q_{target}}) \right] - Q(s, a; \theta_Q)\right\} \nabla_{\theta_Q} Q(s, a; \theta_Q)\\
\nabla_{\theta_{Q_{target}}} L_{Q_{target}} =& \theta_{Q_{target}} - \theta_Q
\end{align*}
$$
**The update rule of DQN**:
$$
\begin{cases}
\theta_Q = \theta_{Q} - \alpha_1 \nabla_{\theta_Q} L_Q(\theta_Q, \theta_{target})\\
\theta_{Q_{target}} = \theta_{Q_{target}} - \alpha_2(\theta_{Q_{target}} - \theta_Q) (\textbf{polyak averaging})
\end{cases}
$$

## 3. SMOOTHED BELLMAN EQUATION

### 3.1  Preliminaries

#### 3.1.1 Normal Reinforcement Learning Target

$$
\max_{\pi} \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) \\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2)+ \\
+ \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (r(s_2, a_2, s_3) +\cdots)\right)\right)
$$

#### 3.1.2 Regularization Based Reinforcement Learning Target

$$
\max_{\pi} \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(a_0 \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(a_1 \vert s_1)) \\
+ \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
$$

### 3.2 Policy Based

For all $$\pi \in \Pi$$, we have $$J(\pi)$$, $$Q^\pi_{soft}(s_0, a_0)$$ and $$V^\pi_{soft}(s_0)$$ defined below:
$$
J(\pi) =  \sum_{s_0} p_0(s_0)\sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(a_0 \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(a_1 \vert s_1)) \\
+ \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
$$

$$
Q^\pi_{soft}(s_0, a_0) = \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(a_0 \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(a_1 \vert s_1)) \\
+ \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
$$

$$
V^\pi_{soft}(s_0) = \sum_{a_0} \pi(a_0 \vert s_0) \sum_{s_1} p(s_1 \vert s_0, a_0)\left(r(s_0, a_0, s_1) + \mathcal{H}(\pi(a_0 \vert s_0))\\ + \sum_{a_1}\pi(a_1 \vert s_1) \sum_{s_2}p(s_2\vert s_1, a_1) \gamma \left(r(s_1, a_1, s_2) + \mathcal{H}(\pi(a_1 \vert s_1)) \\
+ \sum_{a_2}\pi(a_2 \vert s_2) \sum_{s_3} p(s_3 \vert s_2, a_2) \gamma^2 (\cdots)\right)\right)
$$

> **Lemma**:
> $$
> J(\pi) = \sum_{s_0} p_0(s_0) V^\pi_{soft}(s_0) = \sum_{s_0} p_0(s_0) \sum_{a_0} \pi(a_0 \vert s_0) Q^\pi_{soft}(s_0).
> $$

### 3.3 Value Based

For all $$V \in \mathbb{R}^{\vert S \vert}$$, we have the following things.

**Definition** Smoothed Bellman equation: $$\forall V, \pi$$:
$$
T^\pi_{soft}V(s) = \sum_{a}\pi(a \vert s) \left(\sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]\right)+ H(\pi(\cdot \vert s)).
$$
**Definition** Smoothed optimal Bellman equation:
$$
T_{soft}V(s) = \max_{\pi(\cdot \vert s)} T^\pi_{soft} V(s) = \max_{\pi(\cdot \vert s)}\sum_{a}\pi(a \vert s) \left(\sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]\right)+ H(\pi(\cdot \vert s)).
$$
If $$H(\pi(\cdot \vert s)) = - \lambda\sum_{a} \pi(a \vert s) \log(\pi(a \vert s))$$, then
$$
T_{soft} V(s) = \lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]\right\}
$$
$$
\begin{align*}
&\pi_{V, soft}(a \vert s) = \arg\max_\pi T^\pi V(s)\\
=& \frac{\exp\{\frac{1}{\lambda} \sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]\}}{\sum_{a'} \exp\{\frac{1}{\lambda} \sum_{s'}p(s' \vert s, a') \left[r(s, a', s') + \gamma V(s') \right]\}}
= \frac{\exp\{\frac{1}{\lambda} Q_V(s,a)\}}{\sum_{a'} \exp\{\frac{1}{\lambda} Q_V(s,a')\}}
\end{align*}
$$

where we define $$ Q_V(s, a) = \sum_{s'}p(s' \vert s, a) \left[r(s, a, s') + \gamma V(s') \right]$$.

We construct the target:
$$
\begin{align*}
&\min_{V} \sum_s p(s) \left\{T_\lambda V(s) - V(s)\right\}^2\\
=& \min_{V} \sum_{s} p(s)\left\{\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]\right\} - V(s)\right\}^2
\end{align*}
$$

We have the loss:
$$
L(V) =  \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'))\right]\right\} - V(s)
\right\}^2.
$$

$$
L(\theta_V) =  \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma V(s'; \theta_V))\right]\right\} - V(s; \theta_V)
\right\}^2
$$

### 3.3 Q-Loss Function

$$
L(Q) =  \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma {\frac{ \sum_{a'} Q(s',a') \exp\left\{\frac{1}{\lambda}Q(s', a')\right\} }{ \sum_{a'} \exp\left\{\frac{1}{\lambda}Q(s',a')\right\}}} )\right]\right\} \\
- {\frac{ \sum_{a'} Q(s,a') \exp\left\{\frac{1}{\lambda}Q(s, a')\right\} }{ \sum_{a'} \exp\left\{\frac{1}{\lambda}Q(s,a')\right\}}} 
\right\}^2,
\quad where\quad
\pi_{\lambda}(a \vert s) 
=  \frac{\exp\{\frac{1}{\lambda} Q(s,a)\}}{\sum_{a'} \exp\{\frac{1}{\lambda} Q(s,a')\}}
$$

$$
L(\tilde Q) = \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma \lambda \sum_{a'} \pi(a' \vert s') \log(\tilde Q(s', a'))\right]\right\}\\ - \lambda \sum_{a} \pi(a \vert s) \log(\tilde Q(s, a))
\right\}^2,\quad \quad \pi(a \vert s) = \frac{\tilde Q(s,a)}{\sum_a \tilde Q(s, a)}
$$

### 3.4 Q-Loss Function with Function Approximation

$$
L(\theta) = \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma {\frac{ \sum_{a'} Q(s',a'; \theta) \exp\left\{\frac{1}{\lambda}Q(s', a'; \theta)\right\} }{ \sum_{a'} \exp\left\{\frac{1}{\lambda}Q(s',a';\theta)\right\}}} )\right]\right\} \\
- {\frac{ \sum_{a'} Q(s,a';\theta) \exp\left\{\frac{1}{\lambda}Q(s, a';\theta)\right\} }{ \sum_{a'} \exp\left\{\frac{1}{\lambda}Q(s,a';\theta)\right\}}} 
\right\}^2,
\quad where\quad
\pi_{\lambda}(a \vert s; \theta) 
=  \frac{\exp\{\frac{1}{\lambda} Q(s,a; \theta)\}}{\sum_{a'} \exp\{\frac{1}{\lambda} Q(s,a';\theta)\}}
$$
This is a little difficult to take the derivative.
$$
L(\theta) =  \sum_{s} p(s)\left\{
\lambda \log \left\{\sum_{a} \exp\left[\frac{1}{\lambda}\sum_{s'} p(s'\vert s, a) (r(s, a, s') + \gamma\lambda \sum_{a'} \pi(a' \vert s'; \theta) \log \tilde Q(s', a'; \theta))\right]\right\}\\ - \lambda \sum_{a} \pi(a \vert s; \theta) \log \tilde Q(s, a; \theta))
\right\}^2,\quad \pi(a \vert s; \theta) = \frac{\tilde Q(s,a; \theta)}{\sum_a \tilde Q(s, a; \theta)}
$$

### 3.5 SBEED Loss Function

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

$$
\pi_\lambda(a \vert s; \theta) =  \frac{\exp\{\frac{1}{\lambda} Q(s,a; \theta)\}}{\sum_{a'} \exp\{\frac{1}{\lambda} Q(s,a'; \theta)\}}
$$

$$
V(s; \theta) = \sum_{a} Q(s, a; \theta) \pi_{\lambda}(a \vert s; \theta)
$$

$$
L(\theta) = \mathbb{E}_{s,a} \left\{\mathbb{E}_{s' \vert s, a} [r(s, a, s') + \gamma V(s'; \theta)]  - \lambda \log(\pi(a \vert s; \theta)) - V(s;\theta) \right\}^2
$$

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
\nabla_\theta L(\theta) = 2\nu(s, a; w) [\gamma\nabla_\theta V(s'; \theta) - \lambda \nabla_\theta \log(\pi(a \vert s; \theta)) - \nabla_\theta V(s; \theta)]
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


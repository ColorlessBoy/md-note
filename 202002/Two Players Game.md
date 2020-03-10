# Two Players Game

## The Models

- One player MDP:

$$
\begin{align*}
\forall a \in \mathcal{A}, s \in \mathcal{S}:& \pi
\xrightarrow{p(s' \vert s, a)}
MC = \left\{\tau=(s_0, a_0, s_1,\dots, s_{T-1}, a_{T-1}, s_T)\sim p(s_0)\prod_{t=0}^{T-1}\pi(a_t \vert s_t) p(s_{t+1} \vert s_t, a_t) \right\}\\
&\xrightarrow{r(s, a, s'), \gamma}
V^\pi(s_0) = \mathbb{E}\left\{ \sum^{T-1}_{t=0} \gamma^t r(s_t, a_t, s_{t+1}) \Big\vert s_0 \in \mathcal{S}\right\}
\end{align*}
$$

- Two players MDPs:

$$
\begin{align*}
\forall a, b \in \mathcal{A}, s \in \mathcal{S}:& \pi_a, \pi_b 
\xrightarrow{p(s' \vert s, a, b)}
MC = \left\{\tau=(s_0, a_0, b_0, s_1,\dots, s_{T-1}, a_{T-1}, b_{T-1}, s_T) \\
\sim p(s_0)\prod_{t=0}^{T-1}\pi_a(a_t \vert s_t) \pi_b(b_t \vert s_t, a_t) p(s_{t+1} \vert s_t, a_t, b_t) \right\}\\
&\xrightarrow{r(s, a, b, s'), \gamma}
V^\pi(s_0) = \mathbb{E}\left\{ \sum^{T-1}_{t=0} \gamma^t r(s_t, a_t, b_t, s_{t+1}) \Big\vert s_0 \in \mathcal{S}\right\}
\end{align*}
$$

## From one-player to two-players

### Some Definitions

$$
J(\pi_a, \pi_b) =  
\sum_{s_0} p_0(s_0)
\sum_{a_0} \pi_a(a_0 \vert s_0) 
\sum_{b_0} \pi_b(b_0 \vert s_0, a_0)
\sum_{s_1} p(s_1 \vert s_0, a_0, b_0)\left(r(s_0, a_0, b_0, s_1) + \\
\sum_{a_1}\pi_a(a_1 \vert s_1) 
\sum_{b_1} \pi_b(b_1 \vert s_1, a_1) 
\sum_{s_2}p(s_2\vert s_1, a_1, b_1) \gamma \left(r(s_1, a_1, b_1, s_2) + \\
\sum_{a_2}\pi_a(a_2 \vert s_2) 
\sum_{b_2} \pi_b(b_2 \vert s_2, a_2) 
\sum_{s_3} p(s_3 \vert s_2, a_2, b_2) 
\gamma^2 (\cdots)\right)\right)
$$

$$
V(s_0 \vert \pi_a, \pi_b) =
\sum_{a_0} \pi_a(a_0 \vert s_0) 
\sum_{b_0} \pi_b(b_0 \vert s_0, a_0)
\sum_{s_1} p(s_1 \vert s_0, a_0, b_0)\left(r(s_0, a_0, b_0, s_1) + \\
\sum_{a_1}\pi_a(a_1 \vert s_1) 
\sum_{b_1} \pi_b(b_1 \vert s_1, a_1) 
\sum_{s_2}p(s_2\vert s_1, a_1, b_1) \gamma \left(r(s_1, a_1, b_1, s_2) + \\
\sum_{a_2}\pi_a(a_2 \vert s_2) 
\sum_{b_2} \pi_b(b_2 \vert s_2, a_2) 
\sum_{s_3} p(s_3 \vert s_2, a_2, b_2) 
\gamma^2 (\cdots)\right)\right)
$$

$$
Q_1(s_0, a_0 \vert \pi_a, \pi_b) =
\sum_{b_0} \pi_b(b_0 \vert s_0, a_0)
\sum_{s_1} p(s_1 \vert s_0, a_0, b_0)\left(r(s_0, a_0, b_0, s_1) + \\
\sum_{a_1}\pi_a(a_1 \vert s_1) 
\sum_{b_1} \pi_b(b_1 \vert s_1, a_1) 
\sum_{s_2}p(s_2\vert s_1, a_1, b_1) \gamma \left(r(s_1, a_1, b_1, s_2) + \\
\sum_{a_2}\pi_a(a_2 \vert s_2) 
\sum_{b_2} \pi_b(b_2 \vert s_2, a_2) 
\sum_{s_3} p(s_3 \vert s_2, a_2, b_2) 
\gamma^2 (\cdots)\right)\right)
$$

$$
Q_2(s_0, a_0, b_0 \vert \pi_a, \pi_b) =
\sum_{s_1} p(s_1 \vert s_0, a_0, b_0)\left(r(s_0, a_0, b_0, s_1) + \\
\sum_{a_1}\pi_a(a_1 \vert s_1) 
\sum_{b_1} \pi_b(b_1 \vert s_1, a_1) 
\sum_{s_2}p(s_2\vert s_1, a_1, b_1) \gamma \left(r(s_1, a_1, b_1, s_2) + \\
\sum_{a_2}\pi_a(a_2 \vert s_2) 
\sum_{b_2} \pi_b(b_2 \vert s_2, a_2) 
\sum_{s_3} p(s_3 \vert s_2, a_2, b_2) 
\gamma^2 (\cdots)\right)\right)
$$

### Relationships

$$
\begin{cases}
J(\pi_a, \pi_b) = \sum_{s \in \mathcal{S}} p_0(s) V(s \vert \pi_a, \pi_b) \\
V(s \vert \pi_a, \pi_b) = \sum_{a} \pi_a(a \vert s) Q_1(s, a \vert \pi_a, \pi_b) \\
Q_1(s, a \vert \pi_a, \pi_b) = \sum_b \pi_b(b \vert s, a) Q_2(s, a, b \vert \pi_a, \pi_b) \\
Q_2(s, a, b \vert \pi_a, \pi_b) = \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s'))
\end{cases}
$$

And $$ V_{\pi_a, \pi_b} = [V(s \vert \pi_a, \pi_b)]$$, $$Q_{1, \pi_a, \pi_b} = [Q_1(s, a \vert \pi_a, \pi_b)]$$, $$Q_{2, \pi_a, \pi_b} = [Q_{2}(s, a, b \vert \pi_a, \pi_b)]$$.

## Two Players Bellman Equation

For all $V \in \mathbb{R}^{\vert \mathcal{S} \vert}$,
$$
T_{\pi_a, \pi_b} V(s) = \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s')).
$$
If $$T_{\pi_a, \pi_b}$$ is contraction mapping then
$$
V_{\pi_a, \pi_b}  = T_{\pi_a, \pi_b} V_{\pi_a, \pi_b}.
$$
In value iteration:
$$
J(\pi_a, \pi_b) = \frac{1}{2} \Vert V_{\pi_a, \pi_b} - T_{\pi_a, \pi_b} V_{\pi_a, \pi_b} \Vert^2_{\mu}
$$


### Matrix Formation

$$
V_{\pi_a, \pi_b} = \langle \pi_a, \langle\pi_b, P\rangle \rangle (r + \gamma V_{\pi_a, \pi_b})
\Rightarrow 
V_{\pi_a, \pi_b} = (I - \gamma \langle \pi_a, \langle\pi_b, P\rangle \rangle)^{-1} \bar r
$$

$$
P =
\begin{bmatrix}
\alpha_{11}\beta_{111} p_{1111} + \alpha_{11}\beta_{112} p_{1121} + \alpha_{12} \beta_{121}p_{1211} + \alpha_{12} \beta_{122} p_{1221} &

\alpha_{11}\beta_{111} p_{1112} + \alpha_{11}\beta_{112} p_{1122} + \alpha_{12} \beta_{121}p_{1212} + \alpha_{12} \beta_{122} p_{1222} \\

\alpha_{21}\beta_{211} p_{2111} + \alpha_{21}\beta_{212} p_{2121} + \alpha_{22} \beta_{221}p_{2211} + \alpha_{22} \beta_{222} p_{2221} &

\alpha_{21}\beta_{211} p_{2112} + \alpha_{21}\beta_{212} p_{2122} + \alpha_{22} \beta_{221}p_{2212} + \alpha_{22} \beta_{222} p_{2222}

\end{bmatrix}
$$

$$
\pi_a(\cdot \vert s_1) = \begin{bmatrix} \pi_a(a_1 \vert s_1) \\ \pi_a(a_2 \vert s_1) \end{bmatrix}
$$

$$
P(\cdot \vert \cdot, s_1) =
\begin{bmatrix}
p(s_1 \vert a_1, s_1) & p(s_1 \vert a_2, s_1) \\
p(s_2 \vert a_1, s_1) & p(s_2 \vert a_2, s_1)
\end{bmatrix}
$$

$$
P(\cdot \vert s_1) =
\pi_a(a_1 \vert s_1)
\begin{bmatrix}
p(s_1 \vert a_1, s_1)\\
p(s_2 \vert a_1, s_1)
\end{bmatrix}
+
\pi_a(a_2 \vert s_1)
\begin{bmatrix}
p(s_1 \vert a_2, s_1)\\
p(s_2 \vert a_2, s_1)
\end{bmatrix}
$$

---

$$
P(\cdot \vert s_1, a_1) = 
\pi_b(b_1 \vert s_1, a_1)
\begin{bmatrix}
p(s_1 \vert s_1, a_1, b_1)\\
p(s_2 \vert s_1, a_1, b_1)
\end{bmatrix}
+ \pi_b(b_2 \vert s_1, a_1)
\begin{bmatrix}
p(s_1 \vert s_1, a_1, b_2) \\
p(s_2 \vert s_1, a_1, b_2)
\end{bmatrix}
$$

$$
P(\cdot \vert s_1, a_2) = 
\pi_b(b_1 \vert s_1, a_2)
\begin{bmatrix}
p(s_1 \vert s_1, a_2, b_1)\\
p(s_2 \vert s_1, a_2, b_1)
\end{bmatrix}
+ \pi_b(b_2 \vert s_1, a_2)
\begin{bmatrix}
p(s_1 \vert s_1, a_2, b_2) \\
p(s_2 \vert s_1, a_2, b_2)
\end{bmatrix}
$$



## Two Players Optimal Bellman Equation

For all $V \in \mathbb{R}^{\vert \mathcal{S} \vert}$,
$$
T_{\pi_b} V(s) = \max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s')).
$$
In original optimal Bellman equation,
$$
V_{*, \pi_b} = T_{\pi_b} V_{*, \pi_b}.
$$
We denote $$\pi^*_a(\pi_b)$$ that satisfies $$ V_{\pi^*_a(\pi_b), \pi_b} = V_{*, \pi_b}$$.

---

## Cooperation-Two players Optimal Bellman Equation

For all $V \in \mathbb{R}^{\vert \mathcal{S} \vert}$,
$$
T V(s) = \max_{\pi_b(\cdot \vert s, \cdot)}\max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s')).
$$

### Q-Learning Algorithm

$$
\begin{aligned}
L(V) =& \sum_{s}p(s) [TV(s) - V(s)]^2 \\
=& \sum_{s} p(s)\left\{ \max_{\pi_b(\cdot \vert s, \cdot)}\max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s')) - V(s) \right\}^2 \\
L(Q)=& \sum_{s} p(s) \left\{ \max_{\pi_b(\cdot \vert s, \cdot)}\max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \left[\sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') \\ 
+\gamma \sum_{a'} \pi_a(a' \vert s') \sum_b \pi_b(b' \vert s', a') Q(s', a', b')) - Q(s, a, b)\right] \right\}^2 \\
=& \sum_s p(s) \sum_a \pi_{a, Q}(a \vert s) \sum_b \pi_{b, Q}(b \vert s, a)
\left\{\sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') \\
+\gamma \sum_{a'} \pi_a(a' \vert s') \sum_b \pi_b(b' \vert s', a') Q(s', a', b')) - Q(s, a, b)
\right\}^2
\end{aligned}
$$

$$
Q(s, \cdot, \cdot) = 
\begin{bmatrix}
Q(s, a_1, b_1) & Q(s, a_1, b_2) & \cdots & Q(s, a_1, b_n) \\
Q(s, a_2, b_1) & Q(s, a_2, b_2) & \cdots & Q(s, a_2, b_n) \\
\vdots & \vdots & \ddots & \vdots \\
Q(s, a_m, b_1) & Q(s, a_m, b_2) & \cdots & Q(s, a_m, b_n)
\end{bmatrix}
$$

We denote
$$
\pi_{b,Q}(s, a_i) = \arg\max_{b} Q(s, a_i, b)
$$

$$
Q(s, \cdot \vert \pi_{b, Q}) =
\begin{bmatrix}
Q(s, a_1, \pi_{b, Q}) \\
Q(s, a_2, \pi_{b, Q}) \\
\vdots \\
Q(s, a_m, \pi_{b, Q})
\end{bmatrix}
$$

$$
\pi_{a, Q}(s) = \arg\max_{a} Q(s, a, \pi_{b, Q})
$$

### Soft Q-Learning Algorithm

$$\forall V \in \mathbb{R}^{\vert \mathcal{S} \vert}$$,
$$
T V(s) = \max_{\pi_b(\cdot \vert s, \cdot)}\max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s') \\
- \alpha \pi_a(a \vert s) - \beta \pi_b(b \vert s, a)).
$$

$$
\pi_{b, V}(b \vert s, a) = 
\frac{\exp\left\{\frac{1}{\beta} \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s'))\right\}}
{\sum_{b'} \exp\left\{\frac{1}{\beta} \sum_{s'} p(s' \vert s, a, b') (r(s, a, b', s') + \gamma V(s'))\right\}}
$$

$$
\pi_{a, V}(a \vert s) = 
\frac{\exp\left\{\frac{1}{\alpha} \sum_b \pi_{b,V} (b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s'))\right\}}
{\sum_{a'}\exp\left\{\frac{1}{\alpha} \sum_b \pi_{b,V} (b \vert s, a') \sum_{s'} p(s' \vert s, a', b) (r(s, a', b, s') + \gamma V(s'))\right\}}
$$

$$
\begin{aligned}
Q_V(s, \cdot, \cdot) =& \sum_{s'} p(s' \vert s, a, b)(r(s, a, b, s') + \gamma V(s')) \\
=&
\begin{bmatrix}
Q_V(s, a_1, b_1) & Q_V(s, a_1, b_2) & \cdots & Q_V(s, a_1, b_n) \\
Q_V(s, a_2, b_1) & Q_V(s, a_2, b_2) & \cdots & Q_V(s, a_2, b_n) \\
\vdots & \vdots & \ddots & \vdots \\
Q_V(s, a_m, b_1) & Q_V(s, a_m, b_2) & \cdots & Q_V(s, a_m, b_n)
\end{bmatrix}
\end{aligned}
$$

$$
\pi_{b, V}(v \vert s, a) = \frac{\exp\left\{\frac{1}{\beta} Q_V(s, a, b) \right\}}{\sum_{b'} \exp\left\{\frac{1}{\beta} Q_V(s, a, b')\right\}}
$$

$$
\pi_{a, V}(a \vert s) = 
\frac{\exp\left\{\frac{1}{\alpha} \sum_b \pi_{b,V} (b \vert s, a) Q_V(s, a, b)\right\}}
{\sum_{a'}\exp\left\{\frac{1}{\alpha} \sum_b \pi_{b,V} (b \vert s, a') Q_V(s, a, b')\right\}}
$$

## Zero-Sum-Two Players Optimal Bellman Equation

For all $V \in \mathbb{R}^{\vert \mathcal{S} \vert}$,
$$
T V(s) = \min_{\pi_b(\cdot \vert s, \cdot)}\max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s')).
$$

### Q-Learning Algorithm

$$
\begin{aligned}L(V) 
=& \sum_{s}p(s) [TV(s) - V(s)]^2 \\
=& \sum_{s} p(s)\left\{ \min_{\pi_b(\cdot \vert s, \cdot)}\max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s')) - V(s) \right\}^2 \\
L(Q)=& \sum_{s} p(s) \left\{ \min_{\pi_b(\cdot \vert s, \cdot)}\max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \left[\sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') \\ 
+\gamma \sum_{a'} \pi_a(a' \vert s') \sum_b \pi_b(b' \vert s', a') Q(s', a', b')) - Q(s, a, b)\right] \right\}^2 \\
=& \sum_s p(s) \sum_a \pi_{a, Q}(a \vert s) \sum_b \pi_{b, Q}(b \vert s, a)\left\{\sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') \\
+\gamma \sum_{a'} \pi_a(a' \vert s') \sum_b \pi_b(b' \vert s', a') Q(s', a', b')) - Q(s, a, b)\right\}^2
\end{aligned}
$$

$$
Q(s, \cdot, \cdot) = \begin{bmatrix}Q(s, a_1, b_1) & Q(s, a_1, b_2) & \cdots & Q(s, a_1, b_n) \\Q(s, a_2, b_1) & Q(s, a_2, b_2) & \cdots & Q(s, a_2, b_n) \\\vdots & \vdots & \ddots & \vdots \\Q(s, a_m, b_1) & Q(s, a_m, b_2) & \cdots & Q(s, a_m, b_n)\end{bmatrix}
$$

We denote
$$
\pi_{b,Q}(s, a_i) = \arg\min_{b} Q(s, a_i, b)
$$

$$
Q(s, \cdot \vert \pi_{b, Q}) =\begin{bmatrix}Q(s, a_1, \pi_{b, Q}) \\Q(s, a_2, \pi_{b, Q}) \\\vdots \\Q(s, a_m, \pi_{b, Q})\end{bmatrix}
$$

$$
\pi_{a, Q}(s) = \arg\max_{a} Q(s, \cdot \vert \pi_{b, Q})
$$

### Soft Q-Learning Algorithm

$$\forall V \in \mathbb{R}^{\vert \mathcal{S} \vert}$$,
$$
T V(s) = \min_{\pi_b(\cdot \vert s, \cdot)}\max_{\pi_a(\cdot \vert s)} \sum_a \pi_a(a \vert s) \sum_b \pi_b(b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s') \\- \alpha \pi_a(a \vert s) + \beta \pi_b(b \vert s, a)).
$$

$$
\pi_{b, V}(b \vert s, a) = \frac{\exp\left\{-\frac{1}{\beta} \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s'))\right\}}{\sum_{b'} \exp\left\{-\frac{1}{\beta} \sum_{s'} p(s' \vert s, a, b') (r(s, a, b', s') + \gamma V(s'))\right\}}
$$

$$
\pi_{a, V}(a \vert s) = \frac{\exp\left\{\frac{1}{\alpha} \sum_b \pi_{b,V} (b \vert s, a) \sum_{s'} p(s' \vert s, a, b) (r(s, a, b, s') + \gamma V(s'))\right\}}{\sum_{a'}\exp\left\{\frac{1}{\alpha} \sum_b \pi_{b,V} (b \vert s, a') \sum_{s'} p(s' \vert s, a', b) (r(s, a', b, s') + \gamma V(s'))\right\}}
$$

$$
\begin{aligned}Q_V(s, \cdot, \cdot) =& \sum_{s'} p(s' \vert s, a, b)(r(s, a, b, s') + \gamma V(s')) \\=&\begin{bmatrix}Q_V(s, a_1, b_1) & Q_V(s, a_1, b_2) & \cdots & Q_V(s, a_1, b_n) \\Q_V(s, a_2, b_1) & Q_V(s, a_2, b_2) & \cdots & Q_V(s, a_2, b_n) \\\vdots & \vdots & \ddots & \vdots \\Q_V(s, a_m, b_1) & Q_V(s, a_m, b_2) & \cdots & Q_V(s, a_m, b_n)\end{bmatrix}\end{aligned}
$$

$$
\pi_{b, V}(v \vert s, a) = \frac{\exp\left\{-\frac{1}{\beta} Q_V(s, a, b) \right\}}{\sum_{b'} \exp\left\{-\frac{1}{\beta} Q_V(s, a, b')\right\}}
$$

$$
\pi_{a, V}(a \vert s) = \frac{\exp\left\{\frac{1}{\alpha} \sum_b \pi_{b,V} (b \vert s, a) Q_V(s, a, b)\right\}}{\sum_{a'}\exp\left\{\frac{1}{\alpha} \sum_b \pi_{b,V} (b \vert s, a') Q_V(s, a, b')\right\}}
$$

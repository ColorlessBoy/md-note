# GradientDICE

- GradientDICE求解$$c(s, a) = \frac{\pi(a \vert s)}{\rho(a \vert s)}$$；
- 改进算法GenDICE（state of the art)；
- GenDICE 并非一个convex-concave saddle-point(CCSP)问题。balabala。

---

- Hallak & Mannor(2017) and Liu et al.(2018) : single known behavior policy；
- DualDICE: multiple unknown behavior policies and offline training, but it's only stable with the total discounted reward criterion;
- GenDICE: stable with both criteria, but it's not a CCSP problem with non-linearity functions;
- This paper's work: GradientDICE.

---

$$
\partial f^* (y) = \arg\max_{x \in dom f} \{y^T x - f(x)\}???
$$

---

If $$ x_0 \in \arg\max_{x \in dom f} \{y_0^T x - f(x)\} $$, then
$$
\begin{aligned}
& sup_{x \in dom(f)} \{ y^Tx - f(x)\} - \sup_{x \in dom(f)} \{y^T_0 x - f(x)\} \\
\ge& y^Tx_0 - f(x_0) - y^T_0 x_0 - f(x_0) \\
\ge& (y - y_0)^T x_0
\end{aligned}
$$
and $$ x_0 \in \partial g(y_0)$$.

---

$$ g^*(z) = \max_{y \in dom g} \{y^T z - g(y)\}$$
$$
g^*(z) = \max_{y \in dom g} \{y^T z - \max_{x \in dom f} [y^T x - f(x)]\} \\
= \max_{y \in dom g} \max_{x \in dom f}\{y^T z - y^T x + f(x)\} \\
$$













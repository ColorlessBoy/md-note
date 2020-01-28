# The Mechanics of n-Player Differentiable Games

## Introduction

> **Definition**. (Preliminaries) A game is a set of players $$[n] = \{1, \ldots, n\}$$ and twice continuously differentiable losses $$\{l_i : \mathbb{R}^d \rightarrow \mathbb{R}\}^n_{i=1}$$. Parameters are $$w = (w_1, \ldots, w_n) \in \mathbb{R}^d$$ with $$w_i \in \mathbb{R}^{d_i}$$ where $$\sum^n_{i=1} d_i = d$$. Player $$i$$ controls $$w_i$$.

> **Definition**. **simultaneous gradient**
> $$
> \xi(w) = (\nabla_{w_1}l_1, \ldots, \nabla_{w_n}l_n) \in \mathbb{R}^d.
> $$

Here is an zero-sum example:
$$
l_1(w_1, w_2) = w_1^T w_2\quad and\quad l_2(x,y) = -w_1^T w_2,
$$
then
$$
\xi(w_1, w_2) = (w_2, -w_1)
$$
which rotates around the Nash equilibrium.

But the corresponding Hamiltonian function is
$$
\mathcal{H}(w) = \frac{1}{2}(\Vert w_1 \Vert^2_2 + \Vert w_2 \Vert^2_2),
$$
and
$$
\nabla\mathcal{H} = (w_1, w_2)
$$
which points in the opposite direction of Nash equilibrium.

## The generalized Helmholtz decomposition

> **Definition** The Hessian of a game is the $$(d\times d)$$-matrix
> $$
> H(w) = \nabla_w \xi^T(w) = \left(\frac{\partial \xi_\alpha(w)}{\partial w_\beta}\right)^d_{\alpha,\beta = 1}\\
> =
> \begin{bmatrix}
> \nabla^2_{w_1} l_1 & \nabla^2_{w_1, w_2} l_1 & \ldots & \nabla^2_{w1, w_n} l_1\\
> \nabla^2_{w_2, w_1} l_2 & \nabla^2_{w_2} l_2 &\ldots & \nabla^2_{w_2, w_n} l_2\\
> \vdots&&&\vdots\\
> \nabla^2_{w_n, w_1} l_n & \nabla^2_{w_n, w_2} l_n & \ldots & \nabla^2_{w_n} l_n
> \end{bmatrix}.
> $$

> **Lemma 1** (generalized Helmholtz decomposition).
> $$
> H(w) = S(w) + A(w), s.t. S = S^T,A + A^T = 0.
> $$

> **Definition**:
>
> - **Potential game**: $$A(w) = 0$$;
> - Hamiltonian game $$ S(w) = 0$$.

## Stable fixed points

> **Definition**. A fixed point $$w^*$$, with $$\xi(w^*) = 0$$, is stable is $$S(w) \succeq 0$$ and unstable if $$S(w) \prec 0$$ for w in a neighborhood of $$w^*$$.

> **Definition**.(Local Nash equilibrium).
>
> $$\forall i$$, $$\exists$$ neighborhood $$U_i$$ of $$w^*_i$$ such that $$l_i(w_i, w^*_{-i}) \ge l_i(w^*_i, w^*_{-i})$$ for all $$w_i \in U_i$$.

> **Lemma** (Stable fixed points are local Nash equilibria).
>
> If fixed point $$w^*$$ is stable then it is a local Nash equilibrium.

> **Lemma** (Local Nash equilibria are stable fixed points in **two-player zero-sum games**). If a game is two-player zero-sum, then local Nash equilibria are stable fixed points.

## Potential games

(well studied.)

> For our purposes, they are games where simultaneous gradient descent on the
> losses is gradient descent on a single function.  

Introduce potential function with an example.
$$
l_1(x, y) = \frac{x^2}{2} + 2xy, l_2(x, y) = \frac{y^2}{2}+ 2xy.
$$
Then
$$
\xi=(x+2y, y+2x)^T, H = \begin{bmatrix}1& 2\\ 2& 1\end{bmatrix}.
$$
It's potential function is
$$
\phi(x,y) = \frac{x^2}{2} + 2xy + \frac{y^2}{2}.
$$
Note: There are more general definitions.

## Hamiltonian games

> **Definition**(Hamiltonian function)
> $$
> \mathcal{H}(w) = \frac{1}{2}\Vert \xi(w) \Vert^2_2.
> $$
> Note: the name is improper. $$\mathcal{H}$$ is only a Hamiltonian function for $$\xi$$ if the game is Hamiltonian.

> **Theorem** 
>
> - If the game is Hamiltonian then
>   - $$\nabla \mathcal{H} = A^T \xi$$;
>   - $$\xi$$ preserves the level sets of $$\mathcal{H}$$ since $$\langle \xi, \nabla \mathcal{H}\rangle = 0$$.
> - If the Hessian is invertible and $$\lim_{\Vert w \Vert\rightarrow \infty} \mathcal{H}(w) = \infty$$ then gradient descent on $$\mathcal{H}$$ converges to a local Nash equilibrium.

**proof**:

- Direct computation shows $$\nabla \mathcal{H} = H^T \xi$$ for any game.
- For the first statement, if the game is Hamiltonian then $$H = A$$.
- For the second statement,  $$\langle \xi, \nabla \mathcal{H}\rangle =  \xi^T A^T \xi = 0$$. 
  $$
  \xi^T A^T \xi = (\xi^T A^T \xi)^T = \xi^T A \xi = -\xi^T A^T \xi = 0.
  $$

- For the third, gradient descent converges to $$\nabla \mathcal{H} = H^T \xi = 0$$ and $$H$$ is invertible then clearly $$\xi(w) = 0$$. Since $$S = 0$$ (in a Hamiltonian game), the point is local Nash equilibrium.

## Algorithm

### Finding fixed points: Consensus optimization

> **Definition** (Consensus optimization).
> $$
> \xi + \lambda \cdot H^T\xi = \xi + \lambda \cdot \nabla \mathcal{H}.
> $$

> Unfortunately, consensus optimization can converge to unstable fixed points even in simple cases where the ‘game’ is to minimize a single function.

Here is the example:
$$
l_1(x, y) = l_2(x, y) = -\frac{\kappa}{2}(x^2 + y^2),
$$
then
$$
\xi = -\kappa \cdot (x,y)^T, H = -\begin{bmatrix} \kappa& 0\\0 & \kappa \end{bmatrix}.
$$
Note that $$\Vert \xi \Vert^2 = \kappa^2(x^2 + y^2)$$ and
$$
\xi + \lambda \cdot H^T \xi = \kappa(\lambda \kappa - 1)\cdot (x, y)^T.
$$
Descent on $$\xi + \lambda H^T \xi$$ converges to the global maximum $$(x, y) = (0, 0)$$ unless $$ \lambda < \frac{1}{\kappa}$$.

### Finding stable fixed points: Symplectic Gradient Adjustment(SGA)

> **Desiderate**. To find stable fixed points, an adjustment $$\xi_\lambda$$ to the game dynamics should satisfy
>
> - D1: compatible with game dynamics: $$\langle \xi_\lambda, \xi \rangle = \alpha_1 \Vert \xi \Vert^2_2$$;
> - D2: compatible with potential dynamics: if the game is a potential game then $$\langle \xi_\lambda, \nabla \phi \rangle = \alpha_2 \Vert \nabla \phi \Vert^2$$;
> - D3: compatible with Hamiltonian dynamics: If the game is Hamiltonian then $$\langle \xi_\lambda, \nabla \mathcal{H} \rangle = \alpha_3 \Vert \nabla \mathcal{H} \Vert^2_2$$;
> - D4: attracted to stable equilibria: in neighborhoods where $$S \succ 0$$, require $$\theta(\xi_\lambda, \nabla \mathcal{H}) \le \theta(\xi, \nabla \mathcal{H})$$;
> - D5: repelled by unstable equilibria: in neighborhoods where $$S \prec 0$$, require $$\theta(\xi_\lambda, \nabla\mathcal{H}) \ge \theta(\xi, \nabla \mathcal{H})$$.

> **Definition** (Symplectic gradient adjustment(SGA))
> $$
> \xi_\lambda = \xi + \lambda A^T \xi.
> $$

> **Proposition 4**. Symplectic gradient adjustment satisfies D1-D3 for $$\lambda > 0$$.
>
> **Proof**.
>
> - $$\xi^T A^T \xi = 0 \Rightarrow \langle \xi_\lambda, \xi \rangle = \alpha_1 \Vert \xi \Vert^2_2$$;
> - For potential game $$A = 0$$, then $$ \xi_\lambda = \xi = \nabla \phi$$;
> - For Hamiltonian dynamics: $$ \langle \xi_\lambda, \nabla \mathcal{H} \rangle = \langle \xi_\lambda, H^T \xi \rangle = \langle \xi_\lambda, A^T \xi \rangle = \lambda \xi^T A A^T \xi = \lambda\Vert \nabla \mathcal{H} \Vert^2_2$$;

> **Lemma**. If $$S \succeq 0$$ is symmetric positive semidefinite and S commutes with A ($$ AS = SA$$), then
> $$
> \langle \xi_\lambda, \nabla \mathcal{H}\rangle \ge 0, \forall \lambda \ge 0.
> $$
> **Proof**.
> $$
> \begin{align*}
> &\langle \xi_\lambda, \nabla \mathcal{H}\rangle\\
> =& \langle \xi + \lambda A^T \xi, (S + A)^T \xi\rangle \\
> =& \xi^T S \xi + \xi^T A^T \xi + \lambda \xi^T A S^T \xi + \lambda \xi^T A A^T \xi\\
> =& \xi^T S \xi + \lambda \xi^T A A^T \xi \ge 0. 
> \end{align*}
> $$

> **Theorem 5**.  Let S be a symmetric matrix with eigenvalues $$\sigma_{max} \ge \ldots \ge \sigma_{min}$$. The additive condition number of S is $$\kappa = \sigma_{\max} - \sigma_{\min}$$. If $$S \succeq 0$$ is positive semidefinite with additive condition number $$\kappa$$ then $$\lambda \in (0, \frac{4}{\kappa})$$ implies
> $$
> \langle \xi_{\lambda}, \nabla \mathcal{H}\rangle\ge 0.
> $$
> If S is negative semidefinite, then $$\lambda \in (0, \frac{4}{\kappa})$$ implies
> $$
> \langle \xi_{-\lambda}, \nabla\mathcal{H}\rangle \le 0.
> $$
> The inequalities are strict if H is invertible.

**Proof**:

We first proof the case $$ S \succeq 0$$.
$$
\begin{align*}
&\langle \xi_\lambda, \nabla \mathcal{H}\rangle\\
=& \langle \xi + \lambda A^T \xi, (S + A)^T \xi\rangle \\
=& \xi^T S \xi + \xi^T A^T \xi + \lambda \xi^T A S^T \xi + \lambda \xi^T A A^T \xi\\
=& \xi^T S \xi + \lambda \xi^T A S^T \xi + \lambda \xi^T A A^T \xi
\end{align*}
$$
Let $$\tilde S = S - \sigma_{\min} \cdot I$$, where $$I$$ is the identity matrix. Then
$$
\xi^T S \xi + \lambda \xi^T A S \xi + \lambda \xi^T A A^T \xi \ge \xi^T \tilde S \xi + \lambda \xi^T A \tilde S \xi + \lambda \xi^T A A^T \xi
$$
since $$\xi^T S \xi \ge \xi^T \tilde S \xi$$ and $$\xi^T A \tilde S \xi = \xi^T A S \xi - \sigma_{\min} \xi^T A \xi = \xi^T AS \xi$$.

Since S is positive semidefinite, there exists an upper-triangular square-root matrix T such that $$T^T T = \tilde S$$ and so $$\xi^t \tilde S \xi = \Vert T \xi \Vert^2$$. Further,
$$
\vert \xi^T A \tilde S \xi \vert \le \Vert A^T \xi \Vert \cdot \Vert T^T T \xi\Vert \le \sqrt{\sigma_{max} - \sigma_{min}}\cdot \Vert A^T \xi \Vert \cdot \Vert T \xi \Vert.
$$
since $$\Vert T \Vert_2 \le \sqrt{\sigma_{max} - \sigma_{min}}$$ (???). Then
$$
\begin{align*}
&\langle \xi_\lambda, \nabla \mathcal{H}\rangle\\
=& \xi^T S \xi + \lambda \xi^T A S^T \xi + \lambda \xi^T A A^T \xi\\
\ge& \Vert T\xi \Vert^2 + \lambda(\Vert A \xi \Vert^2 + \langle A\xi, \tilde S\xi\rangle)\\
(???)\ge& \Vert T\xi \Vert^2 + \lambda(\Vert A \xi \Vert^2 - \Vert A \xi \Vert \Vert \tilde S \xi \Vert)\\
\ge& \Vert T\xi \Vert^2 + \lambda(\Vert A \xi \Vert^2 - \sqrt{\sigma_{\max}-\sigma_{min}}\Vert A \xi \Vert \Vert T \xi \Vert)\\
=&(\Vert T\xi \Vert - \sqrt{\lambda} \Vert A \xi \Vert)^2
+ \Vert A \xi \Vert \Vert T \xi \Vert(2\sqrt{\lambda} - \lambda\sqrt{\sigma_{max}-\sigma_{min}}).
\end{align*}
$$
If $$S \preceq 0$$: $$\tilde S = S - \sigma_{min} \cdot I$$，
$$
\begin{align*}
&\langle \xi_\lambda, \nabla \mathcal{H}\rangle\\
=& \langle \xi + \lambda A^T \xi, (S + A)^T \xi\rangle \\
=& \xi^T S \xi + \xi^T A^T \xi + \lambda \xi^T A S^T \xi + \lambda \xi^T A A^T \xi\\
=& \xi^T S \xi + \lambda \xi^T A S^T \xi + \lambda \xi^T A A^T \xi\\
\le& \xi^T \tilde S \xi + \lambda \xi^T A \tilde S^T \xi + \lambda \xi^T A A^T \xi\\
(???)\le& \Vert T \xi \Vert^2 + \lambda\Vert A \xi \Vert^2  + \lambda \Vert A \xi \Vert \Vert \tilde S \xi \Vert\\
\le& \Vert T \xi \Vert^2 + \lambda\Vert A \xi \Vert^2  + \lambda \sqrt{\sigma_{\max}-\sigma_{\min}}\Vert A \xi \Vert \Vert T \xi \Vert\\
=&(\Vert T\xi \Vert - \sqrt{\lambda} \Vert A \xi \Vert)^2
+ \Vert A \xi \Vert \Vert T \xi \Vert(2\sqrt{\lambda} + \lambda\sqrt{\sigma_{max}-\sigma_{min}}).
\end{align*}
$$

### How to pick $$sign(\lambda)$$

$$
\langle \xi, \nabla\mathcal{H}\rangle = \xi^T (S + A)^T \xi = \xi^T S \xi.
$$

It follows that for $$\xi \ne 0$$:
$$
\begin{cases}
S\succeq 0 \Rightarrow \langle \xi, \nabla \mathcal{H} \rangle \ge 0;\\
S\preceq 0 \Rightarrow \langle \xi, \nabla \mathcal{H} \rangle \le 0.
\end{cases}
$$

> **Definition** (Infinitesimal alignment) of $$\xi_{\lambda} = u + \lambda v$$ with a third vector w is
> $$
> align(\xi_\lambda, w) := \frac{d}{d\lambda} \{cos^2 \theta_\lambda\}\Big\vert_{\lambda=0},\quad 
> \theta_{\lambda} := \theta(\xi_\lambda, w).
> $$

> **Lemma**
> $$
> sign(align(\xi_\lambda, \nabla\mathcal{H})) = sign(\langle\xi, \nabla \mathcal{H}\rangle \cdot \langle A^T \xi, \nabla \mathcal{H}\rangle).
> $$
> **Proof**. 
>
> Because
> $$
> \begin{align*}
> &cos^2 \theta_{\lambda} = \left(\frac{\langle \xi_\lambda, \nabla\mathcal{H}\rangle}{\Vert \xi_\lambda \Vert \cdot \Vert \nabla \mathcal{H} \Vert}\right)^2\\
> =& \frac{(\langle\xi + \lambda A^T \xi, \nabla\mathcal{H}\rangle)^2}{\Vert \xi + \lambda A^T \xi\Vert^2 \cdot \Vert \nabla\mathcal{H}\Vert^2}\\
> =& \frac{\langle\xi, \nabla\mathcal{H}\rangle + 2\lambda\langle \xi, \nabla \mathcal{H}\rangle\langle A^T \xi, \nabla\mathcal{H}\rangle + O(\lambda^2)}{(\Vert \xi \Vert^2 + \lambda^2 \xi^T A A^T \xi)\Vert \nabla\mathcal{H}\Vert^2},
> \end{align*}
> $$
>
> therefore
> $$
> sign\left\{\frac{d}{d\lambda} cos^2 \theta_{\lambda}\right\}
> = sign\{\langle \xi, \nabla \mathcal{H}\rangle\langle A^T \xi, \nabla\mathcal{H}\rangle\}
> $$

> **Proposition**. Desiderata D4-D5 are satisfied for $$\lambda$$ such that $$\lambda\cdot \langle \xi, \nabla \mathcal{H}\rangle \cdot \langle A^T\xi, \nabla \mathcal{H}\rangle \ge 0$$.
>
> **Proof**.
>
> If we are in a neighborhood of a stable fixed point then $$\langle \xi, \nabla \mathcal{H}\rangle \ge 0$$. Then $$sign(align(\xi_\lambda, \nabla\mathcal{H})) = sign(\langle A^T \xi, \nabla \mathcal{H}\rangle) = sign(\lambda)$$, which means $$cos^2 \theta_{\lambda}$$ will increase and the angle will decrease.
>
> If we are in a neighborhood of a unstable fixed point then $$\langle \xi, \nabla \mathcal{H}\rangle < 0$$. Then $$sign(align(\xi_\lambda, \nabla\mathcal{H})) = -sign(\langle A^T \xi, \nabla \mathcal{H}\rangle) = sign(\lambda)$$,  which means $$cos^2\theta_{\lambda}$$ will decrease and the angle will increase.

### Alignment and convergence rates

> **Theorem**. Suppose f is convex and L-Lipschitz smooth. Let $$w_{t+1} = w_t - \eta \cdot v$$ where $$\Vert v \Vert = \Vert \nabla f(w_t) \Vert$$. Then the optimal step size is $$\eta^* = \frac{cos \theta}{L}$$ where $$\theta = \theta(\nabla f(w_t), v)$$, with
> $$
> f(w_{t+1}) \le f(w_t) - \frac{cos^2\theta}{2L}\cdot \Vert \nabla f(w_t) \Vert^2
> $$
> **proof**:
> $$
> \begin{align*}
> f(w_{t+1}) \le& f(w_t) + \langle\nabla f(w_t), w_{t+1} - w_t\rangle + \frac{L}{2}\Vert w_{t+1} - w_t \Vert^2\\
> =& f(w_t) - \eta\cdot\langle \nabla f, v \rangle + \eta^2 \frac{L}{2} \Vert v\Vert^2\\
> =&f(w_t) - \eta \cos\theta \Vert \nabla f\Vert \Vert v \Vert + \eta^2 \frac{L}{2} \Vert v \Vert^2\\
> =& f(w_t) - \eta\left(\cos \theta - \frac{\eta L}{2} \right) \Vert \nabla f \Vert^2
> \end{align*}
> $$


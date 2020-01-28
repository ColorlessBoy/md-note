# An Emphatic Approach for the Problem of Off-policy Temporal-difference Learning

## 2. On-policy Convergence of TD(0)

$$
\begin{align*}
\theta_{t+1} =& \theta_t + \alpha(R_{t+1} + \gamma \theta^T_t \phi(S_{t+1}) - \theta^T_t\phi(S_t)) \phi(S_t)\\
=& \theta_t + \alpha(R_{t+1}\phi(S_t) - \phi(S_t)(\phi(S_t) - \gamma \phi(S_{t+1}))^T \theta_t)\\
=& \theta_t + \alpha(b_t - A_t w_t)\\
=& (I-\alpha A_t)\theta_t + \alpha b_t
\end{align*}
$$


$$
A = \lim_{t\rightarrow \infty} \mathbb{E}_{\pi}[A_t] = \Phi^T D_\pi(I - \gamma P_\pi) \Phi.
$$

Sutton uses the theorem positive definiteness is assured if all of its columns sum to a nonnegative number.
$$
\begin{align*}
&\sum_i [D_\pi (I - \gamma P_\pi)]_{ij} \\
=& \sum_i \sum_k[D_\pi]_{ik}[I - \gamma P_pi]_{kj}\\
=& \sum_i [D_\pi]_{ii} [I - \gamma P_\pi]_{ij}\\
=& \sum_i d_\pi(i) [I - \gamma P_\pi]_{ij}\\
=& [d^T_\pi (I - \gamma P_\pi)]_{j}\\
=& [d^T_\pi - \gamma d^T_\pi P_\pi]_j\\
=& [d^T_\pi - \gamma d^T_\pi]_j\\
=& (1 - \gamma) d_\pi(i) > 0.
\end{align*}
$$

## 3. Instability of Off-policy TD(0)

$$
\theta_{t+1} = \theta_t + \frac{\pi(A_t\vert S_t)}{\mu(A_t\vert S_t)}\alpha
(R_{t+1} + \gamma \theta^T_t \phi(S_{t+1}) - \theta^T_t \phi(S_t))\phi(S_t)
$$

$$
A = \lim_{t\rightarrow \infty} \mathbb{E}_\mu[\rho_t \phi(S_t)(\phi(S_t) - \gamma \Phi(S_{t+1}))^T] = \Phi^T D_\mu(I - \gamma P_\pi) \Phi.
$$

> **Example 1**:
>
> <img src="C:\Users\pengl\Documents\md-notes\pic\Emphaic_TD_1.png" alt="Emphaic_TD_1" style="zoom:50%;" />
>
> So $$ d_\mu = [0.5, 0.5]^T$$ and $$ P_\pi = \begin{bmatrix} 0 & 1 \\ 0 & 1\end{bmatrix}$$:
> $$
> D_\mu(I - \gamma P_\pi) =
> \begin{bmatrix} 0.5 & 0 \\ 0 & 0.5\end{bmatrix}
> \begin{bmatrix} 0 & 1 \\ 0 & 1\end{bmatrix}
> = \begin{bmatrix} 0.5 & -0.45 \\ 0.05 & 1\end{bmatrix}.
> $$
> If $$x = [1, 2]^T$$, we have
> $$
> x^T D_\mu(I - \gamma P_\pi) x = -0.2 < 0
> $$

## 4. Off-policy Stability of Emphatic TD(0)

> The deep reason for the difficulty of off-policy learning is that the behavior policy may take the process to a distribution of states different from that which would be encountered under the target policy, yet the states might appear to be the same or similar because of function approximation. 

> It is theoretically possible to convert the state weighting from $$d_\mu$$ to $$d_\pi$$ using the product of all importance sampling ratios from time 0, but in practice this approach has extremely high variance. 

We construct a correct distribution:
$$
f = d_\mu + \gamma P^T_\pi d_\mu + (\gamma P^T_\pi)^2 d_\mu + \cdots 
= (I - \gamma P^T_\pi)^{-1} d_\mu
$$
and $$F = diag(f)$$. Then the off-policy TD becomes:
$$
w_{t+1} = w_t + \alpha F_t \rho_t(R_{t+1} + \gamma w^T_t x_{t+1} - w^T_t x_t) x_t.
$$

$$
\begin{align*}
& A = \lim_{t\rightarrow \infty} \mathbb{E}[A_t]
= \lim_{t\rightarrow \infty} \mathbb{E}_\mu[F_t \rho_t x_t(x_t - \gamma x_{t+1})^T]\\
=& \sum_s d_\mu(s) \lim_{t\rightarrow \infty}\mathbb{E}_\mu[F_t \vert S_t = s]
\mathbb{E}_\mu[\rho_t x_t(x_t - \gamma x_{t+1})\vert S_t = s]\\
=& \sum_s f(s) x(s) \left(x(s) - \gamma \sum_{s'} [P_\pi]_{ss'} x(s')\right)^T\\
=& X^T F(I - \gamma P_\pi) X.
\end{align*}
$$

Then
$$
\begin{align*}
&\sum_i [F(I - \gamma P_\pi)]_{ij} \\
=& \sum_i\sum_k [F]_{ik} [I - \gamma P_\pi]_{kj} \\
=& \sum_i [F]_{ii}[I-\gamma P_\pi]_{ij}\\
=& \sum_i [f]_i [I - \gamma P_\pi]_[ij]\\
=& [f^T (I - \gamma P_\pi)]_j\\
=& [d^T_\mu (I - \gamma P_\pi)^{-1} (I - \gamma P_\pi)]_j\\
=& d_\mu(i) > 0
\end{align*}
$$
We estimate $$F$$ by
$$
F_t(s) :=\gamma \rho_{t-1} F_{t-1}(s) + 1.
$$
$$ \forall t > 0, F_{-1} = 0 $$.

## 5. The General Case

### Generalization 1

$$
G_t := R_{t+1} + \gamma(S_{t+1})R_{t+2} + \gamma(S_{t+1})\gamma(S_{t+2}) R_{t+3} + \cdots.
$$

- If $$ \gamma(S_t) = 0$$, then the time of accumulation is fully terminated;
- If $$ \gamma(S_t) \le 1$$, then it is partially terminated, of soft terminated.

### Generalization 2

$$
MSVE(w) := \sum_{s \in S} d_\mu(s) i(s) (v_\pi(s) - w^T x(s))^2.
$$

- In the past it was typically assumed that we were interested in valuing states in direct proportion to how often they occur, but this is not always the case;
- (This generalization is developed for the first time in this paper);
- 

### Generalization 3

$$
G^\lambda_t = R_{t+1} + \gamma_{t+1} [(1 - \lambda_{t+1}) w^T_t x_{t+1} + \lambda_{t+1} G^\lambda_{t+1}].
$$

### Emphatic TD($$\lambda$$)

- $$ w_{t+1} = w_t + \alpha(R_{t+1} + \gamma_{t+1} w^T_t x_{t+1} - w^T_t x_t) e_t$$;
- $$ e_t = \rho_t(\gamma_t \lambda_t e_{t-1} + M_t x_t)$$;
- $$ M_t = \lambda_t i(S_t) + (1 - \lambda_t) F_t$$;
- $$F_t = \rho_{t-1} \gamma_t F_{t-1} + i(S_t)$$, with $$F_{-1} = 0$$.

## 6. Off-policy Stability of Emphatic TD($$\lambda$$)

> **Theorem 1** (Stability of Emphatic TD($$\lambda$$)) For any
>
> - S, A are finite;
> - Behavior policy $$\mu$$ with $$d_\mu(s) > 0, \forall s \in S $$;
> - If $$ \pi(a \vert s) > 0 $$, then $$\mu(a \vert s) > 0$$;
> - $$\prod^{\infty}_{k=1} \gamma(S_{t+k}) = 0$$, w.p.1, $$\forall t > 0$$;
> - $$ \lambda: S \rightarrow [0,1]$$;
> - $$\exists s \in S, i(s) > 0$$;
> - $$ \Phi(s)$$ has linearly independent columns.

## 7. Derivation of the Emphasis Algorithm

???



[1]R. Sutton, A. R. Mahmood, and M. White, “An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning,” vol. 17, Mar. 2015.
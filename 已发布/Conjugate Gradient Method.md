# Conjugate Gradient Method[^1]

## 1. Introduction of Conjugancy

> **Definition 1** (Q- conjugate directions) Let Q be symmetric metrix. $$\{d_1, \ldots, d_k\}$$ vectors ($$d_i \in \mathbb{R}^n, d_i \ne 0$$) are Q-orthogonal (conjugate) w.r.t. Q, if 
> $$
> d^T_i Q d_j = 0, \forall i \ne j.
> $$

> **Lemma 1** Let Q be positively symmetric metrix. If $$\{d_1, \ldots, d_k\}$$ are Q-conjugate, then they are linearly independent.
>
> If $$ d_k = \sum^{k-1}_{i=1}\alpha_i d_i$$, then $$0 \prec d^T_k Q d_k  = \sum^{k-1}_{i=1} \alpha_i d^T_k Q d_i = 0$$, which causes a contradicition.

Then we will show the usefulness of the conjugate.

Here is a quadratic problem

$$
\min_{x} \frac{1}{2} x^T Q x - b^Tx
$$

where Q is positively symmetric metrix. We denote it's optimal point $$x^*$$ which is also the unique solution of equation

$$
Qx = b
$$

Let $$\{d_0, d_1, \ldots, d_{n-1}\}$$ vectors be Q-conjugate, then $$x^*$$ can be written as
$$
x^* = \sum^{n-1}_{i=1} \alpha_i d_i.
$$
So we need to calculate $$\alpha_i$$ for $$x^*$$. The beautiful property is that
$$
d^T_k b = d^T_kQx^* = \sum^{n-1}_{i=1}\alpha_i d^T_kQd_i = \alpha_k d^T_k Q d_k,\\
\Rightarrow \alpha_k = \frac{d^T_k b}{d^T_k Q d_k}, x^* = \sum^{n-1}_{i=1}\frac{d^T_ib}{d^T_iQd_i} d_i.
$$
**Superisingly, the matrix inversion is unnecessary**.

## 2. Conjugate Direction Theorem

> **Theorem 1** (Conjugate Direction Theorem, Conjugate Direction Method). If
>
> - $$\{d_0, d_1, \ldots, d_{n-1}\} \in \mathbb{R}^n$$ are Q-conjugate,
> - $$\forall x_0 \in \mathbb{R}^m$$,
> - $$x_{k+1} = x_k + \alpha_k d_k$$,
> - $$g_k = Qx_k - b$$
> - and $$\alpha_k = -\frac{g^T_kd_k}{d^T_k Q d_k}$$,
>
> then $$x_n = x^* $$.
>
> **proof**: The first four conditions are constructed, so our proof is by getting the last condition.
> $$
> \begin{cases}
> x_1 = x_0 + \alpha_0 d_0\\
> x_2 = x_0 + \alpha_0 d_0 + \alpha_1 d_1\\
> \vdots\\
> x_k = x_0 + \alpha_0 d_0 + \alpha_1 d_1 + \cdots + \alpha_{k-1} d_{k-1}\\
> \vdots\\
> x_n = x_0 + \alpha_0d_0 + \alpha_1 d_1 + \cdots + \alpha_{n-1}d_{n-1}
> \end{cases}
> $$
> If we want $$x_n = x^*$$, how to choose $$a_i$$? Here is some discussion.
> $$
> d^T_k Q (x^* - x_0)= d^T_k Q (x^n - x_0) = d^T_k Q \cdot \sum^{n-1}_{i=0}\alpha_i d_{i} = \alpha_k d^T_k Q d_k \\
> \Rightarrow \alpha_k = \frac{d^T_k Q (x^* - x_0)}{d^T_k Q d_k}
> $$
> Notice that $$d^T_k Q(x_k - x_0) = d^T_k Q\sum^{k-1}_{i=0} \alpha_i d_i = 0$$ and $$Q x^* = b$$ then
> $$
> \begin{align*}
> \alpha_k 
> =& \frac{d^T_k Q (x^* - x_0)}{d^T_k Q d_k}
> = \frac{d^T_k Q (x^* - x_k + x_k - x_0)}{d^T_k Q d_k}\\
> =& \frac{d^T_k Q (x^* - x_k)}{d^T_k Q d_k}
> = \frac{d^T_k (b - Q x_k)}{d^T_k Q d_k}
> = -\frac{d^T_k g_k}{d^T_k Q d_k}
> \end{align*}
> $$

## 3. Expanding Subspace Theorem

> **Theorem 2** (Expanding Subspace Theorem).
>
> - $$\{d_0, d_1, \ldots, d_{n-1}\} \in \mathbb{R}^n$$ are Q-conjugate,
> - $$\forall x_0 \in \mathbb{R}^m$$,
> - $$x_{k+1} = x_k + \alpha_k d_k$$,
> - $$g_k = Qx_k - b$$
> - and $$\alpha_k = -\frac{g^T_kd_k}{d^T_k Q d_k}$$。
>
> Let $$ \mathcal{B}_k = span(d_0, \ldots, d_{k-1})$$, then
> $$
> x_k = \arg\min_{x=x_{k-1} + \alpha d_{k-1}, \alpha\in \mathbb R}
> \frac{1}{2} x^T Q x - b^Tx,\\
> x_k = \arg\min_{x=x_{0} + \mathcal{B}_k}
> \frac{1}{2} x^T Q x - b^Tx.\\
> $$
> **proof**: The first equation is travial:
> $$
> d^T_{k-1}Q(x_{k-1} + \alpha d_{k-1}) - d^T_{k-1} b \Rightarrow \alpha_{k-1} = \frac{d^T_{k-1}(Q x_{k-1} - b)}{d^T_{k-1} Q d_{k-1}}
> $$
> We then need to proof the second equation.
>
> Because the function is strongly convex, if we can guarantee that $$g_k \perp \beta_k$$, then the second equation is proofed.
>
> We will proof $$g_k \perp \mathcal{B}_k$$ by induction:
>
> - $$\mathcal{B}_k = \emptyset$$;
>
> - Assume that $$g_k \perp \mathcal{B}_k$$. We have
>   $$
>   g_{k+1} = Qx_{k+1} - b = Qx_k - b + \alpha_k Q d_k = g_{k} + \alpha_k Q d_k.
>   $$
>   Firstly, 
>   $$
>   d^T_k g_{k+1} = d^T_k g_k + \alpha_k d^T_k Q d_k = d^T_k g_k - \frac{g^T_k d_k}{d^T_k Q d_k} d^T_k Q d_k = 0
>   $$
>   Secondly, we will proof $$g_{k+1} \perp \mathcal{B}_k$$ by proofing $$g_{k+1} \perp b_i, \forall i < k$$.
>   $$
>   d^T_i g_{k+1} = d^T_i g_k - \alpha_k d^T_i Q d_k = 0 (by\ assumption)
>   $$

The expanding subspace theorem can also proof that $$x_n = \arg\min_{x \in \mathbb{R}^n} \frac{1}{2}x^T Q x - b^T x = x^*$$.

## 4. Conjugate Gradient Algorithm

> **Algorithm 1**(Conjugate Gradient Algorithm) For initialization, arbitrarily choose $$x_0 \in \mathbb{R}^n$$ and let $$ d_0 = -g_0 = b-Q{x_0}$$ .
>
> For any $$k (\ge 0)$$ steps: 
>
> - $$g_k = Q x_k - b$$;
>
> - Because $$g_k \perp \{d_0, \ldots, d_{k-1}\}$$, therefore we can exactly construct $$d_k$$ from $$\{d_0, \ldots, d_{k-1}, g_k\}$$:
> $$
> d_k = -g_k + \sum^{k-1}_{i=0}\frac{g^T_k Q d_{i}}{d^T_{i} Q d_{i}}d_{i}
> = -g_k + \frac{g^T_k g_k}{g^T_{k-1} g_{k-1}}d_{k-1} (Explaination\ as\ follows);
> $$
>
> - $$\alpha_k = -\frac{g^T_k d_k}{d^T_k Q d_k}$$;
> - $$x_{k+1} = x_{k} + \alpha_k d_k $$.

We would like to simplify the equation. Firstly, we notice that 

$$
g_{i+1} = Q x_{i+1} - b = Q(x_{i} + \alpha_i d_i) - b = g_{i} + \alpha_i Q d_i\\
\Rightarrow Qd_i = \frac{1}{\alpha_i}(g_{i+1} - g_{i}).
$$

Because $$\forall j > i, g_j \perp\{d_0, \ldots, d_{j-1}\}$$, we have
$$
\forall j > i, g_j^T g_i = g^T_j \left(-d_i + \sum^{k-1}_{i=0}\frac{g^T_j Q d_{i}}{d^T_{i} Q d_{i}}d_{i} \right) = 0.
$$
Then we will get
$$
\begin{align*}
d^T_i Q d_i =& d^T_i Q \left(-g_i + \sum^{k-1}_{i=0}\frac{g^T_k Q d_{i}}{d^T_{i} Q d_{i}} d_i\right) \\
=& -d^T_i Q g_i = -g^T_i Qd_i\\
=& \frac{1}{\alpha_i} g^T_i (g_{i} - g_{i+1}) = \frac{1}{\alpha_i}g^T_i g_i
\end{align*}
$$
and
$$
g^T_k Q d_i = \frac{1}{a_i}g^T_k(g_{i+1} - g_{i}).
$$
the equation becomes
$$
d_k = -g_k + \sum^{k-1}_{i=0}\frac{g^T_k(g_{i+1}-g_{i})}{ g^T_ig_{i}}d_i = -g_k +\frac{g^T_k g_k}{g^T_{k-1} g_{k-1}}d_{k-1}
$$








[^1]: http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/conjugate_direction_methods.pdf


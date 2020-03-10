# MCMC

Markov chain's stationary distribution $$\pi$$:
$$
\pi^T P = \pi^T.
$$
A sufficient condition is
$$
P(i \vert j) \pi(j) = P(j \vert i) \pi(i).
$$
**proof**:
$$
\sum^D_{j=1} P(i \vert j) \pi(j) = \sum^D_{j=1} \pi(i) P(j \vert i) = \pi(i).
$$

## Vanilla MCMC

We construct MCMC from preceding sufficient condition:

1. For any state transition matrix Q, we convincingly assume that
   $$
   Q(j \vert i) \pi(i) \ne Q(i \vert j) \pi(j), \quad i \ne j;
   $$

2. We introduce a matrix $$\alpha$$ satisfies
   $$
   Q(j\vert i) \pi(i) \cdot \alpha(i, j) = Q(i \vert j) \pi(j) \cdot \alpha(j, i), \quad i \ne j
   $$

   $$
   \begin{bmatrix}
   1 - \sum_{i \ne 1}Q(1, i) \alpha(1, i) & Q(1, 2) \alpha(1, 2) & \cdots & Q(1, n) \alpha(1,n)\\
   Q(2, 1) \alpha(2, 1) & 1 - \sum_{i \ne 2} Q(2, i) \alpha(2,i) & \cdots & Q(2, n) \alpha(2, n)\\
   \vdots & \vdots & \ddots & \vdots \\
   Q(n, 1) \alpha(n, 1) & Q(n, 2) \alpha(n, 2) & \cdots & 1 - \sum_{i \ne n} Q(n, i) \alpha(n, i)
   \end{bmatrix}
   $$

   where
   $$
   \alpha(i, j) = Q(i \vert j) \pi(j), \quad \alpha(j, i) = Q(j \vert i) \pi(i).
   $$
   We also call $$\alpha$$ rejection ratio.

   And the corresponding transition matrix is
   $$
   \begin{bmatrix}
   1 - \sum_{i \ne 1}Q(1, i) \pi(i) Q(i, 1) & Q(1, 2) \pi(2) Q(2, 1) & \cdots & Q(1, n) \pi(n) Q(n, 1)\\
   Q(2, 1) \pi(1) Q(1, 2) & 1 - \sum_{i \ne 2} Q(2, i) \pi(i) Q(i, 2) & \cdots & Q(2, n) \pi(n) Q(n, 2)\\
   \vdots & \vdots & \ddots & \vdots \\
   Q(n, 1) \pi(1) Q(1, n) & Q(n, 2) \pi(2) Q(2, n) & \cdots & 1 - \sum_{i \ne n} Q(n, i) \pi(i) Q(i, n) 
   \end{bmatrix}
   $$

## M-H Sampling

We modify rejection ratio by the following equation
$$
Q(j\vert i) \pi(i) \cdot \alpha(i, j) = Q(i \vert j) \pi(j) \cdot \alpha(j, i)
$$

$$
\Rightarrow Q(j\vert i) \pi(i) \cdot \frac{\alpha(i, j)}{\alpha(j, i)} = Q(i \vert j) \pi(j) \cdot 1
$$

$$
\alpha(i, j) = \min\left\{\frac{Q(i \vert j)\pi(j)}{Q(j \vert i)\pi(i)}, 1\right\}.
$$

If we choose a symmetric matrix $$Q$$, we have
$$
\alpha(i, j) = \min\left\{\frac{\pi(j)}{\pi(i)}, 1\right\}.
$$

## Gibbs Sampling

For 2-dimension point $$(x, y)$$, we have
$$
p(x_1, y_1) p(y_2 \vert x_1) = p(x_1) p(y_1 \vert x_1) p(y_2 \vert x_1)
$$

$$
p(x_1, y_2) p(y_1 \vert x_1) = p(x_1) p(y_2 \vert x_1) p(y_1 \vert x_1),
$$

so we have
$$
p(x_1, y_1) p(y_2 \vert x_1) = p(x_1, y_2) p(y_1 \vert x_1).
$$
For two points $$A = (x, y_A)$$ and $$B = (x, y_B)$$, we have
$$
p(A) p(y_B \vert x) = p(B) p(y_A \vert x).
$$
We can construct the matrix
$$
\begin{bmatrix}
p(y = 1 \vert x = 1) & p(y = 2 \vert x = 1) & \cdots & p(y = n \vert x = 1) \\
p(y = 1 \vert x = 2) & p(y = 2 \vert x = 2) & \cdots & p(y = n \vert x = 2) \\
\vdots & \vdots & \ddots & \vdots \\
p(y = 1 \vert x = n) & p(y = 2 \vert x = n) & \cdots & p(y = n \vert x = n)
\end{bmatrix}
$$

$$
\begin{bmatrix}
p(x = 1 \vert y = 1) & p(x = 2 \vert y = 1) & \cdots & p(x = n \vert y = 1) \\
p(x = 1 \vert y = 2) & p(x = 2 \vert y = 2) & \cdots & p(x = n \vert y = 2) \\
\vdots & \vdots & \ddots & \vdots \\
p(x = 1 \vert y = n) & p(x = 2 \vert y = n) & \cdots & p(x = n \vert y = n)
\end{bmatrix}
$$






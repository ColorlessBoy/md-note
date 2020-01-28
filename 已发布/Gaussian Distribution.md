# Gaussian Distribution

假设误差概率密度函数为$$f(x)$$.

已有独立同分布采样的样本$$\{x_1, x_2, \ldots, x_n\}$$.

极大似然函数为
$$
L(x) = f(x_1 - x) f(x_2 - x) \cdots f(x_n-x).
$$
那么，对数极大似然函数为
$$
\ln L(x) = \sum^n_{i=1}\ln(f(x_i-x)).
$$
我们希望极大似然值x满足：
$$
\frac{d}{dx}\ln L(x) = -\sum^n_{i=1}\frac{f'(x_i-x)}{f(x_i-x)} = 0.
$$
高斯假设x的真值为$$\bar x = \frac{1}{n}x_i$$,那么
$$
\sum^n_{i=1} g(x_i - \bar x) := \sum^n_{i=1}\frac{f'(x_i - \bar x)}{f(x_i-\bar x)} = 0.
$$
对$$x_i$$求偏导可得
$$
\begin{cases}
g'(x_1 - \bar x)(1-\frac{1}{n}) + g'(x_2-\bar x)(-\frac{1}{n})+\cdots+g'(x_n-\bar x)(-\frac{1}{n})=0\\
g'(x_1 - \bar x)(-\frac{1}{n})+g'(x_2 - \bar x)(1-\frac{1}{n})+\cdots+g'(x_n-\bar x)(-\frac{1}{n})=0\\
\vdots\\
g'(x_1 - \bar x)(-\frac{1}{n})+g'(x_2 - \bar x)(-\frac{1}{n})+\cdots+g'(x_n-\bar x)(1-\frac{1}{n})=0
\end{cases}
$$
矩阵秩为$$n-1$$, 所以存在非零解 $$g' = c [1, 1, \ldots, 1]^T$$, 即
$$
g'(x_1 - \bar x) = g'(x_2 - \bar x) = \cdots = g'(x_n - \bar x) = c,
$$
则
$$
g(x) = cx + b.
$$
由于
$$
\sum^n_{i=1} c(x_i - \bar x) + nb = 0 \Rightarrow b = 0,
$$
所以
$$
\frac{f'(x)}{f(x)} = cx \Rightarrow f(x) = k e^{\frac{1}{2}c x^2}.
$$
由于
$$
\int^{\infty}_{-\infty}f(x) dx = \int^{\infty}_{-\infty}k e^{-\frac{x^2}{2 \sigma^2}} dx = 1,
$$
可得
$$
k = \frac{1}{\sqrt{2\pi}\sigma}.
$$
最后我们得到正态分布概率密度函数：
$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}} \sim \mathcal{N}(0, \sigma^2).
$$

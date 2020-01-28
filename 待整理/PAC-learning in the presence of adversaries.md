# PAC-learning in the presence of adversaries



## Abstract

- Imperceptible perturbations generate adversarial examples;

- Attack-evasion instead of attack-defense;

- Corrupted hypothesis classes;

- Adversarial VC-dimension.

 

## Contributions:

- Sample complexity bounds;

- Adversarial VC-dimension for halfspace classifiers with some assumptions;

- The relation between VC-dimension and corresponding AVC-dimension.

 

## Adversarial agnostic PAC-learning

- The adversary presents in the test period rather than training.

- $R \subseteq X \times X$ is binary nearest relation; $N(x) = \{ x' in X: (x, x') \in R \}$.

- Definition 1 (Adversarial Expected Risk). $L_D(h, R) = \mathbb{E}_{(x, y)\sim D}[\max_{x'\in N(x)} l(h(x'), y)]$

- Definition 2 (Adversarial Empirical Risk Minimization). $AERM_{H,R}(S) = \arg\min_{h\in H} L_{S} (h, R)$

- Lemma 1: If $R_1 \subseteq R_2$, $\inf_{h \in H} L_D(h, R_1) \le \inf_{h \in H} L_D(h, R_2)$.

- Definition3 (Learnability and Sample Complexity).

 

## Adversarial VC-dimension and sample complexity

- Corrupted hypothesis classes of halfspace are mapping X to {-1, 1, always_wrong}, we denote corrupted classifier $\kappa_R(h)$;

- Lemma 2: $L_D(h, R) = L_D(\kappa_R(h), I_X)$.

- Lemma 3: Let $\lambda(h) = (x', y) \mapsto l(y, h(x'))$, and $\hat{f} = \lambda(\kappa_R(AERM_{H,R}(x, y))$. With probability $1-\delta$, $\mathbb E_{D}(\hat f(x, y)) - \inf_{f \in \tilde{F}} \mathbb{E}_D(f(x, y)) \le 2R(\tilde F(x, y)) + \sqrt{\frac{32\log (4/\delta)}{n}}$.

  For $\tilde F(x, y) = \{ (\tilde f(x_0, c_0), \cdots, \tilde f(x_{n-1}, y_{n-1})): \tilde f \in \tilde F \}$ and $ R(T) = \frac{1}{n 2^n} \sum_{s \in \{-1, 1\}^n} \sup_{t \in T} s^T t.$

- Definition 4 (Equivalent shattering coefficient definitions). $\sigma(H, i) = \max_{x \in X} |\{(h(x_0), \ldots, h(x_{i-1}): h \in H\}|$ or $\sigma'(F, i) = \max_{(x, y) \in X \times Y} |\{(f(x_0, y_0), \ldots, f(x_{i-1}, y_0): f \in F\}|$

- Definition 5 (Adversarial VC-dimension). $AVC(H, R) = \sup \{ n \in \mathbb{N}: \sigma'(\lambda(\tilde H), n) = 2^n\}$
- Theorem 1 (Sample complexity upper bound with an evasion adversary). $m_{H,R}(\delta, \epsilon) \le C \frac{d\log(d/\epsilon)+\log(1/\delta)}{\epsilon^2}$, where $d = AVC(H, R)$.

## The adversarial VC-dimension of halfspace classifiers

- Definition 6 (Convex constraint on binary adversarial relation). 
  - Let B be a nonempty, closed, convex, origin-symmetric set;
  - $\Arrowvert x \Arrowvert_B = \inf\{ \epsilon \ge 0: x \in \epsilon B \}$;
  - $d_B(x, y) = \Arrowvert x - y \Arrowvert_B$；
  -  $V_B$ is the largest linear subspace contained in B;
  - $R = \{ (x, y): y - x \in B\}$, or equivalently $N(x) = x + B$; Notablly, R encompasses all $l_p$ bounded adversaries.
- Definition 7 (Signed distance). $\delta_B(h, x, y) = yh(x)\inf_{x'\in X: h(x) \ne h(x')} d_B(x, x')$, $D_B(H, \mathbb x, \mathbb y) = \{ (\delta_B(h, x_0), \ldots, \delta_B(h, x_{n-1})): h \in H \}.$
- Refine halfspace classifiers: $\{ (x \mapsto sgn(a^T x - b))\}$. And $sgn(0) = \perp$, which means always wrong.
- Theorem 2. Let H be halfspace hypothesis, B and R satisfy def.6. Then $AVC(H, R) = d+1-dim(V_B)$.  If B is a bounded $l_p$ ball, $dim(V_B) = 0$, and $AVC(H, R) = d+1$. (Proof???)


## Adversarial VC Dimension can be larger

- Theorem 3.



## Related works

- 0-1 loss prevents the efficient implementation of Adversarial ERM to obtain robust classifiers;
- Directly finding a classifier that minimizes the  AERM leads to a saddle point problem;
- Heuristics: replacing the 0-1 loss with smooth surrogates;
- Interesting direction:
  - Other classifiers' adversarial version;
  - The behavior of convex learning problems in the presence of adversaries.


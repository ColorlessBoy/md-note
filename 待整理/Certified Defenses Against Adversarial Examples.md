# Certified Defenses Against Adversarial Examples

## Abstract

- Defenses based on regularization and adversarial training have been proposed, but often followed  by new, stronger attacks that defeat these defenses. Can we somehow end this arms race?
- A semidefinite relaxation that outputs a certificate that for a given network and test input, no attack can force the error to exceed a certain value.
- Providing an adaptive regularizer that encourages robustness against all attacks.

## Introduction

- We are the first to demonstrate a certifiable, trainable and scalable method for defending against adversarial examples on two-layer networks.

## Setup

- Score function: $f(s) = \{f^i(s), i \in [k]\}$;
- Pairwise margin: $f^{ij}(x) = f^i(x) - f^j(x)$;
- Classifier: $C(x) = \arg\min_i f^i(x)$;

- This paper focuses on:
  - $f^i(x) = W^T_i x, W \in \mathbb{R}^{k \times d}$;
  - $f^i(x) = V^T_i \sigma(Wx), W \in \mathbb{R}^{m \times d} and V \in \mathbb{R}^{k \times m}$;
- Attacker $A: X \rightarrow X$ takes an input x and returns  a perturbation $\tilde x$ with constraint $B_\epsilon(x) = \{\tilde x | \Arrowvert \tilde x - x \Arrowvert_\infty \le \epsilon\}$; furthermore, we assume A is white-box adversary: $A_{opt}(x) = \arg\max_{\tilde x \in B_\epsilon(x)}\max_i f^{iy}(\tilde x)$;
- Adversarial loss w.r.t. A as $l_A(x, y) = 1\{C(A(x)) \ne y\}$.

## Certificate on the Adversarial Loss

- We first consider binary classification with classes $Y = \{1, 2\}$; and $y^{true} = 2$. So margin function $f(x) = f^1(x) - f^2(x)$;

- Key results: $f(A(x)) \le f(A_{opt}(x)) \le f(x) + \epsilon \max_{\tilde x \in B_\epsilon(x)}\Arrowvert \nabla f(\tilde x) \Arrowvert_1 \le f_{QP}(x) \le f_{SDP}(x)$;

- Linear classifiers: $f(\tilde x) = f(x) + (W_1 - W_2)^T(\tilde x - x) \le f(x) + \epsilon\Arrowvert W_1 - W_2 \Arrowvert_1$;

- General classifiers: $f(\tilde x) = f(x) + \int^1_0 \nabla f(t\tilde x + (1-t)x)^T(\tilde x - x)dt \le f(x) + \max_{\tilde x \in B_\epsilon(x)}\epsilon \Arrowvert \nabla f(\tilde x)\Arrowvert_1$;

- Two-layer neural networks (Note that $g_{relu} \in [0, 1]$ and $g_{sig} \in [0, 0.25]$):
  $$
  \begin{align*}
  \Arrowvert \nabla_x f(x) \Arrowvert_1 
  = \Arrowvert \nabla_x (V_1 - V_2)^T\sigma(Wx)\Arrowvert_1
  \end{align*}
  = \Arrowvert W^T diag(V_1 - V_2) \sigma'(W\tilde x)\Arrowvert_1\\
  \le \max_{s \in [0,1]^m} \Arrowvert W^T diag(v) s\Arrowvert_1
  = \max_{s\in[0,1]^m, t\in[-1, 1]^d} t^T W^T diag(v) s
  $$
  Then, $f(A_{opt}(x)) \le f(x) + \epsilon \max_{s\in[0,1]^m, t\in[-1, 1]^d} t^T W^T diag(v) s := f_{QP}(x)$

- $\max_{s \in [-1, 1]^m, t \in[-1, 1]^d} \frac{1}{2} t^T W^T diag(v)(\vec 1 + s)$;

  Let $y := [1,t,s]^T$

  $M(v, W) = \begin{bmatrix} 0 & 0 & 1^T W^T diag(v)\\ 0 & 0 & W^T diag(v) \\ diag(v)^T W 1 & diag(v)^T W & 0 \end{bmatrix}$, then we get equal problem $\max_{y \in [-1,1]^{m + d + 1}} \frac{1}{4}y^T M(v, W) y = \max_{y\in[-1,1]^{m+d+1}} \frac{1}{4} \langle M(v, W), yy^T \rangle$ and $f_{QP}(x) \le f_{SDP}(x) = f(x) + \frac{\epsilon}{4} \max_{P\succeq 0, diag(P) \le 1} \langle M(v, W), P \rangle$. 

- Generalization to multiple classes:

  $f^{ij}(A(x)) \le f^{ij}_{SDP}(x) = f^{ij}(x) + \frac{\epsilon}{4} \max_{P\succeq 0,diag(P)\le 1} \langle M^{ij}(V^{ij}, W), P \rangle$.

  
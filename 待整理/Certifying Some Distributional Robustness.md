# Certifying Some Distributional Robustness

## Abstract

- NN are vulnerable to adversarial examples;
- We take the view of distributionally robust optimization, by using Lagrangian penalty and Wasserstein ball;
- The method is used on the population loss in supervised and reinforcement learning.

# Introduction

- The preceding methods are either no guarantees or computational expanse;

-  Distributionally robust optimization: $\min_{\theta} \sup_{P \in \mathcal{P}} \mathbb{E}_P[l(\theta; S \sim P_0)]$;

  The choice of $\mathcal{P}$ influences robustness guarantees and computability;

- Our approach:

  - $\mathcal{P} = \{P: W_c(P, P_0)\le\rho\}$;
  - From proposition1, $\min_{\theta} \{F(\theta) := \sup_P\{\mathbb{E}_P[l(\theta;S \sim P_0)] - \gamma W_c(P, P_0)\} = \mathbb{E}_{P_0}[\phi_\gamma(\theta; S)]\}$, where $\phi_\gamma(\theta;z_0):=\sup_{z\in Z}\{l(\theta;z) - \gamma c(z, z_0)\}$.

- The standard robust optimization approach: $\min_\theta \sup_{u \in \mathcal{U}} l(\theta; z + u)$.
  - Intractable except for specially structured losses;
- Distributionally robust optimization: $P \in \mathcal{P}$.

# Proposed approach

- Wasserstein distance: $W_c(P, Q):= \inf_{M \in \Pi(P, Q)} \mathbb{E}_M[c(S, S')]$.

- Proposition 1. 

  - $\sup_{P:W_c(P,Q)\le\rho} \mathbb{E}_P[l(\theta;S)] = \inf_{\gamma \ge 0}\{\gamma \rho + \mathbb{E}_Q[\phi_\gamma(\theta;S)]\}$,

    where $\phi_\gamma(\theta;z_0)=\sup_{z \in Z}\{l(\theta;z) - \gamma c(z, z_0)\}$;

  - $\sup_P \{\mathbb{E}_P[l(\theta; S)] - \gamma W_c(P, Q)\} = \mathbb{E}_Q[\phi_\gamma(\theta;S)]$;

  - $\min_{\theta} \sup_{P \in \mathcal{P}=\{P:W_c(P,P_0)\le\rho\}} \mathbb{E}_P[l(\theta; S \sim P_0)]$ is equal to $\min_\theta\{F_n(\theta):=\sup_P\{\mathbb{E}[l(\theta;S\sim P_0)] - \gamma W_c(P, P_0)\} = \mathbb{E}_{P_0}[\phi_\gamma(\theta;S)]\}$.

- Assumption A. The function c is continuous and $c(\cdot, z_0)$ is 1-strongly convex with respect to the norm $\Arrowvert \cdot \Arrowvert$.

- Assumption B. $l:\Theta \times Z \rightarrow \mathbb R$ is Lipschitzian smoothness.

  - $\Arrowvert \nabla_\theta l(\theta;z) - \nabla_\theta l(\theta'; z)\Arrowvert_* \le L_{\theta\theta}\Arrowvert \theta - \theta'\Arrowvert  $ 
  - $\Arrowvert \nabla_z l(\theta;z) - \nabla_z l(\theta; z')\Arrowvert_* \le L_{zz}\Arrowvert z - z'\Arrowvert $ 
  - $\Arrowvert \nabla_\theta l(\theta;z) - \nabla_\theta l(\theta; z')\Arrowvert_* \le L_{\theta z}\Arrowvert z - z' \Arrowvert$ 
  - $\Arrowvert \nabla_z l(\theta;z) - \nabla_z l(\theta'; z)\Arrowvert_* \le L_{z\theta}\Arrowvert \theta - \theta'\Arrowvert $ 

- We are usually interested in adversarial perturbations only to the feature vectors. So we can construct $c(z, z') = c_x(x, x') + \infty \cdot 1\{y \ne y'\}$

  - I suddenly get this, in reinforcement learning problem, the regularization maybe: $c((s_1, a_1), (s_2, a_2)) = c_A(a_1, a_2) + \infty \cdot 1\{s \ne s'\}$ 

- Lemma 1. 

  - Let $z^*(\theta) = \arg\min_z l(\theta; z)$, then $\Arrowvert z^*(\theta_1) - z^*(\theta_2)\Arrowvert \le \frac{L_{z\theta}}{\lambda}\Arrowvert \theta_1 - \theta_2 \Arrowvert$
  - Let $\bar l(\theta) = \sup_z l(\theta; z)$, then $\Arrowvert \nabla \bar f(\theta) - \nabla \bar f(\theta')\Arrowvert_* \le (L_{\theta\theta} + \frac{L_{\theta z} L_{z\theta}}{\lambda}) \Arrowvert \theta - \theta' \Arrowvert$

- After all, we get $\nabla_\theta \phi_\gamma(\theta; z_0) = \nabla_\theta l(\theta; z^*(z_0, \theta))$ where $z^*(z_0, \theta) = \arg\max_{z} {l(\theta; z) - \gamma c(z, z_0)}$ . Now we can use SGD.

- Theorem 2 (Convergence of Nonconvex SGD).

# Robustness certificate and generalization

- Algorithm provably learns to protect against adversarial training dataset, and we need look into adversarial test dataset.
- $T_\gamma (\theta; z_0) := \arg\max_{z \in Z} \{l(\theta; z) - \gamma c(z, z_0)\}$;
- $P^*_n(\theta) := \arg\max_P\{\mathbb E_P[l(\theta; S) - \gamma W_c(P, \hat P_n)\} = \frac{1}{n} \sum^{n}_{i=1} \delta_{T_\gamma(\theta, z_i)}$
- $\hat \rho_n(\theta) := W_c(P^*_n(\theta), \hat P_n) = \mathbb E_{\hat P_n}[c(T_\gamma(\theta; S), S)]$
- Theorem 3: empirical errors.




# A Convergent O(n) Algorithm for Off-policy TD Learning with Linear Function Approximation

## Introduction

- Prior to the current work, the possibility of instability could not be avoided whenever four individually desirable algorithmic features were combine.
  - Off-policy updates;
  - Temporal-difference learning;
  - Linear function approximation;
  - Linear complexity in memory and per-time-step computation.
- Our algorithm: Gradient Temporal-difference.

## The GTD(0) Algorithm

- TD error: $\delta = r + \gamma \theta^T \phi' - \theta^T \phi$;
- Target is to get $0 = \mathbb{E}[\delta \phi] = -\mathbb{E}[\phi(\phi - \gamma\phi')^T]\theta + \mathbb{E}[r\phi]$;
- Objective: $J(\theta) = \mathbb{E}[\delta \phi]^T \mathbb{E}[\delta\phi]$; $\nabla_\theta J(\theta) = 2\nabla_\theta \mathbb{E}[\delta\theta] \cdot \mathbb{E}[\delta\theta] = - 2 \mathbb{E}[(\phi - \gamma \phi')\phi^T]\cdot\mathbb{E}[\delta\phi]$;
- $A-TD$: 
  - $A_k = \frac{1}{k}\sum^k_{i=1} (\phi_i - \gamma \phi'_i)\phi_i^T$;
  - $\theta_{k+1} = \theta_{k}+\gamma_k A^T_k \delta_k \phi_k$; 
  - $O(n^2)$ memory and computation per time step;
- $GTD(0)$: 
  - $u_{k+1} = u_k + \beta_k (\delta_k \phi_k - u_k)$;
  - $\theta_{k+1} = \theta_k + \alpha_k(\phi_k - \gamma \phi_k')\phi^T_i u_{k}$;
  - $O(n)$ memory and computation per time step;

## Convergence

Pass.
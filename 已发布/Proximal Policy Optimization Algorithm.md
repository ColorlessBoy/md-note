# Proximal Policy Optimization Algorithm

## 1. Introduction

- Q-learning fails on many simple situations and poorly understood;
- Vanilla policy gradient methods have poor data efficiency and robustness;
- TRPO is relatively complicated and not compatible that include noise or parameter sharing.

- PPO attains data efficiency and performance reliable of TRPO.

## 2. Background: Policy Optimization

- Policy gradient method:
  $$
  L^{PG} = \hat{\mathbb{E}}[\log(\pi_\theta(a_t|s_t)) \hat A_t].
  $$

  - It is perform multiple steps of optimization on this loss  $$L^{PG}$$ using the same trajectory, which leads leads to destructively large policy updates;

- Trust region methods:
  $$
  \max_\theta \hat{\mathbb{E}}_t \left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat A_t - \beta KL(\theta_{old} \Arrowvert \theta)\right]
  $$

## 3. Clipped Surrogate Objective

- Conservative policy iteration:
  $$
  L^{CPI}(\theta) = \hat{\mathbb{E}}_t \left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_old}(a_t | s_t)} \hat A_t\right].
  $$

- Clipped surrogate objective:
  $$
  L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[\min\left(r_t(\theta), clip(r_t(\theta),  1-\epsilon, 1+\epsilon)\right) \hat{A}_t\right], 
  \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = r_t(\theta).
  $$

> **Proposition 1**.  PPO-Clip objective can be simplified to
> $$
> L^{CLIP}(\theta) = \hat{\mathbb{E}}_{s, a\sim\theta_k} \left[\min\left(\frac{\pi_{\theta}(a|s)} {\pi_{\theta_k}(a|s)} A^{\theta_k}(s,a), g(\epsilon, A^{\theta_k} (s,a))\right)\right]
> $$
> where,
> $$
> g(\epsilon, A) =
> \begin{cases}
> (1+\epsilon) A, & A \ge 0\\
> (1-\epsilon) A, & otherwise
> \end{cases}
> $$

- Advantage is positive:
  $$
  L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[\min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1 + \epsilon\right) \hat{A}_t\right].
  $$

- Advantage is negative:
  $$
  L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[\min\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1 - \epsilon\right) \hat{A}_t\right].
  $$
  

## 4. Adaptive KL Penalty Coefficient

$$\beta$$ changes according to KL divergence.

## 5. Algorithm

![PPO](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)


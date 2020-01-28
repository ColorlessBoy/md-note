# Ensemble Adversarial Training: Attacks and Defenses

# Abstract

- Adversarial examples;
- Adversarial training remains vulnerable to black-box attacks;
- Ensemble adversarial training.

# Introduction

- White-box attacks is hard to defense, and it is natural to ask whether it is possible to achieve robustness against the class of black-box adversaries;
- We demonstrate that adversarial training with single-step methods admits a degenerate global minimum;
- There are two results:
  - Adversarially trained models using single-step methods remain vulnerable to simple attacks;
  - We proposed Ensemble Adversarial Training that incorporates perturbed inputs transferred from other pre-trained models;

# The Adversarial Training Framework

- Preliminaries: $x \in [0,1]^d \in X^d$, $y_{true} \in Y_k$,  and $(x, y_{true}) \sim \mathcal{D}$. $h \in \mathcal{H}$, $h(x) \in \mathbb{R}^k$ and $L(h(x), y)$.
- Threat Model:
  - Adversarial target: find $(x^{adv}, y^{true})$ closed to $(x, y^{true})$ that $h(x^{adv}) \ne y^{true}$($\Arrowvert x^{adv} - x\Arrowvert_\infty \le \epsilon$);
  - White-box adversaries and black-box adversaries.
- $h^* = \arg\min_{h \in \mathcal{H}} \mathbb{E}_{(x, y^{true})\sim \mathcal{D}} [\max_{\Arrowvert x^{adv} - x \Arrowvert \le \epsilon} L(h(x^{adv}, y^{true}))]$.
- Method:
  - **Fast Gradient Sign Method(FGSM)**. $x^{adv}_{FGSM} := x + \epsilon \cdot sign(\nabla_x L(h(x), y^{true}))$;
  - **Single-Step Least-Likely Class Method(Step-LL)**. $y_{LL}=\arg\min{h(x)}$, $x^{adv}_{LL} := x - \epsilon\cdot sign(\nabla_x L(h(x)), y_{LL})$;
  - **Iterative Attack(I-FGSM or Iter-LL)**. Applies the FGSM or Step-LL k times with step-size $\alpha \ge \epsilon/k$ and projects each step onto the $l_\infty$ ball of norm $\epsilon$ around $x$. (Projected gradient descent).
- Single-step Adversarial Training:
  - $h^* = \arg\min_{h \in \mathcal{H}} \mathbb{E}_{(x, y^{true})\sim \mathcal{D}} [ L(h(x^{adv}_{FGSM}, y^{true}))]$;
- **Ensemble Adversarial Training**:
  - Draw connection between Ensemble Adversarial Training and multiple-source Domain Adaptation;
  - Domain adaption: data sampled from one or more **source distributions** $S_1, \ldots, S_k$ is evaluated on samples x from a different **target distribution** $\mathcal{T}$;
  - $\mathcal{D} \Rightarrow \{S_1, \ldots, S_k\} \Rightarrow \{A_1, \ldots, A_k\} \Rightarrow A^*$;
  - Theorem 1.(Appendix B)

## Appendix A Threat Model: Formal Definitions

- Normal structure: $h \leftarrow train(\mathcal{H}, X_{train}, Y_{train}, r)$;
- $X, Y = \{(x_1, y_1), \ldots, (x_m, y_m)\} \sim \mathcal{D}$, an adversary A produces adversarial examples $X^{adv} = \{x^{adv}_1, \ldots, x^{adv}_m\}$, such that $\Arrowvert x_i - x^{adv}_i \Arrowvert_\infty \le \epsilon$;
- Adversarial empirical error: $\frac{1}{m} \sum^m_{i=1} 1\{\arg\max_h h(x^{adv}_i) \ne y_i\}$;
- Adversaries:
  - **White-Box Adversary** has access to all elements of the training procedure;
  - **Non-Interactive Black-Box Adversary ** gets access to $train$ and $\mathcal{H}$;
  - **Interative Black-Box Adversary** gets access to $train$, $\mathcal{H}$ and target model.

## Appendix B Generalization Bound For Ensemble Adversarial Training

- We assume the model is:
  - Train data: N data points $S^{adv}$, where $\frac{N}{k}$ data points are sampled from each distribution $A_i \in \{A_1, \ldots, A_k\}$;
  - Test data is sampled from $A^*$.
- Adversarial empirical error: $\hat R(h, A_{train}) := \frac{1}{N} \sum_{(x^{adv}, y^{true}) \in Z_{train}} L(h(x^{adv}), y^{true})$ ;
- Average discrepancy distance: $disc_{\mathcal{H}}(A_{train}, A^*) := \frac{1}{k} \sum^{k}_{i=1}\sup_{h_1, h_2 \in \mathcal{H}} |\mathbb{E}_{A_i}[1\{h_1(x^{adv}) = h_2(x^{adv})\}]-\mathbb{E}_{A_*}[1\{h_1(x^{adv}) = h_2(x^{adv})\}]|$
- Theorem 5. With probability at least $1-\delta$, $\sup_{h\in\mathcal{H}}|\hat R(h, A_{train}) - R(h, A^*)|\le disc_{\mathcal{H}}(A_{train}, A^*) + 2R_N(\mathcal{H}) + O(\sqrt{\frac{\ln(1/\delta)}{N}})$

 
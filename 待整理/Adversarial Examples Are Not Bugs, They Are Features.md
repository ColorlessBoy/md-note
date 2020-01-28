# Adversarial Examples Are Not Bugs, They Are Features

## 1. Introduction

- Arguments based on the concentration of measure in high-dimensions are unable to capture behaviors  we observes in practices.
- Previous perspective: adversarial examples are aberrations arising either from the high dimensional nature of input space or statistical fluctuations in the training data.  In this view, the adversarial examples can be disentangled and pursed independently from maximizing accuracy, either through standard regularization methods or pre/post-processing of network inputs/outputs.
- Our perspective: **adversarial vulnerability is a directly result of out models' sensitivity to well-generalizing features in the data**. 
- ???

## 2. The Robust Features Model

- **Setup**: binary classification
- **Useful, robust/non-robust features**:
  - $\rho-$useful features: $\mathbb{E}_{(x,y)\sim\mathcal{D}}[y\cdot f(x)] \ge \rho$;
  - $\gamma-$robustly useful features: $\mathbb{E}_{(x,y)\sim\mathcal{D}}[\inf_{\delta\in\Delta} y \cdot f(x+\delta)] \ge \gamma$;
  - useful, non-robust features;
- **Training**:
  - Standard Training: $\mathbb{E}_{(x,y)\sim\mathcal{D}}[L_\theta(x, y)] = -\mathbb E [y\cdot(b + \sum_{f\in F}w_f \cdot f(x))]$;
  - Robust Training: $\mathbb{E}_{(x,y)\sim\mathcal{D}}[\max_{\delta\in\Delta}L_\theta(x+\delta, y)]$;

## 3. Finding Robust(Non-robust) Features

- On one hand, we construct a "robustified" data set, consisting of samples that primarily contain robustly useful features;

- On the other hand, we construct datasets where the input-label association is based on purely non-robust features;

- Get robust features:

  - Construct a distribution $\hat{\mathcal{D}}_R$ that $\mathbb{E}_{(x,y)\sim\hat{\mathcal{D}}_R}[f(x)\cdot y] = 1_{\{f \in F_C\}} \cdot \mathbb{E}_{(x,y)\sim\mathcal{D}}[f(x) \cdot y]$;
  - $F_C$ corresponds to exactly the set of activations in the penultimate layer;
  - Get robust dataset:

      ```pseudocode
      GetRobustDataset(D)
        CR = AdversarialTraining(D)
        gR is a mapping function learned by CR from input to the representation layer
        DR = {}
        for (x, y) in D:
            sample x' from D
            xR minimizes the norm of gR(xR) - gR(x), optimized from x'
            insert (xR, y) into DR
        return DR
      ```
  
- Get non-robust features:

  ```pseudocode
  GetNonRobustDataset(D, r)
  	DNR = {}
  	C = StandardTraining(D)
  	for (x, y) in D:
  		t [uar] sampled from [C]
  		xNR minimizes L_C(x', t) subject in r-ball whose center is x
  		insert (xNR, y) into DNR
  	return DNR
  ```

## 4. A Theoretical Framework for Studying (Non-)Robust Features

- This section maybe says that:
  - The adversarial vulnerability can be explicitly expressed as a difference between the inherent data metric and the l2norm;
  - Robust learning corresponds exactly to learning a combination of these two metrics;
  - The gradients of adversarial trained models align better with the adversary‘s metric.
- 
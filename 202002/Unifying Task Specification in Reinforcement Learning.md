# Unifying Task Specification in Reinforcement Learning

## 1. Introduction

>First, we demonstrate the utility of this formalism by showing unification of previous tasks specified in reinforcement learning, including options, general value functions and episodic and continuing, and further providing case studies of utility. We demonstrate how to specify episodic and continuing tasks with only modifications to the discount function, without the addition of states and modifications to the underlying Markov decision process. This enables a unification that significantly simplifies implementation and easily generalizes theory to cover both settings.
>
>Second, we prove novel contraction bounds on the Bellman operator for these generalized RL tasks, and show that previous bounds for both episodic and continuing tasks are subsumed by this more general result

## 2. Generalized problem formulation

- Markov decision process: $$M = (\mathcal{S}, \mathcal{A}, P)$$;

  - $$\mathcal{S}$$ is the set of states, $$n = \vert S \vert$$;
  - $$\mathcal{A}$$ is the set of actions;
  - $$P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$$ is the transition probability function;

- A **reinforcement learning task** is specified on top of these transition dynamics, as the tuple $$(\Pi, r, \gamma, i)$$ where

  - $$\Pi$$ is a set of policies $$\pi: \mathcal{S}\times \mathcal{A} \rightarrow [0,1]$$;
  - the reward function $$r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$$;
  - $$\gamma: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$$ is a transition-based discount function;
  - $$i: \mathcal{S} \rightarrow [0, +\infty)$$ is an interest function that specifies the user defined interest in a state.

- The cumulative discounted reward obtained from following that policy:
  $$
  G_t = \sum^{\infty}_{i=0}\left(\prod^{i-1}_{j=0}\gamma(s_{t+j}, a_{t+j}, s_{t+1+j}) \right) R_{t+1+i},
  $$
  where $$\prod^{-1}_{j=0}\gamma(s_{t+j}, a_{t+j}, s_{t+1+j}) := 1$$.

## 4. Objectives and algorithms

- $$ P_\pi (s, s') := \sum_{a\in A} \pi(s,a) Pr(s,a,s')$$;
- $$P_{\pi,\gamma}(s,s') := \sum_{a\in A} \pi(s,a) P(s,a,s') \gamma(s,a,s')$$;
- $$r_\pi(s) := \sum_{a\in A} \pi(s,a) \sum_{s' \in S} Pr(s,a, s') r(s,a,s')$$;
- $$v_\pi(s) := r_\pi(s) + \sum_{s' \in S} P_{\pi,\gamma}(s, s') v_\pi(s')$$.


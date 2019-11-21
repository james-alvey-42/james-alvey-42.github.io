---
layout: post
title: Multi Armed Bandit Problem
date: 2019-11-21 00:00:00
description: This is a classic reinforcement learning problem where the goal is to maximise the reward in a set of actions from a discrete choice of options. We present some of the theory and a simple example to test two different strategies.
img: multi-armed-bandit.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Reinforcement Learning, Multi-Armed Bandit, Q-learning]
---
> This is a classic reinforcement learning problem where the goal is to maximise the reward in a set of actions from a discrete choice of options. We present some of the theory and a simple example to test two different strategies.

This is based on an excerpt from [this book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) by Sutton and Barto on Reinforcement Learning. The code for this post can be found in my Github repository at the following link:

* <a href="https://github.com/james-alvey-42/ReinforcementLearning/tree/master/Code/Multi-Armed-Bandit" target="blank_"><i class="fa fa-github" aria-hidden="true"></i> Reinforcement Learning Repository</a>

## Statement of the Problem

The problem to be solved can be stated as follows. Suppose we have a bandit who can pull any one of $$n$$ lever. Each lever triggers a reward from the corresponding machine. This reward is distributed according to the given parameters of the machine which are fixed throughout, but are not accessible. For simplicity say that a given machine $$a$$ outputs rewards according to a normal distribution with mean $$q_a$$ and variance $$\sigma_a^2$$. The goal is the find an optimal strategy to maximise the rewards recieved over $$N$$ trials.

## Exploitation vs Exploration

Suppose we have $$M$$ machines with true mean rewards $$q_a$$, $$a = 1, \ldots M$$, then before pulling a single lever as a bandit, our knowledge of the true distributions is limited. To be more precise, our estimated *value* of each machine is independent of the machine. Clearly to proceed we should try and update this belief by pulling a variety of levers. To estimate our perceived value of machine $$a$$ after some time $$t$$, we construct the following quantity,

$$Q_t(a) := \frac{R_1 + R_2 + \cdots + R_{N_t(a)}}{N_t(a)}$$

where $$R_i$$ is the reward received the $$i$$th time that lever $$a$$ is pulled and $$N_t(a)$$ is the total number of times lever $$a$$ has been pulled. In other words we calculate the average reward that we have seen thus far and define that to be the current value of that lever.

The optimal solution then is one where these $$Q_t(a)$$ converge to the true values $$q(a)$$ in as few steps as possible, whilst still balancing the rewards from the "best" lever. This is the balance between *exploration* and *exploitation* - I want to check that there are no other better levers, but once I have found the best I want to stick there.

Now, there are a number of very sophisticated methods to solve this truly optimally, but here we consider just the simplest way to balance these two concepts:

* The Greedy Method: In this case we always choose the lever we *currently perceive to have the highest value to us*.
* The $$\epsilon$$-Greedy Method: On the other hand, we might take the approach that *most* of the time we pull the lever that we currently think has the highest value, but some of the time randomly explore the other levers just to check we're not missing out on anything.

To put this more formally so we can write some code around it, we define a parameter $$\epsilon$$ so that with probability $$(1 - \epsilon)$$ we choose the action with the highest value $$A_t = \mathrm{argmax} Q_t(a)$$ and with probability $$\epsilon$$ we randomly choose another machine. The Greedy Method case is then where $$\epsilon = 0$$.

## Implementation


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

Implementation
---

To actually implement the scheme and compare the two different methods, we start by defining a `Machine` class which will output the rewards when it is "pulled".

{%highlight python%}
class Machine():
	
	def __init__(self, mean, sigma, label):
		self.mean = mean
		self.sigma = sigma
		self.label = label

	def sample(self):
		return np.random.normal(loc=self.mean, scale=self.sigma, size=1)[0]
{%endhighlight%}

We also create an `Agent` class that has a given $$\epsilon$$ parameter and number of trials.

{%highlight python%}
class Agent():
	
	def __init__(self, epsilon, Nsteps, Nmachines, label=None):
		self.epsilon = epsilon
		self.Nsteps = Nsteps
		self.label = label
		self.Q = np.zeros((Nmachines, Nsteps))
		self.rewards = np.zeros(Nsteps)
		self.numbers = np.zeros(Nmachines)
		self.pulls = 0
		self.averages = None

	def get_reward(self, machine):
		return machine.sample()

	def pull(self, Q, machines):
		best_machine = np.argmax(Q)
		if (np.random.uniform(0, 1, size=1)[0] > self.epsilon):
			return machines[best_machine]
		else:
			index_sample = np.random.randint(0, len(machines) - 1, size=1)[0]
			while index_sample == best_machine:
				index_sample = np.random.randint(0, len(machines) - 1, size=1)[0]
			return machines[index_sample]

	def update(self, Q, machines):
		pulled_machine = self.pull(Q, machines)
		reward = self.get_reward(pulled_machine)
		num = self.numbers[pulled_machine.label]
		self.numbers[pulled_machine.label] += 1
		Qnew = Q
		Qnew[pulled_machine.label] = (reward + Q[pulled_machine.label]*num)/(num + 1)
		self.pulls += 1
		return Qnew, reward

	def get_averages(self):
		return self.averages

	def get_numbers(self):
		return self.numbers
{%endhighlight%}

Finally, for a given agent and set of machines, we can run a full simulation of the trials with the following function:

{%highlight python%}

def run_simulation(agent, Nsteps, machines):
	for i in range(Nsteps - 1):
		Qold = agent.Q[:, i]
		Qnew, reward = agent.update(Qold, machines)
		agent.Q[:, i + 1] = Qnew
		agent.rewards[i] = reward
	averages = []
	for i in range(1, len(agent.rewards)):
		averages.append(np.sum(agent.rewards[:i])*(1/i))
	agent.averages = averages

{%endhighlight%}

The end of this function computes the rolling averages of the total reward gained whcih can then be plotted. To implement these and trial a few strategies, we just need:

{%highlight python%}

Nmachines = 10
means = np.random.uniform(-1, 1, Nmachines)
sigma = np.random.uniform(0, 1, Nmachines)
machines = [Machine(mean=means[i], sigma=sigma[i], label=i) for i in range(Nmachines)]
Nsteps = 10000

agent1 = Agent(epsilon=0.1, Nsteps=Nsteps, Nmachines=Nmachines, label='0.1')
agent2 = Agent(epsilon=0.01, Nsteps=Nsteps, Nmachines=Nmachines, label='0.01')
agent3 = Agent(epsilon=0.0, Nsteps=Nsteps, Nmachines=Nmachines, label='0.0')

run_simulation(agent1, Nsteps=Nsteps, machines=machines)
run_simulation(agent2, Nsteps=Nsteps, machines=machines)
run_simulation(agent3, Nsteps=Nsteps, machines=machines)

{%endhighlight%}

With some plotting we find the following results:

![fullresults]({{site.baseurl}}/assets/img/multi-armed-bandit-full.png)

## Conclusions: $$\epsilon$$-greedy policies

If we take a look at the results, we see that strategies with non-zero $$\epsilon$$ work significantly better than there purely greedy strategy. We can also understand why this is the case: the greedy strategy doesn't bother discovering lever 7, whilst the ones that do explore find it at various points and then take advantage thereafter. Indeed we see that $$\epsilon = 0.01$$ takes longer to find the lever, but as soon as it does it catches up with $$\epsilon = 0.1$$ realtively quickly once it does. 

This is of course a brief interlude into this problem, but it illustrates the key ideas and provides a simple implementation. A more thorough analysis might take into account a suite of simulations for different distributions of means and variances for the machines to see which strategy is globally the best.

## Gradient Bandits

Suppose instead of estimating the *actual* value of an action $$a$$ at some time $$t$$ ($$Q_t(a)$$), we now consider learning a *preference* $$H_t(a)$$. For some set of preferences, we can then choose an action based on the probability distribution:

$$\pi_t(a) = \mathbb{P}(A_t = a) = \frac{\exp[H_t(a)]}{\sum_{b}{\exp[H_t(b)]}}$$

This has the advantage that it removes an initial bias on the expected value of a reward. If we initially give a flat prior to all of the machines, then we can update our preferences according to,

$$H_{t + 1}(A_t) = H_t(A_t) + \alpha (R_t - \bar{R}_t) (1 - \pi_t(A_t)$$

$$H_{t + 1}(a) = H_t(a) - \alpha (R_t - \bar{R}_t) \pi_t(a) \quad \forall a \neq A_t$$

where $$R_t$$ is the reward received at timestep $$t$$, $$\bar{R}_t$$ is the average reward received over all times previous to $$t$$, $$A_t$$ is the action taken at time $$t$$, and $$\alpha$$ is a parameter than controls the step size. At this point, these definitions look a little arbitrary. It is nice that we have the notion of a probability distribution, but why should we update it in the way given above? The derivation is in [Sutton and Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), but the key points are the following:

* The update preserves the total preference i.e. $$\sum_{a}{H_t(a)} = \mathrm{const.}$$. In other words, the preferences just get reshared out from one timestep to the next. To show this, you need the fact that the probabilities add to unity.

* It can be shown that the update above is the sampling equivalent to maximising the expected reward $$\mathbb{E}(R_t)$$ over the probability distribution $$\pi$$. If you believe me that this is indeed the case, then the above is equivalent to;

$$H_{t + 1}(a) = H_t(a) + \alpha \frac{\partial \mathbb{E}(R_t)}{\partial H_t(a)}$$

which is just a very simple form of gradient ascent.

## Implementation

To implement the above we define a new class `ProbabilityAgent`;

{% highlight python %}

class ProbabilityAgent():
	
	def __init__(self, Nsteps, alpha, Nmachines, label=None):
		self.Nsteps = Nsteps
		self.label = label
		self.alpha = alpha
		self.H = np.zeros((Nmachines, Nsteps))
		self.probabilities  = (1/Nmachines) * np.ones(Nmachines)
		self.rewards = np.zeros(Nsteps)
		self.numbers = np.zeros(Nmachines)
		self.pulls = 0
		self.averages = np.zeros(Nsteps)

	def get_reward(self, machine):
		return machine.sample()

	def pull(self, machines):
		machine = np.random.choice([i for i in range(len(machines))], 
									size=1,
									p=list(self.probabilities))
		return machines[machine[0]]

	def first_step(self, machines):
		pulled_machine = self.pull(machines)
		reward = pulled_machine.sample()
		self.numbers[pulled_machine.label] += 1
		self.rewards[self.pulls] = reward
		self.averages[0] = reward
		machine_index = pulled_machine.label
		mask = np.ones(len(machines), dtype='bool')
		mask[machine_index] = False
		H = self.H[:, 0]
		H[mask] = H[mask] - self.alpha * (reward - self.averages[0]) * self.probabilities[mask]
		H[machine_index] = H[machine_index] + self.alpha * (reward - self.averages[0]) * (1 - self.probabilities[machine_index])
		self.H[:, 1] = H
		self.probabilities = calculate_probabilities(H)
		self.pulls += 1

	def update(self, machines):
		pulled_machine = self.pull(machines)
		reward = pulled_machine.sample()
		self.numbers[pulled_machine.label] += 1
		self.rewards[self.pulls] = reward
		self.averages[self.pulls] = (reward + self.pulls*self.averages[self.pulls - 1])/(self.pulls + 1)
		machine_index = pulled_machine.label
		mask = np.ones(len(machines), dtype='bool')
		mask[machine_index] = False
		H = self.H[:, self.pulls]
		H[mask] = H[mask] - self.alpha * (reward - self.averages[self.pulls - 1]) *self.probabilities[mask]
		H[machine_index] = H[machine_index] + self.alpha * (reward - self.averages[self.pulls - 1]) * (1 - self.probabilities[machine_index])
		self.H[:, self.pulls + 1] = H
		self.probabilities = calculate_probabilities(H)
		self.pulls += 1

	def get_averages(self):
		return self.averages

	def get_numbers(self):
		return self.numbers

{% endhighlight %}

Running the simulations in the same case as the $$\epsilon$$-greedy bandit, we find the following results;

![h-results]({{site.baseurl}}/assets/img/multi-armed-bandit-h.png)

## Conclusions: Gradient Bandit

We see from the results that the convergence is quite sensitive to $$\alpha$$. Furthemore, we see that we indeed find a better strategy compared to the $$\epsilon$$-greedy case, with a higher average reward. One thing that we did not test here is whether this secondary method performs better with different initialisations of the preferences compared to varying the initial action value priors. One would expect as such, since the action priors can be biased if the order of magnitude expected for the reward is not known. 

<a href="{{site.baseurl}}/"><i class="fa fa-home" aria-hidden="true"></i> Homepage</a>
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

tfd = tfp.distributions

normal_dist = tfd.Normal(loc=0, scale=1)
print('Normal distribution:', normal_dist)

# Sample from the distribution
sample = normal_dist.sample(10)

# Probability's density
prob = normal_dist.prob(0)

# Log probability
log_prob = normal_dist.log_prob(0)

# Verify log prob
np_log_prob = np.log(normal_dist.prob(0))

# Histogram
plt.hist(normal_dist.sample(10000).numpy().flatten(), bins=50, density=True)
plt.show()

# Exponential distribution
exponential_dist = tfd.Exponential(rate=1)
plt.hist(exponential_dist.sample(10000).numpy().flatten(), bins=50, density=True)
plt.show()

# Sample
exp_sample = exponential_dist.sample(10)

# Bernoulli distribution
bernoulli_dist = tfd.Bernoulli(probs=0.8)

# Calculate Bernoulli prob and see that 0.5 and -1 do not give the correct probability!

for k in [0,0.5,1,-1]:
    print('Prob result {} for k = {} '.format(bernoulli_dist.prob(k), k))
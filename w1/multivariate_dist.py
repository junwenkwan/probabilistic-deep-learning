import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

tfd = tfp.distributions

# 2D multivariate Gaussian with diagonal covariance matrix
normal_diag = tfd.MultivariateNormalDiag(loc=[0,1],scale_diag=[1,2])

print('MultivariateNormalDiag:', normal_diag)

plt_sample = normal_diag.sample(10000)
plt.scatter(plt_sample[:, 0], plt_sample[:, 1], marker='.', alpha=0.05)
plt.axis('equal')
plt.show()

# Batches of multivariate normals
normal_diag_batch = tfd.MultivariateNormalDiag(loc=[[0,0],[0,0],[0,0]],scale_diag=[[1,2],[2,1],[2,2]])
print('MultivariateNormalDiag(Batch):', normal_diag_batch)

# Sample
samples = normal_diag_batch.sample(5)

# Compute log probs
print('Log probs:', normal_diag_batch.log_prob(samples))

# Plot
plt_sample_batch = normal_diag_batch.sample(10000)
fig, axs = (plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 3)))
titles = ['cov_diag=[1, 2]','cov_diag=[2, 1]', 'cov_diag=[2, 2]']

print('plt_sample_batch.shape:', plt_sample_batch.shape)

for i, (ax, title) in enumerate(zip(axs,titles)):
    samples = plt_sample_batch[:,i,:] #take the ith batch [samples x event_shape]
    ax.scatter(samples[:, 0], samples[:, 1], marker='.', alpha=0.05)
    ax.set_title(title)
plt.show()
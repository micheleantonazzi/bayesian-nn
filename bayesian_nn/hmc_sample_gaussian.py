import torch
import hamiltorch
import matplotlib.pyplot as plt

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print(hamiltorch.__version__)

def log_prob_func(omega):
    mean = torch.tensor([1.5,1.,0.])
    stddev = torch.tensor([.5,1.,2.])
    return torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega).sum()


print(log_prob_func(torch.Tensor([1, 2, 3])))

N = 400
step_size = .3
L = 5

hamiltorch.set_random_seed(123)
params_init = torch.zeros(3)
params_hmc = hamiltorch.sample(log_prob_func=log_prob_func, params_init=params_init, num_samples=10000,
                               step_size=step_size, num_steps_per_sample=L)

plt.close()
plt.hist(torch.tensor(torch.stack(params_hmc)).T[0],density=True,bins=50)
plt.show()
print('LORO', torch.stack(params_hmc).mean(dim=0), torch.stack(params_hmc).std(dim=0))
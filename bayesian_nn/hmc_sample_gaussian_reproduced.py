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

def gibbs(params):
    dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    return dist.sample()

def hamiltonian(params, momentum):

    log_prob = log_prob_func(params)

    potential = -log_prob
    kinetic = 0.5 * torch.dot(momentum, momentum)

    hamiltonian = potential + kinetic

    return hamiltonian

def collect_gradients(log_prob, params, pass_grad = None):
    params.grad = torch.autograd.grad(log_prob,params)[0]
    return params

def leap_frog(params, momentum, step, step_size, ):
    def params_grad(p):
        p = p.detach().requires_grad_()
        log_prob = log_prob_func(p)
        p = collect_gradients(log_prob, p)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return p.grad

    ret_params = []
    ret_momentum = []

    params = params.clone()
    momentum = momentum.clone()

    momentum += 0.5 * step_size * params_grad(params)

    for n in range(step):

        params = params + step_size * momentum

        p_grad = params_grad(params)
        momentum += step_size * p_grad
        ret_params.append(params.clone())
        ret_momentum.append(momentum.clone())

    ret_momentum[-1] = ret_momentum[-1] - 0.5 * step_size * p_grad.clone()
    return ret_params, ret_momentum

params = torch.zeros(3).requires_grad_()
num_samples = 10000

samples = [params.clone().tolist()]
accepted = 0
rejected = 0

for n in range(num_samples):
    momentum = gibbs(params=params)
    ham = hamiltonian(params, momentum)

    leapfrog_params, leapfrog_momenta = leap_frog(params, momentum, step=5, step_size=0.3)
    new_params = leapfrog_params[-1].to('cpu').detach().requires_grad_()
    new_momentum = leapfrog_momenta[-1].to(device)

    new_ham = hamiltonian(new_params, new_momentum)

    rho = min(0., float(-new_ham + ham))

    # Accepted
    if rho >= torch.log(torch.rand(1)):
        samples.append(leapfrog_params[-1].clone().tolist())
        params = new_params.clone().to('cpu')
        accepted += 1
    else:
        rejected += 1

print(accepted, rejected)

#samples = samples[1000:]
print('MIO', torch.torch.tensor(samples).mean(dim=0), torch.torch.tensor(samples).std(dim=0))
plt.close()
plt.hist(torch.tensor(samples).T[0],density=True,bins=50)
plt.show()
print(torch.tensor(samples).T)
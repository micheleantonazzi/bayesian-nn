import torch
import hamiltorch
import matplotlib.pyplot as plt

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

print(hamiltorch.__version__)

def log_prob_func(omega):
    mean = torch.tensor([0.,0.,0.])
    stddev = torch.tensor([.5,1.,2.])
    return torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega).sum()


print(log_prob_func(torch.Tensor([1, 2, 3])))

N = 10000
step_size = .3
L = 5

hamiltorch.set_random_seed(123)
params_init = torch.zeros(3) + 5
burn=500
N_nuts = burn + N
params_hmc = hamiltorch.sample(log_prob_func=log_prob_func, params_init=params_init,
                               num_samples=N_nuts,step_size=step_size,num_steps_per_sample=L,
                               sampler=hamiltorch.Sampler.HMC_NUTS, burn=burn,
                               desired_accept_rate=0.8)


print(len(params_hmc),torch.stack(params_hmc).mean(dim=0), torch.stack(params_hmc).std(dim=0))


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

        params += step_size * momentum

        momentum += step_size * params_grad(params)

        ret_params.append(params.clone())
        ret_momentum.append(momentum.clone())
    ret_momentum[-1] = 0.5 * step_size * params_grad(params)
    return ret_params, ret_momentum

params = torch.zeros(3)
num_samples = 10000

samples = []
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
        samples.append(new_params.detach().tolist())
        params = new_params.to('cpu')
        accepted += 1
    else:
        params = new_params.to('cpu')
        rejected += 1

print(accepted, rejected)

samples = samples[1000:]

print(torch.tensor(samples), torch.torch.tensor(samples).mean(dim=0), torch.torch.tensor(samples).std(dim=0))
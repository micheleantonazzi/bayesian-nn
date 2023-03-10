import torch
import hamiltorch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

hamiltorch.set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)

print(hamiltorch.__version__)

class Net(nn.Module):

    def __init__(self, layer_sizes, loss = 'multi_class', bias=True):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.loss = loss
        self.bias = bias
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1],bias = self.bias)

    def forward(self, x):
        x = self.l1(x)

        return x

layer_sizes = [4,3]
net = Net(layer_sizes)


print(net)

from sklearn.datasets import load_iris
import numpy as np
np.random.seed(0)
data = load_iris()
x_ = data['data']
y_ = data['target']
print(x_.shape)
N_tr = 10#50
N_val = 140
a = np.arange(x_.shape[0])
train_index = np.random.choice(a, size = N_tr, replace = False)
val_index = np.delete(a, train_index, axis=0)
x_train = x_[train_index]
y_train = y_[train_index]
x_val = x_[val_index][:]
y_val = y_[val_index][:]
x_m = x_train.mean(0)
x_s = x_train.std(0)
x_train = (x_train-x_m)/ x_s
x_val = (x_val-x_m)/ x_s
D_in = x_train.shape[1]
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_val = torch.FloatTensor(x_val)
y_val = torch.FloatTensor(y_val)
plt.scatter(x_train.numpy()[:,0],y_train.numpy())

x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

tau_list = []
tau = 1.#/100. # iris 1/10
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

hamiltorch.set_random_seed(123)
net = Net(layer_sizes)
print([p.grad for p in net.parameters()])
params_flatten = hamiltorch.util.flatten(net).to(device).clone()
print(params_flatten)

step_size = 0.1
num_samples = 300
L = 20
tau_out = 1.

criterion = nn.CrossEntropyLoss(reduction='sum')
output = net(x_train)

print(output)
print(y_train.long().view(-1))

print([p.grad for p in net.parameters()])

def classify(net, x_train, y_train, criterion):
    params_flatten = hamiltorch.util.flatten(net)

    dist = torch.distributions.Normal(torch.zeros_like(torch.tensor(0.0)), torch.tensor(1.0))

    prior = dist.log_prob(params_flatten).sum()

    output = net(x_train)
    loss = -criterion(output, y_train.long().view(-1))

    loss_prior = prior + loss

    return net, loss_prior

def gibbs(params):
    dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    return dist.sample()

def hamiltonian(net, x_train, y_train, criterion, momentum):

    net, loss_prior = classify(net, x_train, y_train, criterion)


    potential = -loss_prior
    kinetic = 0.5 * torch.dot(momentum, momentum)

    hamiltonian = potential + kinetic

    return hamiltonian

def leap_frog(net, x_train, y_train, criterion, momentum, step, step_size, ):

    ret_params = []
    ret_momentum = []
    try:
        for p in net.parameters():
            p.grad.detach_()
            p.grad.zero_()
    except:
        pass
    net, loss_prior = classify(net, x_train, y_train, criterion)
    loss_prior.backward()

    grads = torch.cat([p.grad.flatten() for p in net.parameters()])

    momentum += 0.5 * step_size * grads

    for n in range(step):
        params_flatten = hamiltorch.util.flatten(net)
        params_flatten = params_flatten + step_size * momentum
        sd = net.state_dict()
        sd['l1.weight'] = torch.reshape(params_flatten[:12], (3,4))
        sd['l1.bias'] = params_flatten[12:]
        net.load_state_dict(sd)
        for p in net.parameters():
            p.grad.detach_()
            p.grad.zero_()
        net, loss_prior = classify(net, x_train, y_train, criterion)
        loss_prior.backward()


        grads = torch.cat([p.grad.flatten() for p in net.parameters()])

        momentum += step_size * grads
        ret_params.append(params_flatten.clone())
        ret_momentum.append(momentum.clone())

    ret_momentum[-1] = ret_momentum[-1] - 0.5 * step_size * grads.clone()
    return ret_params, ret_momentum

samples = []
accepted = 0
rejected = 0

for n in range(num_samples):

    sd = net.state_dict()
    sd['l1.weight'] = torch.reshape(params_flatten[:12], (3, 4))
    sd['l1.bias'] = params_flatten[12:]
    net.load_state_dict(sd)
    momentum = gibbs(params=hamiltorch.util.flatten(net))
    ham = hamiltonian(net, x_train, y_train, criterion, momentum)

    leapfrog_params, leapfrog_momenta = leap_frog(net, x_train, y_train, criterion, momentum, step=L, step_size=step_size)
    new_params_flatten = leapfrog_params[-1].detach().clone()
    new_momentum = leapfrog_momenta[-1]
    sd = net.state_dict()
    sd['l1.weight'] = torch.reshape(new_params_flatten[:12], (3,4))
    sd['l1.bias'] = new_params_flatten[12:]
    net.load_state_dict(sd)
    for p in net.parameters():
        p.grad.detach_()
        p.grad.zero_()

    new_ham = hamiltonian(net, x_train, y_train, criterion, new_momentum)

    rho = min(0., float(-new_ham + ham))

    # Accepted
    if rho >= torch.log(torch.rand(1)):
        samples.append(leapfrog_params[-1].clone().tolist())
        params_flatten = new_params_flatten.clone()
        accepted += 1
    else:
        rejected += 1
print(accepted, rejected)
print('MIO', torch.torch.tensor(samples).mean(dim=0), torch.torch.tensor(samples).std(dim=0))
plt.close()
plt.hist(torch.tensor(samples).T[1],density=True,bins=50)
plt.show()
print(torch.tensor(samples).T)

pred_list = []
loss_priors = []

for sample in samples:
    sample = torch.tensor(sample)
    dist = torch.distributions.Normal(torch.zeros_like(torch.tensor(0.0)), torch.tensor(1.0))

    prior = dist.log_prob(sample).sum()
    sd = net.state_dict()
    sd['l1.weight'] = torch.reshape(sample[:12], (3,4))
    sd['l1.bias'] = sample[12:]
    net.load_state_dict(sd)
    for p in net.parameters():
        p.grad.detach_()
        p.grad.zero_()
    output = net(x_val)
    loss = -criterion(output, y_val.long().view(-1))

    loss_prior = prior + loss

    pred_list.append(output)
    loss_priors.append(loss_prior)

print(loss_priors)
pred_list = torch.stack(pred_list)
print(pred_list.size())
_, pred = torch.max(pred_list, 2)
acc = torch.zeros( len(pred_list)-1)
nll = torch.zeros( len(pred_list)-1)
ensemble_proba = F.softmax(pred_list[0], dim=-1)
for s in range(1,len(pred_list)):
    _, pred = torch.max(pred_list[:s].mean(0), -1)
    acc[s-1] = (pred.float() == y_val.flatten()).sum().float()/y_val.shape[0]
    ensemble_proba += F.softmax(pred_list[s], dim=-1)
    nll[s-1] = F.nll_loss(torch.log(ensemble_proba.cpu()/(s+1)), y_val[:].long().cpu().flatten(), reduction='mean')


fs = 20
plt.figure(figsize=(10,5))
plt.plot(acc.detach_())
plt.grid()
# plt.xlim(0,3000)
plt.xlabel('Iteration number',fontsize=fs)
plt.ylabel('Sample accuracy',fontsize=fs)
plt.tick_params(labelsize=15)
# plt.savefig('mnist_acc_100_training.png')
plt.show()

fs = 20
plt.figure(figsize=(10,5))
plt.plot(nll.detach_())
plt.grid()
# plt.xlim(0,3000)
plt.xlabel('Iteration number',fontsize=fs)
plt.ylabel('Negative Log-Likelihood',fontsize=fs)
plt.tick_params(labelsize=15)
# plt.savefig('mnist_acc_100_training.png')
plt.show()
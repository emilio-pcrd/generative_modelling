import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_W(embed_dim, in_dim):
    k = np.sqrt(1 / in_dim)
    return 2*k*torch.rand(embed_dim, in_dim) - k


def init_b(embed_dim, in_dim):
    k = np.sqrt(1 / in_dim)
    return 2*k*torch.rand(embed_dim,) - k


class CascadeGradNet(nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, activation):
        super().__init__()

        self.num_layers = num_layers
        self.nonlinearity = nn.ModuleList([activation() for i in range(num_layers)])

        self.W = nn.Parameter(init_W(embed_dim, in_dim), requires_grad=True)
        self.bias = nn.ParameterList([nn.Parameter(init_b(embed_dim, embed_dim), requires_grad=True) for i in range(num_layers+1)])
        self.bias[0] = nn.Parameter(init_b(embed_dim, in_dim), requires_grad=True)
        self.bias[-1] = nn.Parameter(init_b(in_dim, embed_dim), requires_grad=True)

        self.beta = nn.ParameterList([nn.Parameter(torch.rand(embed_dim)-0.5, requires_grad=True) for i in range(num_layers)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.rand(embed_dim)-0.5, requires_grad=True) for i in range(num_layers)])

    def forward(self, x):
        z = self.beta[0].view(1,-1) * F.linear(x, self.W, self.bias[0])
        for i in range(self.num_layers - 1):
            skip = self.beta[i+1].view(1,-1) * F.linear(x, self.W, self.bias[i+1])
            z = skip + self.alpha[i].view(1,-1) * self.nonlinearity[i](z)

        z = self.alpha[-1].view(1,-1) * self.nonlinearity[i](z)
        z = F.linear(z, self.W.T, self.bias[-1])

        return z


# OUR CASCADE NET
class CascadeGradNetOURS(nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, activation):
        super().__init__()

        self.num_layers = num_layers
        self.nonlinearity = nn.ModuleList([activation() for _ in range(num_layers)])

        k_W = np.sqrt(1 / in_dim)
        self.W = nn.Parameter(2 * k_W * torch.rand(embed_dim, in_dim) - k_W, requires_grad=True)

        self.bias = nn.ParameterList([
            nn.Parameter(2 * np.sqrt(1 / embed_dim) * torch.rand(embed_dim) - np.sqrt(1 / embed_dim), requires_grad=True)
            for _ in range(num_layers)
        ])
        self.bias.insert(0, nn.Parameter(2 * k_W * torch.rand(embed_dim) - k_W, requires_grad=True))
        self.bias.append(nn.Parameter(2 * np.sqrt(1 / embed_dim) * torch.rand(in_dim) - np.sqrt(1 / embed_dim), requires_grad=True))

        self.beta = nn.ParameterList([
            nn.Parameter(torch.rand(embed_dim) - 0.5, requires_grad=True) for _ in range(num_layers)
        ])
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.rand(embed_dim) - 0.5, requires_grad=True) for _ in range(num_layers)
        ])

    def forward(self, x):
        z = self.beta[0].view(1, -1) * F.linear(x, self.W, self.bias[0])

        for i, (alpha, beta, nonlinearity) in enumerate(zip(self.alpha[:-1], self.beta[1:], self.nonlinearity)):
            skip = beta.view(1, -1) * F.linear(x, self.W, self.bias[i + 1])
            z = skip + alpha.view(1, -1) * nonlinearity(z)

        z = self.alpha[-1].view(1, -1) * self.nonlinearity[-1](z)
        z = F.linear(z, self.W.T, self.bias[-1])

        return z


class Module_ModularGN(nn.Module):
    def __init__(self, in_dim, embed_dim, activation):
        super().__init__()

        self.beta = nn.Parameter(torch.rand(1), requires_grad=True)
        self.W = nn.Parameter(init_W(embed_dim, in_dim), requires_grad=True)
        self.b = nn.Parameter(init_b(embed_dim, in_dim), requires_grad=True)
        self.act = activation()

    def forward(self, x):

        z = F.linear(x, weight=self.W, bias=self.b)
        z = self.act(z * F.softplus(self.beta))
        z = F.linear(z, weight=self.W.T)

        return z


class ModularGradNet(nn.Module):
    def __init__(self, num_modules, in_dim, embed_dim, activation):
        super().__init__()

        self.num_modules = num_modules
        self.mmgn_modules = nn.ModuleList([Module_ModularGN(in_dim, embed_dim, activation) for i in range(num_modules)])
        self.alpha = nn.Parameter(torch.randn(num_modules,), requires_grad=True)
        self.bias = nn.Parameter(init_b(in_dim, embed_dim), requires_grad=True)

    def forward(self, x):
        out = 0
        for i in range(self.num_modules):
            out += self.alpha[i] * self.mmgn_modules[i](x)
        out += self.bias
        return out

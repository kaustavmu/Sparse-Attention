"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            if i not in self.u:
                self.u[i] = 0
            grad = ndl.Tensor(param.grad, device=param.device, dtype='float32').data + self.weight_decay * param.data       
            self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * grad
            param.data = param.data - self.u[i] * self.lr
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped



class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for w in self.params:
            if w.grad is None:
                continue
            if self.weight_decay > 0:
                grad = w.grad.data + self.weight_decay * w.data
            else:
                grad = w.grad.data

            if w not in self.m:
                self.m[w] = ndl.init.zeros(*w.shape, device=w.device, dtype=w.dtype)
            if w not in self.v:
                self.v[w] = ndl.init.zeros(*w.shape, device=w.device, dtype=w.dtype)

            self.m[w] = self.m[w].detach()
            self.v[w] = self.v[w].detach()

            self.m[w] = self.beta1 * self.m[w] + (1 - self.beta1) * grad
            self.v[w] = self.beta2 * self.v[w] + (1 - self.beta2) * (grad ** 2)

            
            m_hat = self.m[w] / (1 - self.beta1 ** self.t)
            v_hat = self.v[w] / (1 - self.beta2 ** self.t)

            grad = grad.detach()
            m_hat = m_hat.detach()
            v_hat = v_hat.detach()

            w.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION

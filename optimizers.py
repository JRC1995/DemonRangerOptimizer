from torch.optim.optimizer import Optimizer, required
import torch
import math
import numpy as np


class LRangerMod(Optimizer):

    # AMSGrad/Adam + AdaMod + QH Momentum + Iterate Averaging + Lookahead + Rule of Thumb Linear Warmup (instead of RAdam Rectification) + P from PAdam

    def __init__(self, params, lr=1e-3,
                 betas=(0.999, 0.999, 0.999),
                 nus=(0.7, 1.0),
                 p=0.5,
                 eps=1e-8,
                 k=5,
                 alpha=0.8,
                 amsgrad=True,
                 AdaMod=True,
                 warmup=True,
                 AdaMod_bias_correct=True,
                 IA=True,
                 use_gc=False,
                 IA_cycle=1000,
                 epochs=100,
                 step_per_epoch=None,
                 weight_decay=0):

        # betas = (beta1 for first order moments, beta2 for second order moments, beta3 for ema over adaptive learning rates (AdaMod))
        # nus = (nu1,nu2) (for quasi hyperbolic momentum)
        # eps = small value for numerical stability (avoid divide by zero)
        # k = lookahead cycle
        # alpha = outer learning rate (lookahead)
        # amsgrad = bool to decide whether to use amsgrad instead of adam as the core optimizer
        # AdaMod_bias_correct = bool to decide whether to add bias correction to AdaMod
        # IA = bool to decide if Iterate Averaging is ever going to be used
        # IA_cycle = Iterate Averaging Cycle (Recommended to initialize with no. of iterations in Epoch) (doesn't matter if you are not using IA)
        # epochs = No. of epochs you plan to use (Only relevant if using DEMON)
        # step_per_epoch = No. of iterations in an epoch (only relevant if using DEMON)
        # weight decay = decorrelated weight decay value

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= p <= 0.5:
            raise ValueError("Invalid p value: {}".format(p))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= nus[0] <= 1.0:
            raise ValueError(
                "Invalid nu parameter at index 0: {}".format(nus[0]))
        if not 0.0 <= nus[1] <= 1.0:
            raise ValueError(
                "Invalid nu parameter at index 1: {}".format(nus[1]))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))

        self.k = k
        self.epochs = epochs
        self.amsgrad = amsgrad
        self.warmup_period = 2 / (1 - betas[1])
        self.IA_cycle = IA_cycle
        self.IA = IA
        self.AdaMod = AdaMod
        self.use_gc = use_gc
        self.AdaMod_bias_correct = AdaMod_bias_correct
        self.warmup = warmup
        if step_per_epoch is None:
            self.step_per_epoch = IA_cycle
        else:
            self.step_per_epoch = step_per_epoch

        self.T = self.epochs * self.step_per_epoch

        defaults = dict(lr=lr,
                        betas=betas,
                        nus=nus,
                        eps=eps,
                        p=p,
                        alpha=alpha,
                        weight_decay=weight_decay)
        super(LRangerMod, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LRangerMod, self).__setstate__(state)

    def apply_AdaMod(self, beta3, n_avg, n, step):
        n_avg.mul_(beta3).add_(1 - beta3, n)
        if self.AdaMod_bias_correct:
            n_avg_ = n_avg.clone()
            n_avg_.div_(1 - (beta3 ** step))
            torch.min(n, n_avg_, out=n)
        else:
            torch.min(n, n_avg, out=n)
        return n

    def step(self, activate_IA=False, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'LRangerMod does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['num_models'] = 0
                    state['cached_params'] = p.data.clone()
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.AdaMod:
                        state['n_avg'] = torch.zeros_like(p.data)

                state['step'] += 1

                w = min([1.0, state['step'] / self.warmup_period])
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2, beta3 = group['betas']
                nu1, nu2 = group['nus']

                if self.warmup:
                    lr = w * group['lr']
                else:
                    lr = group['lr']

                wd = group['weight_decay']
                alpha = group['alpha']

                do_IA = False
                lookahead_step = False

                if self.IA and activate_IA:
                    lookahead_step = False
                    if state['step'] % self.IA_cycle == 0:
                        do_IA = True
                elif self.k == 0:
                    lookahead_step = False
                else:
                    if state['step'] % self.k == 0:
                        lookahead_step = True
                    else:
                        lookahead_step = False

                if self.use_gc and grad.view(-1).size(0) > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1,
                                                         len(list(grad.size())))), keepdim=True))

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                momentum = exp_avg.clone()
                momentum.div_(
                    1 - (beta1 ** state['step'])).mul_(nu1).add_(1 - nu1, grad)

                if wd != 0:
                    p.data.add_(-wd * lr, p.data)

                beta2_t = beta2 ** state['step']

                if self.amsgrad and state['step'] > 1:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    vt = max_exp_avg_sq.clone()
                else:
                    vt = exp_avg_sq.clone()

                bias_correction2 = 1 - beta2_t
                vt.div_(bias_correction2)
                if nu2 != 1.0:
                    vt.mul_(nu2).addcmul_(1 - nu2, grad, grad)
                denom = vt.pow_(group['p']).add_(group['eps'])
                n = lr / denom
                if self.AdaMod:
                    n_avg = state['n_avg']
                    n = self.apply_AdaMod(beta3, n_avg, n, step=state['step'])

                p.data.add_(-n * momentum)

                if lookahead_step:
                    p.data.mul_(alpha).add_(
                        1.0 - alpha, state['cached_params'])
                    state['cached_params'].copy_(p.data)

                if do_IA:
                    p.data.add_(state["num_models"], state['cached_params']
                                ).div_(state["num_models"] + 1.0)
                    state['cached_params'].copy_(p.data)
                    state["num_models"] += 1

        return loss


class DemonRanger(Optimizer):

    # Rectified-AMSGrad/RAdam + AdaMod + QH Momentum + Iterat Averaging + Lookahead + DEMON (decaying Momentum) + gradient centralization + grad noise

    def __init__(self, params, lr=1e-3,
                 betas=(0.999, 0.999, 0.999),
                 nus=(0.7, 1.0),
                 eps=1e-8,
                 k=5,
                 alpha=0.8,
                 gamma=0.55,
                 use_demon=True,
                 rectify=True,
                 amsgrad=True,
                 AdaMod=True,
                 AdaMod_bias_correct=True,
                 IA=True,
                 IA_cycle=1000,
                 epochs=100,
                 step_per_epoch=None,
                 weight_decay=0,
                 use_gc=True,
                 use_grad_noise=False,
                 use_diffgrad=False,
                 dropout=0.0):

        # betas = (beta1 for first order moments, beta2 for second order moments, beta3 for ema over adaptive learning rates (AdaMod))
        # nus = (nu1,nu2) (for quasi hyperbolic momentum)
        # eps = small value for numerical stability (avoid divide by zero)
        # k = lookahead cycle
        # alpha = outer learning rate (lookahead)
        # gamma = gradient noise control parameter (for regularization)
        # use_demon = bool to decide whether to use DEMON (Decaying Momentum) or not
        # rectify = bool to decide whether to apply the recitification term (from RAdam) or not
        # amsgrad = bool to decide whether to use amsgrad instead of adam as the core optimizer
        # AdaMod_bias_correct = bool to decide whether to add bias correction to AdaMod
        # IA = bool to decide if Iterate Averaging is ever going to be used
        # IA_cycle = Iterate Averaging Cycle (Recommended to initialize with no. of iterations in Epoch) (doesn't matter if you are not using IA)
        # epochs = No. of epochs you plan to use (Only relevant if using DEMON)
        # step_per_epoch = No. of iterations in an epoch (only relevant if using DEMON)
        # weight decay = decorrelated weight decay value
        # use_gc = bool to determine whether to use gradient centralization or not.
        # use_grad_noise = bool to determine whether to use gradient noise or not.
        # use_diffgrad = bool to determine whether to use diffgrad or not.
        # dropout = learning rate dropout, probability of setting learning rate to zero

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= nus[0] <= 1.0:
            raise ValueError(
                "Invalid nu parameter at index 0: {}".format(nus[0]))
        if not 0.0 <= nus[1] <= 1.0:
            raise ValueError(
                "Invalid nu parameter at index 1: {}".format(nus[1]))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid dropout parameter: {}".format(dropout))

        self.use_gc = use_gc
        self.use_grad_noise = use_grad_noise
        self.use_diffgrad = use_diffgrad
        self.k = k
        self.epochs = epochs
        self.amsgrad = amsgrad
        self.use_demon = use_demon
        self.IA_cycle = IA_cycle
        self.IA = IA
        self.rectify = rectify
        self.AdaMod = AdaMod
        self.AdaMod_bias_correct = AdaMod_bias_correct
        if step_per_epoch is None:
            self.step_per_epoch = IA_cycle
        else:
            self.step_per_epoch = step_per_epoch

        self.T = self.epochs * self.step_per_epoch

        defaults = dict(lr=lr,
                        betas=betas,
                        nus=nus,
                        eps=eps,
                        alpha=alpha,
                        gamma=gamma,
                        weight_decay=weight_decay,
                        dropout=dropout)
        super(DemonRanger, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DemonRanger, self).__setstate__(state)

    def apply_AdaMod(self, beta3, n_avg, n, step):
        n_avg.mul_(beta3).add_(1 - beta3, n)
        if self.AdaMod_bias_correct:
            n_avg_ = n_avg.clone()
            n_avg_.div_(1 - (beta3 ** step))
            torch.min(n, n_avg_, out=n)
        else:
            torch.min(n, n_avg, out=n)
        return n

    def step(self, activate_IA=False, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'DemonRanger does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.use_diffgrad:
                        state['previous_grad'] = torch.zeros_like(p.data)
                    state['num_models'] = 0
                    state['cached_params'] = p.data.clone()
                    if self.amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.AdaMod:
                        state['n_avg'] = torch.zeros_like(p.data)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1_init, beta2, beta3 = group['betas']
                rho_inf = (2 / (1 - beta2)) - 1
                nu1, nu2 = group['nus']
                lr = group['lr']
                wd = group['weight_decay']
                alpha = group['alpha']
                gamma = group['gamma']

                do_IA = False
                lookahead_step = False

                if self.IA and activate_IA:
                    lookahead_step = False
                    if state['step'] % self.IA_cycle == 0:
                        do_IA = True
                elif self.k == 0:
                    lookahead_step = False
                else:
                    if state['step'] % self.k == 0:
                        lookahead_step = True
                    else:
                        lookahead_step = False

                if self.use_demon:
                    temp = 1 - (state['step'] / self.T)
                    beta1 = beta1_init * temp / \
                        ((1 - beta1_init) + beta1_init * temp)
                else:
                    beta1 = beta1_init

                if self.use_grad_noise:
                    grad_var = lr / ((1 + state['step'])**gamma)
                    grad_noise = torch.empty_like(grad).normal_(
                        mean=0.0, std=math.sqrt(grad_var))
                    grad.add_(grad_noise)

                if self.use_gc and grad.dim() > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                if self.use_diffgrad:
                    previous_grad = state['previous_grad']
                    diff = abs(previous_grad - grad)
                    dfc = 1. / (1. + torch.exp(-diff))
                    state['previous_grad'] = grad.clone()
                    exp_avg = exp_avg * dfc

                momentum = exp_avg.clone()
                momentum.div_(
                    1 - (beta1 ** state['step'])).mul_(nu1).add_(1 - nu1, grad)

                if wd != 0:
                    p.data.add_(-wd * lr, p.data)

                beta2_t = beta2 ** state['step']

                if self.amsgrad and state['step'] > 1:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    vt = max_exp_avg_sq.clone()
                else:
                    vt = exp_avg_sq.clone()

                if self.rectify:
                    rho_t = rho_inf - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)

                    # more conservative since it's an approximated value
                    if rho_t >= 5:
                        R = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) /
                                      ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                        bias_correction2 = 1 - beta2_t
                        vt.div_(bias_correction2)
                        if nu2 != 1.0:
                            vt.mul_(nu2).addcmul_(1 - nu2, grad, grad)
                        denom = vt.sqrt_().add_(group['eps'])

                        n = (lr * R) / denom

                        if self.AdaMod:
                            n_avg = state['n_avg']
                            n = self.apply_AdaMod(
                                beta3, n_avg, n, step=state['step'])

                        if group['dropout'] > 0.0:
                            mask = torch.bernoulli(
                                torch.ones_like(p.data) - group['dropout'])
                            n = n * mask

                        p.data.add_(-n * momentum)
                    else:
                        if self.AdaMod:
                            n_avg = state['n_avg']
                            n_avg.mul_(beta3).add_(1 - beta3, lr)
                        p.data.add_(-lr, momentum)
                else:
                    bias_correction2 = 1 - beta2_t
                    vt.div_(bias_correction2)
                    if nu2 != 1.0:
                        vt.mul_(nu2).addcmul_(1 - nu2, grad, grad)
                    denom = vt.sqrt_().add_(group['eps'])
                    n = lr / denom
                    if self.AdaMod:
                        n_avg = state['n_avg']
                        n = self.apply_AdaMod(
                            beta3, n_avg, n, step=state['step'])

                    if group['dropout'] > 0.0:
                        mask = torch.bernoulli(
                            torch.ones_like(p.data) - group['dropout'])
                        n = n * mask

                    p.data.add_(-n * momentum)

                if lookahead_step:
                    p.data.mul_(alpha).add_(
                        1.0 - alpha, state['cached_params'])
                    state['cached_params'].copy_(p.data)

                if do_IA:
                    p.data.add_(state["num_models"], state['cached_params']
                                ).div_(state["num_models"] + 1.0)
                    state['cached_params'].copy_(p.data)
                    state["num_models"] += 1

        return loss


class HyperRanger(Optimizer):

    # Nostalgic PAdam + QH Momentum + Iterate Averaging + Lookahead + DEMON (decaying Momentum) + gradient centralization + hypergradient descent on lr and nu1

    def __init__(self, params, lr=1e-3,
                 betas=(0.999, 0.999),
                 nus=(0.7, 1.0),
                 eps=1e-8,
                 gamma=0.0001,
                 nostalgia=True,
                 use_demon=True,
                 hypergrad_lr=1e-7,
                 HDM=False,
                 hypertune_nu1=False,
                 p=0.25,
                 k=5,
                 alpha=0.8,
                 IA=True,
                 IA_cycle=1000,
                 epochs=100,
                 step_per_epoch=None,
                 weight_decay=0,
                 use_gc=True,
                 use_diffgrad=False):

        # betas = (beta1 for first order moments, beta2 for second order moments)
        # nus = (nu1,nu2) (for quasi hyperbolic momentum)
        # eps = small value for numerical stability (avoid divide by zero)
        # k = lookahead cycle
        # alpha = outer learning rate (lookahead)
        # gamma = used for nostalgia
        # nostalgia = bool to decide whether to use nostalgia (from Nostalgic Adam or NosAdam)
        # use_demon = bool to decide whether to use DEMON (Decaying Momentum) or not
        # hypergrad_lr = learning rate for updating hyperparameters (like lr) through hypergradient descent (probably need to increase around 0.02 if HDM is True). Set to 0.0 to disable hypergradient descent.
        # HDM = bool to decide whether to use Multiplicative rule for updating hyperparameters or not
        # hypertune_nu1 = bool to decide whether apply hypergradient descent on nu1 as well or not.
        # p = p from PAdam
        # IA = bool to decide if Iterate Averaging is ever going to be used
        # IA_cycle = Iterate Averaging Cycle (Recommended to initialize with no. of iterations in Epoch) (doesn't matter if you are not using IA)
        # epochs = No. of epochs you plan to use (Only relevant if using DEMON)
        # step_per_epoch = No. of iterations in an epoch (only relevant if using DEMON)
        # weight decay = decorrelated weight decay value
        # use_gc = bool to determine whether to use gradient centralization or not.
        # use_diffgrad = bool to determine whether to use diffgrad or not.

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= hypergrad_lr:
            raise ValueError(
                "Invalid hypergradient learning rate: {}".format(hypergrad_lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= nus[0] <= 1.0:
            raise ValueError(
                "Invalid nu parameter at index 0: {}".format(nus[0]))
        if not 0.0 <= nus[1] <= 1.0:
            raise ValueError(
                "Invalid nu parameter at index 1: {}".format(nus[1]))
        if not 0.0 <= p <= 0.5:
            raise ValueError("Invalid p parameter: {}".format(p))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))

        self.nostalgia = nostalgia
        self.use_demon = use_demon
        self.k = k
        self.IA = IA
        self.IA_cycle = IA_cycle
        self.epochs = epochs
        if step_per_epoch is None:
            self.step_per_epoch = IA_cycle
        else:
            self.step_per_epoch = step_per_epoch
        self.use_gc = use_gc
        self.use_diffgrad = use_diffgrad
        self.T = self.epochs * self.step_per_epoch
        self.hypertune_nu1 = hypertune_nu1
        self.HDM = HDM

        defaults = dict(lr=lr,
                        betas=betas,
                        nu1=nus[0],
                        nu2=nus[1],
                        eps=eps,
                        alpha=alpha,
                        gamma=gamma,
                        p=p,
                        hypergrad_lr=hypergrad_lr,
                        weight_decay=weight_decay)
        super(HyperRanger, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HyperRanger, self).__setstate__(state)

    def step(self, activate_IA=False, display=False, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'HyperRanger does not support sparse gradients')

                state = self.state[p]

                hypergrad_lr = group['hypergrad_lr']
                beta1_init, beta2 = group['betas']
                wd = group['weight_decay']
                alpha = group['alpha']
                gamma = group['gamma']

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.use_diffgrad:
                        state['previous_grad'] = torch.zeros_like(p.data)
                    state['lr'] = group['lr']

                    if self.IA:
                        state['num_models'] = 0
                    if self.IA or (self.k > 0):
                        state['cached_params'] = p.data.clone()
                    if self.nostalgia:
                        state['B_old'] = 0
                        state['B_new'] = 1
                    if hypergrad_lr > 0.0:
                        state['nu1'] = group['nu1']
                        state['prev_lr_grad'] = torch.zeros_like(grad.view(-1))
                        if self.hypertune_nu1:

                            state['prev_nu_grad'] = torch.zeros_like(
                                grad.view(-1))

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if self.use_demon:
                    temp = 1 - (state['step'] / self.T)
                    beta1 = beta1_init * temp / \
                        ((1 - beta1_init) + beta1_init * temp)
                else:
                    beta1 = beta1_init

                if self.nostalgia:
                    beta2 = state['B_old'] / state['B_new']
                    state['B_old'] += math.pow(state['step'], -gamma)
                    state['B_new'] += math.pow(state['step'] + 1, -gamma)

                do_IA = False
                lookahead_step = False

                if self.IA and activate_IA:
                    lookahead_step = False
                    if state['step'] % self.IA_cycle == 0:
                        do_IA = True
                elif self.k == 0:
                    lookahead_step = False
                else:
                    if state['step'] % self.k == 0:
                        lookahead_step = True
                    else:
                        lookahead_step = False

                if state['step'] > 1 and hypergrad_lr > 0.0:
                    prev_lr_grad = state['prev_lr_grad']
                    h = torch.dot(grad.view(-1), prev_lr_grad)

                    if self.HDM:
                        grad_norm = grad.view(-1).norm()
                        norm_denom = grad_norm * (prev_lr_grad.norm())
                        norm_denom.add_(group['eps'])
                        state['lr'] = state['lr'] * \
                            (1 - hypergrad_lr * (h / norm_denom))
                    else:
                        state['lr'] -= hypergrad_lr * h

                    torch.max(state['lr'], torch.zeros_like(
                        state['lr']), out=state['lr'])

                    if display:
                        print("lr", state['lr'])

                    if self.hypertune_nu1:
                        prev_nu_grad = state['prev_nu_grad']
                        h = torch.dot(grad.view(-1), prev_nu_grad)
                        if self.HDM:
                            norm_denom = grad_norm * (prev_nu_grad.norm())
                            norm_denom.add_(group['eps'])
                            state['nu1'] = state['nu1'] * \
                                (1 - hypergrad_lr * (h / norm_denom))
                        else:
                            state['nu1'] -= hypergrad_lr * h

                        torch.max(state['nu1'], torch.zeros_like(
                            state['nu1']), out=state['nu1'])
                        torch.min(state['nu1'], torch.ones_like(
                            state['nu1']), out=state['nu1'])

                    if display:
                        print("nu", state['nu1'])

                if self.use_gc and grad.dim() > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                nu1 = state['nu1']
                nu2 = group['nu2']
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                if self.use_diffgrad:
                    previous_grad = state['previous_grad']
                    diff = abs(previous_grad - grad)
                    dfc = 1. / (1. + torch.exp(-diff))
                    state['previous_grad'] = grad.clone()
                    exp_avg = exp_avg * dfc

                momentum = exp_avg.clone()
                bias_correction1 = 1 - (beta1 ** state['step'])
                momentum.div_(bias_correction1)

                vt = exp_avg_sq.clone()

                if not self.nostalgia:
                    vt.div_(1 - (beta2 ** state['step']))
                if nu2 != 1.0:
                    vt.mul_(nu2).addcmul_(1 - nu2, grad, grad)

                denom = vt.pow_(group['p']).add_(group['eps'])

                n = state['lr'] / denom

                if lookahead_step:
                    dalpha = alpha
                elif do_IA:
                    dalpha = (1 / (state["num_models"] + 1.0))
                else:
                    dalpha = 1.0

                if hypergrad_lr > 0.0 and self.hypertune_nu1:
                    state['prev_nu_grad'] = (-dalpha *
                                             n * (momentum - grad)).view(-1)

                # quasi hyperbolic momentum
                momentum.mul_(nu1).add_(1 - nu1, grad)

                if hypergrad_lr > 0.0:
                    temp = dalpha * (-(momentum / denom) - wd * p.data)
                    state['prev_lr_grad'] = temp.view(-1)

                p.data.add_(-n * momentum)

                if wd != 0:
                    p.data.add_(-wd * state['lr'], p.data)

                if lookahead_step:
                    p.data.mul_(alpha).add_(
                        1.0 - alpha, state['cached_params'])
                    state['cached_params'].copy_(p.data)

                if do_IA:
                    p.data.add_(state["num_models"], state['cached_params']
                                ).div_(state["num_models"] + 1.0)
                    state['cached_params'].copy_(p.data)
                    state["num_models"] += 1

        return loss


class HyperRangerMod(Optimizer):

    # Different from HyperRanger integrates AdaMod, and hypergradient descent through it. Slower, however.
    # doesn't have hypertunability for nu1 though.

    def __init__(self, params, lr=1e-3,
                 betas=(0.999, 0.999, 0.999),
                 nus=(0.7, 1.0),
                 eps=1e-8,
                 AdaMod_bias_correct=True,
                 gamma=0.0001,
                 nostalgia=True,
                 use_demon=True,
                 hypergrad_lr=1e-7,
                 p=0.5,
                 k=5,
                 alpha=0.8,
                 IA=True,
                 IA_cycle=1000,
                 epochs=100,
                 step_per_epoch=None,
                 weight_decay=0,
                 use_gc=True,
                 use_diffgrad=False):

        # betas = (beta1 for first order moments, beta2 for second order moments, beta3 for AdaMod) # set beta3 = 0 to disable AdaMod
        # nus = (nu1,nu2) (for quasi hyperbolic momentum)
        # eps = small value for numerical stability (avoid divide by zero)
        # AdaMod_bias_correct = bool to determine whether to apply bias correction on AdaMod or not
        # k = lookahead cycle
        # alpha = outer learning rate (lookahead)
        # gamma = used for nostalgia
        # nostalgia = bool to decide whether to use nostalgia (from Nostalgic Adam or NosAdam)
        # use_demon = bool to decide whether to use DEMON (Decaying Momentum) or not
        # hypergrad_lr = learning rate for updating hyperparameters (like lr) through hypergradient descent (probably need to increase around 0.02 if HDM is True). Set to 0.0 to disable hypergradient descent.
        # HDM = bool to decide whether to use Multiplicative rule for updating hyperparameters or not
        # hypertune_nu1 = bool to decide whether apply hypergradient descent on nu1 as well or not.
        # p = p from PAdam
        # IA = bool to decide if Iterate Averaging is ever going to be used
        # IA_cycle = Iterate Averaging Cycle (Recommended to initialize with no. of iterations in Epoch) (doesn't matter if you are not using IA)
        # epochs = No. of epochs you plan to use (Only relevant if using DEMON)
        # step_per_epoch = No. of iterations in an epoch (only relevant if using DEMON)
        # weight decay = decorrelated weight decay value
        # use_gc = bool to determine whether to use gradient centralization or not.
        # use_diffgrad = bool to determine whether to use diffgrad or not.

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= hypergrad_lr:
            raise ValueError(
                "Invalid hypergradient learning rate: {}".format(hypergrad_lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= nus[0] <= 1.0:
            raise ValueError(
                "Invalid nu parameter at index 0: {}".format(nus[0]))
        if not 0.0 <= nus[1] <= 1.0:
            raise ValueError(
                "Invalid nu parameter at index 1: {}".format(nus[1]))
        if not 0.0 <= p <= 0.5:
            raise ValueError("Invalid p parameter: {}".format(p))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))

        self.AdaMod_bias_correct = AdaMod_bias_correct
        self.nostalgia = nostalgia
        self.use_demon = use_demon
        self.k = k
        self.IA = IA
        self.IA_cycle = IA_cycle
        self.epochs = epochs
        if step_per_epoch is None:
            self.step_per_epoch = IA_cycle
        else:
            self.step_per_epoch = step_per_epoch
        self.use_gc = use_gc
        self.use_diffgrad = use_diffgrad
        self.T = self.epochs * self.step_per_epoch

        defaults = dict(lr=lr,
                        betas=betas,
                        nus=nus,
                        eps=eps,
                        alpha=alpha,
                        gamma=gamma,
                        p=p,
                        hypergrad_lr=hypergrad_lr,
                        weight_decay=weight_decay)
        super(HyperRangerMod, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HyperRangerMod, self).__setstate__(state)

    def step(self, display=False, activate_IA=False, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'HyperRangerMod does not support sparse gradients')

                state = self.state[p]

                hypergrad_lr = group['hypergrad_lr']
                beta1_init, beta2, beta3 = group['betas']
                nu1, nu2 = group['nus']
                wd = group['weight_decay']
                alpha = group['alpha']
                gamma = group['gamma']

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.use_diffgrad:
                        state['previous_grad'] = torch.zeros_like(p.data)
                    state['lr'] = group['lr']

                    if self.IA:
                        state['num_models'] = 0
                    if self.IA or self.k > 0:
                        state['cached_params'] = p.data.clone()
                    if beta3 > 0.0:
                        state['n_avg'] = torch.zeros_like(p.data)
                    if self.nostalgia:
                        state['B_old'] = 0
                        state['B_new'] = 1
                    if hypergrad_lr > 0.0:
                        state['cached_hypergrad_comp'] = torch.zeros_like(
                            grad.view(-1))

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if self.use_demon:
                    temp = 1 - (state['step'] / self.T)
                    beta1 = beta1_init * temp / \
                        ((1 - beta1_init) + beta1_init * temp)
                else:
                    beta1 = beta1_init

                if self.nostalgia:
                    beta2 = state['B_old'] / state['B_new']
                    state['B_old'] += math.pow(state['step'], -gamma)
                    state['B_new'] += math.pow(state['step'] + 1, -gamma)

                do_IA = False
                lookahead_step = False

                if self.IA and activate_IA:
                    lookahead_step = False
                    if state['step'] % self.IA_cycle == 0:
                        do_IA = True
                elif self.k == 0:
                    lookahead_step = False
                else:
                    if state['step'] % self.k == 0:
                        lookahead_step = True
                    else:
                        lookahead_step = False

                if state['step'] > 1 and hypergrad_lr > 0.0:
                    du = state['cached_hypergrad_comp']
                    h = torch.dot(grad.view(-1), du)
                    state['lr'] -= hypergrad_lr * h
                    torch.max(state['lr'], torch.zeros_like(
                        state['lr']), out=state['lr'])
                    if display:
                        print(state['lr'])

                if self.use_gc and grad.dim() > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                if self.use_diffgrad:
                    previous_grad = state['previous_grad']
                    diff = abs(previous_grad - grad)
                    dfc = 1. / (1. + torch.exp(-diff))
                    state['previous_grad'] = grad.clone()
                    exp_avg = exp_avg * dfc

                momentum = exp_avg.clone()
                momentum.div_(
                    1 - (beta1 ** state['step'])).mul_(nu1).add_(1 - nu1, grad)
                vt = exp_avg_sq.clone()

                if not self.nostalgia:
                    vt.div_(1 - (beta2 ** state['step']))
                if nu2 != 1.0:
                    vt.mul_(nu2).addcmul_(1 - nu2, grad, grad)

                denom = vt.pow_(group['p']).add_(group['eps'])

                n = state['lr'] / denom

                if beta3 > 0.0:  # apply AdaMod
                    n_avg = state['n_avg']
                    n_avg.mul_(beta3).add_(1 - beta3, n)
                    if self.AdaMod_bias_correct:
                        n_avg_ = n_avg.clone()
                        bias_correction3 = 1 - (beta3 ** state['step'])
                        n_avg_.div_(bias_correction3)
                        torch.min(n, n_avg_, out=n)
                    else:
                        torch.min(n, n_avg, out=n)

                p.data.add_(-n * momentum)

                if lookahead_step:
                    dalpha = alpha
                elif do_IA:
                    dalpha = (1 / (state["num_models"] + 1.0))
                else:
                    dalpha = 1.0

                if hypergrad_lr > 0.0:

                    if beta3 > 0.0:
                        grad_from_n = dalpha * \
                            (-(momentum / denom) - wd * p.data)

                        if self.AdaMod_bias_correct:
                            grad_from_n_avg_ = dalpha * \
                                (-((1 - beta3) / bias_correction3)
                                 * (momentum / denom) - wd * p.data)
                            du = torch.where(n_avg_ < n,
                                             grad_from_n_avg_,
                                             grad_from_n)
                        else:
                            grad_from_n_avg = dalpha * \
                                (-(1 - beta3) * (momentum / denom) - wd * p.data)
                            du = torch.where(n_avg < n,
                                             grad_from_n_avg,
                                             grad_from_n)

                    else:
                        du = dalpha * (-(momentum / denom) - wd * p.data)

                    state['cached_hypergrad_comp'] = du.view(-1)

                if wd != 0:
                    p.data.add_(-wd * state['lr'], p.data)

                if lookahead_step:
                    p.data.mul_(alpha).add_(
                        1.0 - alpha, state['cached_params'])
                    state['cached_params'].copy_(p.data)

                if do_IA:
                    p.data.add_(state["num_models"], state['cached_params']
                                ).div_(state["num_models"] + 1.0)
                    state['cached_params'].copy_(p.data)
                    state["num_models"] += 1

        return loss


class HDQHSGDW(Optimizer):
    def __init__(self, params, lr=1e-3,
                 beta=0.999,
                 nu=0.7,
                 hypergrad_lr=1e-3,
                 HDM=False,
                 k=5,
                 alpha=0.5,
                 eps=1e-8,
                 weight_decay=0,
                 use_gc=True,
                 use_diffgrad=False):

        # BASIC SGD + Momentum but with QHMomentum and Hypergradient descent over all beta, lr, and nu + Lookahead and decorrelated weight decay
        # they say the best of them all is still SGD + Momentum?
        # use_diffgrad = bool to determine whether to use diffgrad or not.

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= hypergrad_lr:
            raise ValueError(
                "Invalid hypergradient learning rate: {}".format(hypergrad_lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if not 0.0 <= nu <= 1.0:
            raise ValueError("Invalid nu parameter: {}".format(nu))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))

        self.k = k
        self.use_gc = use_gc
        self.use_diffgrad = use_diffgrad
        self.HDM = HDM

        defaults = dict(lr=lr,
                        beta=beta,
                        nu=nu,
                        alpha=alpha,
                        hypergrad_lr=hypergrad_lr,
                        eps=eps,
                        weight_decay=weight_decay)
        super(HDQHSGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HDQHSGDW, self).__setstate__(state)

    def hyperupdate(self, update, grad, grad_comp, hypergrad_lr, eps):
        h = torch.dot(grad.view(-1), grad_comp)

        if self.HDM:
            grad_norm = grad.view(-1).norm()
            norm_denom = grad_norm * (grad_comp.norm())
            norm_denom.add_(eps)
            update = update * (1 - hypergrad_lr * (h / norm_denom))
        else:
            update -= hypergrad_lr * h

        return update

    def step(self, display=False, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'HDQHSGDW does not support sparse gradients')

                state = self.state[p]

                hypergrad_lr = group['hypergrad_lr']
                wd = group['weight_decay']
                alpha = group['alpha']

                if len(state) == 0:
                    state['step'] = 0
                    state['lr'] = group['lr']
                    state['nu'] = group['nu']
                    state['beta'] = group['beta']
                    state['exp_avg'] = torch.zeros_like(p.data)
                    if self.use_diffgrad:
                        state['previous_grad'] = torch.zeros_like(p.data)
                    if self.k > 0:
                        state['cached_params'] = p.data.clone()
                    if hypergrad_lr > 0.0:
                        state['prev_lr_grad'] = torch.zeros_like(grad.view(-1))
                        state['prev_nu_grad'] = torch.zeros_like(grad.view(-1))
                        state['prev_beta_grad'] = torch.zeros_like(
                            grad.view(-1))

                state['step'] += 1
                exp_avg = state['exp_avg']

                if self.use_diffgrad:
                    previous_grad = state['previous_grad']
                    diff = abs(previous_grad - grad)
                    dfc = 1. / (1. + torch.exp(-diff))
                    state['previous_grad'] = grad.clone()
                    exp_avg = exp_avg * dfc

                lookahead_step = False

                if self.k == 0:
                    lookahead_step = False
                else:
                    if state['step'] % self.k == 0:
                        lookahead_step = True
                    else:
                        lookahead_step = False

                if state['step'] > 1 and hypergrad_lr > 0.0:

                    prev_lr_grad = state['prev_lr_grad']
                    prev_beta_grad = state['prev_beta_grad']
                    prev_nu_grad = state['prev_nu_grad']

                    state['lr'] = self.hyperupdate(update=state['lr'],
                                                   grad=grad,
                                                   grad_comp=prev_lr_grad,
                                                   hypergrad_lr=hypergrad_lr,
                                                   eps=group['eps'])

                    torch.max(state['lr'], torch.zeros_like(
                        state['lr']), out=state['lr'])

                    if display:
                        print("lr", state['lr'])

                    state['beta'] = self.hyperupdate(update=state['beta'],
                                                     grad=grad,
                                                     grad_comp=prev_beta_grad,
                                                     hypergrad_lr=hypergrad_lr,
                                                     eps=group['eps'])

                    torch.max(state['beta'], torch.zeros_like(
                        state['beta']), out=state['beta'])
                    torch.min(state['beta'], torch.ones_like(
                        state['beta']), out=state['beta'])

                    if display:
                        print("beta", group['beta'])

                    state['nu'] = self.hyperupdate(update=state['nu'],
                                                   grad=grad,
                                                   grad_comp=prev_beta_grad,
                                                   hypergrad_lr=hypergrad_lr,
                                                   eps=group['eps'])

                    torch.max(state['nu'], torch.zeros_like(
                        state['nu']), out=state['nu'])
                    torch.min(state['nu'], torch.ones_like(
                        state['nu']), out=state['nu'])

                    if display:
                        print("nu", state['nu'])

                if self.use_gc and grad.dim() > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                nu = state['nu']
                beta = state['beta']
                lr = state['lr']

                if lookahead_step:
                    dalpha = alpha
                else:
                    dalpha = 1.0

                gx = 1 - (beta ** state['step'])
                fx = beta * exp_avg + (1 - beta) * grad

                if hypergrad_lr > 0.0:
                    dfx = exp_avg - grad
                    dgx = - state['step'] * beta**(state['step'] - 1)
                    dbeta = (gx * dfx + fx * dgx) / \
                        (math.pow(gx, 2) + group['eps'])
                    dbeta = - dalpha * lr * nu * dbeta
                    state['prev_beta_grad'] = dbeta.view(-1)

                momentum = fx / gx
                group['exp_avg'] = fx

                if hypergrad_lr > 0.0:
                    state['prev_nu_grad'] = (-dalpha *
                                             lr * (momentum - grad)).view(-1)

                # quasi hyperbolic momentum
                momentum.mul_(nu).add_(1 - nu, grad)

                if hypergrad_lr > 0.0:
                    temp = dalpha * (-momentum - wd * p.data)
                    state['prev_lr_grad'] = temp.view(-1)

                p.data.add_(-group['lr'] * momentum)

                if wd != 0:
                    p.data.add_(-wd * group['lr'], p.data)

                if lookahead_step:
                    p.data.mul_(alpha).add_(
                        1.0 - alpha, state['cached_params'])
                    state['cached_params'].copy_(p.data)

        return loss


class HyperProp(Optimizer):

    # LaProp + hypergradient descent on lr and nu (for QH momentum) + QH Momentum + Decaying Momentum (DEMON) + Lookahead + Iterate Averaging + Nostalgia (from NosAdam) + P from PAdam
    # + gradient centralization + weight decay
    def __init__(self, params, lr=1e-3,
                 betas=(0.999, 0.999),
                 nu=0.7,
                 eps=1e-8,
                 gamma=0.0001,
                 nostalgia=True,
                 use_demon=True,
                 hypergrad_lr=0.02,
                 HDM=True,
                 hypertune_nu=True,
                 p=0.25,
                 k=5,
                 alpha=0.8,
                 IA=True,
                 IA_cycle=1000,
                 epochs=100,
                 step_per_epoch=None,
                 weight_decay=0,
                 use_gc=True,
                 use_diffgrad=False):

        # betas = (beta1 for first order moments, beta2 for second order moments)
        # nu = for quasi hyperbolic momentum
        # eps = small value for numerical stability (avoid divide by zero)
        # k = lookahead cycle
        # alpha = outer learning rate (lookahead)
        # gamma = used for nostalgia
        # nostalgia = bool to decide whether to use nostalgia (from Nostalgic Adam or NosAdam)
        # use_demon = bool to decide whether to use DEMON (Decaying Momentum) or not
        # hypergrad_lr = learning rate for updating hyperparameters (like lr) through hypergradient descent (probably need to increase around 0.02 if HDM is True). Set to 0.0 to disable hypergradient descent.
        # HDM = bool to decide whether to use Multiplicative rule for updating hyperparameters or not
        # hypertune_nu1 = bool to decide whether apply hypergradient descent on nu1 as well or not.
        # p = p from PAdam
        # IA = bool to decide if Iterate Averaging is ever going to be used
        # IA_cycle = Iterate Averaging Cycle (Recommended to initialize with no. of iterations in Epoch) (doesn't matter if you are not using IA)
        # epochs = No. of epochs you plan to use (Only relevant if using DEMON)
        # step_per_epoch = No. of iterations in an epoch (only relevant if using DEMON)
        # weight decay = decorrelated weight decay value
        # use_gc = bool to determine whether to use gradient centralization or not.
        # use_diffgrad = bool to determine whether to use diffgrad or not.

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= hypergrad_lr:
            raise ValueError(
                "Invalid hypergradient learning rate: {}".format(hypergrad_lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= nu <= 1.0:
            raise ValueError("Invalid nu parameter: {}".format(nu))
        if not 0.0 <= p <= 0.5:
            raise ValueError("Invalid p parameter: {}".format(p))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))

        self.nostalgia = nostalgia
        self.use_demon = use_demon
        self.k = k
        self.IA = IA
        self.IA_cycle = IA_cycle
        self.epochs = epochs
        if step_per_epoch is None:
            self.step_per_epoch = IA_cycle
        else:
            self.step_per_epoch = step_per_epoch
        self.use_gc = use_gc
        self.use_diffgrad = use_diffgrad
        self.T = self.epochs * self.step_per_epoch
        self.hypertune_nu = hypertune_nu
        self.HDM = HDM

        defaults = dict(lr=lr,
                        betas=betas,
                        nu=nu,
                        eps=eps,
                        alpha=alpha,
                        gamma=gamma,
                        p=p,
                        hypergrad_lr=hypergrad_lr,
                        weight_decay=weight_decay)
        super(HyperProp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(HyperProp, self).__setstate__(state)

    def step(self, activate_IA=False, display=False, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        'HyperProp does not support sparse gradients')

                state = self.state[p]

                hypergrad_lr = group['hypergrad_lr']
                beta1_init, beta2 = group['betas']
                wd = group['weight_decay']
                alpha = group['alpha']
                gamma = group['gamma']

                if len(state) == 0:
                    state['lr'] = group['lr']
                    state['nu'] = group['nu']
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if self.use_diffgrad:
                        state['previous_grad'] = torch.zeros_like(p.data)

                    if self.IA:
                        state['num_models'] = 0
                    if self.IA or (self.k > 0):
                        state['cached_params'] = p.data.clone()
                    if self.nostalgia:
                        state['B_old'] = 0
                        state['B_new'] = 1
                    if hypergrad_lr > 0.0:
                        state['prev_lr_grad'] = torch.zeros_like(grad.view(-1))
                        if self.hypertune_nu:
                            state['prev_nu_grad'] = torch.zeros_like(
                                grad.view(-1))

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if self.use_demon:
                    temp = 1 - (state['step'] / self.T)
                    beta1 = beta1_init * temp / \
                        ((1 - beta1_init) + beta1_init * temp)
                else:
                    beta1 = beta1_init

                if self.nostalgia:
                    beta2 = state['B_old'] / state['B_new']
                    state['B_old'] += math.pow(state['step'], -gamma)
                    state['B_new'] += math.pow(state['step'] + 1, -gamma)

                do_IA = False
                lookahead_step = False

                if self.IA and activate_IA:
                    lookahead_step = False
                    if state['step'] % self.IA_cycle == 0:
                        do_IA = True
                elif self.k == 0:
                    lookahead_step = False
                else:
                    if state['step'] % self.k == 0:
                        lookahead_step = True
                    else:
                        lookahead_step = False

                if state['step'] > 1 and hypergrad_lr > 0.0:
                    prev_lr_grad = state['prev_lr_grad']
                    h = torch.dot(grad.view(-1), prev_lr_grad)

                    if self.HDM:
                        grad_norm = grad.view(-1).norm()
                        norm_denom = grad_norm * (prev_lr_grad.norm())
                        norm_denom.add_(group['eps'])
                        state['lr'] = state['lr'] * \
                            (1 - hypergrad_lr * (h / norm_denom))
                    else:
                        state['lr'] -= hypergrad_lr * h

                    torch.max(state['lr'], torch.zeros_like(
                        state['lr']), out=state['lr'])

                    if display:
                        print("lr", state['lr'])

                    if self.hypertune_nu:
                        prev_nu_grad = state['prev_nu_grad']
                        h = torch.dot(grad.view(-1), prev_nu_grad)
                        if self.HDM:
                            norm_denom = grad_norm * (prev_nu_grad.norm())
                            norm_denom.add_(group['eps'])
                            state['nu'] = state['nu'] * \
                                (1 - hypergrad_lr * (h / norm_denom))
                        else:
                            state['nu'] -= hypergrad_lr * h

                        torch.max(state['nu'], torch.zeros_like(
                            state['nu']), out=state['nu'])
                        torch.min(state['nu'], torch.ones_like(
                            state['nu']), out=state['nu'])

                    if display:
                        print("nu", state['nu'])

                if self.use_gc and grad.dim() > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                nu = state['nu']
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                vt = exp_avg_sq.clone()

                if not self.nostalgia:
                    vt.div_(1 - (beta2 ** state['step']))

                denom = vt.pow_(group['p']).add_(group['eps'])

                exp_avg.mul_(beta1).addcdiv_(1 - beta1, grad, denom)

                if self.use_diffgrad:
                    previous_grad = state['previous_grad']
                    diff = abs(previous_grad - grad)
                    dfc = 1. / (1. + torch.exp(-diff))
                    state['previous_grad'] = grad.clone()
                    exp_avg = exp_avg * dfc

                momentum = exp_avg.clone()
                bias_correction1 = 1 - (beta1 ** state['step'])
                momentum.div_(bias_correction1)

                if lookahead_step:
                    dalpha = alpha
                elif do_IA:
                    dalpha = (1 / (state["num_models"] + 1.0))
                else:
                    dalpha = 1.0

                if hypergrad_lr > 0.0 and self.hypertune_nu:
                    state['prev_nu_grad'] = (-dalpha * state['lr']
                                             * (momentum - grad)).view(-1)

                # quasi hyperbolic momentum
                momentum.mul_(nu).add_(1 - nu, grad)

                if hypergrad_lr > 0.0:
                    temp = dalpha * (-momentum - wd * p.data)
                    state['prev_lr_grad'] = temp.view(-1)

                p.data.add_(-state['lr'] * momentum)

                if wd != 0:
                    p.data.mul_(1 - state['lr'] * wd)

                if lookahead_step:
                    p.data.mul_(alpha).add_(
                        1.0 - alpha, state['cached_params'])
                    state['cached_params'].copy_(p.data)

                if do_IA:
                    p.data.add_(state["num_models"], state['cached_params']
                                ).div_(state["num_models"] + 1.0)
                    state['cached_params'].copy_(p.data)
                    state["num_models"] += 1

        return loss

# new stuffs to try: https://arxiv.org/pdf/1607.04381.pdf

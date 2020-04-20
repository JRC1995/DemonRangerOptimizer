from torch.optim.optimizer import Optimizer, required
import torch
import math
import numpy as np


class DemonRanger(Optimizer):

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
                 use_grad_noise=False):
        
        #betas = (beta1 for first order moments, beta2 for second order moments, beta3 for ema over adaptive learning rates (AdaMod))
        #nus = (nu1,nu2) (for quasi hyperbolic momentum)
        #eps = small value for numerical stability (avoid divide by zero)
        #k = lookahead cycle
        #alpha = outer learning rate (lookahead)
        #gamma = gradient noise control parameter (for regularization)
        #use_demon = bool to decide whether to use DEMON (Decaying Momentum) or not
        #rectify = bool to decide whether to apply the recitification term (from RAdam) or not
        #amsgrad = bool to decide whether to use amsgrad instead of adam as the core optimizer
        #AdaMod_bias_correct = bool to decide whether to add bias correction to AdaMod
        #IA = bool to decide if Iterate Averaging is ever going to be used
        #IA_cycle = Iterate Averaging Cycle (Recommended to initialize with no. of iterations in Epoch) (doesn't matter if you are not using IA)
        #epochs = No. of epochs you plan to use (Only relevant if using DEMON)
        #step_per_epoch = No. of iterations in an epoch (only relevant if using DEMON)
        #weight decay = decorrelated weight decay value
        #use_gc = bool to determine whether to use gradient centralization or not.
        #use_grad_noise = bool to determine whether to use gradient noise or not. 

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[2]))
        if not 0.0 <= nus[0] <= 1.0:
            raise ValueError("Invalid nu parameter at index 0: {}".format(nus[0]))
        if not 0.0 <= nus[1] <= 1.0:
            raise ValueError("Invalid nu parameter at index 1: {}".format(nus[1]))

        self.use_gc = use_gc
        self.use_grad_noise = use_grad_noise
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

        self.T = self.epochs*self.step_per_epoch

        defaults = dict(lr=lr,
                        betas=betas,
                        nus=nus,
                        eps=eps,
                        alpha=alpha,
                        gamma=gamma,
                        weight_decay=weight_decay)
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
        
        # disables lookahead and starts doing IA if activate_IA is true

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('DemonRanger does not support sparse gradients')

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
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1_init, beta2, beta3 = group['betas']
                rho_inf = (2/(1-beta2)) - 1
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
                    temp = 1-(state['step']/self.T)
                    beta1 = beta1_init * temp / ((1-beta1_init)+beta1_init*temp)
                else:
                    beta1 = beta1_init

                if self.use_grad_noise:
                    grad_var = lr/((1+state['step'])**gamma)
                    grad_noise = torch.empty_like(grad).normal_(mean=0.0, std=math.sqrt(grad_var))
                    grad.add_(grad_noise)

                if self.use_gc:
                    grad.add_(-grad.mean(dim=tuple(range(1, len(list(grad.size())))), keepdim=True))

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                momentum = exp_avg.clone()
                momentum.div_(1 - (beta1 ** state['step'])).mul_(nu1).add_(1-nu1, grad)

                if wd != 0:
                    p.data.add_(-wd*lr, p.data)

                beta2_t = beta2 ** state['step']

                if self.amsgrad and state['step'] > 1:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    vt = max_exp_avg_sq.clone()
                else:
                    vt = exp_avg_sq.clone()

                if self.rectify:
                    rho_t = rho_inf - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    # more conservative since it's an approximated value
                    if rho_t >= 5:
                        R = math.sqrt(((rho_t-4)*(rho_t-2)*rho_inf) /
                                      ((rho_inf-4)*(rho_inf-2)*rho_t))
                        bias_correction2 = 1 - beta2_t
                        vt.div_(bias_correction2)
                        if nu2 != 1.0:
                            vt.mul_(nu2).addcmul_(1-nu2, grad, grad)
                        denom = vt.sqrt_().add_(group['eps'])

                        n = (lr*R)/denom

                        if self.AdaMod:
                            n_avg = state['n_avg']
                            n = self.apply_AdaMod(beta3, n_avg, n, step=state['step'])

                        p.data.add_(-n*momentum)
                    else:
                        if self.AdaMod:
                            n_avg = state['n_avg']
                            n_avg.mul_(beta3).add_(1 - beta3, lr)
                        p.data.add_(-lr, momentum)
                else:
                    bias_correction2 = 1 - beta2_t
                    vt.div_(bias_correction2)
                    if nu2 != 1.0:
                        vt.mul_(nu2).addcmul_(1-nu2, grad, grad)
                    denom = vt.sqrt_().add_(group['eps'])
                    n = lr/denom
                    if self.AdaMod:
                        n_avg = state['n_avg']
                        n = self.apply_AdaMod(beta3, n_avg, n, step=state['step'])

                    p.data.add_(-n*momentum)

                if lookahead_step:
                    p.data.mul_(alpha).add_(1.0 - alpha, state['cached_params'])
                    state['cached_params'].copy_(p.data)

                if do_IA:
                    p.data.add_(state["num_models"], state['cached_params']
                                ).div_(state["num_models"]+1.0)
                    state['cached_params'].copy_(p.data)
                    state["num_models"] += 1

        return loss
    
    

class HyperRanger(Optimizer):

    def __init__(self, params, lr=1e-3,
                 betas=(0.999, 0.999),
                 nus=(0.7, 1.0),
                 eps=1e-7,
                 gamma=0.0001,
                 nostalgia=True,
                 use_demon=True,
                 hypergrad_lr=1e-7,
                 HDM=False,
                 hypertune_nu1=False,
                 p=0.5,
                 k=5,
                 alpha=0.8,
                 IA=True,
                 IA_cycle=1000,
                 epochs=100,
                 step_per_epoch=None,
                 weight_decay=0,
                 use_gc=True):
        
        #betas = (beta1 for first order moments, beta2 for second order moments)
        #nus = (nu1,nu2) (for quasi hyperbolic momentum)
        #eps = small value for numerical stability (avoid divide by zero)
        #k = lookahead cycle
        #alpha = outer learning rate (lookahead)
        #gamma = used for nostalgia
        #nostalgia = bool to decide whether to use nostalgia (from Nostalgic Adam or NosAdam)
        #use_demon = bool to decide whether to use DEMON (Decaying Momentum) or not
        #hypergrad_lr = learning rate for updating hyperparameters (like lr) through hypergradient descent (probably need to increase around 0.02 if HDM is True). Set to 0.0 to disable hypergradient descent.
        #HDM = bool to decide whether to use Multiplicative rule for updating hyperparameters or not
        #hypertune_nu1 = bool to decide whether apply hypergradient descent on nu1 as well or not.
        #p = p from PAdam
        #IA = bool to decide if Iterate Averaging is ever going to be used
        #IA_cycle = Iterate Averaging Cycle (Recommended to initialize with no. of iterations in Epoch) (doesn't matter if you are not using IA)
        #epochs = No. of epochs you plan to use (Only relevant if using DEMON)
        #step_per_epoch = No. of iterations in an epoch (only relevant if using DEMON)
        #weight decay = decorrelated weight decay value
        #use_gc = bool to determine whether to use gradient centralization or not.
        #use_grad_noise = bool to determine whether to use gradient noise or not. 

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= nus[0] <= 1.0:
            raise ValueError("Invalid nu parameter at index 0: {}".format(nus[0]))
        if not 0.0 <= nus[1] <= 1.0:
            raise ValueError("Invalid nu parameter at index 1: {}".format(nus[1]))

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
        self.T = self.epochs*self.step_per_epoch
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
                    raise RuntimeError('HyperRanger does not support sparse gradients')

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

                    if self.IA:
                        state['num_models'] = 0
                    if self.IA or (self.k > 0):
                        state['cached_params'] = p.data.clone()
                    if self.nostalgia:
                        state['B_old'] = 0
                        state['B_new'] = 1
                    if hypergrad_lr > 0.0:
                        state['prev_lr_grad'] = torch.zeros_like(grad.view(-1))
                        if self.hypertune_nu1:
                            state['prev_nu_grad'] = torch.zeros_like(grad.view(-1))

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                if self.use_demon:
                    temp = 1-(state['step']/self.T)
                    beta1 = beta1_init * temp / ((1-beta1_init)+beta1_init*temp)
                else:
                    beta1 = beta1_init

                if self.nostalgia:
                    beta2 = state['B_old']/state['B_new']
                    state['B_old'] += math.pow(state['step'], -gamma)
                    state['B_new'] += math.pow(state['step']+1, -gamma)

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

                if self.use_gc:
                    grad.add_(-grad.mean(dim=tuple(range(1, len(list(grad.size())))), keepdim=True))

                if state['step'] > 1 and hypergrad_lr > 0.0:
                    prev_lr_grad = state['prev_lr_grad']
                    h = torch.dot(grad.view(-1), prev_lr_grad)

                    if self.HDM:
                        grad_norm = grad.view(-1).norm()
                        norm_denom = grad_norm*(prev_lr_grad.norm())
                        norm_denom.add_(group['eps'])
                        group['lr'] = group['lr']*(1-hypergrad_lr*(h/norm_denom))
                    else:
                        group['lr'] -= hypergrad_lr * h

                    if display:
                        print(group['lr'])

                    if self.hypertune_nu1:
                        prev_nu_grad = state['prev_nu_grad']
                        h = torch.dot(grad.view(-1), prev_nu_grad)
                        if self.HDM:
                            norm_denom = grad_norm*(prev_nu_grad.norm())
                            norm_denom.add_(group['eps'])
                            group['nu1'] = group['nu1']*(1-hypergrad_lr*(h/norm_denom))
                        else:
                            group['nu1'] -= hypergrad_lr * h

                nu1 = group['nu1']
                nu2 = group['nu2']
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                momentum = exp_avg.clone()
                bias_correction1 = 1 - (beta1 ** state['step'])
                momentum.div_(bias_correction1)

                vt = exp_avg_sq.clone()

                if not self.nostalgia:
                    vt.div_(1 - (beta2 ** state['step']))
                if nu2 != 1.0:
                    vt.mul_(nu2).addcmul_(1-nu2, grad, grad)

                denom = vt.pow_(group['p']).add_(group['eps'])

                n = group['lr']/denom

                if hypergrad_lr > 0.0 and self.hypertune_nu1:
                    state['prev_nu_grad'] = (-n * (momentum - grad)).view(-1)

                momentum.mul_(nu1).add_(1-nu1, grad)  # quasi hyperbolic momentum

                if hypergrad_lr > 0.0:
                    state['prev_lr_grad'] = -(momentum/denom).view(-1)

                p.data.add_(-n*momentum)

                if wd != 0:
                    p.data.add_(-wd*group['lr'], p.data)

                if lookahead_step:
                    p.data.mul_(alpha).add_(1.0 - alpha, state['cached_params'])
                    state['cached_params'].copy_(p.data)

                if do_IA:
                    p.data.add_(state["num_models"], state['cached_params']
                                ).div_(state["num_models"]+1.0)
                    state['cached_params'].copy_(p.data)
                    state["num_models"] += 1

        return loss


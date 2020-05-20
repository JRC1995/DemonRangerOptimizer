# DemonRangerOptimizer

Quasi Hyperbolic Rectified DEMON (Decaying Momentum) Adam/Amsgrad with AdaMod,  Lookahead, iterate averaging, and decorrelated weight decay.

Also, other variants with Nostalgia (NosAdam), P (from PAdam), LaProp, and Hypergradient Descent (see HyperRanger and HyperRangerMod and others in optimizers.py) 

## Notes:

* Hyperxxx series optimizers implements hypergradient descent for dynamic learning rate updates. Some optimizers like  HDQHSGDW implements hypergradient descent for all hyperparameters - beta, nu, lr. Unlike the original implementation (https://arxiv.org/abs/1703.04782, https://github.com/gbaydin/hypergradient-descent) they take care of the gradients due to the weight decay and other things. (I also implement state level lr so that lr for each parameters will be hypertuned through hypergradient descent separately instead of in the group level like in the original implementation)

* LRangerMod uses Linear Warmup within Adam/AMSGrad based on the rule of thumb as in (https://arxiv.org/abs/1910.04209v1). Note Rectified Adam boils down to a fixed (not dynamic) form of learning rate scheduling similar to a linear warmup. 

* The file explains the parameters for each different synergistic optimizers. 

## How to use:


```
from optimizers import DemonRanger
from dataloader import batcher # some random function to batch data

class config:
   def __init__(self):
       self.batch_size = ...
       self.wd = ...
       self.lr = ...
       self.epochs = ...
       
       
config = config()
   

train_data = ...
step_per_epoch = count_step_per_epoch(train_data,config.batch_size)

model = module(stuff)

optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        weight_decay=config.wd,
                        epochs=config.epochs,
                        step_per_epoch=step_per_epoch,
                        IA_cycle=step_per_epoch)
IA_activate = False                      
for epoch in range(config.epochs):
    batches = batcher(train_data, config.batch_size)
    
    for batch in batches:
        loss = do stuff
        loss.backward()
        optimizer.step(IA_activate=IA_activate)
    
    # automatically enable IA (Iterate Averaging) near the end of training (when metric of your choice not improving for a while)
    if (IA_patience running low) and IA_activate is False:
        IA_activate = True 
        

```

## Recover AdamW:

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        betas=(0.9,0.999,0.999), # restore default AdamW betas
                        nus=(1.0,1.0), # disables QHMomentum
                        k=0,  # disables lookahead
                        alpha=1.0, 
                        weight_decay=config.wd,
                        IA=False, # disables Iterate Averaging
                        rectify=False, # disables RAdam Recitification
                        AdaMod=False, #disables AdaMod
                        use_demon=False, #disables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
                        

# just do optimizer.step() when necessary

```

## Recover AMSGrad:

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        betas=(0.9,0.999,0.999), # restore default AdamW betas
                        nus=(1.0,1.0), # disables QHMomentum
                        k=0,  # disables lookahead
                        alpha=1.0, 
                        weight_decay=config.wd,
                        IA=False, # disables Iterate Averaging
                        rectify=False, # disables RAdam Recitification
                        AdaMod=False #disables AdaMod
                        use_demon=False, #disables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=True # disables amsgrad
                        )
                        
# just do optimizer.step() when necessary

```

## Recover QHAdam

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        k=0,  # disables lookahead
                        alpha=1.0, 
                        weight_decay=config.wd,
                        IA=False, # disables Iterate Averaging
                        rectify=False, # disables RAdam Recitification
                        AdaMod=False, #disables AdaMod
                        use_demon=False, #disables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
                        
# just do optimizer.step() when necessary

```

## Recover RAdam

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        betas=(0.9,0.999,0.999), # restore default AdamW betas
                        nus=(1.0,1.0), # disables QHMomentum
                        k=0,  # disables lookahead
                        alpha=1.0, 
                        weight_decay=config.wd,
                        IA=False, # disables Iterate Averaging
                        AdaMod=False, #disables AdaMod
                        use_demon=False, #disables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
# just do optimizer.step() when necessary
```

## Recover Ranger (RAdam + LookAhead)

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        betas=(0.9,0.999,0.999), # restore default AdamW betas
                        nus=(1.0,1.0), # disables QHMomentum
                        weight_decay=config.wd,
                        IA=False, # disables Iterate Averaging
                        AdaMod=False, #disables AdaMod
                        use_demon=False, #disables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
# just do optimizer.step() when necessary
```


## Recover QHRanger (QHRAdam + LookAhead)

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        weight_decay=config.wd,
                        IA=False, # disables Iterate Averaging
                        AdaMod=False, #disables AdaMod
                        use_demon=False, #disables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
# just do optimizer.step() when necessary
```

## Recover AdaMod

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        weight_decay=config.wd,
                        betas=(0.9,0.999,0.999), # restore default AdamW betas
                        nus=(1.0,1.0), # disables QHMomentum
                        k=0,  # disables lookahead
                        alpha=1.0, 
                        IA=False, # disables Iterate Averaging
                        rectify=False, # disables RAdam Recitification
                        AdaMod_bias_correct=False, #disables AdaMod bias corretion (not used originally)
                        use_demon=False #disables Decaying Momentum (DEMON)
                        use_gc=False #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
# just do optimizer.step() when necessary
```

## Recover GAdam

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        weight_decay=config.wd,
                        betas=(0.9,0.999,0.999), # restore default AdamW betas
                        nus=(1.0,1.0), # disables QHMomentum
                        k=0,  # disables lookahead
                        alpha=1.0, 
                        IA=True, # enables Iterate Averaging
                        rectify=False, # disables RAdam Recitification
                        AdaMod=False, #disables AdaMod
                        use_demon=False, #disables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
# just do optimizer.step(IA_activate=IA_activate) when necessary (change IA_activate to True near the end of training based on some scheduling scheme or tuned hyperparameter--- alternative to learning rate scheduling)
```

## Recover GAdam + LookAhead

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        weight_decay=config.wd,
                        betas=(0.9,0.999,0.999), # restore default AdamW betas
                        nus=(1.0,1.0), # disables QHMomentum
                        k=5,  # enables lookahead
                        alpha=0.88, 
                        IA=True, # enables Iterate Averaging
                        rectify=False, # disables RAdam Recitification
                        AdaMod=False, #disables AdaMod
                        use_demon=False, #disables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
# just do optimizer.step(IA_activate=IA_activate) when necessary (change IA_activate to True near the end of training based on some scheduling scheme or tuned hyperparameter--- alternative to learning rate scheduling)
```

## Recover DEMON Adam

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        weight_decay=config.wd,
                        epochs = config.epochs,
                        step_per_epoch = step_per_epoch, 
                        betas=(0.9,0.999,0.999), # restore default AdamW betas
                        nus=(1.0,1.0), # disables QHMomentum
                        k=0,  # disables lookahead
                        alpha=1.0, 
                        IA=False, # enables Iterate Averaging
                        rectify=False, # disables RAdam Recitification
                        AdaMod=False, #disables AdaMod
                        AdaMod_bias_correct=False, #disables AdaMod bias corretion (not used originally)
                        use_demon=True, #enables Decaying Momentum (DEMON)
                        use_gc=False, #disables gradient centralization
                        amsgrad=False # disables amsgrad
                        )
# just do optimizer.step() when necessary
```

## Use Variance Rectified DEMON QHAMSGradW with AdaMod, LookAhead, Iterate Averaging, and Gradient Centralization

```
optimizer = DemonRanger(params=model.parameters(),
                        lr=config.lr,
                        weight_decay=config.wd,
                        epochs=config.epochs,
                        step_per_epoch=step_per_epoch,
                        IA_cycle=step_per_epoch)
 # just do optimizer.step(IA_activate=IA_activate) when necessary (change IA_activate to True near the end of training based on some scheduling scheme or tuned hyperparameter--- alternative to learning rate scheduling)
 ```
 
## Stuffs to try or add:

* Dense-sparse-Dense Training: https://arxiv.org/pdf/1607.04381.pdf
* Bayesian Deep Learning: SWAG/SWA 

 ## References:
 
 * Adam: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
 * AMSGrad: https://arxiv.org/abs/1904.09237
 * QHAdam: https://arxiv.org/abs/1810.06801
 * Gradient Noise: https://arxiv.org/abs/1511.06807
 * AdamW: https://arxiv.org/abs/1711.05101
 * RAdam: https://arxiv.org/abs/1908.03265, https://github.com/LiyuanLucasLiu/RAdam
 * More on RAdam: https://arxiv.org/abs/1910.04209v1
 * Lookahead: https://arxiv.org/abs/1907.08610
 * Ranger: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
 * Gradient Centralization: https://arxiv.org/abs/2004.01461v2
 * DEMON (Decaying Momentum): https://arxiv.org/abs/1910.04952
 * AdaMod: https://arxiv.org/abs/1910.12249
 * GAdam (Iterate Averaging): https://arxiv.org/abs/2003.01247, https://github.com/diegogranziol/Gadam
 * Hypergradient Descent: https://arxiv.org/abs/1703.04782, https://github.com/gbaydin/hypergradient-descent
 * Nostalgic Adam: https://arxiv.org/abs/1805.07557, https://github.com/andrehuang/NostalgicAdam-NosAdam
 * PAdam: https://arxiv.org/abs/1806.06763, https://github.com/uclaml/Padam, https://arxiv.org/pdf/1901.09517.pdf
 * LaProp: https://arxiv.org/abs/2002.04839

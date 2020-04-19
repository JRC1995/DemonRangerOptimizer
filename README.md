# DemonRangerOptimizer
Quasi Hyperbolic Rectified DEMON (Decaying Momentum) Adam/Amsgrad with AdaMod, Gradient Centralization, Lookahead, iterative averaging, and decorrelated weight decay


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
    
    # automatically enable IA near the end of training (when metric of your choice not improving for a while)
    if (IA_patience running low) and IA_activate is False:
        IA_activate = True 
        

```

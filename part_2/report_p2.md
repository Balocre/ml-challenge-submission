# Hyperpramaters Optimization

For this part I chose to use the Rat Tune library and a small CNN ressembling ResNet.


## Tuning process

For the optimization process we define a config dictionary containing the 
hyperparameters we want to tune. 

We define a training loop like we would for a normal training, in which the model,
the optimizer and the other training objects are initialized by an instanciation of the
hyperparamter config.

In this loop we do a training pass and a validation pass for each epoch.

Once the validation is done the loss and accuracy are sent to the hyperparamter 
optimization framework.

Since we are evaluating our model over the CIFAR-100 dataset, we mesure the 
classification accuracy.

The optimizer is set to monitor the loss, and there are early stopping mechanisms 
triggered if the loss does not diminish enough.

We keep the checkpoints of the best performing models in terms of accuracy n validation.


## About the model

This is a small CNN based model with skip connections.

The architecture is inspired by ResNet.

You can customize the number of hidden feature between each blocks.


## How to use

Simply run the `hparam_search.py` python file, you can modify the values from the 
`param_space` dict to try other hyperparamters.

You can run a tensorboard of the output directory to monitor some metrics such as
loss accuracy, time elapsed etc...


## Results

If the tuning is succesful you should get a recap like such :

```
Trial status: 12 TERMINATED
Current time: 2024-11-15 02:09:00. Total running time: 42min 46s
Logical resource usage: 6.0/12 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:G)
Current best trial: 3845f_00003 with loss=2.063175916671753 and params={'epochs': 20, 'h_1': 128, 'h_2': 1024, 'p_drop': 0.10072527948682011, 'lr': 0.0054913895023098545, 'batch_size': 64}
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                status         p_drop            lr     h_1     h_2     iter     total time (s)      loss     accuracy │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_cifar_3845f_00000   TERMINATED   0.272951   0.00385297       64     512       21           394.622    2.19799       0.5397 │
│ train_cifar_3845f_00001   TERMINATED   0.331844   0.000141037     128     512        3            81.1105   3.73514       0.1486 │
│ train_cifar_3845f_00002   TERMINATED   0.153036   2.45343e-05      64    1024        3            99.0799   3.94285       0.1118 │
│ train_cifar_3845f_00003   TERMINATED   0.100725   0.00549139      128    1024       12           490.365    2.06318       0.5435 │
│ train_cifar_3845f_00004   TERMINATED   0.105379   0.0207525        64     512        3            59.2342   3.33976       0.1978 │
│ train_cifar_3845f_00005   TERMINATED   0.122865   0.0120914       128     512       21           573.507    2.16692       0.5885 │
│ train_cifar_3845f_00006   TERMINATED   0.444873   0.0029448        64    1024        3           102.883    3.6483        0.1642 │
│ train_cifar_3845f_00007   TERMINATED   0.223607   0.00335549      128    1024        6           250.92     2.26556       0.4166 │
│ train_cifar_3845f_00008   TERMINATED   0.208023   0.00229886       64     512        6           120.154    2.26649       0.4156 │
│ train_cifar_3845f_00009   TERMINATED   0.226076   0.000125113     128     512        3            82.8989   3.53941       0.1764 │
│ train_cifar_3845f_00010   TERMINATED   0.193704   4.70692e-05      64    1024        3           100.98     3.75676       0.1436 │
│ train_cifar_3845f_00011   TERMINATED   0.131377   0.0257601       128    1024        3           123.554    3.9407        0.0903 │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

(train_cifar pid=229962) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/home/balocre/Documents/ki/ML-Engineer-Challenge/part_2/outputs/ray-experiement/train_cifar_3845f_00011_11_h_1=128,h_2=1024,lr=0.0258,p_drop=0.1314_2024-11-15_01-26-13/checkpoint_000002)
Best trial config: {'epochs': 20, 'h_1': 128, 'h_2': 512, 'p_drop': 0.1228650546400748, 'lr': 0.012091409647477234, 'batch_size': 64}
Best trial final validation loss: 2.1669225692749023
Best trial final validation accuracy: 0.5885
```

This indicates the trials that have been run, and gives you an overview of the 
hyperparameters values that have been tried.

The best performing instance I've got achieved an accuracy of 58%.

The optimizer seems to stop the trials a bit to fast in my opinion, this could be 
improved by trying another scheduler.

Nonetheless the results are good for such a small model. Looking at the accuracy and 
and loss curves (plateau all over the place) I doubt we could squeeze much more 
performances from this network.


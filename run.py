# MSTOP Greedy Rollout, train with instance augmentation (v45)

import os
import json
import pprint as pp
import time
import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline

from nets.attention_model import AttentionModel
# from nets.attention_model_original import AttentionModel

from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem


def run(opts, spec=None, prob=None, distribution=None):
    opts.baseline = 'rollout' 
    opts.problem = prob # 'top', 'tsp', 'cvrp'
    opts.data_distribution = 'const'
    opts.n_encode_layers = 3 # 4 for DDTM, 3 for AM
    opts.n_decode_layers = 2 # irrevelant for AM
    opts.num_veh = 2  # irrevelant for AM
    opts.graph_size = 20
    
    opts.n_epochs = 200
    
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    
    if spec=='A':
        opts.max_ent = False
        opts.with_replace = False
        opts.aug_factor = 1
        opts.epoch_size = 1280000
        opts.batch_size = 512
        opts.log_step = 250
    elif spec=='B':
        opts.max_ent = True
        opts.with_replace = False
        opts.aug_factor = 1
        opts.epoch_size = 1280000
        opts.batch_size = 512
        opts.log_step = 250
    elif spec=='C':
        opts.max_ent = False
        opts.with_replace = True
        opts.aug_factor = 8
        opts.epoch_size = 160000
        opts.batch_size = 64
        opts.log_step = 250
    elif spec=='D':
        opts.max_ent = False
        opts.with_replace = False
        opts.aug_factor = 8
        opts.epoch_size = 160000
        opts.batch_size = 64
        opts.log_step = 250
    elif spec =='E':
        opts.max_ent = True
        opts.with_replace = False
        opts.aug_factor = 8
        opts.epoch_size = 160000
        opts.batch_size = 64
        opts.log_step = 250
    
    # Use the same validation set as Kool
    if opts.problem == 'tsp':
        opts.val_dataset = 'data/tsp/tsp20_validation_seed4321.pkl'
    elif opts.problem == 'cvrp':
        opts.val_dataset = 'data/vrp/vrp20_validation_seed4321.pkl'
    
    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)
    print(problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        n_decode_layers=opts.n_decode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if not opts.max_ent:
        if opts.use_cuda and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts) # reinforce_baselines.py ->"Evaluating baseline model on evaluation dataset"
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()


    # if opts.bl_warmup_epochs > 0:
    #     baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # Start the actual training loop
    if opts.problem == 'top':
        val_dataset = problem.make_dataset(
            size=opts.graph_size, num_veh=opts.num_veh, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
    else:
        val_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)


    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        start_train = time.time()
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )
        end_train = time.time()
        tot_train_time = end_train - start_train
        print(f">>> Training complete in {time.strftime('%H:%M:%S', time.gmtime(tot_train_time))} s")
        print(f">>> Training complete in {time.gmtime(tot_train_time)} s")


if __name__ == "__main__":
    
    for prob in ['top']:
        for distribution in ['const', 'unif']: # uncomment for TSP/CVRP
            for spec in ['A','B','C','D','E']:
                print(f'problem : {prob}, spec = {spec}')
                run(get_options(), spec, prob, distribution=distribution) # set distribution as None for TSP/CVRP
                print("--------------------------------------------------")
    
import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from utils.functions import augment_xy_data_by_8_fold

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    if opts.problem == 'top':
        training_dataset = problem.make_dataset(
            size=opts.graph_size, num_veh=opts.num_veh, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    else:
        training_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)
    
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        # Put model in train mode!
        model.train()
        set_decode_type(model, "sampling")           

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward (per step)', avg_reward, step)
        tb_logger.log_value('val_avg_reward (per epoch)', avg_reward, epoch)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    
    #x = batch['data'] 
    #x, bl_val = baseline.unwrap_batch(batch)
    batch = move_to(batch, opts.device)
        
    # Multiple Samples 
    if opts.aug_factor > 1: # batch['depot'][0::batch_size]
        aug_factor = opts.aug_factor
        
        if opts.problem == 'tsp':
            if not opts.with_replace:
                batch = augment_xy_data_by_8_fold(batch)
            else:
                batch = batch.repeat(aug_factor,1,1)
        elif opts.problem == 'cvrp':
            if not opts.with_replace:
                batch['depot'] = augment_xy_data_by_8_fold(batch['depot'][:,None]).squeeze(1)
                batch['loc'] = augment_xy_data_by_8_fold(batch['loc'])
                batch['demand'] = batch['demand'].repeat(aug_factor, 1)
            else:
                batch['depot'] = batch['depot'].repeat(aug_factor, 1)
                batch['loc'] = batch['loc'].repeat(aug_factor, 1, 1)
                batch['demand'] = batch['demand'].repeat(aug_factor, 1)
        else: # MSTOP
            # D. Train with x8 instance augmentation (possibly non-duplicate samples)
            if not opts.with_replace:
                batch['depot'] = augment_xy_data_by_8_fold(batch['depot'][:,None]).squeeze(1)
                batch['loc'] = augment_xy_data_by_8_fold(batch['loc'])
                batch['cur_loc'] = augment_xy_data_by_8_fold(batch['cur_loc'])
            
            # C.Train with multiple samples with replacement
            else:
                batch['depot'] = batch['depot'].repeat(aug_factor,1)
                batch['loc'] = batch['loc'].repeat(aug_factor,1,1)
                batch['cur_loc'] = batch['cur_loc'].repeat(aug_factor,1,1)
            
            # Common
            batch['prize'] = batch['prize'].repeat(aug_factor, 1)
            batch['cur_tlen'] = batch['cur_tlen'].repeat(aug_factor, 1)
            batch['num_veh'] = batch['num_veh'].repeat(aug_factor)
            batch['max_length'] = batch['max_length'].repeat(aug_factor)   
        
        # Evaluate model, get costs and log probabilities  
        bl_loss = 0
        if opts.max_ent:
            # E. Train with x8 instance augmentation + Entropy Regularisation
            cost, log_likelihood, pi, entropy = model(batch, return_pi=True) 
        else:
            # D. Train with x8 instance augmentation
            cost, log_likelihood = model(batch) # (batch_size*aug_factor)
            entropy = 0
            
        # Evaluate local mean baseline
        batch_size = len(cost)//opts.aug_factor
        cost_mean = torch.stack(list(cost.split(batch_size)),dim=0).mean(dim=0).repeat(opts.aug_factor)
        
        # Calculate loss
        reinforce_loss = ((cost - cost_mean)*log_likelihood - 0.1*entropy).mean()
        
    else: # Single Sample 
        # Evaluate current model
        if opts.max_ent:
            # B. Greedy Rollout Basline + Max Entropy regularization
            cost, log_likelihood, pi, entropy = model(batch, return_pi=True) 
        else:
            # A.  Greedy Rollout Basline only
            cost, log_likelihood = model(batch)
            entropy = 0
            
        # Evaluate greedy rollout baseline
        bl_val, bl_loss = baseline.eval(batch, cost) 
        
        # Calculate loss
        reinforce_loss = ((cost - bl_val)*log_likelihood - 0.1*entropy).mean()# alpha: 0.01, 0.04
        
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)

import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
from torch.utils.data import DataLoader
import time
from datetime import timedelta
from utils.functions import parse_softmax_temperature
from utils.functions import augment_xy_data_by_8_fold, do_batch_rep, sample_many, permute_vehicle_order
from problems.top.tsiligirides import top_tsiligirides
from problems.top.problem_top import TOP
mp = torch.multiprocessing.get_context('spawn')
from itertools import permutations


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]

def eval_dataset(dataset_path, width, softmax_temp, opts, aug_factor=None):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    assert not opts.multiprocessing # only one cuda device available
    
    torch.manual_seed(opts.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
        
    # Generate dataset on the fly
    dataset = model.problem.make_dataset(filename=None, size=opts.graph_size, num_veh=opts.num_veh, num_samples=opts.val_size, offset=opts.offset, distribution=opts.data_distribution)
    results = _eval_dataset(model, dataset, width, softmax_temp, opts, device, aug_factor=aug_factor)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, tours, durations = zip(*results)  # Not really costs since they should be negative

    print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
    print("Average serial duration: {} +- {}".format(
        np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
    print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
    print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

    #dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, model_name)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}{}-t{}-{}-{}".format(
            model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs)
        ))
    else:
        out_file = opts.o

    assert opts.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."

    save_dataset((results, parallelism), out_file)

    return costs, tours, durations


def _eval_dataset(model, dataset, width, softmax_temp, opts, device, aug_factor=None):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('greedy', 'inst_aug') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else: # Sample
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            elif opts.decode_strategy == 'inst_aug':
                assert aug_factor is not None
                batch_rep = 1
                iter_rep = 1
                num_veh = batch['num_veh'][0] # 2, 3
                permute = list(permutations(range(0,num_veh))) # len(permute) is num_veh!
                num_veh_f = len(permute)

                if aug_factor == num_veh_f:
                    batch['depot'] = batch['depot'].repeat(num_veh_f, 1)
                    batch['loc'] = batch['loc'].repeat(num_veh_f, 1, 1)
                    batch['cur_loc'] = batch['cur_loc'].repeat(num_veh_f, 1, 1)
                    batch['cur_tlen'] = batch['cur_tlen'].repeat(num_veh_f, 1)
                    batch['prize'] = batch['prize'].repeat(num_veh_f, 1)
                    batch['num_veh'] = batch['num_veh'].repeat(num_veh_f)
                    batch['max_length'] = batch['max_length'].repeat(num_veh_f)
                    # Swap vehicle order
                    batch['cur_loc'] = permute_vehicle_order(batch['cur_loc'], permute=permute)
                    batch['cur_tlen'] = permute_vehicle_order(batch['cur_tlen'], permute=permute)
                    
                elif aug_factor == 8:
                    batch['depot'] = augment_xy_data_by_8_fold(batch['depot'][:,None]).squeeze(1)
                    batch['loc'] = augment_xy_data_by_8_fold(batch['loc'])
                    batch['cur_loc'] = augment_xy_data_by_8_fold(batch['cur_loc'])
                    batch['cur_tlen'] = batch['cur_tlen'].repeat(aug_factor, 1)
                    batch['prize'] = batch['prize'].repeat(aug_factor, 1)
                    batch['num_veh'] = batch['num_veh'].repeat(aug_factor)
                    batch['max_length'] = batch['max_length'].repeat(aug_factor)
   
                else: #aug_factor == 8*num_veh_f:        
                    batch['depot'] = augment_xy_data_by_8_fold(batch['depot'][:,None]).squeeze(1).repeat(num_veh_f, 1)
                    batch['loc'] = augment_xy_data_by_8_fold(batch['loc']).repeat(num_veh_f, 1, 1)
                    batch['cur_loc'] = augment_xy_data_by_8_fold(batch['cur_loc']).repeat(num_veh_f, 1, 1)
                    batch['cur_tlen'] = batch['cur_tlen'].repeat(aug_factor, 1) # aug_factor is already 8*2! or 8*3!
                    batch['prize'] = batch['prize'].repeat(aug_factor, 1)
                    batch['num_veh'] = batch['num_veh'].repeat(aug_factor)
                    batch['max_length'] = batch['max_length'].repeat(aug_factor)
                    # Swap vehicle order
                    batch['cur_loc'] = permute_vehicle_order(batch['cur_loc'], permute=permute)
                    batch['cur_tlen'] = permute_vehicle_order(batch['cur_tlen'], permute=permute)

                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                
                # Check if have duplicate samples
                batch_size = torch.div(len(costs), aug_factor, rounding_mode='trunc')
                # 20//2, 80//8, 160//16 
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
                
                # Get best results from augmentation
                aug_reward = costs.cpu().reshape(aug_factor, batch_size)
                costs, max_reward_idx = aug_reward.min(dim=0) # note negative rewards
                aug_tours = sequences.cpu().reshape(aug_factor, batch_size, -1)
                sequences = aug_tours[max_reward_idx[:,None], torch.arange(batch_size)[:,None], :].squeeze(1) 
                #wo_aug_reward = aug_reward[0]       
      
            else: # tsilis
                if width == 0: # greedy
                    batch_rep = 1
                    iter_rep = 1
                    sample=False
                else: # sample
                    batch_rep = width
                    iter_rep = 1
                    sample=True
                    
                sequences, costs = sample_many(
                    lambda input: top_tsiligirides(input, sample),
                    lambda input, pi, p_idx: TOP.get_costs(input, pi, p_idx),
                    batch, batch_rep=batch_rep, iter_rep=iter_rep)
                
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
        duration = time.time() - start
        for seq, cost in zip(sequences, costs):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp", "sdvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            elif model.problem.NAME in ("top", "op", "pctsp"): # TODO: Change top to mstop
                seq = np.trim_zeros(seq)  # We have the convention to exclude the depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append((cost, seq, duration))
    
        #delete batch from cuda
        # print(torch.cuda.memory_allocated(device) / 1024 /1024)
        # del batch
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated(device) / 1024 /1024)
        
    # Check CUDA memory
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))    
    
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    
    parser.add_argument('--problem', default='mstop', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--num_veh', type=int, default=2, help="Number of vehicles")
    parser.add_argument('--data_distribution', type=str, default='const',
                        help='Data distribution to use during training, defaults and options depend on problem.')
    
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed for test")
    
    parser.add_argument('--aug_factor', type=int, default=[16,8,2], help="Instance augmentation [num_veh, 8, 16]")
    parser.add_argument('--width', type=int, nargs='+',
                        help='Number of samples for sampling, '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str,
                        help='Greedy (greedy), Sampling (sample), Instance Augmentation (inst_aug)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    opts = parser.parse_args()
    
    # Change opts
    opts.multiprocessing = False # one GPU available
    opts.model = 'pretrained/UNIF/10_new/E/'
    opts.graph_size = 20 # 10,20,50,70
    opts.num_veh = 2 # 2,2,3,3
    opts.data_distribution = 'unif' # 'const', 'unif'
    decode_strategy = ['inst_aug'] #['greedy','sample','inst_aug'] #, 'tsili']
    dataset_path = None
    opts.val_size = 10000
    opts.f = True # overwrite

    # assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
    #     "Cannot specify result filename with more than one dataset or more than one width"

    #widths = opts.width if opts.width is not None else [0]
    for decode in decode_strategy:
           opts.decode_strategy = decode
           if decode == 'sample':
               opts.eval_batch_size = 1
               opts.width = [128, 1280]
               for width in opts.width:
                   print(f">>>Begin sample{width}")
                   eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
           elif decode == 'greedy' :
               opts.eval_batch_size = 1000 
               opts.width = 0
               print(f">>>Begin {decode}")
               eval_dataset(dataset_path, opts.width, opts.softmax_temperature, opts)
           elif decode == 'inst_aug':
               if opts.graph_size > 20:
                   opts.eval_batch_size = 50 # for n=50, n=70
               else:
                   opts.eval_batch_size = 1000 # for n=20 
               opts.width = 0
               for aug_factor in opts.aug_factor:
                   if aug_factor > 8: 
                       aug_factor = 8 * np.math.factorial(opts.num_veh) # 8*2! or 8*3!
                   elif aug_factor < 8: 
                       aug_factor = np.math.factorial(opts.num_veh) # num_veh!
                   print(f">>>Begin {decode}{aug_factor}")
                   eval_dataset(dataset_path, opts.width, opts.softmax_temperature, opts, aug_factor=aug_factor)
           else: # tsiligirides
                opts.eval_batch_size = 10000
                opts.width = 0
                print(f">>>Begin {decode}(greedy)")
                eval_dataset(dataset_path, opts.width, opts.softmax_temperature, opts)
                opts.eval_batch_size = 1
                opts.width = [128,1280]
                for width in opts.width:
                    print(f">>>Begin {decode}(sample{width})")
                    eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
            



    # 사용하지 않는 텐서가 있다면 del tensor로 지운 다음 torch.cuda.empty_cache() 를 실행하시면 됩니다.
    # torch.cuda.memory_allocated()와 torch.cuda.memory_reserved() 를 이용하시면 사용하고 있는 메모리와 cache 메모리를 각각 볼 수 있습니다.
    # torch.cuda.empty_cache()는 torch.cuda.memory_reserved() 에서 보이는 만큼을 free하게 해줍니다.

    #import pickle
    #with open('results\\top\\pretrained_mstop_const_20_v4.4_ep200\\pretrained_mstop_const_20_v4.4_ep200-sample1280-t1-0-10.pkl', 'rb') as f:
        #data = pickle.load(f)
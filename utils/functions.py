import warnings

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F
import math

def load_problem(name):
    from problems import TSP, CVRP, SDVRP, OP, TOP, PCTSPDet, PCTSPStoch
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
        'sdvrp': SDVRP,
        'op': OP,
        'top':TOP,
        'pctsp_det': PCTSPDet,
        'pctsp_stoch': PCTSPStoch,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None):
    from nets.attention_model import AttentionModel
    from nets.pointer_network import PointerNetwork

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        n_decode_layers=args['n_decode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None)
    )
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})}) 
    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v) # v[0] : dict, v[1] : embeddings (1, 1+problem+num_veh, emb_dim)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)
    # input[0]['loc'] : (1280, 20, 2), input[0]['cur_tlen'] : (1280, 2), input[1] : (1280, 23, 128)
    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi, pad_idx = inner_func(input) # _log_p : (1280, 19, 21), pi : (1280, 19), pad_idx : [9, 10]
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi, pad_idx) # mask is None

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2) 
    # xy_data[0] : tensor([[0.0967, 0.7253]])
    x = xy_data[:, :, [0]] # tensor([[0.0967]])
    y = xy_data[:, :, [1]] # tensor([[0.7253]])
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2) # dat1.shape:(batch, N, 2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data


def permute_vehicle_order(batch_tensor, permute=None):
    assert permute is not None
    num_veh = len(permute)
    chunk_batch = list(batch_tensor.chunk(num_veh, dim=0))
    for i in range(num_veh-1):
        perm = list(permute[i+1])
        chunk_batch[i+1] = chunk_batch[i+1][:, perm]
        
    return torch.cat(chunk_batch, dim=0)

def json_processing(path_to_json, x_step):
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
    valid_data = np.zeros((len(json_files), x_step)) # 200 epochs
    for idx, json_obj in enumerate(json_files):
        with open(os.path.join(path_to_json, json_obj)) as file:
            valid_cost = np.array(json.load(file)) # nested list to array, (200,3)
            valid_data[idx,:] = valid_cost[:,2] 
    
    return valid_data

def smooth(input_array, weight=0.6):
    last = input_array[0] # First value in the plot (at first time step)
    smoothed = list()
    for data in input_array:
        smoothed_data = last*weight + (1-weight)*data
        smoothed.append(smoothed_data)
        last = smoothed_data
    return np.array(smoothed)

# # Instance augmentation (aug_factor*num_veh)
# if opts.aug_factor > 1: 
#     aug_factor = opts.aug_factor
#     batch['depot'] = augment_xy_data_by_8_fold(batch['depot'][:,None]).squeeze(1).repeat(num_veh, 1)
#     batch['loc'] = augment_xy_data_by_8_fold(batch['loc']).repeat(num_veh, 1, 1)
#     batch['cur_loc'] = augment_xy_data_by_8_fold(batch['cur_loc']).repeat(num_veh, 1, 1)
#     batch['cur_tlen'] = batch['cur_tlen'].repeat(aug_factor*num_veh, 1)
#     batch['prize'] = batch['prize'].repeat(aug_factor*num_veh, 1)
#     batch['num_veh'] = batch['num_veh'].repeat(aug_factor*num_veh)
#     batch['max_length'] = batch['max_length'].repeat(aug_factor*num_veh)
    
#     # Check cur_loc and cur_tlen
#     chunk_cur_loc = list(batch['cur_loc'].chunk(num_veh, dim=0))
#     chunk_cur_tlen = list(batch['cur_tlen'].chunk(num_veh, dim=0))
    
#     for i in range(len(permute)-1): # swap cur_loc and cur_tlen
#         perm = list(permute[i+1])
#         chunk_cur_loc[i+1] = chunk_cur_loc[i+1][:, perm]
#         chunk_cur_tlen[i+1] = chunk_cur_tlen[i+1][:,perm]
        
#     batch['cur_loc'] = torch.cat(chunk_cur_loc, dim=0)
#     batch['cur_tlen'] = torch.cat(chunk_cur_tlen, dim=0)
#     # batch now has 8*num_veh instances                              

















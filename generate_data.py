import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_top_data(dataset_size, top_size, num_veh, prize_type='const'):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, top_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, top_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, top_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Max length is approximately half of optimal TSP tour, such that half (a bit more) of the nodes can be visited
    # which is maximally difficult as this has the largest number of possibilities
    MAX_LENGTHS = {
        10: 1.5,
        20: 2.,
        50: 3.,
        100: 4.
    }
    
    ###################### 0523: add num_veh TODO: this is harded coded. Change later#############################
    num_veh = 2*np.ones(dataset_size, dtype=np.uint8) # TO CHECK !!!!
    ###############################################################################################################

    # 0715: add vehicle current location and tour length so far 
    cur_loc = np.random.uniform(size=(dataset_size, num_veh[0], 2))
    # np.linalg.norm(cur_loc.reshape(-1,2) - np.repeat(depot, num_veh, axis=0), ord=2, axis=1)
    dist_to_depot = (cur_loc - depot).norm(p=2,dim=-1) # = distance required to be left
    cur_tlen = [torch.FloatTensor(1).uniform_(0, MAX_LENGTHS[graph_size] - dist_to_depot[i].item() - 1e-6) for i in range(num_veh)] 
    cur_tlen = torch.tensor([cur_tlen[i].item() for i in range(num_veh)])


    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, MAX_LENGTHS[top_size]).tolist(),  # T_max, same for whole dataset
        num_veh.tolist() # to check!!
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, help="Name to identify dataset") # required=True,
    parser.add_argument("--problem", type=str, default='mstop',
                        help="MSTOP Problem")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset (default 2, 3, 4)")   
    parser.add_argument('--num_veh', type=int, default=[2, 3, 4], help="Number of vehicles")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()
    
    # Specify options 
    opts.name = 'MSTOP20'
    opts.problem = 'mstop'
    opts.num_veh = [2]
    opts.graph_sizes = [20]

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'mstop': ['const', 'unif']
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        } # {'mstop': ['const', 'unif']}

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for i, graph_size in enumerate(opts.graph_sizes):
                
                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                num_veh = opts.num_veh[i] # 2,3,4
                dataset = generate_top_data(opts.dataset_size, graph_size, num_veh, prize_type=distribution)

                print(dataset[0])

                save_dataset(dataset, filename)

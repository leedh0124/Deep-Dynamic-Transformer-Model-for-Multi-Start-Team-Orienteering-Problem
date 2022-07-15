from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.top.state_top import StateTOP
from utils.beam_search import beam_search


class TOP(object):

    NAME = 'top'  # Team Orienteering problem 0713: dynamic version
    #num_veh = 2   # 0720: Specify num_veh for TOP

    @staticmethod
    def get_costs(dataset, pi, pad_idx):
        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == 0).all(), "If all length 1 tours, they should be zero"
            # Return
            return torch.zeros(pi.size(0), dtype=torch.float, device=pi.device), None
        
        
        batch_size, steps = pi.size()
        max_len = max(pad_idx)
        pi_split = torch.split(pi, pad_idx, dim=1) # this is tuple
        num_veh = len(pi_split)
        
        prize_with_depot = torch.cat(
            (
                torch.zeros_like(dataset['prize'][:, :1]),
                dataset['prize']
            ),
            1
        )
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        
        # loop over num_veh to check length
        total_p = []
        for n in range(num_veh):
            
            # 0913: Reshape pi before getting costs
            if pi_split[n].shape[-1] != max_len:
                pi_reshape =  torch.cat((pi_split[n], torch.zeros((batch_size, max_len-pi_split[n].shape[-1]), dtype=torch.uint8, device=pi.device)), dim=-1)
                # torch.Size([1024, 1]) -> torch.Size([1024, 11]) with zeros filled in
            else:
                pi_reshape = pi_split[n] # torch.Size([1024, 11])
            
            p = prize_with_depot.gather(1, pi_reshape) 

            # Gather dataset in order of tour
            d = loc_with_depot.gather(1, pi_reshape[..., None].expand(*pi_reshape.size(), loc_with_depot.size(-1))) # coordinates in order of tour pi
        
            # Python *, ** unpacking operators:  The single asterisk operator * can be used on any iterable that Python provides, while the double asterisk operator ** can only be used on dictionaries. 
            length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
                + (d[:, 0] - dataset['cur_loc'][:,n]).norm(p=2, dim=-1)  # cur_loc to first
                + (d[:, -1] - dataset['depot']).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last TODO: Is this always true? can last be nondepot?
                + dataset['cur_tlen'][:,n]
                )
            
            if (length > dataset['max_length'] + 1e-6).all():
                name = input('Here\n')
            
            assert (length <= dataset['max_length'] + 1e-6).all(), "Max length exceeded by {}".format((length - dataset['max_length']).max())
            
            total_p.append(p)
        
        # We want to maximize total prize but code minimizes so return negative
        # return -p.sum(-1), None
        return -((torch.stack(total_p, -1)).sum(1)).sum(1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TOPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTOP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TOP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def generate_instance(size, num_veh, prize_type):
    # Details see paper
    MAX_LENGTHS = {
        10: 1.5,
        20: 2.,
        50: 3.,
        70: 3.0
    }

    loc = torch.FloatTensor(size, 2).uniform_(0, 1) #0509: loc.shape = torch.Size([20, 2])
    depot = torch.FloatTensor(2).uniform_(0, 1)     #0509: depot.shape = torch.Size([2])
    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = torch.ones(size)
    elif prize_type == 'unif':
        prize = (1 + torch.randint(0, 100, size=(size, ))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = (depot[None, :] - loc).norm(p=2, dim=-1)
        prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.
        
    ###################### 0523: add num_veh TODO: this is harded coded. Change later#############################
    #num_veh = torch.tensor(2, dtype=torch.uint8)
    ###############################################################################################################
    
    # 0715: add vehicle current location and tour length so far 
    cur_loc = torch.FloatTensor(num_veh,2).uniform_(0, 1)
    dist_to_depot = (cur_loc - depot).norm(p=2,dim=-1) # = distance required to be left
    cur_tlen = [torch.FloatTensor(1).uniform_(0, MAX_LENGTHS[size] - dist_to_depot[i].item() - 1e-6) for i in range(num_veh)] 
    cur_tlen = torch.tensor([cur_tlen[i].item() for i in range(num_veh)])
        
    assert (MAX_LENGTHS[size] - cur_tlen > dist_to_depot).all() , "not enough remaining tour length to return to depot" # check for an error
    
    return {
        'loc': loc,
        # Uniform 1 - 9, scaled by capacities
        'prize': prize,
        'depot': depot,
        'max_length': torch.tensor(MAX_LENGTHS[size]),
        'num_veh': num_veh,
        'cur_loc': cur_loc,
        'cur_tlen': cur_tlen
    }


class TOPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_veh=2, num_samples=1000000, offset=0, distribution='const'):
        super(TOPDataset, self).__init__()
        assert distribution is not None, "Data distribution must be specified for OP"
        # Currently the distribution can only vary in the type of the prize
        prize_type = distribution

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [
                    {
                        'loc': torch.FloatTensor(loc),
                        'prize': torch.FloatTensor(prize),
                        'depot': torch.FloatTensor(depot),
                        'max_length': torch.tensor(max_length),
                        'num_veh': torch.tensor(num_veh), # 0523: add num_veh
                        'cur_loc': torch.tensor(cur_loc),
                        'cur_tlen': torch.tensor(cur_tlen)
                    }
                    for depot, loc, prize, max_length, num_veh, cur_loc, cur_tlen in (data[offset:offset+num_samples])
                ]
        else:
            self.data = [
                generate_instance(size, num_veh, prize_type)
                for i in range(num_samples) # 0509: 512 -> generate 512 instances each with 20 nodes and one depot 
            ]

        self.size = len(self.data) # 512

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

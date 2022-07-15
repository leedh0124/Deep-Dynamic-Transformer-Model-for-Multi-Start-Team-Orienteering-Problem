import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import torch.nn.functional as F


class StateTOP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    prize: torch.Tensor
    # Max length is not a single value, but one for each node indicating max length that each tour should have when arriving
    # at this node, so this is max_length - d(depot, node)
    max_length: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and prizes tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows = index of TOP instances (0720)

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    cur_total_prize: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    num_veh: torch.Tensor  # 0523: add num_veh
    cur_loc: torch.Tensor  # 0719: add cur_loc
    cur_tlen: torch.Tensor # 0719: add cur_tlen

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.coords.size(-2))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key],
            cur_total_prize=self.cur_total_prize[key],
            num_veh = self.num_veh[key],        # 0523: add num_veh
            cur_loc = self.cur_loc[key],        # 0719: add cur_loc
            cur_tlen = self.cur_tlen[key]       # 0719: add cur_tlen
        )

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):
        depot = input['depot']
        loc = input['loc']
        prize = input['prize']
        max_length = input['max_length']
        num_veh = input['num_veh']      # 0523: add num_veh
        cur_loc = input['cur_loc']      # 0719: add cur_loc  
        cur_tlen = input['cur_tlen']    # 0719: add cur_tlen

        batch_size, n_loc, _ = loc.size()
        coords = torch.cat((depot[:, None, :], loc), -2) # [1024, 1, 2] & [1024, 20, 2] -> [1024, 21, 2]
        
        # 0720: concatenate max_length tensor for each num_veh , [1024, 21] -> [1024, 1, 21] -> [1024, 2, 21], along dim=1
        # max_length now has cur_tlen subtracted
        max_length_temp = (max_length[:, None] - cur_tlen[:,0][:,None] - (depot[:, None, :] - coords).norm(p=2, dim=-1)  -  1e-6)[:,None,:] # first route
        for n in range(num_veh[0]-1):
            max_length_temp = torch.cat((max_length_temp, (max_length[:, None] - cur_tlen[:,n+1][:,None] - (depot[:, None, :] - coords).norm(p=2, dim=-1)  -  1e-6)[:,None,:]), dim=1) # next routes
        
        # 0720: prev_a is initialized with index of the first vehicle's loc. 20+1 at start
        prev_a = n_loc + torch.arange(1, 2, device=loc.device) # 20 + 1
        prev_a = prev_a[None,:].expand(batch_size,1) # torch.Size([1024, 1])
        
        
        # prev_a_temp = torch.arange(1,num_veh[0]+1, dtype=torch.long, device=loc.device) + n_loc # +20
        # prev_a_temp = prev_a_temp[None,:].expand(batch_size, num_veh[0]) # torch.Size([1024, 2])
        
        
        return StateTOP(
            coords=coords, # depot + n_locs 
            prize=torch.cat( (F.pad(prize, (1, 0), mode='constant', value=0), torch.zeros(batch_size,num_veh[0], device=loc.device) ), dim=1),  
            # add 0 for depot and cur_loc # [1...,  ..., 1.] -> [0., 1.,  ..., 1., 0., 0...]
            # max_length is max length allowed when arriving at node, so subtract distance to return to depot
            # Additionally, substract epsilon margin for numeric stability
            max_length=max_length_temp, # This is now torch.Size([1024, 2, 21])
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension torch.Size([1024]) -> torch.Size([1024,1]) of 0,1,2,...,1023
            prev_a=prev_a, # [1024,1]
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently (if there is an action for depot)
                torch.zeros(
                    batch_size, n_loc + 1, # 0906: self.visited shape: [1024,21] 
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 1 + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ), 
            lengths=torch.zeros(batch_size, 1, device=loc.device), # torch.Size([1024,1]) 
            cur_coord=input['cur_loc'][:,0,:],  # 0719: torch.Size([1024,2]) : first vehicle's cur_coord
            cur_total_prize=torch.zeros(batch_size, 1, device=loc.device), # torch.Size([1024,1]) 
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            num_veh = num_veh[:,None],              # torch.Size([1024,1]) 
            cur_loc = input['cur_loc'],            # 0719: add cur_loc
            cur_tlen = input['cur_tlen'][:,None,:] # 0719: add cur_tlen with steps dimension
            
            # 0907: prev_a, visited_, lengths, cur_coord, cur_total_prize,  to be UPDATED in subsequent steps
            # 0908: coords, prize, max_length, cur_loc, cur_tlen, num_veh remains same throughout
        )

    def get_remaining_length(self, inner_i):
        # max_length[:, 0] is max length arriving at depot so original max_length
        return self.max_length[:,inner_i,0].unsqueeze(1) - self.lengths # torch.Size([1024,2]) /  self.max_length[self.ids, 0] - self.lengths[:,None,:] 
        # (previouisly) TOP:  self.max_length[:,0] = [1024, 1]...depot, self.lengths = zeros with dim [1024,1] 
        # 0720's Note: negative entries indicate nodes that cannot be visited (even at cur_loc when start)

    def get_final_cost(self):

        assert self.all_finished()
        # The cost is the negative of the collected prize since we want to maximize collected prize
        return -self.cur_total_prize

    def update(self, input, inner_i):
        
        # Update cur_loc, lengths, prev_a, i
        assert (self.prev_a == 0).all(), "all vehicles returned to depot once partial finished"
        
        loc = input['loc']
        batch_size, n_loc, _ = loc.size()
        cur_coord = input['cur_loc'][:,inner_i,:]
        visited_ = self.visited_.scatter(-1, torch.zeros((batch_size,1), dtype=torch.int64, device=loc.device), 0)
        lengths = torch.zeros(batch_size, 1, device=loc.device)
        prev_a = (n_loc + inner_i) + torch.arange(1, 2, device=loc.device) # 20 + inner_i + 1
        prev_a = prev_a[None,:].expand(batch_size,1) # torch.Size([1024, 1])
        i = torch.zeros(1, dtype=torch.int64, device=loc.device)
        
        return self._replace(visited_=visited_, cur_coord=cur_coord, lengths=lengths, prev_a=prev_a, i=i)

    def inner_update(self, selected, inner_i):

        assert self.i.size(0) == 1, "Can only update if state represents single step"
        
        batch_size, step = selected.size()
        
        # Update the state
        prev_a = selected

        # Add the length
        cur_coord = self.coords[self.ids,selected].squeeze(1) # torch.Size([1024,2])
        lengths = (self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)[:,None])  # torch.Size([1024,1])
        #lengths = (self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1))[:,None]  # torch.Size([1024,1])


        # Add the collected prize
        cur_total_prize = self.cur_total_prize + self.prize[self.ids, selected] # torch.Size([1024,1])

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a.view(batch_size, step), 1) #0523: DH: visited에 1 넣기
        else:
            # This works, by check_unset=False it is allowed to set the depot visited a second a time
            visited_ = mask_long_scatter(self.visited_, prev_a, check_unset=False)
            
        # Update num_veh and lengths: selected가0(depot)이면 num_veh -= 1, <- NOT NECESSARY
        # to_update = torch.tensor([(torch.max(self.num_veh[i]-1, torch.tensor(0)), 0.0)  if selected[i]==0 else (self.num_veh[i], lengths[i]) for i in range(selected.size()[0])])[:,None].to(selected.device)
        # num_veh = to_update[:,:,0]
        # lengths = to_update[:,:,1]
        
        return self._replace(
            prev_a=prev_a, visited_=visited_, lengths=lengths,
            cur_coord=cur_coord, cur_total_prize=cur_total_prize, i=self.i + 1)

    def all_finished(self, inner_i):
        # All must be returned to depot (and at least 1 step since at start also prev_a == 0)
        # This is more efficient than checking the mask
        return self.i.item() > 0 and (self.prev_a == 0).all() and (inner_i == self.num_veh[0])      
        #return self.i.item() > 0 and (self.num_veh == 0).all() 
        
    def partial_finished(self):
        return self.i.item() > 0 and (self.prev_a == 0).all()
    
    @staticmethod
    def outer_pr(a, b):
        """Outer product of matrices
        """
        return torch.einsum('ki,kj->kij', a, b)
    
    # Trial v4.5 - failed
    # def get_att_mask(self, inner_i):
    #     # We don't want to mask depot 
    #     visited_node = self.visited[:,1:] # torch.Size([batch_size, graph_size-1])   
        
    #     # Number of nodes in new instance after masking
    #     batch_size, graph_size = self.visited.shape # graph_size = 21
    #     cur_num_nodes = graph_size - visited_node.sum(-1)[:,None] # torch.Size([batch_size,1])
        
    #     # veh_mask to mask vehicles arrived at depot
    #     veh_mask = torch.zeros(batch_size, self.num_veh[0], dtype=torch.uint8, device=self.visited.device)
    #     veh_mask[:,0:inner_i] = torch.tensor(1)
        
    #     # Create square attention mask from row-like mask: torch.Size([batch_size, 20+2]) ->  torch.Size([batch_size, 22, 22])
    #     # For each problem in batch_size, mask row and column corresponding to each visited (masked) node 
    #     att_mask = torch.cat((visited_node, veh_mask), dim=-1)
    #     ones_mask = torch.ones_like(att_mask)
    #     att_mask = (self.outer_pr(att_mask, ones_mask) + self.outer_pr(ones_mask, att_mask) - self.outer_pr(att_mask, att_mask)).to(torch.bool)
        
    #     return att_mask, cur_num_nodes
    
    # v4.4
    def get_att_mask(self):
        # We don't want to mask depot 
        visited_node = self.visited[:,1:] # torch.Size([batch_size, graph_size-1])   
        
        # Number of nodes in new instance after masking
        _, graph_size = self.visited.shape
        cur_num_nodes = graph_size - visited_node.sum(-1)[:,None] # torch.Size([batch_size,1])
        
        # Create square attention mask from row-like mask: torch.Size([batch_size, graph_size-1]) ->  torch.Size([batch_size, graph_size-1, graph_size-1])
        # For each problem in batch_size, mask row and column corresponding to each visited (masked) node 
        att_mask = visited_node[:]
        ones_mask = torch.ones_like(att_mask)
        att_mask = (self.outer_pr(att_mask, ones_mask) + self.outer_pr(ones_mask, att_mask) - self.outer_pr(att_mask, att_mask)).to(torch.bool)
        
        return att_mask, cur_num_nodes    
    

    def get_current_node(self):
        """
        Returns the current node where 0 is depot, 1...n are nodes
        :return: (batch_size, num_steps) tensor with current nodes
        """
        return self.prev_a

    def get_mask(self, inner_i): 
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        
        # 1. mask negative distance nodes in self.max_length
        
        # 2. Compute exceeds_lengths with torch.Size([1024,1,21]) where 21 is depot + n_loc
        # => 현재까지 traveled length + 현 위치로부터 다른 모든 점까지 거리 > maxlength(=2)에서 각 점으로부터 depot까지 거리를 이미 뺀 값
        # self.lengths      = torch.Size([1024,1])
        # self.coords       = torch.Size([1021,21,2])
        # self.cur_coord    = torch.Size([1024,2])
        # self.max_length   = torch.Size([1021,2,21]) -> self.max_length[self.ids,:] = [1024, 1, 2, 21]
        # self.ids          = torch.Size([1024,1])
        # exceeds_length    = torch.Size([1024,21])

        exceeds_length = (
            self.lengths + (self.coords - self.cur_coord[:,None,:]).norm(p=2, dim=-1)
            > self.max_length[:,inner_i] 
        )
        # Note: this always allows going to the depot, but that should always be suboptimal so be ok
        # Cannot visit if already visited or if length that would be upon arrival is too large to return to depot
        # If the depot has already been visited then we cannot visit anymore (DH: This is for OP. Must be changed for TOP)
        visited_ = self.visited.to(exceeds_length.dtype) # exceeds_length.dtype = torch.bool, convert self.visited to torch.bool
        # torch.Size([1024,2,21]) 
        
        # Depot can always be visited
        # (so we do not hardcode knowledge that this is strictly suboptimal if other options are available)
        #mask1[:, :, 0] = 0 # do not mask depot node 
    
        batch_size, num_veh = self.lengths.shape
        ######################################################################################################
        # 0. Check if vehicle has departed. 
        # visited_[:,0] = ~ (self.lengths[:,inner_i] > 0) 
        # 1. If num_veh == 0 or has returned to depot, mask all nodes including depot
        mask = visited_ | exceeds_length | visited_[:,0][:,None]
        #mask[:,0] = ~(torch.all(mask,dim=-1))
        # 2. If all nodes (including depot) is masked, set num_veh_ = 0, and route is over
        num_veh_ = self.num_veh * ~(torch.all(mask,dim=-1))[:,None]
        # 3. Hardcode: unmask depot if route is over so that vehicle stays in depot until other prob is solved
        mask[:,0] = mask[:,0] * ~(num_veh_==0).squeeze(1)
        ########################################################################################################
        
        # 0802: 3. Return reshaped mask so that it has cur_loc node always masked using concat.
        return torch.cat((mask,torch.ones((batch_size,num_veh), dtype=torch.bool, device=mask.device)), dim=-1) # this must be torch.Size([1024,22])

    def construct_solutions(self, actions):
        return actions

import torch
from problems.top.state_top import StateTOP

# Adapted for MSTOP
def top_tsiligirides(batch, sample=False, power=4.0):
    state = StateTOP.initialize(batch)

    num_veh = state.num_veh[0]
    inner_i=0
    all_a = []
    p_idx = []
    
    while not state.all_finished(inner_i):
        if inner_i > 0:
            mask,_ = state.get_att_mask()
            state = state.update(batch, inner_i) # batch=input in _inner function
        
        # Construct partial solution
        p_i = 0
        
        while not state.partial_finished():
        
            # Compute scores
            mask = state.get_mask(inner_i)
            p = (
                    (mask[..., 1:-1] == 0).unsqueeze(1).float() * # (batch, n_nodes)
                    state.prize[state.ids, 1:-num_veh] /
                    ((state.coords[state.ids, 1:, :] - state.cur_coord[:, None, None, :]).norm(p=2, dim=-1) + 1e-6)
            ) ** power
            # p.shape : (batch, 1, n_nodes)
            bestp, besta = p.topk(4, dim=-1)
            # bestp, besta: (batch, 1, 4)
            bestmask = mask[..., 1:-1].gather(-1, besta.squeeze(1))
    
            # If no feasible actions, must go to depot
            # mask == 0 means feasible, so if mask == 0 sums to 0 there are no feasible and
            # all corresponding ps should be 0, so we need to add a column with a 1 that corresponds
            # to selecting the end destination
            to_depot = ((bestmask == 0).sum(-1, keepdim=True) == 0).float()
            # best_p should be zero if we have to go to depot, but because of numeric stabilities, it isn't
            p_ = torch.cat((to_depot[:,None,:], bestp), -1)
            # p_ : (batch, 1, 1+4)
            pnorm = p_ / p_.sum(-1, keepdim=True)
            # pnorm : (batch, 1, 1+4)
            
            if sample:
                a = pnorm[:, 0, :].multinomial(1)  # Sample action
            else:
                # greedy
                a = pnorm[:, 0, :].max(-1)[1].unsqueeze(-1)  # Add 'sampling dimension'
    
            # a == 0 means depot, otherwise subtract one
            final_a = torch.cat((torch.zeros_like(besta[..., 0:1]), besta + 1), -1)[:, 0, :].gather(-1, a)
            # final_a: (batch, 1)
            state = state.inner_update(final_a, inner_i)
            selected = final_a[..., 0]  
            # selected: (batch)
            all_a.append(selected)
            
            p_i += 1
            
        # Collect output of partial finished
        p_idx.append(p_i)
        pi = torch.stack(all_a, -1)    
        
        inner_i += 1
        
        
    return None, pi, p_idx #torch.stack(all_a, -1)


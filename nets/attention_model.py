import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches
from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import torch.nn.functional as F
import numpy as np

def MHA_decoder(Q, K, V, n_heads, mask=None):
    """
    Compute multi-head attention (MHA) for a given query Q, key K, value V and mask, used during decoding step of partial soln.
    h = Softmax(QK^T)V and concatenate across all heads
    Parameters
    ----------
    Q : size [batch_size, 1, emb_dim]
        batch of queries.
    K : size [batch_size, 1++, emb_dim] for step 1: self-attention and [batch_size, 1+n_nodes+1, emb_dim] for step 2: enc-dec attention
        batch of keys.
    V : size [batch_size, 1++, emb_dim]
        batch of values.
    Returns
    -------
    attn_output : size [batch_size, 1, emb_dim]
        batch of attention vectors
    attn_weights : size [batch_size, 1, n_nodes+1]
        batch of attention weights/scores
    """
    assert n_heads > 1, "nb_heads greater than 1 for multi-head attention"
    batch_size, n_nodes_partial , emb_dim = K.size() # [512, 1++, 128] or ([512, 1, 128])
    head_dim = emb_dim//n_heads # 128 // 8 = 16
    Q = Q.contiguous().view(batch_size*n_heads, 1, head_dim) # now size is [512*8, 1, 16]
    K = K.contiguous().view(batch_size*n_heads, n_nodes_partial, head_dim) # now size is [512*8, 1++, 16] or [512*8, 1+n_nodes+1, 16] 
    V = V.contiguous().view(batch_size*n_heads, n_nodes_partial, head_dim) # now size is [512*8, 1++, 16]
    attn_weights = torch.bmm(Q, K.transpose(1,2)) / head_dim**0.5    # size[512*8, 1, n_nodes+1]
    
    if mask is not None:
        mask = torch.repeat_interleave(mask, repeats=n_heads, dim=0) 
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-1e9')) # [batch_size*n_heads, 1, 1++]
    
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_weights, V) # [batch_size*n_heads, 1, head_dim]
    
    # Reshape attn_output and attn_weights to [batch_size, 1, emb_dim] and [batch_size, 1, 1++]
    attn_output = attn_output.contiguous().view(batch_size, 1, emb_dim)
    attn_weights = attn_weights.view(batch_size, n_heads, 1, n_nodes_partial)
    attn_weights = attn_weights.mean(dim=1) # average over all heads
    
    return attn_output, attn_weights


class AutoRegressiveDecoderLayer(nn.Module):
    """
    Single decoder layer for self-attention and query-attention
    Inputs:
        h_t   : input queries, [batch_size, 1, emb_dim] 
        K_att : enc_dec-attention keys, [batch_size, n_nodes+1, emb_dim]
        V_att : enc_dec-attention values, [batch_size, n_nodes+1, emb_dim]
        mask  : mask of visited + cannot be visited cities, [batch_size, n_nodes+1]
    Output:
        h_t   : transformed queries, [batch_size, 1, emb_dim]
    """
    def __init__(self, emb_dim, n_heads):
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.Wq_sa = nn.Linear(emb_dim, emb_dim)
        self.Wk_sa = nn.Linear(emb_dim, emb_dim)
        self.Wv_sa = nn.Linear(emb_dim, emb_dim)
        self.W0_sa = nn.Linear(emb_dim, emb_dim)
        self.W0_att = nn.Linear(emb_dim, emb_dim)
        self.Wq_att = nn.Linear(emb_dim, emb_dim)
        self.W1_FF = nn.Linear(emb_dim, emb_dim)
        self.W2_FF = nn.Linear(emb_dim, emb_dim)
        self.LN_sa = nn.LayerNorm(emb_dim)
        self.LN_att = nn.LayerNorm(emb_dim)
        self.LN_FF = nn.LayerNorm(emb_dim)
        self.act_FF = nn.ReLU()
        self.K_sa = None
        self.V_sa = None
        
    def reset_sa_keys_values(self):
        # To be called at the start of each partial solution
        self.K_sa = None
        self.V_sa = None
        
    def forward(self, h_t, K_att, V_att, mask):
        # Embed h_t for self-attention
        q_sa = self.Wq_sa(h_t) # [batch_size, 1, emb_dim]
        k_sa = self.Wk_sa(h_t) # [batch_size, 1, emb_dim]
        v_sa = self.Wv_sa(h_t) # [batch_size, 1, emb_dim]
        
        # Concatenate new self-attention keys and values to the previous keys and values
        if self.K_sa is None:
            self.K_sa = k_sa
            self.V_sa = v_sa
        else:
            self.K_sa = torch.cat([self.K_sa, k_sa], dim=1) #[batch_size, 1++, emb_dim]
            self.V_sa = torch.cat([self.V_sa, v_sa], dim=1) #[batch_size, 1++, emb_dim]
        
        # Step 1: Compute self-attention with current node and the nodes in the partial solution
        attn_output_sa, _  = MHA_decoder(h_t, self.K_sa, self.V_sa, self.n_heads, mask=None)
        h_t = h_t + self.W0_sa(attn_output_sa) # h_t size : [batch_size, 1, emb_dim]
        h_t = self.LN_sa(h_t.squeeze(1))    # h_t size : [batch_size, emb_dim]
        h_t = h_t.unsqueeze(1)              # h_t size : [batch_size, 1, emb_dim]
        
        # Step 2: Compution encoder-decoder-attention between self-attention of partial soln (h_t) \
        #         and context node embeddings with appropriate mask (K_att and V_att derived from encoder's H_enc)
        q_att = self.Wq_att(h_t)
        attn_output_att, _  = MHA_decoder(q_att, K_att, V_att, self.n_heads, mask=mask)
        h_t = h_t + self.W0_att(attn_output_att) # h_t size : [batch_size, 1, emb_dim]
        h_t = self.LN_sa(h_t.squeeze(1))    # h_t size : [batch_size, emb_dim]
        h_t = h_t.unsqueeze(1)              # h_t size : [batch_size, 1, emb_dim]
        
        # Fully Connected layer
        h_t = h_t + self.W2_FF(self.act_FF(self.W1_FF(h_t)))
        h_t = self.LN_FF(h_t.squeeze(1)) 
        h_t = h_t.unsqueeze(1)              # h_t size : [batch_size, 1, emb_dim]

        return h_t

def PositionalEmbedding(d_model, max_len=100):
    """
    Create standard transformer PEs.
    Inputs :  
      d_model is a scalar correspoding to the hidden dimension
      max_len is the maximum length of the sequence
    Output :  
      pe of size (max_len, d_model), where d_model=dim_emb, max_len=100 (max. 100 nodes)
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:,0::2] = torch.sin(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)
    return pe

def set_decode_type(model, decode_type):

    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
        )


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=3,
                 n_decode_layers=3,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.n_decode_layers = n_decode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_team_orienteering = problem.NAME == 'top'
        self.is_pctsp = problem.NAME == 'pctsp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem # <class 'problems.top.problem_top.TOP'>
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_team_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1
            
            # if self.is_team_orienteering: #0720: remaining length per route/num_veh
            #     step_context_dim = step_context_dim + problem.num_veh - 1 # 128+2 = 130
                
            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand for VRP / prize for TOP

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
                
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations
            
        # 0718: Modify node_dim so that cur_loc and cur_tlen info are included
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.init_embed_cur_loc = nn.Linear(node_dim, embedding_dim)

        # Encoder 
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        
        # Decoder layer
        self.decoder_layer = nn.ModuleList( [AutoRegressiveDecoderLayer(embedding_dim, n_heads) for _ in range(self.n_decode_layers)] )
        
        # Key and value for encoder-decoder attention in decoder
        self.WK_att_decoder = nn.Linear(embedding_dim, n_decode_layers*embedding_dim)
        self.WV_att_decoder = nn.Linear(embedding_dim, n_decode_layers*embedding_dim)
        
        # Final query and key in decoder
        self.WQ_att_f = nn.Linear(embedding_dim, embedding_dim)
        self.WK_att_f = nn.Linear(embedding_dim, embedding_dim)
        
        # Positional encoding : encode the order of nodes explicitly
        self.PE = PositionalEmbedding(d_model=embedding_dim) #, max_len=4*embedding_dim)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim) #0204: dont use this
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim) #0204: dont use this

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
        ## Transformer의 Encoder 부분 ##
        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input), mask=None) # 0719:  Now, embeddings is torch.Size([512, 23, 128])
            
        ## Transformer의 Decoder 부분 ##
        # This embeddings is the original graph instance
        _log_p, pi, pad_idx = self._inner(input, embeddings) 
        
        # Get problem costs
        cost, mask = self.problem.get_costs(input, pi, pad_idx)
        
        # Log likelihood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask) # mask here is None. Not same as mask inside self._inner
        if return_pi:
            ## Previous version
            # log_p = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)
            # exp_log_p = log_p.exp()
            # entropy = (-exp_log_p*log_p).sum(-1)
            ## New version
            exp_log_p = _log_p.exp()
            # assert ~torch.isnan((-exp_log_p*_log_p)).any(), "Nan in entropy!"
            entropy = (-exp_log_p*_log_p).sum(-1).sum(-1) # _log_p[x][a_t]
            return cost, ll, pi, entropy
            
        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]
      
    def _calc_log_likelihood(self, _log_p, a, mask):
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1) # _log_p.shape = torch.Size([1000, 29, 21]), log_p.shape = torch.Size([1000, 29]) --- (v4.5)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
    
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
    
        # Calculate log_likelihood
        return log_p.sum(1) # torch.Size([1000])
    
    def _init_embed(self, input):
        # 0718: modified input to include cur_loc and cur_tlen
        
        if self.is_vrp or self.is_orienteering or self.is_team_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering or self.is_team_orienteering:
                features = ('prize', ) # tuple
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :], # torch.Size([1024, 1, 128])
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features) # 0719: torch.Size([1024, 20, 3]) -> torch.Size([1024, 20, 128])              
                    ), -1)), 
                    self.init_embed_cur_loc(torch.cat((input['cur_loc'], input['cur_tlen'][:,:,None]),-1)) 
                    # 0719: add cur_loc and cur_tlen info, torch.Size([1024, 2, 3]) -> torch.Size([1024, 2, 128])
                ),
                1
            ) # 0719: torch.Size([1024, 23, 128]): depot (1) + nodes/prizes (20) + cur_loc/cur_tlen for each vehicle (2) 
        # TSP
        return self.init_embed(input)

    def _inner(self, input, embeddings):
        """
        Parameters
        ----------
        input : dictionary
            contains cur_loc, cur_tlen, depot, loc, max_length, num_veh, prize.
        embeddings : tensor
            size [batch_size, number of nodes + 3, emb_dim] = [512, 23, 128].

        Returns
        """
        state = self.problem.make_state(input) # 0719: Make necessary modifications for TOP state / access Namedtuple by state.coords or state.cur_tlen

        # Namedtuple AttentionFiexModel(embeddings, context_node_projected, glimpse_key, glimpse_val, logit_key)
        fixed = self._precompute(embeddings)
        batch_size = state.ids.size(0)
        
        # Generate Positional Embedding and send to GPU
        self.PE = self.PE.to(state.ids.device)
        
        # Perform decoding steps
        #1223: So far, n_decoder_layers = 1
        inner_i = 0
        outputs = []
        sequences = []
        pad_idx = []
        while not (self.shrink_size is None and state.all_finished(inner_i)):
            if inner_i > 0:
                # After vehicle has returned to depot, update embeddings of the remaining nodes using self.embedder
                # 1. get current att_mask and cur_num_nodes
                att_mask, cur_num_nodes = state.get_att_mask() #state.get_att_mask(inner_i) --- v4.5
                # 2. Update embeddings by adding mask to attention weights
                new_embeddings, context_vectors = self.embedder(self._init_embed(input), mask=att_mask, cur_num_nodes=cur_num_nodes, inner_i=inner_i)
                # 3. Update AttentionModelFixed / fixed
                fixed = self._precompute(embeddings=new_embeddings, context_vectors=context_vectors)
                # 4. Update state: next vehicle starts at a different cur_loc...
                state = state.update(input, inner_i)
            
            # Construct partial solution
            pad_i = 0
            
            # Set self-attention keys and values to None at the start of partial solution
            for i in range(self.n_decode_layers):    
                self.decoder_layer[i].reset_sa_keys_values()
                
            while not state.partial_finished():  
                
                # 1221:_get_log_p에서 Step 1 시작   
                log_p, mask = self._get_log_p(fixed, state, inner_i, pad_i) #check torch.nonzero(~mask[:,:,1:23]) for termination
    
                # Select the indices of the next nodes in the sequences, result (batch_size) long
                selected = self._select_node(log_p.exp(), mask)  # 0906: selected is [512,1]
                
                # Update state
                state = state.inner_update(selected, inner_i)
    
                # Collect output of partial step
                outputs.append(log_p[:,0,:])
                sequences.append(selected.squeeze(1))   
                
                pad_i += 1                
                
            # Collect output of partial_finished  (TODO: check dimension, zeros use torch.nn.functional.pad)
            pad_idx.append(pad_i)
            _log_p = torch.stack(outputs,1)      
            pi = torch.stack(sequences,1)    
                               
            inner_i += 1
            
        # Collected lists, return Tensor
        return _log_p, pi, pad_idx # torch.Size([1024, n, 2, 23]), torch.Size([1024, n, 2]) 다시!

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi, pad_idx: self.problem.get_costs(input[0], pi, pad_idx),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.squeeze(1).max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.squeeze(1).multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected[:,None]

    def _precompute(self, embeddings, num_steps=1, context_vectors=None):

        # The fixed context projection of the graph embedding is calculated only once for efficiency <- done per route/vehicle
        if context_vectors is None: 
            graph_embed = embeddings.mean(1)
            # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
            fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        else:
            # for updating fixed 
            fixed_context = self.project_fixed_context(context_vectors)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )
           
    def _get_log_p(self, fixed, state, inner_i, pad_i, normalize=True):
        """
        Parameters
        ----------
        fixed.context_node_projected : context node embedding/H_enc, [batch_size, 1, emb_dim] = [512, 1, 128]
        fixed.node_embeddings : [batch_size, depot+n_nodes+2, emb_dim] = [512, 23, 128]
        # H_enc : fixed.context_node_projected.
        # h_t : node embedding of previously selected node i_t, i.e. cur_coord, size [batch_size, 1, emb_dim]
        state : top.state_top.StateTOP
            size 13.
        inner_i : int
            counter for routes/num_veh.
        normalize : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        log_p : TYPE
            DESCRIPTION.
        mask : TYPE
            DESCRIPTION.
        """
        # Step 0 : Compute the mask (nodes cannot be visited + cur_loc)
        mask = state.get_mask(inner_i) # torch.Size([batch_size, 1+n_nodes+1]); last one : per vehicle/cur_loc
        # Step 1 : Self-attention to the partial tour at current step t, starting from cur_loc and extract its embedding, h_t 
        # Step 2 : Query next node with non-visited nodes using query-attention layer
        # Get h_t of current node, of size [batch_size, 1, emb_dim]
        h_t = self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, inner_i))
        # Add positional encoding to h_t 
        h_t = h_t + self.PE[pad_i].expand(mask.size(0), self.embedding_dim).unsqueeze(1) # [batch_size, 1, emb_dim]
        # Get H_enc which is context node embedding, obtained from encoder
        H_enc = fixed.context_node_projected
        # Get H_node_emb which is node embedding including up to current vehicle's cur_loc, size [batch_size, 1+n_nodes+1, emb_dim]
        _, n_nodes, _ = state.coords.shape # here, n_nodes = 21
        H_node_emb = torch.cat((fixed.node_embeddings[:,0:n_nodes,:],fixed.node_embeddings[:,n_nodes+inner_i,:].unsqueeze(1)), dim=1)  
        K_att_decoder = self.WK_att_decoder(H_node_emb)
        V_att_decoder = self.WV_att_decoder(H_node_emb)
        # Use multiple decode layers
        for i in range(self.n_decode_layers):
            K_att_i = K_att_decoder[:, :, i*self.embedding_dim:(i+1)*self.embedding_dim].contiguous() # 0~127, 128~255,...
            V_att_i = V_att_decoder[:, :, i*self.embedding_dim:(i+1)*self.embedding_dim].contiguous()
            h_t = self.decoder_layer[i](h_t, K_att_i, V_att_i, mask)
            #h_t = self.decoder_layer[i](h_t, K_att_decoder, V_att_decoder, mask)
        
        # Step 3 : Final query to get distribution over non-visited nodes <- _one_to_many_logits(...)
        # Compute query = context node embedding (pet step t) with torch.Size([1024, 23, 128])
        #query = H_enc + h_t 
        h_t = H_enc + h_t
        query = self.WQ_att_f(h_t)
        H_node_tilda = H_node_emb[:,0:n_nodes,:] # only depot + nodes, excluding vehicle's cur_loc
        key = self.WK_att_f(H_node_tilda) 
        final_attn = torch.bmm(query, key.transpose(1,2)) / query.size(-1)**0.5 # [batch_size, 1, n_nodes+1]
        if self.tanh_clipping > 0:
            final_attn = torch.tanh(final_attn) * self.tanh_clipping
        if self.mask_inner:
            final_attn[mask[:,None,0:n_nodes]] = float('-1e9')
        # logits = torch.softmax(final_attn, dim=-1)
        # if self.mask_logits:
        #     logits[mask[:,None,0:n_nodes]] = float('-1e9')
        # # Compute keys and values for the nodes and cur_loc
        # # size of glimpse_K : [8, 512, 1, 23, 16]
        # # size of logit_K : [512, 1, 23, 128]
        # glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # # Compute logits (unnormalized log_p) 0907: Take one cur_loc pertaining to the vehicle
        # log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, state, inner_i)

        log_p = torch.log_softmax(final_attn / self.temp, dim=-1)
         

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, inner_i, from_depot=False):
        """
        Returns the context node embedding per step t, i.e. of current node

        """

        current_node = state.get_current_node() # now equivalent to cur_loc node 
        batch_size, num_steps = current_node.size()

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous()
                                .view(batch_size, num_steps, 1)
                                .expand(batch_size, num_steps, embeddings.size(-1))
                        ).view(batch_size, num_steps, embeddings.size(-1)),
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]
                    ),
                    -1
                )
        elif self.is_orienteering or self.is_team_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,                       
                        current_node.contiguous() # current_node의 embedding된 값들을 불러오기 [512,1] -> [512,1,128]
                            .view(batch_size, num_steps, 1)
                            .expand(batch_size, num_steps, embeddings.size(-1))
                    ).view(batch_size, num_steps, embeddings.size(-1)), 
                    (
                        state.get_remaining_length(inner_i)[:, None] #0907: Take only those pertaining to current vehicle; torch.Size([512, 1, 1])
                        if self.is_orienteering or self.is_team_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            ) # By torch.cat, we want [512, 1, 128+1] where +1: remaining_length
        else:  # TSP
        
            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    return embeddings.gather(
                        1,
                        torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, state, inner_i): 
        # new glimpse_K should have graph size 1+20+1=22, includes one cur_loc
        # mew mask should be torch.Size([1024,22])

        _, graph_size , _ = state.coords.shape
        graph_size -= 1 # 21 -> 20
        new_glimpse_K = torch.cat((glimpse_K[:,:,:,0:graph_size+1,:], glimpse_K[:,:,:,graph_size+inner_i+1,:].unsqueeze(-2)), dim=-2) # torch.Size([8, 512, 1, 22, 16]) since nodes&cur_loc + cur_tlen
        new_glimpse_V = torch.cat((glimpse_V[:,:,:,0:graph_size+1,:], glimpse_V[:,:,:,graph_size+inner_i+1,:].unsqueeze(-2)), dim=-2)
        new_logit_K = torch.cat((logit_K[:,:,0:graph_size+1,:], logit_K[:,:,graph_size+inner_i+1,:].unsqueeze(-2)), dim=-2)
        
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size) 
        # = torch.Size([8, 512, 1, 1, 16]) 0730
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        # = torch.Size([8, 512, 1, 1, 22]) just one cur_lo`c
        compatibility = torch.matmul(glimpse_Q, new_glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            
            compatibility[mask[None, :, None, None].expand_as(compatibility)] = -1e10 #-math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), new_glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, new_logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1)) 
        #0906: logits.shape = torch.Size([1024, 1, 22])

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask[:,None,:]] = -1e10 #-math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        if self.is_vrp and self.allow_partial:

            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

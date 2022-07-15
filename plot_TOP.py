USE_CUDA = True
POMO = False
INST_AUG = True

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from generate_data import generate_top_data 
from utils import load_model, move_to
from utils.functions import augment_xy_data_by_8_fold
from problems import TOP
#from problems.top.top_MILP import solve_euclidian_dtop as solve_euclidian_dtop_MILP
from problems.top.top_MILP_original import solve_euclidian_dtop as solve_euclidian_dtop_MILP
import time
from scipy.io import savemat, loadmat
import subprocess
import math
from tqdm import tqdm
from itertools import permutations

#%% plot_TOP.py for MSTOP

# matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from adjustText import adjust_text
import seaborn as sns

# IEEE Style Plot Presetting
import Preset
Preset.setPlotStyle()
# plt.style.use(['ieee'])

# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def discrete_cmap(N, base_cmap=None):
  """
    Create an N-bin discrete colormap from the specified input map
    """
  # Note that if base_cmap is a string or None, you can simply do
  #    return plt.cm.get_cmap(base_cmap, N)
  # The following works for string, None, or a colormap instance:

  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)

def plot_vehicle_routes_TOP(data, route, max_vehicle_num, ax1, markersize=5, prize_scale=1, round_prize=False, isMILP=False, prob_no=0, perm_i=None):
    """
    Plot the vehicle routes on matplotlib axis ax1.
    """
    K = max_vehicle_num
    if not isMILP: 
        # route is one sequence, separating different routes with 0 (depot), and removes the last zeros 
        routes = [r[r!=0] for r in np.split(route.cpu().numpy(), np.where(route==0)[0]) if (r != 0).any()]
        # routes = [array([15, 20, 17,  5, 16, 11], dtype=int64), array([14,  2,  1,  4, 12, 18], dtype=int64), array([ 6, 19, 13,  7], dtype=int64), array([ 8,  3, 10,  9], dtype=int64)]
        
        # TO DO:  straight-to-depot route 처리 for 1st vehicle
        if route[0] == 0: 
              routes.insert(0, []) #
    else:
        # Ex. route = [[0, 2, 1, 4, 3, 10, 8, 9, 12, 14], [0, 11, 16, 5, 6, 19, 17, 20, 15]]
        # 0818: route = [[0, 15, 20, 17, 19, 6, 13, 7, 18, 21], [0, 11, 5, 16, 14, 2, 4, 1, 3, 10, 9, 12, 8, 21]]
        # reverse route!
        routes = [route[k][::-1][1:-1] for k in range(K)]
        
    # TODO: straight-to-depot route 처리 for 2nd vehicle
    if len(routes) < K:
        for k in range(K-len(routes)):
            routes.append([]) #np.zeros(1, dtype=np.int64)
    
    depot = data['depot'].cpu().numpy()
    locs = data['loc'].cpu().numpy()
    prizes = data['prize'].cpu().numpy() 
    max_length = data['max_length'].cpu().numpy()
    cur_loc = data['cur_loc'].cpu().numpy()
    cur_tlen = data['cur_tlen'].cpu().numpy()
    capacity = prize_scale # Capacity is always 1
    
    # cur_loc and cur_tlen according to permute
    if not isMILP:
        permute = list(permutations(range(0,K)))
        cur_loc = cur_loc[permute[perm_i],:]
        cur_tlen = cur_tlen[None,:][:,permute[perm_i]].reshape(K)
        
    # plot depot
    x_dep, y_dep = depot
    ax1.plot(x_dep, y_dep, 'sk', markersize=markersize*4)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # remove x,y axis ticks 
    x_axis = np.arange(0,1.01, 0.5)
    y_axis = np.arange(0,1.01, 0.5)
    ax1.set_xticks(x_axis)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.set_yticks(y_axis)
    ax1.tick_params(axis='y', labelsize=14)
    
    # plot nodes
    ax1.plot(locs[:,0], locs[:,1], 'sb', markersize=markersize*2)
    legend = ax1.legend(loc='best')
    
    # 0817: plot cur_loc 
    x_cur_loc, y_cur_loc = cur_loc[:,0], cur_loc[:,1]
    ax1.plot(x_cur_loc, y_cur_loc, 'sr', markersize=markersize*2)
    
    # 0817: plot cur_tlen->remaining fuel and use adjustText
    texts = [ax1.text(cur_loc[k][0], cur_loc[k][1], f'{max_length-cur_tlen[k]:.2f}', fontsize=14) for k in range(len(cur_loc))]
    adjust_text(texts, expand_points=(1.4, 1.4))
    
    # plot prizes and use adjustText
    if not np.all(prizes == prizes[0]):
        texts = [ax1.text(locs[k][0], locs[k][1], f'{prizes[k]:.2f}', fontsize=14) for k in range(len(locs))]
        adjust_text(texts, expand_points=(1.4, 1.4))
    
    cmap = discrete_cmap(len(routes) + 2, 'nipy_spectral')
    dem_rects = []
    used_rects = []
    cap_rects = []
    qvs = []
    total_dist = 0
    total_prize = 0
    for veh_number, r in enumerate(routes):
        
        # veh_number = 0, r = array([15, 20, 17,  5, 16, 11], dtype=int64)
        # veh_number = 1, r = array([14,  2,  1,  4, 12, 18], dtype=int64)
        # veh_number = 2, r = array([ 6, 19, 13,  7], dtype=int64)
        # veh_number = 3, r = array([ 8,  3, 10,  9], dtype=int64)
        
        color = cmap(len(routes) - veh_number) # Invert to have in rainbow order
        
        
        # 기존: route_prizes = prizes[r - 1] # node number: 1~20 
        # 0925 TODO: straight-to-depot route 처리 추가 
        r_idx = [x-1 for x in r]
        route_prizes = prizes[r_idx]
        coords = locs[r_idx, :]
        coords = np.concatenate((cur_loc[veh_number].reshape(1,2),coords,(depot.reshape(1,2))))
        xs, ys = coords.transpose() # x좌표, y좌표 따로 

        total_route_prize = sum(route_prizes)
        total_prize += total_route_prize
        # assert total_route_demand <= capacity (no need)
        #if not visualize_prizes:
        #    ax1.plot(xs, ys, 'o', mfc=color, markersize=markersize, markeredgewidth=0.0)
        
        dist = 0
        x_prev, y_prev = x_cur_loc[veh_number], y_cur_loc[veh_number]
        cum_prize = 0
        for (x, y), p in zip(coords, route_prizes):
            dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)
            
            cap_rects.append(Rectangle((x, y), 0.01, 0.1))
            used_rects.append(Rectangle((x, y), 0.01, 0.1 * total_route_prize))
            dem_rects.append(Rectangle((x, y + 0.1 * cum_prize), 0.01, 0.1 * p))
            
            x_prev, y_prev = x, y
            cum_prize += p
            
        dist += np.sqrt((x_dep - x_prev) ** 2 + (y_dep - y_prev) ** 2)
        total_dist += dist
        qv = ax1.quiver(
            xs[:-1],
            ys[:-1],
            xs[1:] - xs[:-1],
            ys[1:] - ys[:-1],
            scale_units='xy',
            angles='xy',
            scale=1,
            color=color,
            label='Vehicle {}. Prizes: {:.2f}, Avail. fuel: {:.2f}, Re-planned dist: {:.2f}'.format(
                veh_number, 
                int(total_route_prize) if round_prize else total_route_prize, 
                max_length-cur_tlen[veh_number],
                dist,
                # dist + cur_tlen[veh_number]
            )
        )
        
        qvs.append(qv)
    
    if not isMILP:
        ax1.set_title('Deep Dynamic Attention Model (DDTM) Result for Prob # {}: {} routes'.format(prob_no, len(routes)), fontsize=24)
        if not round_prize: # unif dist
                # ax1.set_title('DDTM Result for Prob # {}: {} routes, total prize = {:.2f}'.format(prob_no, len(routes), total_prize), fontsize=20)
                ax1.set_title('DDTM Total prize: {:.2f}'.format(total_prize), fontsize=24)

    else:
        ax1.set_title('MILP Result for Prob # {}: {} routes'.format(prob_no, len(routes)), fontsize=24)
        if not round_prize: # unif dist
                # ax1.set_title('MILP Result for Prob # {} : {} routes, total prize = {:.2f}'.format(prob_no, len(routes), total_prize), fontsize=20)
                ax1.set_title('MILP Total prize: {:.2f}'.format(total_prize), fontsize=24)

    ax1.legend(fontsize=16, handles=qvs)
    
    pc_cap = PatchCollection(cap_rects, facecolor='whitesmoke', alpha=1.0, edgecolor='lightgray')
    pc_used = PatchCollection(used_rects, facecolor='lightgray', alpha=1.0, edgecolor='lightgray')
    pc_dem = PatchCollection(dem_rects, facecolor='black', alpha=1.0, edgecolor='black')

#%% Load model 
aug_factor = 8
model, _ = load_model('pretrained/UNIF/20_new/E/') 
prize_distribution = 'unif'
torch.manual_seed(1234)
max_vehicle_num = 2
graph_size = 20     
num_samples = 10000
batch_size = 1000
dataset = TOP.make_dataset(size=graph_size, num_veh=max_vehicle_num, num_samples=num_samples, distribution=prize_distribution)
round_prize = False; 

# Clear GPU cache
torch.cuda.empty_cache()

# Setup device
device = torch.device("cuda:0" if USE_CUDA else "cpu")

# Send model to GPU
model.to(device)

# Set the model as evaluation mode
model.eval()

# Set the decode type: greedy or sampling
model.set_decode_type('greedy')

# Set dataloader to batch instances
dataloader = DataLoader(dataset, batch_size=batch_size)

# Store results 
results = []
T = 0
for batch in tqdm(dataloader):
    
    # Send batch to GPU c.f. 'dict' object has no attribute 'to'
    batch = move_to(batch, device)

    num_veh = batch['num_veh'][0]
    permute = list(permutations(range(0,num_veh))) # len(permute) is np.math.factorial(num_veh)

    # Instance augmentation (aug_factor*num_veh!)
    if aug_factor > 1: 
        batch['depot'] = augment_xy_data_by_8_fold(batch['depot'][:,None]).squeeze(1).repeat(np.math.factorial(num_veh), 1)
        batch['loc'] = augment_xy_data_by_8_fold(batch['loc']).repeat(np.math.factorial(num_veh), 1, 1)
        batch['cur_loc'] = augment_xy_data_by_8_fold(batch['cur_loc']).repeat(np.math.factorial(num_veh), 1, 1)
        batch['cur_tlen'] = batch['cur_tlen'].repeat(aug_factor*np.math.factorial(num_veh), 1)
        batch['prize'] = batch['prize'].repeat(aug_factor*np.math.factorial(num_veh), 1)
        batch['num_veh'] = batch['num_veh'].repeat(aug_factor*np.math.factorial(num_veh))
        batch['max_length'] = batch['max_length'].repeat(aug_factor*np.math.factorial(num_veh))
        
        # Check cur_loc and cur_tlen
        chunk_cur_loc = list(batch['cur_loc'].chunk(np.math.factorial(num_veh), dim=0))
        chunk_cur_tlen = list(batch['cur_tlen'].chunk(np.math.factorial(num_veh), dim=0))
        
        for i in range(len(permute)-1): # swap cur_loc and cur_tlen
            perm = list(permute[i+1])
            chunk_cur_loc[i+1] = chunk_cur_loc[i+1][:, perm]
            chunk_cur_tlen[i+1] = chunk_cur_tlen[i+1][:,perm]
            
        batch['cur_loc'] = torch.cat(chunk_cur_loc, dim=0)
        batch['cur_tlen'] = torch.cat(chunk_cur_tlen, dim=0)
        # batch now has 8*np.math.factorial(num_veh) instances

    # Run the model
    start = time.time()
    with torch.no_grad():
        reward, log_p, pi, entropy = model(batch, return_pi=True)
    end = time.time()
    AM_time = (end-start)/batch_size
    
    # Get best results from augmentation
    aug_reward = reward.cpu().reshape(aug_factor*np.math.factorial(num_veh), batch_size)
    max_aug_reward, max_reward_idx = aug_reward.min(dim=0) # note negative rewards
    aug_tours = pi.cpu().reshape(aug_factor*np.math.factorial(num_veh), batch_size, -1)
    max_tours = aug_tours[max_reward_idx[:,None], torch.arange(batch_size)[:,None], :].squeeze(1) # (1000,30)
    wo_aug_reward = aug_reward[0]
    if aug_tours.size(-1) >  T:
        T = aug_tours.size(-1)
    
    # Update results 
    results.append((max_aug_reward, max_reward_idx, max_tours, wo_aug_reward, AM_time))
    
# Stack all tours
max_aug_reward, max_reward_idx, max_tours, wo_aug_reward, AM_time = zip(*results)
max_reward_idx = torch.stack(list(max_reward_idx)).reshape(num_samples)
final_reward = torch.stack(max_aug_reward).reshape(num_samples)
final_tours = []
for i in range(len(AM_time)):
    t = T - max_tours[i].size(-1)
    final_tours.append(F.pad(input=max_tours[i], pad=(0,t,0,0), mode='constant', value=0))
final_tours = torch.stack(final_tours).reshape(num_samples, -1)
wo_aug_reward = torch.stack(wo_aug_reward).reshape(num_samples)
AM_time = np.asarray(AM_time).mean()

#%% Plot Results
MILP_timelog = np.zeros(num_samples)
MILP_costlog = np.zeros(num_samples)

for i, (data, tour, max_idx) in enumerate(zip(dataset, final_tours, max_reward_idx)):
        fig = plt.figure(figsize=(22,8))
       
        # AM results 
        ax = fig.add_subplot(1,2,1)
        plot_vehicle_routes_TOP(data, tour, max_vehicle_num, ax, prize_scale=50, 
                                round_prize=round_prize, isMILP=False, prob_no=i,
                                perm_i=torch.div(max_idx, aug_factor, rounding_mode='trunc'))
        
        # MILP results
        ax = fig.add_subplot(1,2,2)
        depot = data['depot'].numpy()
        loc = data['loc'].numpy()
        prize = data['prize'].numpy()
        max_length = data['max_length'].numpy()
        cur_loc = data['cur_loc'].numpy()
        cur_tlen = data['cur_tlen'].numpy()
        start = time.time()
        cost, GRBtour = solve_euclidian_dtop_MILP(depot, loc, prize, max_length, cur_loc, cur_tlen, max_vehicle_num=2, threads=16)
        plot_vehicle_routes_TOP(data, GRBtour, max_vehicle_num, ax, prize_scale=50,
                                round_prize=round_prize, isMILP=True, prob_no=i)
        duration = time.time() - start  # Measure clock time
        MILP_timelog[i] = duration # Save MILP time
        MILP_costlog[i] = cost
        plt.show()
        fig.savefig(os.path.join('images', f'MSTOP_{prize_distribution}_{i}.svg'), bbox_inches="tight")
    
MILP_time = np.mean(MILP_timelog)
MILP_cost = -np.mean(MILP_costlog)
AM_cost_wo_aug = np.mean(-wo_aug_reward.numpy())
AM_cost_w_aug = np.mean(-final_reward.numpy())
print(f"Time taken for MILP to solve {num_samples} MSTOP problems on average is {MILP_time:.5f} sec")
print(f"Average cost for MILP is {MILP_cost:.5f}")
print(f"Time taken for Deep Dynamic Attention Model to solve {num_samples} MSTOP problems on average is {AM_time:.5f} sec")
print(f"Average cost for Deep Dynamic Attention Model (DDTM) with original set is {AM_cost_wo_aug:.5f} with opt. gap {(MILP_cost-AM_cost_wo_aug)/MILP_cost * 100:.3f}%")
print(f"Average cost for Deep Dynamic Attention Model (DDTM) with augmented set is {AM_cost_w_aug:.5f} with opt. gap {(MILP_cost-AM_cost_w_aug)/MILP_cost * 100:.3f}%")
MILPmdic = {"MILP_costlog":MILP_costlog, "MILP_timelog": MILP_timelog}
savemat("MILP_Result.mat", MILPmdic)
AMmdic = {"AM_costlog":final_reward.numpy(), "AM_time":AM_time}
savemat("AM_Result.mat", AMmdic)

#%% Do scatterplot
prize_distribution = 'unif'
graph_size=20

# Ram's colors
seshadri = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
#            0sangre,   1neptune,  2pumpkin,  3clover,   4denim,    5cocoa,    6cumin,    7berry


# Color Match
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def addlabels(x,y):
    for i in range(len(x)):
    #    plt.text(i, y[i], y[i], ha = 'center',
    #         Bbox = dict(facecolor = 'white', alpha =.8))
        plt.annotate(y[i], (i, y[i]+1000), ha="center", 
                     bbox = dict(boxstyle="square", fc='white', alpha=1.0))

# Load data 
AMmdic = loadmat('AM_Result.mat')
MILPmdic = loadmat('MILP_Result.mat')
AM_costlog = AMmdic['AM_costlog']
MILP_costlog = MILPmdic['MILP_costlog']


if prize_distribution == 'const':
    # Compute difference (int)
    AM_costlog = np.around(AM_costlog).astype(int)
    MILP_costlog = np.around(MILP_costlog).astype(int)
    MILP_costlog = -MILP_costlog
    AM_costlog = -AM_costlog
    
    # Compute difference
    diff = (MILP_costlog - AM_costlog).reshape(-1)
    
    tally = np.zeros((4,1))
    tally[0] = sum(diff<=0) 
    tally[1] = sum(diff==1) 
    tally[2] = sum(diff==2) 
    tally[3] = sum(diff>2) 
    
    # Normalize tally
    min_tally = min(tally)
    max_tally = max(tally)
    tally_factor = (tally-min_tally) / (max_tally-min_tally)
    tally_factor = tally_factor*50+1
    
    sz = np.zeros((len(diff),1))
    for i in range(len(diff)):
        if diff[i] > 3:
            diff[i] = 3
        if diff[i] < 0: # TODO: How to treat this case?
            diff[i] = 0
        sz[i] = tally_factor[diff[i]]
    
    # Size of scatter according to true tally
    for i in range(0,graph_size+1): #0~20 for MILP_costlog
        GRBidx = np.where(MILP_costlog[0] == i)
        AMval = AM_costlog[0][GRBidx]
        for j in set(AMval): # {4,5,6}
            cnt = np.count_nonzero(AMval == j) 
            idx = np.where(AMval == j)
            sz[GRBidx[0][idx]] = cnt
            
    # Normalize sizes for scatter plot [1~100]
    sz = (sz-min(sz) + 1) / (max(sz) - min(sz) + 1) * 100 
    
    # Color map
    cmap = plt.get_cmap('viridis')
    new_cmap = truncate_colormap(cmap, 0.0, 0.75, n=1000)

    # Prepare multipanel plot
    fig = plt.figure(num=1, figsize=(5,7)) # default (6.4, 4.8)
    gs = gridspec.GridSpec(nrows=7, ncols=5)
    gs.update(hspace=8.0)
    
    # Generate first panel
    axes1 = fig.add_subplot(gs[0:5, 0:5])
    axes1.plot([0, graph_size], [0, graph_size], color='red', linestyle='--', linewidth=2.0, transform=axes1.transAxes, zorder=1) 
    scatter = axes1.scatter(x=MILP_costlog, y=AM_costlog, c=diff, s=sz,
                            cmap=new_cmap, alpha=0.8, linewidths=1.5, zorder=0)
    classes = [r'$0$', '1', '2', r'$\geq3$']
    legend1 = axes1.legend(handles=scatter.legend_elements()[0], labels=classes, loc="center left", bbox_to_anchor=(0.6, 0.3), fontsize=10, ncol=1, title=r'$\Delta p$', frameon=False)
    #kw = dict(prop="sizes", num=tally.astype(int), color='black', fmt="{x:2.0f}", func=lambda s: (s-1.0)/50.0 * (max_tally-min_tally) + min_tally) 
    #legend2 = axes1.legend(*scatter.legend_elements(**kw), loc="lower right", fontsize=14, title=r'Frequency')
    #axes1.add_artist(legend1) # manually add legend1 
    axes1.set_xlabel('MILP', fontsize=10)
    axes1.set_ylabel('DDTM', fontsize=10)
    ticks = np.arange(0, graph_size+1, 5) # 0, 5, 15, 20, ...
    axes1.minorticks_on()
    axes1.tick_params(direction='in', which='minor', length=5, 
                      bottom=True, top=True, left=True, right=True)
    axes1.tick_params(direction='in', which='major', length=10, 
                      bottom=True, top=True, left=True, right=True)
    axes1.set_xticks(ticks)
    axes1.set_yticks(ticks)
    # plt.tick_params(axis='both', which='major', labelsize=8)
    axes1.set_xlim(0, graph_size+1)
    axes1.set_ylim(0, graph_size+1)

    # Generate second panel 
    axes2 = fig.add_subplot(gs[5:7, 0:5])
    new_color = [new_cmap(i) for i in np.linspace(0,1.0,len(classes))]    
    axes2.bar(x=classes, height=tally.reshape(-1), color=np.asarray(new_color))
    addlabels(x=classes, y=tally.reshape(-1).astype(int))
    axes2.minorticks_off()
    axes2.tick_params(direction='in', which='major', length=5, 
                      bottom=True, top=False, left=True, right=False)
    xticks = np.arange(0, 4)
    yticks = np.arange(0, 10001, 5000)
    axes2.set_xticks(xticks)
    axes2.set_yticks(yticks)
    axes2.set_ylim(bottom=0, top=num_samples+1000)
    axes2.set_xlabel(r'$\Delta p$', fontsize=10)
    #axes2.set_ylabel('Frequency')
    #axes2.yaxis.set_label_coords(0.5, 0.5)
    
    # savefig
    fig.set_size_inches(3.4, 5)
    plt.savefig('Summary of Result.svg', bbox_inches="tight")
    plt.savefig('Summary of Result.png', dpi=600, bbox_inches="tight")
    
else: # 'unif' 
    MILP_costlog = -MILP_costlog.reshape(-1)
    AM_costlog = -AM_costlog.reshape(-1)
    # In case MILP sol = 0
    MILP_costlog = np.where(MILP_costlog==0, 0.001, MILP_costlog)
    AM_costlog = np.where(AM_costlog==0, 0.001, AM_costlog)
    # Compute difference
    diff = (MILP_costlog - AM_costlog) / MILP_costlog # relative percentage difference

    tally = np.zeros((4,1)) 
    tally[0] = np.count_nonzero(diff <= 0.05) 
    tally[1] = np.count_nonzero(np.logical_and(diff > 0.05, diff <= 0.10)) 
    tally[2] = np.count_nonzero(np.logical_and(diff > 0.10, diff <= 0.15))
    tally[3] = np.count_nonzero(diff > 0.15)
    
    # Normalize tally
    min_tally = min(tally)
    max_tally = max(tally)
    tally_factor = (tally-min_tally) / (max_tally-min_tally)
    tally_factor = tally_factor*50+1
    
    # (,5%] and (15%,) 
    diff[diff<0] = 0
    diff[diff>0.15] = 0.151
    
    # Color map
    cmap = plt.get_cmap('viridis')
    new_cmap = truncate_colormap(cmap, 0.0, 0.75, n=1000) #(cmap, 0.1, 0.7, n=1000)

    # Prepare multipanel plot
    fig = plt.figure(num=1, figsize=(5,7)) # default (6.4, 4.8)
    gs = gridspec.GridSpec(nrows=7, ncols=5)
    gs.update(hspace=8.0)
    
    # Generate first panel
    axes1 = fig.add_subplot(gs[0:5, 0:5])
    axes1.plot([0, max(MILP_costlog)+1], [0, max(MILP_costlog)+1], color='red', linestyle='--', linewidth=2.0, transform=axes1.transAxes, zorder=1) 
    scatter = axes1.scatter(x=MILP_costlog, y=AM_costlog, c=diff, edgecolors='None',
                            cmap=new_cmap, alpha=0.8, linewidths=1.5, zorder=0)
    classes = [r'$[ , 5]$', r'$(5, 10]$', r'$(10, 15]$', r'$(15 , )$']
    # labels = [f'{item}: {count}' for item, count in Counter(diff).items()]
    #legend1 = axes1.legend(*scatter.legend_elements(), loc="best", ncol=1, title='$\Delta p_{rel}  (\%)$')
    legend1 = axes1.legend(handles=[scatter.legend_elements()[0][idx] for idx in [0,3,5,7]], labels=classes, loc="center left", bbox_to_anchor=(0.55, 0.3), fontsize=10, ncol=1, title=r'$\Delta p_{rel}  (\%)$', frameon=False)
    axes1.set_xlabel('MILP', fontsize=10)
    axes1.set_ylabel('DDTM', fontsize=10)
    ticks = np.arange(0, max(MILP_costlog)+1, 2) # 0, 2, 4, 6, ...
    axes1.minorticks_on()
    axes1.tick_params(direction='in', which='minor', length=5, 
                      bottom=True, top=True, left=True, right=True)
    axes1.tick_params(direction='in', which='major', length=10, 
                      bottom=True, top=True, left=True, right=True)
    axes1.set_xticks(ticks)
    axes1.set_yticks(ticks)
    axes1.set_xlim(0, max(MILP_costlog)+1)
    axes1.set_ylim(0, max(MILP_costlog)+1)

    # Generate second panel 
    axes2 = fig.add_subplot(gs[5:7, 0:5])
    new_color = [new_cmap(i) for i in np.linspace(0,1.0,len(classes))]    
    axes2.bar(x=classes, height=tally.reshape(-1), color=np.asarray(new_color))
    #axes2.tick_params(axis='x', labelsize=12)
    addlabels(x=classes, y=tally.reshape(-1).astype(int))
    axes2.minorticks_off()
    axes2.tick_params(direction='in', which='major', length=5, 
                      bottom=True, top=False, left=True, right=False)
    xticks = np.arange(0, 4)
    yticks = np.arange(0, 10001, 5000)
    axes2.set_xticks(xticks)
    axes2.set_yticks(yticks)
    axes2.set_ylim(bottom=0, top=num_samples+1000)
    axes2.set_xlabel(r'$\Delta p_{rel} (\%)$', fontsize=10)
    # axes2.set_ylabel('Frequency')
    # axes2.yaxis.set_label_coords(-0.2, 0.5)
    
    # savefig
    fig.set_size_inches(3.4, 5)
    plt.savefig('Summary of Result.svg', bbox_inches="tight")
    plt.savefig('Summary of Result.png', dpi=600, bbox_inches="tight")   

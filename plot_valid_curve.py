import json
import os
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from utils.functions import json_processing, smooth
import matplotlib.ticker as ticker

# https://dailyheumsi.tistory.com/97 pyplot 그래프의 범주박스 위치 변경하기
# https://www.bastibl.net/publication-quality-plots/ plot IEEE figures
# IEEE Figsize: one column wide (3.5 in) or one page wide (7.16), maximum depth of 8.5, allow space for caption

#%% 
n_epochs = 200
x = np.arange(n_epochs)
max_step = 500000
ylim_l = 10.3
ylim_u = 11.4
width = 3.487 # for one column fig
height = width / 1.618 
caption_size = 8 
main_font_size = 8
linewidth=1.5
linestyle=(0, (3,1,1,1))

#%%
import Preset
Preset.setPlotStyle()
# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=caption_size)
# plt.rc('ytick', labelsize=caption_size)
# plt.rc('axes', labelsize=caption_size)
# plt.rcParams['legend.title_fontsize'] = main_font_size

#%% Load json data for validation score
path_to_json = 'results/top/CONST/20/validation_score_compare_baselines/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
valid_data = json_processing(path_to_json, n_epochs)
print(json_files) 

#%% Load json data for validation score
path_to_json = 'results/top/CONST/20/validation_score_batch_size_epoch/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
valid_data_epoch = json_processing(path_to_json, n_epochs)
print(json_files) 

path_to_json = 'results/top/CONST/20/validation_score_batch_size_step/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
valid_data_step = np.zeros((len(json_files), n_epochs)) # 200 epochs
valid_data_k = np.zeros((len(json_files), n_epochs)) # 200 epochs
for idx, json_obj in enumerate(json_files):
    with open(os.path.join(path_to_json, json_obj)) as file:
        valid_score = np.array(json.load(file)) # nested list to array, (200,3)
        valid_data_k[idx,:] = valid_score[:,1]
        valid_data_step[idx,:] = valid_score[:,2]        
print(json_files) 

#%% Plot Validation score  (find critical batch size)
#plt.style.use(['science', 'notebook']) #plt.style.use(['science','ieee'])
#fig = plt.figure(figsize=(10,5))

# 1. Per (gradient) step
fig, ax1 = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
plt1, = ax1.plot(valid_data_k[0], -valid_data_step[0], c='tab:orange', linestyle='solid', linewidth=linewidth, label='1024')
plt2, = ax1.plot(valid_data_k[1], -valid_data_step[1], c='tab:purple', linestyle='solid', linewidth=linewidth, label='2048')
plt3, = ax1.plot(valid_data_k[2], -valid_data_step[2], c='tab:green', linestyle='solid', linewidth=linewidth, label='4096')
plt4, = ax1.plot(valid_data_k[3], -valid_data_step[3], c='tab:blue', linestyle='solid', linewidth=linewidth, label='512')


ax1.legend(handles=[plt4, plt1, plt2, plt3],  loc="best", fontsize=caption_size, ncol=1, title='Batch Size', edgecolor='k')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Validation score')
ax1.minorticks_on()
ax1.set_xlim(-10000, max_step+10000)
ax1.set_ylim(ylim_l, ylim_u)
xticks1 = np.arange(0, max_step+10, 50000) # 0k, 50k, 150k, 200k
ax1.set_xticks(xticks1)
ax1.tick_params(direction='in', which='minor', bottom=False, left=False)
ax1.tick_params(direction='in', which='major', length=2.5, 
                bottom=True, top=False, left=True, right=False)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x/1000)) + 'k'))
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

# Save fig
fig.set_size_inches(width, height)
plt.savefig('results/top/CONST/20/validation_score_batch_size_step/Comparison of Different Batch Sizes_per_step.png', dpi=600, bbox_inches="tight")
plt.savefig('results/top/CONST/20/validation_score_batch_size_step/Comparison of Different Batch Sizes_per_step.pdf', bbox_inches="tight")
plt.savefig('results/top/CONST/20/validation_score_batch_size_step/Comparison of Different Batch Sizes_per_step.svg', bbox_inches="tight")

# 2. Per epoch
fig, ax2 = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

plt5, = ax2.plot(x, -valid_data_epoch[0], c='tab:orange', linestyle='solid', linewidth=linewidth, label='1024')
plt6, = ax2.plot(x, -valid_data_epoch[1], c='tab:purple', linestyle='solid', linewidth=linewidth, label='2048')
plt7, = ax2.plot(x, -valid_data_epoch[2], c='tab:green', linestyle='solid', linewidth=linewidth, label='4096')
plt8, = ax2.plot(x, -valid_data_epoch[3], c='tab:blue', linestyle='solid', linewidth=linewidth, label='512')


ax2.legend(handles=[plt8, plt5, plt6, plt7], loc="best", fontsize=caption_size, ncol=1, title='Batch Size', edgecolor='k')
ax2.set_xlabel('Epochs') # [500k,250k, 125k, 62.4k]
ax2.set_ylabel('Validation score')
ax2.minorticks_on()
ax2.set_xlim(-10, n_epochs+10)
ax2.set_ylim(ylim_l, ylim_u)
xticks2 = np.arange(0, n_epochs+1, 50) # 0, 50, 150, 200
ax2.set_xticks(xticks2)
ax2.tick_params(direction='in', which='minor', bottom=False, left=False)
ax2.tick_params(direction='in', which='major', length=2.5, 
                bottom=True, top=False, left=True, right=False)

# Common legend for all subplots
# lines, labels = [], []
# for ax in fig.axes:
#     Line, Label = ax.get_legend_handles_labels()
#     lines.extend(Line)
#     labels.extend(Label)
# fig.legend(lines, labels, loc='center right', title='Batch Size')

# savefig
fig.set_size_inches(width, height)
plt.savefig('results/top/CONST/20/validation_score_batch_size_epoch/Comparison of Different Batch Sizes_per_epoch.png', dpi=600, bbox_inches="tight")
plt.savefig('results/top/CONST/20/validation_score_batch_size_epoch/Comparison of Different Batch Sizes_per_epoch.pdf', bbox_inches="tight")
plt.savefig('results/top/CONST/20/validation_score_batch_size_epoch/Comparison of Different Batch Sizes_per_epoch.svg', bbox_inches="tight")

#%% Load json data for validation score
path_to_json = 'results/top/CONST/20/validation_score_inst_aug/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
valid_data = json_processing(path_to_json, n_epochs)
print(json_files) 

#%%
# Plot Validation score  (training with and without instance augmentation)
# Full View + Zoom

# 1. Batch size 1024
fig, ax3 = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

plt1, = ax3.plot(x, -valid_data[0], c='tab:blue', linestyle='solid', linewidth=linewidth, label='No augmentation (1024)')
plt2, = ax3.plot(x, -valid_data[2], c='tab:orange', linestyle='solid', linewidth=linewidth, label='x8 augmentation (128)')

ax3.legend(handles=[plt2, plt1], loc='center', bbox_to_anchor=(0.5, 0.22), fontsize=caption_size, ncol=1)
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Validation score')
ax3.minorticks_on()
xticks = np.arange(0, n_epochs+1, 50) # 0, 50, 150, 200
ax3.tick_params(direction='in', which='minor', bottom=False, left=False)
ax3.tick_params(direction='in', which='major', length=2.5, 
                bottom=True, top=False, left=True, right=False)
ax3.set_xticks(xticks)
ax3.set_xlim(-10, n_epochs+10)

# Add child inset axes (https://codetorial.net/matplotlib/add_inset_graph.html)
axins = ax3.inset_axes([0.2, 0.45, 0.6, 0.3]) # [x0,y0,width,height]
axins.plot(x[100:], -valid_data[0][100:], 'tab:orange', x[100:], -valid_data[2][100:], 'tab:blue', linestyle='solid', linewidth=linewidth)
axins.set_xticklabels('')
axins.set_yticklabels('')
axins.tick_params(left=False, right=False, top=False, bottom=False)
axins.set_xticks([])
axins.set_yticks([])
for axis in ['top', 'bottom', 'left', 'right']:
  axins.spines[axis].set_linewidth(1.5)
  # axins.spines[axis].set_color('r')

#ax1.indicate_inset_zoom(axins)
indicator = ax3.indicate_inset_zoom(axins)
# indicator[0].set_linewidth(3)
# indicator[0].set_edgecolor('r')
# indicator[0].set_color('r')
indicator[1][0].set_linewidth(1)
indicator[1][1].set_linewidth(1)
indicator[1][2].set_linewidth(1)
indicator[1][3].set_linewidth(1)
# indicator[1][1].set_color('r')
# indicator[1][2].set_color('r')

# Save fig
fig.set_size_inches(width, height)
plt.savefig('results/top/CONST/20/validation_score_inst_aug/Comparison of Training with Instance Augmentation_bs128.png', dpi=600, bbox_inches="tight")
plt.savefig('results/top/CONST/20/validation_score_inst_aug/Comparison of Training with Instance Augmentation_bs128.pdf', bbox_inches="tight")
plt.savefig('results/top/CONST/20/validation_score_inst_aug/Comparison of Training with Instance Augmentation_bs128.svg', bbox_inches="tight")

# 2. Batch size 2048
# fig, ax2 = plt.subplots()
# fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

# plt3, = ax2.plot(x, -valid_data[1], c='tab:purple', linestyle='solid', linewidth=linewidth, label='No augmentation (2048)')
# plt4, = ax2.plot(x, -valid_data[3], c='tab:green', linestyle='solid', linewidth=linewidth, label='x8 augmentation (256)')

# ax2.legend(handles=[plt4, plt3], loc='center', bbox_to_anchor=(0.5, 0.22), fontsize=caption_size, ncol=1)
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Validation score')
# ax2.minorticks_on()
# xticks = np.arange(0, n_epochs+1, 50) # 0, 50, 150, 200
# ax2.tick_params(direction='in', which='minor', bottom=False, left=False)
# ax2.tick_params(direction='in', which='major', length=2.5, 
#                 bottom=True, top=False, left=True, right=False)
# ax2.set_xticks(xticks)
# ax2.set_xlim(-10, n_epochs+10)

# # Add child inset axes (https://codetorial.net/matplotlib/add_inset_graph.html)
# axins = ax2.inset_axes([0.2, 0.45, 0.6, 0.3]) # [x0,y0,width,height]
# axins.plot(x[100:], -valid_data[1][100:], 'tab:purple', x[100:], -valid_data[3][100:], 'tab:green', linestyle='solid', linewidth=linewidth)
# axins.set_xticklabels('')
# axins.set_yticklabels('')
# axins.tick_params(left=False, right=False, top=False, bottom=False)
# axins.set_xticks([])
# axins.set_yticks([])
# for axis in ['top', 'bottom', 'left', 'right']:
#   axins.spines[axis].set_linewidth(1.5)

# indicator = ax2.indicate_inset_zoom(axins)
# indicator[1][0].set_linewidth(1)
# indicator[1][1].set_linewidth(1)
# indicator[1][2].set_linewidth(1)
# indicator[1][3].set_linewidth(1)

# # Save fig
# fig.set_size_inches(width, height)
# plt.savefig('results/top/CONST/20/validation_Score_inst_aug/Comparison of Training with Instance Augmentation_bs256.png', dpi=600, bbox_inches="tight")
# plt.savefig('results/top/CONST/20/validation_Score_inst_aug/Comparison of Training with Instance Augmentation_bs256.pdf', bbox_inches="tight")
# plt.savefig('results/top/CONST/20/validation_Score_inst_aug/Comparison of Training with Instance Augmentation_bs256.svg', bbox_inches="tight")


#%% Compare baselines for MSTOP_CONST
path_to_json = 'results/top/CONST/20/validation_score_compare_baselines/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
valid_data = json_processing(path_to_json, n_epochs)
print(json_files) 

# A. Greedy Rollout Baseline only
# B. Greedy Rollout Baseline + max. entropy regularisation
# C. Multiple samples with replacement baseline (local baseline)
# D. x8 instance augmentation baseline (local baseline)
# E. x8 instance augmentation baseline + max. entropy regularisation

# Original (transparent) + Smoothed 
fig, ax4 = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

plt1, = ax4.plot(x, -valid_data[0], c='tab:green', linestyle=linestyle, linewidth=linewidth, alpha = 0.2, label='A. Greedy Rollout')
plt2, = ax4.plot(x, -valid_data[4], c='tab:blue', linestyle=linestyle, linewidth=linewidth, alpha = 0.2, label='B. Greedy Rollout + Entropy')
#plt3, = ax4.plot(x, -valid_data[2], c='tab:brown', linestyle='-', linewidth=linewidth, alpha = 0.2, label='Samples with Replacement')
plt4, = ax4.plot(x, -valid_data[1], c='tab:purple', linestyle='-', linewidth=linewidth, alpha = 0.2, label='C. Proposed Baseline')
plt5, = ax4.plot(x, -valid_data[3], c='tab:orange', linestyle='-', linewidth=linewidth, alpha = 0.2, label='D. Proposed Baseline + Entropy')

plt6, = ax4.plot(x, smooth(-valid_data[0]), c='tab:green', linestyle=linestyle, linewidth=linewidth, label='A. Greedy Rollout')
plt7, = ax4.plot(x, smooth(-valid_data[4]), c='tab:blue', linestyle=linestyle, linewidth=linewidth, label='B. Greedy Rollout + Entropy')
#plt8, = ax4.plot(x, smooth(-valid_data[2]), c='tab:brown', linestyle='-', linewidth=linewidth, label='Samples with Replacement')
plt9, = ax4.plot(x, smooth(-valid_data[1]), c='tab:purple', linestyle='-', linewidth=linewidth, label='C. Proposed Baseline')
plt10, = ax4.plot(x, smooth(-valid_data[3]), c='tab:orange', linestyle='-', linewidth=linewidth, label='D. Proposed Baseline + Entropy')

# ax4.legend(handles=[plt10, plt9, plt7, plt6], loc='center', bbox_to_anchor=(0.5, -.40), fontsize=caption_size, ncol=2)
ax4.legend(handles=[plt6, plt7, plt9, plt10], loc='best', fontsize=caption_size, ncol=1, frameon=False)

ax4.set_xlabel('Epochs')
ax4.set_ylabel('Validation score')
ax4.minorticks_on()
xticks = np.arange(0, n_epochs+1, 50) # 0, 50, 150, 200
yticks = np.arange(10.9, 11.31, 0.10)
ax4.tick_params(direction='in', which='minor', length=2.0,
                bottom=True, top=True, left=True, right=True)
ax4.tick_params(direction='in', which='major', length=4.0, 
                bottom=True, top=True, left=True, right=True)
ax4.set_xticks(xticks)
ax4.set_yticks(yticks)
ax4.set_xlim(-1, n_epochs)
ax4.set_ylim(10.85, 11.375)

'''
# Add child inset axes (https://codetorial.net/matplotlib/add_inset_graph.html)
axins = ax4.inset_axes([0.25, 0.10, 0.7, 0.5]) # [x0,y0,width,height]
axins.plot(x[100:], -valid_data[0][100:], 'tab:green',
           x[100:], -valid_data[4][100:], 'tab:blue',
           #x[100:], -valid_data[2][100:], 'tab:brown',
           x[100:], -valid_data[1][100:], 'tab:purple',
           x[100:], -valid_data[3][100:], 'tab:orange', 
           linestyle='solid', alpha = 0.2, linewidth=linewidth)

axins.plot(x[100:], smooth(-valid_data[0][100:]), 'tab:green',
           x[100:], smooth(-valid_data[4][100:]), 'tab:blue',
           #x[100:], smooth(-valid_data[2][100:]), 'tab:brown',
           x[100:], smooth(-valid_data[1][100:]), 'tab:purple',
           x[100:], smooth(-valid_data[3][100:]), 'tab:orange', 
           linestyle='solid', linewidth=linewidth)

axins.set_xticklabels('')
axins.set_yticklabels('')
axins.tick_params(left=False, right=False, top=False, bottom=False)
axins.set_xticks([])
axins.set_yticks([])
for axis in ['top', 'bottom', 'left', 'right']:
  axins.spines[axis].set_linewidth(1.5)

indicator = ax4.indicate_inset_zoom(axins)
indicator[1][0].set_linewidth(1)
indicator[1][1].set_linewidth(1)
indicator[1][2].set_linewidth(1)
indicator[1][3].set_linewidth(1)
'''

# Save fig
fig.set_size_inches(width, height)
plt.savefig('results/top/CONST/20/validation_score_compare_baselines/Comparison of Training with many Baselines.svg', bbox_inches="tight")
plt.savefig('results/top/CONST/20/validation_score_compare_baselines/Comparison of Training with many Baselines.png', dpi=600, bbox_inches="tight")
plt.savefig('results/top/CONST/20/validation_score_compare_baselines/Comparison of Training with many Baselines.pdf', bbox_inches="tight")

#%% Compare baselines for MSTOP_UNIF (new entropy)
path_to_json = 'results/top/UNIF/20/validation_score_MSTOP20_UNIF_new_entropy/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
valid_data = json_processing(path_to_json, n_epochs)
print(json_files) 

# A. Greedy Rollout Baseline only
# B. Greedy Rollout Baseline + max. entropy regularisation
# D. x8 instance augmentation baseline (local baseline)
# E. x8 instance augmentation baseline + max. entropy regularisation

# Original (transparent) + Smoothed 
fig, ax4 = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

plt1, = ax4.plot(x, -valid_data[1], c='tab:green', linestyle=linestyle, linewidth=linewidth, alpha = 0.2, label='A. Greedy Rollout')
plt2, = ax4.plot(x, -valid_data[2], c='tab:blue', linestyle=linestyle, linewidth=linewidth, alpha = 0.2, label='B. Greedy Rollout + Entropy')
plt4, = ax4.plot(x, -valid_data[0], c='tab:purple', linestyle='-', linewidth=linewidth, alpha = 0.2, label='C. Proposed Baseline')
plt5, = ax4.plot(x, -valid_data[3], c='tab:orange', linestyle='-', linewidth=linewidth, alpha = 0.2, label='D. Proposed Baseline + Entropy')

plt6, = ax4.plot(x, smooth(-valid_data[1]), c='tab:green', linestyle=linestyle, linewidth=linewidth, label='A. Greedy Rollout')
plt7, = ax4.plot(x, smooth(-valid_data[2]), c='tab:blue', linestyle=linestyle, linewidth=linewidth, label='B. Greedy Rollout + Entropy')
plt9, = ax4.plot(x, smooth(-valid_data[0]), c='tab:purple', linestyle='-', linewidth=linewidth, label='C. Proposed Baseline')
plt10, = ax4.plot(x, smooth(-valid_data[3]), c='tab:orange', linestyle='-', linewidth=linewidth, label='D. Proposed Baseline + Entropy')

ax4.legend(handles=[plt6, plt7, plt9, plt10], loc='best', fontsize=caption_size, ncol=1, frameon=False)

ax4.set_xlabel('Epochs')
ax4.set_ylabel('Validation score')
ax4.minorticks_on()
xticks = np.arange(0, n_epochs+1, 50) # 0, 50, 150, 200
yticks = np.arange(5.7, 6.01, 0.1)
ax4.tick_params(direction='in', which='minor', length=2.0,
                bottom=True, top=True, left=True, right=True)
ax4.tick_params(direction='in', which='major', length=4.0, 
                bottom=True, top=True, left=True, right=True)
ax4.set_xticks(xticks)
ax4.set_yticks(yticks)
ax4.set_xlim(-1, n_epochs)
ax4.set_ylim(5.65, 6.075)

# Save fig
fig.set_size_inches(width, height)
plt.savefig('results/top/UNIF/20/validation_score_compare_baselines/Comparison of Training with many Baselines.svg', bbox_inches="tight")
plt.savefig('results/top/UNIF/20/validation_score_compare_baselines/Comparison of Training with many Baselines.png', dpi=600, bbox_inches="tight")
plt.savefig('results/top/UNIF/20/validation_score_compare_baselines/Comparison of Training with many Baselines.pdf', bbox_inches="tight")

# plt.savefig('results/top/UNIF/20/validation_score_compare_baselines/wo_legend/Comparison of Training with many Baselines.svg', bbox_inches="tight")
# plt.savefig('results/top/UNIF/20/validation_score_compare_baselines/wo_legend/Comparison of Training with many Baselines.png', dpi=600, bbox_inches="tight")
# plt.savefig('results/top/UNIF/20/validation_score_compare_baselines/wo_legend/Comparison of Training with many Baselines.pdf', bbox_inches="tight")


#%% Compare baselines for TSP_50
path_to_json = 'results/tsp/50/validation_score_TSP50_new_entropy/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
valid_data = json_processing(path_to_json, x_step=100)
print(json_files) 

n_epochs = 100
x = np.arange(n_epochs)

# A. Greedy Rollout Baseline only
# B. Greedy Rollout Baseline + max. entropy regularisation
# D. x8 instance augmentation baseline (local baseline)
# E. x8 instance augmentation baseline + max. entropy regularisation

# Original (transparent) + Smoothed 
fig, ax5 = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

plt1, = ax5.plot(x, valid_data[0], c='tab:green', linestyle=linestyle, linewidth=linewidth, alpha = 0.2, label='A. Greedy Rollout')
plt2, = ax5.plot(x, valid_data[2], c='tab:blue', linestyle=linestyle, linewidth=linewidth, alpha = 0.2, label='B. Greedy Rollout + Entropy')
plt3, = ax5.plot(x, valid_data[1], c='tab:purple', linestyle='-', linewidth=linewidth, alpha = 0.2, label='C. Proposed Baseline')
plt4, = ax5.plot(x, valid_data[3], c='tab:orange', linestyle='-', linewidth=linewidth, alpha = 0.2, label='D. Proposed Baseline + Entropy')

plt5, = ax5.plot(x, smooth(valid_data[0]), c='tab:green', linestyle=linestyle, linewidth=linewidth, label='A. Greedy Rollout')
plt6, = ax5.plot(x, smooth(valid_data[2]), c='tab:blue', linestyle=linestyle, linewidth=linewidth, label='B. Greedy Rollout + Entropy')
plt7, = ax5.plot(x, smooth(valid_data[1]), c='tab:purple', linestyle='-', linewidth=linewidth, label='C. Proposed Baseline')
plt8, = ax5.plot(x, smooth(valid_data[3]), c='tab:orange', linestyle='-', linewidth=linewidth, label='D. Proposed Baseline + Entropy')

ax5.legend(handles=[plt5, plt6, plt7, plt8], loc='best', fontsize=caption_size, ncol=1, frameon=False)

ax5.set_xlabel('Epochs')
ax5.set_ylabel('Validation score')
ax5.minorticks_on()
xticks = np.arange(0, n_epochs+1, 25) # 0, 25, 50, 75, 100
yticks = np.arange(5.70, 6.05, 0.05)
ax5.tick_params(direction='in', which='minor', length=2.0,
                bottom=True, top=True, left=True, right=True)
ax5.tick_params(direction='in', which='major', length=4.0, 
                bottom=True, top=True, left=True, right=True)
ax5.set_xticks(xticks)
ax5.set_yticks(yticks)
ax5.set_xlim(-1, n_epochs)
ax5.set_ylim(5.74, 6.03)

'''
# Add child inset axes (https://codetorial.net/matplotlib/add_inset_graph.html)
axins = ax5.inset_axes([0.30, 0.1, 0.5, 0.3]) # [x0,y0,width,height]
axins.plot(x[50:], valid_data[0][50:], 'tab:green',
            x[50:], valid_data[2][50:], 'tab:blue',
            x[50:], valid_data[1][50:], 'tab:purple',
            x[50:], valid_data[3][50:], 'tab:orange',
            linestyle='solid', alpha = 0.2, linewidth=linewidth)

axins.plot(x[50:], smooth(valid_data[0][50:]), 'tab:green',
            x[50:], smooth(valid_data[2][50:]), 'tab:blue',
            x[50:], smooth(valid_data[1][50:]), 'tab:purple',
            x[50:], smooth(valid_data[3][50:]), 'tab:orange',
            linestyle='solid', linewidth=linewidth)

axins.set_xticklabels('')
axins.set_yticklabels('')
axins.tick_params(left=False, right=False, top=False, bottom=False)
axins.set_xticks([])
axins.set_yticks([])
for axis in ['top', 'bottom', 'left', 'right']:
  axins.spines[axis].set_linewidth(1.5)

indicator = ax5.indicate_inset_zoom(axins)
indicator[1][0].set_linewidth(1)
indicator[1][1].set_linewidth(1)
indicator[1][2].set_linewidth(1)
indicator[1][3].set_linewidth(1)
'''

# Save fig
fig.set_size_inches(width, height)
plt.savefig('results/tsp/50/Comparison of Baselines.png', dpi=600, bbox_inches="tight")
plt.savefig('results/tsp/50/Comparison of Baselines.pdf', bbox_inches="tight")
plt.savefig('results/tsp/50/Comparison of Baselines.svg', bbox_inches="tight")

# plt.savefig('results/tsp/50/wo_legend/Comparison of Baselines.png', dpi=600, bbox_inches="tight")
# plt.savefig('results/tsp/50/wo_legend/Comparison of Baselines.pdf', bbox_inches="tight")
# plt.savefig('results/tsp/50/wo_legend/Comparison of Baselines.svg', bbox_inches="tight")

#%% Compare baselines for CVRP_50
path_to_json = 'results/cvrp/50/validation_score_CVRP50/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
valid_data = json_processing(path_to_json, x_step=100)
print(json_files) 

n_epochs = 100
x = np.arange(n_epochs)

# A. Greedy Rollout Baseline only
# B. Greedy Rollout Baseline + max. entropy regularisation
# D. x8 instance augmentation baseline (local baseline)
# E. x8 instance augmentation baseline + max. entropy regularisation

# Original (transparent) + Smoothed 
fig, ax5 = plt.subplots()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)

plt1, = ax5.plot(x, valid_data[0], c='tab:green', linestyle=linestyle, linewidth=linewidth, alpha = 0.2, label='A. Greedy Rollout')
plt2, = ax5.plot(x, valid_data[1], c='tab:blue', linestyle=linestyle, linewidth=linewidth,  alpha = 0.2, label='B. Greedy Rollout + Entropy')
plt3, = ax5.plot(x, valid_data[2], c='tab:purple', linestyle='-', linewidth=linewidth, alpha = 0.2, label='C. Proposed Baseline')
plt4, = ax5.plot(x, valid_data[3], c='tab:orange', linestyle='-', linewidth=linewidth, alpha = 0.2, label='D. Proposed Baseline + Entropy')

plt5, = ax5.plot(x, smooth(valid_data[0]), c='tab:green', linestyle=linestyle, linewidth=linewidth, label='A. Greedy Rollout')
plt6, = ax5.plot(x, smooth(valid_data[1]), c='tab:blue', linestyle=linestyle, linewidth=linewidth, label='B. Greedy Rollout + Entropy')
plt7, = ax5.plot(x, smooth(valid_data[2]), c='tab:purple', linestyle='-', linewidth=linewidth, label='C. Proposed Baseline')
plt8, = ax5.plot(x, smooth(valid_data[3]), c='tab:orange', linestyle='-', linewidth=linewidth, label='D. Proposed Baseline + Entropy')

# ax5.legend(handles=[plt5, plt6, plt7, plt8], loc='best',fontsize=caption_size, ncol=1, frameon=False)

ax5.set_xlabel('Epochs')
ax5.set_ylabel('Validation score')
ax5.minorticks_on()
xticks = np.arange(0, n_epochs+1, 25) # 0, 25, 50, 75, 100
yticks = np.arange(10.9, 11.75, 0.2)
ax5.tick_params(direction='in', which='minor', length=2.0,
                bottom=True, top=True, left=True, right=True)
ax5.tick_params(direction='in', which='major', length=4.0, 
                bottom=True, top=True, left=True, right=True)
ax5.set_xticks(xticks)
ax5.set_yticks(yticks)
ax5.set_xlim(-1, n_epochs)
ax5.set_ylim(10.85, 11.7)

# Save fig
fig.set_size_inches(width, height)
# plt.savefig('results/cvrp/50/Comparison of Baselines.png', dpi=600, bbox_inches="tight")
# plt.savefig('results/cvrp/50/Comparison of Baselines.pdf', bbox_inches="tight")
# plt.savefig('results/cvrp/50/Comparison of Baselines.svg', bbox_inches="tight")

plt.savefig('results/cvrp/50/wo_legend/Comparison of Baselines.png', dpi=600, bbox_inches="tight")
plt.savefig('results/cvrp/50/wo_legend/Comparison of Baselines.pdf', bbox_inches="tight")
plt.savefig('results/cvrp/50/wo_legend/Comparison of Baselines.svg', bbox_inches="tight")

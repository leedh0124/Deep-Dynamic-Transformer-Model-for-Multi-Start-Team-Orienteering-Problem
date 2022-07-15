import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
caption_size = 10 
main_font_size = 12
bar_width = 0.15
width = 7.16 # for two column fig
height = width / 1.618 

import Preset
Preset.setPlotStyle()
plt.rcParams['hatch.color'] = 'dimgrey'
plt.rcParams['hatch.linewidth'] = 2

# color
color = ['#6cb33e', '#c0ce21', '#007b85', '#30c0d9', '#b32417', '#f39920', '#7a003c', '#f0607c']

# #%%1a. MSTOP1020 (2 vehicles) Train Env: Constant Prizes
# fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4)
# Model = ['MSTOP10 (B)', 'MSTOP10 (D)', 'MSTOP20 (B)', 'MSTOP20 (D)']
# Problem = ['10','20']
# x_axis = np.arange(1)

# # (a1) Train: Const, Test: Const
# MSTOP10_Const_B1 = [0.28]
# MSTOP10_Const_D1 = [0.21]
# MSTOP20_Const_B1 = [1.75]
# MSTOP20_Const_D1 = [1.15]
# ax0.bar(x_axis-bar_width*1.5-0.04, MSTOP10_Const_B1, width=bar_width, label=Model[0], align='center', color=color[0], 
#         linewidth=2, edgecolor=color[0], alpha=0.85)
# ax0.bar(x_axis-bar_width/2-0.03, MSTOP10_Const_D1, width=bar_width, label=Model[1], align='center', color=color[1],
#         linewidth=2, facecolor=color[1], hatch='//')
# ax0.bar(x_axis+bar_width/2+0.03, MSTOP20_Const_B1, width=bar_width, label=Model[2], align='center', color=color[2],
#        linewidth=2, edgecolor=color[2], alpha=0.85)
# ax0.bar(x_axis+bar_width*1.5+0.04, MSTOP20_Const_D1, width=bar_width, label=Model[3], align='center', color=color[3],
#         linewidth=2, facecolor=color[3], hatch='//')
# ax0.margins(x=0.05)
# ax0.set_ylim(0.2, 1.8)
# y_axis = np.arange(0.2, 1.8, 0.4)
# ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax0.set_xticks(x_axis)
# ax0.set_yticks(y_axis)
# ax0.set_ylabel("Optimality gap (%)", fontsize=main_font_size)
# Problem_a1 = ['MSTOP10\n(Const)']
# ax0.set_xticklabels(Problem_a1, fontsize=main_font_size)
# for c in ax0.containers:
#     ax0.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# # (a2) Train: Const, Test: Const
# MSTOP10_Const_B2 = [8.58]
# MSTOP10_Const_D2 = [8.90]
# MSTOP20_Const_B2 = [0.79]
# MSTOP20_Const_D2 = [0.77] 
# ax1.bar(x_axis-bar_width*1.5-0.04, MSTOP10_Const_B2, width=bar_width, label=Model[0], align='center', color=color[0], 
#         linewidth=2, edgecolor=color[0], alpha=0.85)
# ax1.bar(x_axis-bar_width/2-0.03, MSTOP10_Const_D2, width=bar_width, label=Model[1], align='center', color=color[1],
#         linewidth=2, facecolor=color[1], hatch='//')
# ax1.bar(x_axis+bar_width/2+0.03, MSTOP20_Const_B2, width=bar_width, label=Model[2], align='center', color=color[2],
#        linewidth=2, edgecolor=color[2], alpha=0.85)
# ax1.bar(x_axis+bar_width*1.5+0.04, MSTOP20_Const_D2, width=bar_width, label=Model[3], align='center', color=color[3],
#         linewidth=2, facecolor=color[3], hatch='//')
# ax1.margins(x=0.05)
# ax1.set_ylim(0.5, 9.0)
# y_axis = np.arange(0.5, 9.0, 5.0)
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax1.set_xticks(x_axis)
# ax1.set_yticks(y_axis)
# Problem_a2 = ['MSTOP20\n(Const)']
# ax1.set_xticklabels(Problem_a2, fontsize=main_font_size)
# for c in ax1.containers:
#     ax1.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)
   
# # (b1) Train: Const, Test: Uniform
# MSTOP10_Const_B1 = [1.54]
# MSTOP10_Const_D1 = [1.68]
# MSTOP20_Const_B1 = [2.28]
# MSTOP20_Const_D1 = [1.54] 
# ax2.bar(x_axis-bar_width*1.5-0.04, MSTOP10_Const_B1, width=bar_width, label=Model[0], align='center', color=color[0], 
#         linewidth=2, edgecolor=color[0], alpha=0.85)
# ax2.bar(x_axis-bar_width/2-0.03, MSTOP10_Const_D1, width=bar_width, label=Model[1], align='center', color=color[1],
#         linewidth=2, facecolor=color[1], hatch='//')
# ax2.bar(x_axis+bar_width/2+0.03, MSTOP20_Const_B1, width=bar_width, label=Model[2], align='center', color=color[2],
#        linewidth=2, edgecolor=color[2], alpha=0.85)
# ax2.bar(x_axis+bar_width*1.5+0.04, MSTOP20_Const_D1, width=bar_width, label=Model[3], align='center', color=color[3],
#         linewidth=2, facecolor=color[3], hatch='//')
# ax2.margins(x=0.05)
# ax2.set_ylim(1.5, 2.5)
# y_axis = np.arange(1.5, 2.5, 0.2)
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax2.set_xticks(x_axis)
# ax2.set_yticks(y_axis)
# Problem_b1 = ['MSTOP10\n(Unif)']
# ax2.set_xticklabels(Problem_b1, fontsize=main_font_size)
# for c in ax2.containers:
#     ax2.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)
    
# # (b2) Train: Const, Test: Uniform
# MSTOP10_Const_B2 = [11.20]
# MSTOP10_Const_D2 = [10.19]
# MSTOP20_Const_B2 = [6.13]
# MSTOP20_Const_D2 = [4.51] 
# ax3.bar(x_axis-bar_width*1.5-0.04, MSTOP10_Const_B2, width=bar_width, label=Model[0], align='center', color=color[0], 
#         linewidth=2, edgecolor=color[0], alpha=0.85)
# ax3.bar(x_axis-bar_width/2-0.03, MSTOP10_Const_D2, width=bar_width, label=Model[1], align='center', color=color[1],
#         linewidth=2, facecolor=color[1], hatch='//')
# ax3.bar(x_axis+bar_width/2+0.03, MSTOP20_Const_B2, width=bar_width, label=Model[2], align='center', color=color[2],
#        linewidth=2, edgecolor=color[2], alpha=0.85)
# ax3.bar(x_axis+bar_width*1.5+0.04, MSTOP20_Const_D2, width=bar_width, label=Model[3], align='center', color=color[3],
#         linewidth=2, facecolor=color[3], hatch='//')
# ax3.margins(x=0.05)
# ax3.set_ylim(4.0, 12.0)
# y_axis = np.arange(4.0, 12.0, 3.0)
# ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax3.set_xticks(x_axis)
# ax3.set_yticks(y_axis)
# Problem_b2 = ['MSTOP20\n(Unif)']
# ax3.set_xticklabels(Problem_b2, fontsize=main_font_size)
# for c in ax3.containers:
#     ax3.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# # Common legend for all subplots
# handles, labels = ax2.get_legend_handles_labels()
# plt.legend(handles, labels, bbox_to_anchor=(-1.6, 1.2), loc="center", ncol=4, fontsize=caption_size-2, \
#             title="DDTM train environment: Constant prizes", title_fontsize=main_font_size-2)
# plt.subplots_adjust(wspace=0.4)

# # Save fig
# fig.set_size_inches(width, height/2)
# plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Const_v2.png', dpi=600, bbox_inches="tight")
# plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Const_v2.pdf', bbox_inches="tight")
# plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Const_v2.svg', bbox_inches="tight")

#%% 1a. MSTOP1020 (2 vehicles) Train Env: Constant Prizes
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
Model = ['MSTOP10 (B)', 'MSTOP10 (D)', 'MSTOP20 (B)', 'MSTOP20 (D)']
Problem = ['10','20']
x_axis = np.arange(len(Problem))

# (a) Train: Const, Test: Const 
MSTOP10_Const_B = [0.24, 7.96]
MSTOP10_Const_D = [0.19, 7.00]
MSTOP20_Const_B = [1.70, 0.90]
MSTOP20_Const_D = [1.59, 0.78]
ax0.bar(x_axis-bar_width*1.5-0.04, MSTOP10_Const_B, width=bar_width, label=Model[0], align='center', color=color[0], 
        linewidth=2, edgecolor=color[0], alpha=0.85)
ax0.bar(x_axis-bar_width/2-0.03, MSTOP10_Const_D, width=bar_width, label=Model[1], align='center', color=color[1],
        linewidth=2, facecolor=color[1], hatch='//')
ax0.bar(x_axis+bar_width/2+0.03, MSTOP20_Const_B, width=bar_width, label=Model[2], align='center', color=color[2],
        linewidth=2, edgecolor=color[2], alpha=0.85)
ax0.bar(x_axis+bar_width*1.5+0.04, MSTOP20_Const_D, width=bar_width, label=Model[3], align='center', color=color[3],
        linewidth=2, facecolor=color[3], hatch='//')
ax0.margins(x=0.05)
ax0.set_ylim(0.0, 14)
ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax0.set_ylabel("Optimality gap (%)", fontsize=main_font_size)
ax0.set_xticks(x_axis)
Problem_a = ['MSTOP10\n(Const)','MSTOP20\n(Const)']
ax0.set_xticklabels(Problem_a, fontsize=main_font_size)
for c in ax0.containers:
    ax0.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-2)

# (b) Train: Const, Test: Uniform
MSTOP10_Const_B = [2.17, 12.61]
MSTOP10_Const_D = [1.91, 10.59]
MSTOP20_Const_B = [2.52, 5.43]
MSTOP20_Const_D = [1.59, 4.50]
ax1.bar(x_axis-bar_width*1.5-0.04, MSTOP10_Const_B, width=bar_width, label=Model[0], align='center', color=color[0], 
        linewidth=2, edgecolor=color[0], alpha=0.85)
ax1.bar(x_axis-bar_width/2-0.03, MSTOP10_Const_D, width=bar_width, label=Model[1], align='center', color=color[1],
        linewidth=2, facecolor=color[1], hatch='//')
ax1.bar(x_axis+bar_width/2+0.03, MSTOP20_Const_B, width=bar_width, label=Model[2], align='center', color=color[2],
        linewidth=2, edgecolor=color[2], alpha=0.85)
ax1.bar(x_axis+bar_width*1.5+0.04, MSTOP20_Const_D, width=bar_width, label=Model[3], align='center', color=color[3],
        linewidth=2, facecolor=color[3], hatch='//')
ax1.margins(x=0.05)
ax1.set_xticks(x_axis)
Problem_b = ['MSTOP10\n(Unif)','MSTOP20\n(Unif)']
ax1.set_xticklabels(Problem_b, fontsize=main_font_size)
for c in ax1.containers:
    ax1.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-2)

# Common legend for all subplots
handles, labels = ax0.get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(-0.05, 1.15), loc="center", ncol=4, fontsize=caption_size-2, \
            title="DDTM train environment: Constant prizes", title_fontsize=main_font_size-2)
plt.subplots_adjust(wspace=0.1)

# Save fig
# fig.set_size_inches(width/2, height/4)
fig.set_size_inches(width, height/2)
plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Const.png', dpi=600, bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Const.pdf', bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Const.svg', bbox_inches="tight")

#%% 1b. MSTOP1020 (2 vehicles) Train Env: Uniformly Distributed Prizes
fig, (ax2, ax3) = plt.subplots(nrows=1, ncols=2, sharey=True)
Model = ['MSTOP10 (B)', 'MSTOP10 (D)', 'MSTOP20 (B)', 'MSTOP20 (D)']
Problem = ['10','20']
x_axis = np.arange(len(Problem))

# (c) Train: Uniform, Test: Uniform
MSTOP10_Unif_B = [0.62, 10.01]
MSTOP10_Unif_D = [0.56, 9.20]
MSTOP20_Unif_B = [2.58, 1.52]
MSTOP20_Unif_D = [2.30, 1.34]
ax2.bar(x_axis-bar_width*1.5-0.04, MSTOP10_Unif_B, width=bar_width, label=Model[0], align='center', color=color[4], 
        linewidth=2, edgecolor=color[4], alpha=0.85)
ax2.bar(x_axis-bar_width/2-0.03, MSTOP10_Unif_D, width=bar_width, label=Model[1], align='center', color=color[5],
        linewidth=2, facecolor=color[5], hatch='//')
ax2.bar(x_axis+bar_width/2+0.03, MSTOP20_Unif_B, width=bar_width, label=Model[2], align='center', color=color[6],
       linewidth=2, edgecolor=color[6], alpha=0.85)
ax2.bar(x_axis+bar_width*1.5+0.04, MSTOP20_Unif_D, width=bar_width, label=Model[3], align='center', color=color[7],
        linewidth=2, facecolor=color[7], hatch='//')
ax2.margins(x=0.05)
ax2.set_ylim(0.0, 12.2)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.set_ylabel("Optimality gap (%)", fontsize=main_font_size)
ax2.set_xticks(x_axis)
Problem_c = ['MSTOP10\n(Unif)','MSTOP20\n(Unif)']
ax2.set_xticklabels(Problem_c, fontsize=main_font_size)
for c in ax2.containers:
    ax2.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-2)

# (d) Train: Uniform, Test: Const
MSTOP10_Unif_B = [0.20, 7.37]
MSTOP10_Unif_D = [0.19, 7.95]
MSTOP20_Unif_B = [1.53, 0.89]
MSTOP20_Unif_D = [1.45, 0.75]
ax3.bar(x_axis-bar_width*1.5-0.04, MSTOP10_Unif_B, width=bar_width, label=Model[0], align='center', color=color[4], 
        linewidth=2, edgecolor=color[4], alpha=0.85)
ax3.bar(x_axis-bar_width/2-0.03, MSTOP10_Unif_D, width=bar_width, label=Model[1], align='center', color=color[5],
        linewidth=2, facecolor=color[5], hatch='//')
ax3.bar(x_axis+bar_width/2+0.03, MSTOP20_Unif_B, width=bar_width, label=Model[2], align='center', color=color[6],
       linewidth=2, edgecolor=color[6], alpha=0.85)
ax3.bar(x_axis+bar_width*1.5+0.04, MSTOP20_Unif_D, width=bar_width, label=Model[3], align='center', color=color[7],
        linewidth=2, facecolor=color[7], hatch='//')
ax3.margins(x=0.05)
ax3.set_xticks(x_axis)
Problem_d = ['MSTOP10\n(Const)','MSTOP20\n(Const)']
ax3.set_xticklabels(Problem_d, fontsize=main_font_size)
for c in ax3.containers:
    ax3.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-2)


# Common legend for all subplots
handles, labels = ax2.get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(-0.05, 1.15), loc="center", ncol=4, fontsize=caption_size-2, \
           title="DDTM train environment: Uniformly distributed prizes", title_fontsize=main_font_size-2)
plt.subplots_adjust(wspace=0.1)

# Save fig
fig.set_size_inches(width, height/2)
plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Unif.png', dpi=600, bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Unif.pdf', bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/MSTOP1020_Train_Unif.svg', bbox_inches="tight")

#%%1a. MSTOP5070 (3 vehicles) Train Env: Constant Prizes
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4)
Model = ['MSTOP50 (B)', 'MSTOP50 (D)', 'MSTOP70 (B)', 'MSTOP70 (D)']
Problem = ['50', '70']
x_axis = np.arange(1)

# (a1) Train: Const, Test: Const
MSTOP50_Const_B1 = [40.91]
MSTOP50_Const_D1 = [41.01]
MSTOP70_Const_B1 = [40.40]
MSTOP70_Const_D1 = [40.67]
ax0.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Const_B1, width=bar_width, label=Model[0], align='center', color=color[0], 
        linewidth=2, edgecolor=color[0], alpha=0.85)
ax0.bar(x_axis-bar_width/2-0.03, MSTOP50_Const_D1, width=bar_width, label=Model[1], align='center', color=color[1],
        linewidth=2, facecolor=color[1], hatch='//')
ax0.bar(x_axis+bar_width/2+0.03, MSTOP70_Const_B1, width=bar_width, label=Model[2], align='center', color=color[2],
       linewidth=2, edgecolor=color[2], alpha=0.85)
ax0.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Const_D1, width=bar_width, label=Model[3], align='center', color=color[3],
        linewidth=2, facecolor=color[3], hatch='//')
ax0.margins(x=0.05)
ax0.set_ylim(40.0, 41.3)
y_axis = np.arange(40.0, 41.3, 0.4)
ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax0.set_xticks(x_axis)
ax0.set_yticks(y_axis)
ax0.set_ylabel("Test score", fontsize=main_font_size)
Problem_a1 = ['MSTOP50\n(Const)']
ax0.set_xticklabels(Problem_a1, fontsize=main_font_size)
for c in ax0.containers:
    ax0.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# (a2) Train: Const, Test: Const
MSTOP50_Const_B2 = [51.80]
MSTOP50_Const_D2 = [52.19]
MSTOP70_Const_B2 = [52.22]
MSTOP70_Const_D2 = [52.49] 
ax1.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Const_B2, width=bar_width, label=Model[0], align='center', color=color[0], 
        linewidth=2, edgecolor=color[0], alpha=0.85)
ax1.bar(x_axis-bar_width/2-0.03, MSTOP50_Const_D2, width=bar_width, label=Model[1], align='center', color=color[1],
        linewidth=2, facecolor=color[1], hatch='//')
ax1.bar(x_axis+bar_width/2+0.03, MSTOP70_Const_B2, width=bar_width, label=Model[2], align='center', color=color[2],
       linewidth=2, edgecolor=color[2], alpha=0.85)
ax1.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Const_D2, width=bar_width, label=Model[3], align='center', color=color[3],
        linewidth=2, facecolor=color[3], hatch='//')
ax1.margins(x=0.05)
ax1.set_ylim(51.5, 52.8)
y_axis = np.arange(51.5, 52.8, 0.4)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.set_xticks(x_axis)
ax1.set_yticks(y_axis)
Problem_a2 = ['MSTOP70\n(Const)']
ax1.set_xticklabels(Problem_a2, fontsize=main_font_size)
for c in ax1.containers:
    ax1.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)
   
# (b1) Train: Const, Test: Uniform
MSTOP50_Const_B1 = [18.74]
MSTOP50_Const_D1 = [19.26]
MSTOP70_Const_B1 = [19.02]
MSTOP70_Const_D1 = [19.07] 
ax2.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Const_B1, width=bar_width, label=Model[0], align='center', color=color[0], 
        linewidth=2, edgecolor=color[0], alpha=0.85)
ax2.bar(x_axis-bar_width/2-0.03, MSTOP50_Const_D1, width=bar_width, label=Model[1], align='center', color=color[1],
        linewidth=2, facecolor=color[1], hatch='//')
ax2.bar(x_axis+bar_width/2+0.03, MSTOP70_Const_B1, width=bar_width, label=Model[2], align='center', color=color[2],
       linewidth=2, edgecolor=color[2], alpha=0.85)
ax2.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Const_D1, width=bar_width, label=Model[3], align='center', color=color[3],
        linewidth=2, facecolor=color[3], hatch='//')
ax2.margins(x=0.05)
ax2.set_ylim(18.5, 19.5)
y_axis = np.arange(18.5, 19.51, 0.3)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.set_xticks(x_axis)
ax2.set_yticks(y_axis)
Problem_b1 = ['MSTOP50\n(Unif)']
ax2.set_xticklabels(Problem_b1, fontsize=main_font_size)
for c in ax2.containers:
    ax2.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)
    
# (b2) Train: Const, Test: Uniform
MSTOP50_Const_B2 = [21.91]
MSTOP50_Const_D2 = [23.27]
MSTOP70_Const_B2 = [22.44]
MSTOP70_Const_D2 = [22.99] 
ax3.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Const_B2, width=bar_width, label=Model[0], align='center', color=color[0], 
        linewidth=2, edgecolor=color[0], alpha=0.85)
ax3.bar(x_axis-bar_width/2-0.03, MSTOP50_Const_D2, width=bar_width, label=Model[1], align='center', color=color[1],
        linewidth=2, facecolor=color[1], hatch='//')
ax3.bar(x_axis+bar_width/2+0.03, MSTOP70_Const_B2, width=bar_width, label=Model[2], align='center', color=color[2],
       linewidth=2, edgecolor=color[2], alpha=0.85)
ax3.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Const_D2, width=bar_width, label=Model[3], align='center', color=color[3],
        linewidth=2, facecolor=color[3], hatch='//')
ax3.margins(x=0.05)
ax3.set_ylim(21.8, 23.45)
y_axis = np.arange(21.8, 23.41, 0.5)
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.set_xticks(x_axis)
ax3.set_yticks(y_axis)
Problem_b2 = ['MSTOP70\n(Unif)']
ax3.set_xticklabels(Problem_b2, fontsize=main_font_size)
for c in ax3.containers:
    ax3.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# Common legend for all subplots
handles, labels = ax2.get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(-1.6, 1.2), loc="center", ncol=4, fontsize=caption_size-2, \
            title="DDTM train environment: Constant prizes", title_fontsize=main_font_size-2)
plt.subplots_adjust(wspace=0.4)

# Save fig
fig.set_size_inches(width, height/2)
plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Const_v2.png', dpi=600, bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Const_v2.pdf', bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Const_v2.svg', bbox_inches="tight")

#%% backup
# #%% 2. MSTOP5070 (3 vehicles) 
# fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
# Model = ['MSTOP50 (B)', 'MSTOP50 (D)', 'MSTOP70 (B)', 'MSTOP70 (D)']
# Problem = ['50','70']

# # (a) Train: Const, Test: Const 
# MSTOP50_Const_B = [40.87, 51.92]
# MSTOP50_Const_D = [41.02, 52.12]
# MSTOP70_Const_B = [40.40, 52.22]
# MSTOP70_Const_D = [40.67, 52.49]
# ax0.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Const_B, width=bar_width, label=Model[0], align='center', color=color[0], 
#         linewidth=2, edgecolor=color[0], alpha=0.85)
# ax0.bar(x_axis-bar_width/2-0.03, MSTOP50_Const_D, width=bar_width, label=Model[1], align='center', color=color[1],
#         linewidth=2, facecolor=color[1], hatch='//')
# ax0.bar(x_axis+bar_width/2+0.03, MSTOP70_Const_B, width=bar_width, label=Model[2], align='center', color=color[2],
#        linewidth=2, edgecolor=color[2], alpha=0.85)
# ax0.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Const_D, width=bar_width, label=Model[3], align='center', color=color[3],
#         linewidth=2, facecolor=color[3], hatch='//')
# ax0.margins(x=0.05)
# ax0.set_ylim(40.0, 55.0)
# y_axis = np.arange(40.0, 55.1, 5.0)
# ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax0.set_xticks(x_axis)
# ax0.set_yticks(y_axis)
# Problem_a = ['MSTOP50\n(Const)','MSTOP70\n(Const)']
# ax0.set_ylabel("Test score", fontsize=main_font_size)
# ax0.set_xticklabels(Problem_a, fontsize=main_font_size)
# for c in ax0.containers:
#     ax0.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# # (b) Train: Const, Test: Uniform
# MSTOP50_Const_B = [18.50, 22.24]
# MSTOP50_Const_D = [18.91, 22.96]
# MSTOP70_Const_B = [19.02, 22.44]
# MSTOP70_Const_D = [19.07, 22.99]
# ax1.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Const_B, width=bar_width, label=Model[0], align='center', color=color[0], 
#         linewidth=2, edgecolor=color[0], alpha=0.85)
# ax1.bar(x_axis-bar_width/2-0.03, MSTOP50_Const_D, width=bar_width, label=Model[1], align='center', color=color[1],
#         linewidth=2, facecolor=color[1], hatch='//')
# ax1.bar(x_axis+bar_width/2+0.03, MSTOP70_Const_B, width=bar_width, label=Model[2], align='center', color=color[2],
#        linewidth=2, edgecolor=color[2], alpha=0.85)
# ax1.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Const_D, width=bar_width, label=Model[3], align='center', color=color[3],
#         linewidth=2, facecolor=color[3], hatch='//')
# ax1.margins(x=0.05)
# ax1.set_ylim(18.0, 24.0)
# y_axis = np.arange(18.0, 24.1, 2.0)
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax1.set_xticks(x_axis)
# ax1.set_yticks(y_axis)
# Problem_b = ['MSTOP50\n(Unif)','MSTOP70\n(Unif)']
# ax1.set_xticklabels(Problem_b, fontsize=main_font_size)
# for c in ax1.containers:
#     ax1.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# # Common legend for all subplots
# handles, labels = ax0.get_legend_handles_labels()
# plt.legend(handles, labels, bbox_to_anchor=(-0.10, 1.15), loc="center", ncol=4, fontsize=caption_size-2, \
#            title="DDTM train environment: Constant prizes", title_fontsize=main_font_size-2)
# plt.subplots_adjust(wspace=0.2)

# # Save fig
# # fig.set_size_inches(width/2, height/4)
# fig.set_size_inches(width, height/2)
# plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Const.png', dpi=600, bbox_inches="tight")
# plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Const.pdf', bbox_inches="tight")
# plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Const.svg', bbox_inches="tight")

#%%1b. MSTOP5070 (3 vehicles) Train Env: Uniformly Distributed Prizes
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4)
Model = ['MSTOP50 (B)', 'MSTOP50 (D)', 'MSTOP70 (B)', 'MSTOP70 (D)']
Problem = ['50', '70']
x_axis = np.arange(1)

# (c1) Train: Uniform, Test: Uniform
MSTOP50_Unif_B1 = [21.42]
MSTOP50_Unif_D1 = [21.45]
MSTOP70_Unif_B1 = [21.25]
MSTOP70_Unif_D1 = [21.27]
ax0.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Unif_B1, width=bar_width, label=Model[0], align='center', color=color[4], 
        linewidth=2, edgecolor=color[4], alpha=0.85)
ax0.bar(x_axis-bar_width/2-0.03, MSTOP50_Unif_D1, width=bar_width, label=Model[1], align='center', color=color[5],
        linewidth=2, facecolor=color[5], hatch='//')
ax0.bar(x_axis+bar_width/2+0.03, MSTOP70_Unif_B1, width=bar_width, label=Model[2], align='center', color=color[6],
       linewidth=2, edgecolor=color[6], alpha=0.85)
ax0.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Unif_D1, width=bar_width, label=Model[3], align='center', color=color[7],
        linewidth=2, facecolor=color[7], hatch='//')
ax0.margins(x=0.05)
ax0.set_ylim(21.0, 21.6)
y_axis = np.arange(21.0, 21.6, 0.2)
ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax0.set_xticks(x_axis)
ax0.set_yticks(y_axis)
ax0.set_ylabel("Test score", fontsize=main_font_size)
# Problem_c = ['MSTOP50\n(Unif)','MSTOP70\n(Unif)']
Problem_c1 = ['MSTOP50\n(Unif)']
ax0.set_xticklabels(Problem_c1, fontsize=main_font_size)
for c in ax0.containers:
    ax0.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# (c2) Train: Uniform, Test: Uniform
MSTOP50_Unif_B2 = [27.69]
MSTOP50_Unif_D2 = [27.78]
MSTOP70_Unif_B2 = [27.86]
MSTOP70_Unif_D2 = [27.98] 
ax1.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Unif_B2, width=bar_width, label=Model[0], align='center', color=color[4], 
        linewidth=2, edgecolor=color[4], alpha=0.85)
ax1.bar(x_axis-bar_width/2-0.03, MSTOP50_Unif_D2, width=bar_width, label=Model[1], align='center', color=color[5],
        linewidth=2, facecolor=color[5], hatch='//')
ax1.bar(x_axis+bar_width/2+0.03, MSTOP70_Unif_B2, width=bar_width, label=Model[2], align='center', color=color[6],
       linewidth=2, edgecolor=color[6], alpha=0.85)
ax1.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Unif_D2, width=bar_width, label=Model[3], align='center', color=color[7],
        linewidth=2, facecolor=color[7], hatch='//')
ax1.margins(x=0.05)
ax1.set_ylim(27.5, 28.05)
y_axis = np.arange(27.5, 28.1, 0.2)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.set_xticks(x_axis)
ax1.set_yticks(y_axis)
Problem_c2 = ['MSTOP70\n(Unif)']
ax1.set_xticklabels(Problem_c2, fontsize=main_font_size)
for c in ax1.containers:
    ax1.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)
   
# (d1) Train: Uniform, Test: Const
MSTOP50_Unif_B1 = [40.82]
MSTOP50_Unif_D1 = [40.90]
MSTOP70_Unif_B1 = [40.62]
MSTOP70_Unif_D1 = [40.63] 
ax2.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Unif_B1, width=bar_width, label=Model[0], align='center', color=color[4], 
        linewidth=2, edgecolor=color[4], alpha=0.85)
ax2.bar(x_axis-bar_width/2-0.03, MSTOP50_Unif_D1, width=bar_width, label=Model[1], align='center', color=color[5],
        linewidth=2, facecolor=color[5], hatch='//')
ax2.bar(x_axis+bar_width/2+0.03, MSTOP70_Unif_B1, width=bar_width, label=Model[2], align='center', color=color[6],
       linewidth=2, edgecolor=color[6], alpha=0.85)
ax2.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Unif_D1, width=bar_width, label=Model[3], align='center', color=color[7],
        linewidth=2, facecolor=color[7], hatch='//')
ax2.margins(x=0.05)
ax2.set_ylim(40.5, 41.05)
y_axis = np.arange(40.5, 41.1, 0.2)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.set_xticks(x_axis)
ax2.set_yticks(y_axis)
Problem_d1 = ['MSTOP50\n(Const)']
ax2.set_xticklabels(Problem_d1, fontsize=main_font_size)
for c in ax2.containers:
    ax2.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)
    
# (d2) Train: Uniform, Test: Const
MSTOP50_Unif_B2 = [51.90]
MSTOP50_Unif_D2 = [51.82]
MSTOP70_Unif_B2 = [52.04]
MSTOP70_Unif_D2 = [52.26] 
ax3.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Unif_B2, width=bar_width, label=Model[0], align='center', color=color[4], 
        linewidth=2, edgecolor=color[4], alpha=0.85)
ax3.bar(x_axis-bar_width/2-0.03, MSTOP50_Unif_D2, width=bar_width, label=Model[1], align='center', color=color[5],
        linewidth=2, facecolor=color[5], hatch='//')
ax3.bar(x_axis+bar_width/2+0.03, MSTOP70_Unif_B2, width=bar_width, label=Model[2], align='center', color=color[6],
       linewidth=2, edgecolor=color[6], alpha=0.85)
ax3.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Unif_D2, width=bar_width, label=Model[3], align='center', color=color[7],
        linewidth=2, facecolor=color[7], hatch='//')
ax3.margins(x=0.05)
ax3.set_ylim(51.6, 52.5)
y_axis = np.arange(51.6, 52.51, 0.3)
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax3.set_xticks(x_axis)
ax3.set_yticks(y_axis)
Problem_d2 = ['MSTOP70\n(Const)']
ax3.set_xticklabels(Problem_d2, fontsize=main_font_size)
for c in ax3.containers:
    ax3.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# Common legend for all subplots
handles, labels = ax2.get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(-1.6, 1.2), loc="center", ncol=4, fontsize=caption_size-2, \
            title="DDTM train environment: Uniformly distributed prizes", title_fontsize=main_font_size-2)
plt.subplots_adjust(wspace=0.4)

# Save fig
fig.set_size_inches(width, height/2)
plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Unif_v2.png', dpi=600, bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Unif_v2.pdf', bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Unif_v2.svg', bbox_inches="tight")

#%% backup
# #%%1b. MSTOP5070 (3 vehicles) Train Env: Uniformly Distributed Prizes
# fig, (ax2, ax3) = plt.subplots(nrows=1, ncols=2)
# Model = ['MSTOP50 (B)', 'MSTOP50 (D)', 'MSTOP70 (B)', 'MSTOP70 (D)']
# Problem = ['50', '70']

# # (c) Train: Uniform, Test: Uniform
# MSTOP50_Unif_B = [21.42, 27.73]
# MSTOP50_Unif_D = [21.51, 27.82]
# MSTOP70_Unif_B = [21.25, 27.86]
# MSTOP70_Unif_D = [21.27, 27.98] 
# ax2.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Unif_B, width=bar_width, label=Model[0], align='center', color=color[4], 
#         linewidth=2, edgecolor=color[4], alpha=0.85)
# ax2.bar(x_axis-bar_width/2-0.03, MSTOP50_Unif_D, width=bar_width, label=Model[1], align='center', color=color[5],
#         linewidth=2, facecolor=color[5], hatch='//')
# ax2.bar(x_axis+bar_width/2+0.03, MSTOP70_Unif_B, width=bar_width, label=Model[2], align='center', color=color[6],
#        linewidth=2, edgecolor=color[6], alpha=0.85)
# ax2.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Unif_D, width=bar_width, label=Model[3], align='center', color=color[7],
#         linewidth=2, facecolor=color[7], hatch='//')
# ax2.margins(x=0.05)
# ax2.set_ylim(21.0, 29.0)
# y_axis = np.arange(21.0, 29.1, 2.0)
# ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax2.set_xticks(x_axis)
# ax2.set_yticks(y_axis)
# ax2.set_ylabel("Test score", fontsize=main_font_size)
# Problem_c = ['MSTOP50\n(Unif)','MSTOP70\n(Unif)']
# ax2.set_xticklabels(Problem_c, fontsize=main_font_size)
# for c in ax2.containers:
#     ax2.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)

# # (d) Train: Uniform, Test: Const
# MSTOP50_Unif_B = [40.79, 51.81]
# MSTOP50_Unif_D = [41.01, 52.02]
# MSTOP70_Unif_B = [40.62, 52.04]
# MSTOP70_Unif_D = [40.63, 52.26]
# ax3.bar(x_axis-bar_width*1.5-0.04, MSTOP50_Unif_B, width=bar_width, label=Model[0], align='center', color=color[4], 
#         linewidth=2, edgecolor=color[4], alpha=0.85)
# ax3.bar(x_axis-bar_width/2-0.03, MSTOP50_Unif_D, width=bar_width, label=Model[1], align='center', color=color[5],
#         linewidth=2, facecolor=color[5], hatch='//')
# ax3.bar(x_axis+bar_width/2+0.03, MSTOP70_Unif_B, width=bar_width, label=Model[2], align='center', color=color[6],
#        linewidth=2, edgecolor=color[6], alpha=0.85)
# ax3.bar(x_axis+bar_width*1.5+0.04, MSTOP70_Unif_D, width=bar_width, label=Model[3], align='center', color=color[7],
#         linewidth=2, facecolor=color[7], hatch='//')
# ax3.margins(x=0.05)
# ax3.set_ylim(40.0, 55.0)
# y_axis = np.arange(40.0, 55.1, 5.0)
# ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax3.set_xticks(x_axis)
# ax3.set_yticks(y_axis)
# Problem_d = ['MSTOP50\n(Const)','MSTOP70\n(Const)']
# ax3.set_xticklabels(Problem_d, fontsize=main_font_size)
# for c in ax3.containers:
#     ax3.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-3)


# # Common legend for all subplots
# handles, labels = ax2.get_legend_handles_labels()
# plt.legend(handles, labels, bbox_to_anchor=(-0.10, 1.15), loc="center", ncol=4, fontsize=caption_size-2, \
#            title="DDTM train environment: Uniformly distributed prizes", title_fontsize=main_font_size-2)
# plt.subplots_adjust(wspace=0.2)

# # Save fig
# fig.set_size_inches(width, height/2)
# plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Unif.png', dpi=600, bbox_inches="tight")
# plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Unif.pdf', bbox_inches="tight")
# plt.savefig('Result_plot/[2] generalization/MSTOP5070_Train_Unif.svg', bbox_inches="tight")


#%% 3. TSP/CVRP
fig, ((ax0, ax1)) = plt.subplots(nrows=1, ncols=2)
bar_width = 0.18
# (a) TSP
Model = ['TSP50 (B)', 'TSP50 (D)', 'TSP100 (B)', 'TSP100 (D)']
Problem = ['50','100']
TSP50_B = [0.31, 2.71]
TSP50_D = [0.24, 2.61]
TSP100_B = [1.07, 2.44]
TSP100_D = [0.61, 1.70]
x_axis = np.arange(len(Problem))
ax0.bar(x_axis-bar_width*1.5-0.04, TSP50_B, width=bar_width, label=Model[0], align='center', color='tab:blue',
        linewidth=2, edgecolor='tab:blue', alpha=0.85)
ax0.bar(x_axis-bar_width/2-0.03, TSP50_D, width=bar_width, label=Model[1], align='center', color='tab:purple',
        linewidth=2, facecolor='tab:purple', hatch='//')
ax0.bar(x_axis+bar_width/2+0.03, TSP100_B, width=bar_width, label=Model[2], align='center', color='tab:green',
        linewidth=2, edgecolor='tab:green', alpha=0.85)
ax0.bar(x_axis+bar_width*1.5+0.04, TSP100_D, width=bar_width, label=Model[3], align='center', color='tab:orange', 
        linewidth=2, facecolor='tab:orange', hatch='//')
ax0.margins(x=0.05)
ax0.set_ylim(0.0, 3.1)
ax0.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax0.set_xlabel("Problem size", fontsize=caption_size)
ax0.set_ylabel("Optimality gap (%)", fontsize=caption_size)
ax0.set_xticks(x_axis)
ax0.set_xticklabels(Problem)
# ax0.set_title('(a) TSP', fontsize=main_font_size, y=0, pad=-35, verticalalignment="top")
ax0.legend(fontsize=caption_size)
for c in ax0.containers:
    ax0.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-1)
    
# (b) CVRP
Model = ['CVRP50 (B)', 'CVRP50 (D)', 'CVRP100 (B)', 'CVRP100 (D)']
Problem = ['50','100']
CVRP50_B = [2.84, 5.79]
CVRP50_D = [2.64, 6.44]
CVRP100_B = [4.69, 4.24]
CVRP100_D = [4.20, 3.42]
x_axis = np.arange(len(Problem))
ax1.bar(x_axis-bar_width*1.5-0.04, CVRP50_B, width=bar_width, label=Model[0], align='center', color='tab:blue',
        linewidth=2, edgecolor='tab:blue', alpha=0.85)
ax1.bar(x_axis-bar_width/2-0.03, CVRP50_D, width=bar_width, label=Model[1], align='center', color='tab:purple',
        linewidth=2, facecolor='tab:purple', hatch='//')
ax1.bar(x_axis+bar_width/2+0.03, CVRP100_B, width=bar_width, label=Model[2], align='center', color='tab:green',
        linewidth=2, edgecolor='tab:green', alpha=0.85)
ax1.bar(x_axis+bar_width*1.5+0.04, CVRP100_D, width=bar_width, label=Model[3], align='center', color='tab:orange', 
        linewidth=2, facecolor='tab:orange', hatch='//')
ax1.margins(x=0.05)
ax1.set_ylim(0.0, 7.1)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.set_xlabel("Problem size", fontsize=caption_size)
ax1.set_ylabel("Optimality gap (%)", fontsize=caption_size)
ax1.set_xticks(x_axis)
ax1.set_xticklabels(Problem)
# ax1.set_title('(b) CVRP', fontsize=main_font_size, y=0, pad=-35, verticalalignment="top")
ax1.legend(fontsize=caption_size)
for c in ax1.containers:
    ax1.bar_label(c, padding=1, fmt='%.2f', fontsize=caption_size-1)

fig.tight_layout()
plt.subplots_adjust(wspace=0.3)

# Save fig
fig.set_size_inches(width, height/1.2)
plt.savefig('Result_plot/[2] generalization/TSP_CVRP.png', dpi=900)
plt.savefig('Result_plot/[2] generalization/TSP_CVRP.pdf', bbox_inches="tight")
plt.savefig('Result_plot/[2] generalization/TSP_CVRP.svg', bbox_inches="tight")
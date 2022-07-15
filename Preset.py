# -*- coding: utf-8 -*-
# reference: https://jonathansoma.com/lede/data-studio/matplotlib/list-all-fonts-available-in-matplotlib-plus-samples/

import numpy as np
import matplotlib as mpl
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/Arial.ttf").get_name()
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/Helvetica.ttf").get_name()
# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/NANUMBARUNPENR.TTF").get_name()

def setPlotStyle():    
    # plt.figure(dpi=300)  
    mpl.rcParams['font.family'] = "Times New Roman" #font_name
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['legend.title_fontsize'] = 12
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['axes.labelsize'] = 8
    mpl.rcParams['axes.grid'] = 0 # boolean    
    mpl.rcParams['axes.xmargin'] = 0.1     
    mpl.rcParams['axes.ymargin'] = 0.1     
    mpl.rcParams["mathtext.fontset"] = "dejavuserif" #"cm", "stix", etc.
    mpl.rcParams['figure.dpi'] = 600
    mpl.rcParams['savefig.dpi'] = 600
    





    
#--------------------------------------------- Non-task-specific imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
#warnings.filterwarnings('ignore')
import time
import seaborn as sns
import json
import os

# Run-finishing alert for long-running cells (copy down where needed)
#os.system('say "your program has finished"')

# Plotting-specific commands to enhance the plots
# Importing the color palettes
# with open('colors/thesis_specific_colors.json') as json_file:
#     specific_colors = json.load(json_file)

# with open("colors/color_palette.json", "r") as fp:
#     color_palette = json.load(fp)
color_palette = mpl.cycler(color=sns.color_palette("hls", 8))
color_palette = color_palette.by_key()['color'] # Converting the cycler to a list
deep_colors = sns.color_palette('deep')
pastel_colors = sns.color_palette('pastel')

# Font sizes
plt.rc('axes', titlesize=13)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('font', size=10)          # controls default text sizes

sns.set_style('darkgrid') # better-looking background grids

# Font similar to Latex font Computer Modern
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Change standard color cycle for plots to custom color cycle created for thesis
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=sns.color_palette("hls", 8)) 

latex_width = 455.8843 # width of latex document in pts
plots_filepath = 'images/plots/' # filepath for plots

# Function to set figure size to avoid scaling in LaTeX
def set_size(width=455.8843, fraction=0.75, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.
       source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 455.8843
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

# Function to slightly darken or lighten a color
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    source: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# Function to interpolate between to colors
# source: https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python (direct answer: https://stackoverflow.com/a/50784012)
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

# Function based on the colorFader to create num_colors between and including the two given colors
def rangeColorFader(c1, c2, num_colors):
    """Creates a list of num_colors colors between and including the two given colors
    Parameters----------
    c1: string
        First color
    c2: string
        Second color
    num_colors: int
        Number of colors to be created
    Returns -------
    pal: list
        List of num_colors colors between and including the two given colors
        """
    pal = []
    for x in np.linspace(0, num_colors, num_colors):    # interpolating the colors between the first and last color
        pal += [colorFader(c1, c2, x/num_colors)]
    return pal
# usage example:
# pal = [color_1]   # initializing the color palette with the first color
# for x in range(1, ridgeplot_df.model.nunique()-1):    # interpolating the colors between the first and last color
#       print(x/ridgeplot_df.model.nunique())
#       pal += [colorFader(color_1, color_2, x/ridgeplot_df.model.nunique())]
# pal += [color_2]   # adding the last color
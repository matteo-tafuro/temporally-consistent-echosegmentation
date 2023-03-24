import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def split_axis(ax, pad=0.05):
    """
    Split an axis into two axes.
    args:
        ax: Axis to split
        pad: Padding between the two axes
    returns:
        The two axes
    """

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", size="100%", pad=pad)
    fig1 = ax.get_figure()
    fig1.add_axes(ax2)
    
    return ax, ax2

def plot_contour_class(ax, mask, label, color, thickness=8):
    """
    Plot a contour of a specific class.
    args:
        ax: Axis to plot on
        
        mask: Mask to plot
        label: Label to plot
        color: Color of the contour
        thickness: Thickness of the contour
    """

    cmap = mpl.colors.ListedColormap([color])

    mask_label = mask == label
    
    img = (mask_label[..., np.newaxis]* 255).astype(np.uint8)

    mask = np.ones(img.shape, dtype=np.uint8) * 255
    cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(mask, [c], -1, (0, 0, 0), thickness=thickness)

    # mask using np.ma.masked
    masked = np.ma.masked_where(mask == 255, mask)
    ax.imshow(masked, cmap=cmap)
    ax.set_xlim([0,masked.shape[1]])


def define_box_properties(plot_name, color_code, label):
    """
    Utility function to chenge the attributes and the legend of a boxplot. It sets the outer color of the boxes 
    and the median line to black, while the inner color and the outliers are set to `color_code`.
    args:
        plot: the plot object returned by the `plt.boxplot` function
        color_code: the color code of the inner box
        label: the label of the boxplot
    """	

    for patch, median, outliers, color in zip(plot_name['boxes'], plot_name['medians'], plot_name['fliers'], color_code):
        patch.set_facecolor(color)
        median.set_color('white')
        outliers.set(markerfacecolor=color_code , markeredgecolor=None, markeredgewidth=0)
        
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend()

def boxplot_statistics(list1, list2, title, xlabel, ylabel, colors=['#D7191C', '#2C7BB6'], labels=['2CH', '4CH'], ticks=['Poor', 'Medium', 'Good']):
    
    fig = plt.figure(figsize = (8,6), dpi = 300)

    boxplot1 = plt.boxplot(list1, positions=np.array(np.arange(len(list1)))*2.0-0.35, widths=0.6, patch_artist=True)

    boxplot2 = plt.boxplot(list2, positions=np.array(np.arange(len(list1)))*2.0+0.35, widths=0.6, patch_artist=True)
    
    
    # setting colors for each groups
    define_box_properties(boxplot1, colors[0], labels[0])
    define_box_properties(boxplot2, colors[1], labels[1])
    
    # set the x label values
    plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)
    # Make title italics
    plt.xlabel(xlabel, fontstyle='italic', fontsize=12)
    plt.ylabel(ylabel, fontstyle='italic', fontsize=12)

    # set the limit for x axis
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(ymin=0)
    plt.grid(axis='y', alpha=0.3)
    
    # set the title
    plt.title(title, fontsize=14)

    plt.tight_layout()
    plt.show()

models_colors = ["#2596be", "#EAA70B", "#be2596"]
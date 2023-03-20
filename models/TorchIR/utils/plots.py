import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.eval import compute_dice
from matplotlib.collections import LineCollection
import numpy as np
import torch
import copy
import cv2


def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def plot_grid(u,v, step = 10, grids = None, ax=None, **kwargs):
    """
    Plot a deformation grid on the current axis.
    args:
        u, v: 2D arrays of the same shape
        step: plot every step-th vector. Used for downscaling and visualizing large grids.
        ax: matplotlib axis to plot on. If None, the current axis is used.
        kwargs: passed to matplotlib.collections.LineCollection
    """
    if grids is None:
        grid_x, grid_y = np.meshgrid(np.arange(0, u.shape[0], 1), np.arange(0, v.shape[1], 1))
    else:
        grid_x, grid_y = grids
    f = lambda x, y : (x + u, y + v)
    distx, disty = f(grid_x, grid_y)

    ax = ax or plt.gca()
    segs1 = np.stack((distx[::step, ::step], disty[::step, ::step]), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

    return (distx, disty)


def average_warped_image(model, frames, out_path, quality, view, patient, device='cpu'):
    """
    Plot the average warped image of a sequence of frames. This serves as a qualitative evaluation:
    the less blurry the average warped image is, the better the registration.
    args:
        model: the registration model
        frames: a sequence of frames of shape [no_frames,H,W]
        out_path: path to save the figure
        device: device to use for computation
    """
    
   # Initialize average images
    avg_moving = np.zeros((frames.shape[1], frames.shape[2]), dtype=np.float32)
    avg_warped = np.zeros((frames.shape[1], frames.shape[2]), dtype=np.float32)

    fixed = torch.from_numpy(frames[0]).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        for moving in frames[1:,:,:]:
            moving = torch.from_numpy(moving).unsqueeze(0).unsqueeze(0).to(device)
            warped = model(fixed, moving)
            # Add result to average
            avg_moving += moving.squeeze().cpu().numpy() / frames.shape[0]
            avg_warped += warped.squeeze().cpu().numpy() / frames.shape[0]

    # Compute absolute difference
    res_moving = cv2.absdiff(fixed.squeeze().cpu().numpy(), avg_moving)
    res_warped = cv2.absdiff(fixed.squeeze().cpu().numpy(), avg_warped)

    # Generate figure with results
    fig, ax = plt.subplots(2, 3, figsize = (7.5, 5))
    ax[0,0].imshow(fixed.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    ax[0,0].set_title('Fixed image')
    ax[0,1].imshow(avg_moving, cmap='gray', vmin=0, vmax=1)
    ax[0,1].set_title('Average moving image')
    ax[0,2].imshow(avg_warped, cmap='gray', vmin=0, vmax=1)
    ax[0,2].set_title('Average warped image')
    ax[1,1].imshow(res_moving, cmap='afmhot')
    ax[1,2].imshow(res_warped, cmap='afmhot')

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(out_path, f'{quality.lower()}_{view}_{str(patient).zfill(4)}_avg_warped_img.png'))
    plt.close(fig)


def plot_jacobian_determinant(jacobian, jacobian_masked, ax=None, redraw=True, **kwargs):
    """
    Plot the jacobian determinant of a displacement field.
    args:
        jacobian: jacobian determinant of shape [H,W]
        jacobian_masked: masked jacobian determinant of shape [H,W], specifying where the values are less than zero
        ax: matplotlib axis to plot on. If None, the current axis is used.
        kwargs: passed to matplotlib.collections.LineCollection
    """
    cmap = copy.copy(plt.cm.PuOr)
    cmap.set_under('red')
    ax = ax or plt.gca()
    # Plot jacobian determinant
    im1 = ax.imshow(jacobian, cmap=cmap, vmin=0, vmax=2)
    # Plot critical points, i.e. where the jacobian determinant is less than zero
    im2 = ax.imshow(jacobian_masked, cmap='bwr', vmin=-1e8, vmax=-1e7)
    # Plot colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    if not redraw: cax.set_axis_off()
    cbar = plt.colorbar(im1, cax=cax, cmap=cmap, ticks=[0.0, 0.5, 1.0, 1.5, 2.0], extend='min')
    cbar.ax.set_yticklabels(['< 0', '0.5', '1.0', '1.5', '> 2'])  



def generate_video(frames, warped_masks_masked, out_path, quality, view, patient, jacobians=None, jacobian_masked=None):
    """
    Generate a video of the original video, the warped mask, the warped mask on the original video and the jacobian determinant.
    args:
        frames: original video of shape [no_frames,H,W]
        warped_masks_masked: warped masks of shape [no_frames,H,W]
        jacobians: jacobian determinant of shape [no_frames,H,W]
        jacobian_masked: masked jacobian determinant of shape [no_frames,H,W], specifying where the values are less than zero
        out_path: path to save the video
        quality: quality of the video
        view: view of the video
        patient: patient number
    """

    fig, ax = plt.subplots(1, 3 if (jacobians is None and jacobian_masked is None) else 4, figsize=(10.5 if (jacobians is None and jacobian_masked is None) else 15.5, 4.5), dpi=125)
    size = None
    fps = 10
    filename = f'{quality.lower()}_{view}_{str(patient).zfill(4)}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i in range(frames.shape[0]):
        # Plot video
        ax[0].imshow(frames[i], cmap='gray', vmin=0, vmax=1)
        ax[0].set_title('Original video')

        # Plot masks
        ax[1].imshow(warped_masks_masked[i])
        ax[1].set_title('Warped mask')

        # Plot both
        ax[2].imshow(frames[i], cmap='gray', vmin=0, vmax=1)
        ax[2].imshow(warped_masks_masked[i], alpha=0.6)
        ax[2].set_title('Warped mask on original video')

        # Plot the pairwise jacobian determinant
        if (jacobians is not None) and (jacobian_masked is not None):
            plot_jacobian_determinant(jacobians[i], jacobian_masked[i], ax=ax[3], redraw=True if i==0 else False)
            ax[3].set_title('Jacobian determinant')

        # set the spacing between subplots
        plt.subplots_adjust(left=0.025,
                            bottom=0.15,
                            right=0.975,
                            top=0.75,
                            wspace=0.1,
                            hspace=0.1)
        plt.suptitle(f"{quality.upper()}_{view}_{str(patient).zfill(4)}", fontweight="bold", fontsize=15, y=0.9)

        im = figure_to_array(fig)
        
        if size is None:
            size = im.shape[1], im.shape[0] # W,H
            video = cv2.VideoWriter(os.path.join(out_path, filename), fourcc, fps, size)

        im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
        video.write(im)
        
        for a in ax:
            a.clear()
    
    plt.close(fig)
    video.release()
    

def plot_masks_comparison(ground_truth, warped_mask_masked, out_path, quality, view, patient):
    """
    Plot a comparison of ground truth and the warped mask. Also compute and include the DICE scores.
    args:
        ground_truth: ground truth mask of shape [H,W]
        warped_mask_masked: warped mask of shape [H,W]
        out_path: path to save the plot
        quality: quality of the video
        view: view of the video
        patient: patient number
    """

    # Compute dice scores
    BG = compute_dice(warped_mask_masked, ground_truth, 0)
    LV_end = compute_dice(warped_mask_masked, ground_truth, 1)
    LV_epi = compute_dice(warped_mask_masked, ground_truth, 2)
    LA = compute_dice(warped_mask_masked, ground_truth, 3)

    tmp = "DICE SCORES:\n"
    # tmp += f'Dice score for background: {BG:.3f}\n'
    tmp += f'LV_endo: {LV_end:.3f}\n'
    tmp += f'LV_epi: {LV_epi:.3f}\n'
    tmp += f'LA: {LA:.3f}\n'

    # Plot the two masks and paste the dice scores
    fig, ax = plt.subplots(1,3,figsize=(15,5), dpi=150)
    ax[0].imshow(ground_truth)
    ax[0].set_title('Ground truth mask')
    ax[1].imshow(warped_mask_masked)
    ax[1].set_title('Generated mask')
    ax[2].axis('off')
    plt.text(0, 0.95, tmp, fontsize=20, horizontalalignment='left', verticalalignment='top', transform=ax[2].transAxes)
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, f'{quality.lower()}_{view}_{str(patient).zfill(4)}.png'))
    plt.close(fig)


def jacobian_std_boxplot(jac_std, out_path):
    """
    Plot a boxplot of the jacobian determinant standard deviation for each patient.
    args:
        jac_std: dictionary of the jacobian determinant standard deviation for each of the six chosen patients:
            jac_std = {VIDEOQUALITY_VIEW: std}...
        out_path: path to save the plot
    """
    # Boxplot with the jacobian std of the different patients
    fig = plt.figure(figsize = (8,6), dpi = 150)
    boxplot = plt.boxplot(
                jac_std.values(),       # data
                vert=True,              # vertical box alignment
                patch_artist=True,      # fill with color
                labels=jac_std.keys(),  # xlabels
            )

    # Customize the appearance of the plot
    reds = mpl.cm.get_cmap('Reds')
    blues = mpl.cm.get_cmap('Blues')

    reds = [reds(i) for i in [0.7, 0.5, 0.3]]
    blues = [blues(i) for i in [0.8, 0.6, 0.4]]
    colors = reds + blues

    for patch, median, outliers, color in zip(boxplot['boxes'], boxplot['medians'], boxplot['fliers'], colors):
        median.set_color('black')
        outliers.set(markerfacecolor=color)
        patch.set_facecolor(color)

    plt.ylim(ymin=0)
    plt.ylabel('STD of the Jacobian Determinant')
    plt.grid(True, linestyle=':', alpha=0.75)
    plt.tight_layout() 
    fig.savefig(os.path.join(out_path, f'jacobian_boxplot.png'))
    plt.close(fig)
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.widgets import Slider

def create_visualization(frame_heights_mm, frame_names, overlays, layer_ids, args):
    """Create visualization with height plots and overlays."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 3]})
    plt.subplots_adjust(bottom=0.25)

    # Format frame names for display
    display_names = [os.path.basename(x).replace("." + os.path.basename(x).split(".")[-1], "") for x in frame_names]
    
    # Plot with or without layer coloring
    if args.color_layers:
        axs[0].plot(
            range(len(frame_heights_mm)),
            frame_heights_mm,
            linestyle='-',
            color='lightgray',
            alpha=0.5,
            zorder=1
        )
        
        # Get unique layer IDs and assign colors
        unique_layers = np.unique(layer_ids)
        num_layers = len(unique_layers)
        colors = cm.get_cmap('tab20', num_layers)
        
        for layer_idx in unique_layers:
            layer_indices = np.where(layer_ids == layer_idx)[0]
            layer_heights = frame_heights_mm[layer_indices]
            
            axs[0].scatter(
                layer_indices,
                layer_heights,
                marker='o',
                color=colors(layer_idx),
                label=f'Layer {layer_idx+1}',
                s=50,  
                zorder=2  
            )
            
            # Calculate and display statistics
            mean_height = np.mean(layer_heights)
            std_dev = np.std(layer_heights)
            mid_idx = layer_indices[len(layer_indices)//2]
            
            axs[0].text(
                mid_idx, mean_height + 0.2, 
                f"μ={mean_height:.2f}\nσ={std_dev:.2f}", 
                ha='center', va='bottom',
                color=colors(layer_idx),
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor=colors(layer_idx), pad=1.5)
            )
    else:
        # Simple line plot
        label = "Our Model" if args.enable_temporal else "Baseline model"
        axs[0].plot(
            range(len(frame_names)), 
            frame_heights_mm, 
            marker='o', 
            linestyle='-', 
            color='blue', 
            label=label
        )
    
    # Set up plot labels and formatting
    axs[0].set_title('CURRENT_PART Height per Frame')
    axs[0].set_ylabel('Height (mm)')
    axs[0].set_xlabel('Frame')
    axs[0].set_xticks(range(len(frame_names)))
    axs[0].set_xticklabels(display_names, rotation=45, ha='right', fontsize=8)
    axs[0].legend(loc='best', fontsize=8)

    # Display initial overlay
    overlay_ax = axs[1]
    overlay_img_obj = overlay_ax.imshow(overlays[0])
    overlay_ax.axis('off')
    overlay_ax.set_title(f"Frame: {display_names[0]}")

    # Create slider for interactivity
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame Index',
        valmin=0,
        valmax=len(overlays) - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        idx = int(frame_slider.val)
        overlay_img_obj.set_data(overlays[idx])
        overlay_ax.set_title(f"Frame: {display_names[idx]}")
        fig.canvas.draw_idle()

    frame_slider.on_changed(update)
    
    # Adjust layout and save figure
    plt.tight_layout()
    y_min, y_max = axs[0].get_ylim()
    axs[0].set_ylim(y_min, y_max * 1.15)
    
    return fig, axs


def get_output_filename(args):
    """Generate output filename based on input source."""
    if args.image_dir:
        dir_name = os.path.basename(os.path.normpath(args.image_dir))
        file_prefix = f"{dir_name}_"
    elif args.video:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        file_prefix = f"{video_name}_"
    elif args.image:
        img_name = os.path.splitext(os.path.basename(args.image))[0]
        file_prefix = f"{img_name}_"
    else:
        file_prefix = ""
        
    filters_used = []
    if args.use_median:
        filters_used.append("median")
    if args.enable_temporal:
        filters_used.append("temporal")
        
    filename_suffix = "_".join(filters_used) if filters_used else "unfiltered"
    return f"{file_prefix}height_plot_{filename_suffix}.png"

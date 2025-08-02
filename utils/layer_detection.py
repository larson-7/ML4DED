import numpy as np

def group_layers_by_height(heights, threshold=0.2):
    """Group frame indices into layers based on height changes."""
    layers = []
    current_layer = [0]
    
    for i in range(1, len(heights)):
        current_avg = np.mean([heights[j] for j in current_layer])
        if heights[i] > current_avg + threshold:
            layers.append(current_layer)
            current_layer = [i]
        else:
            current_layer.append(i)
    
    # Add the last layer
    if current_layer:
        layers.append(current_layer)
        
    return layers

def convert_layers_to_ids(layers, total_length):
    """Convert layer groups to a layer ID array for each frame."""
    layer_ids = np.zeros(total_length, dtype=int)
    for layer_idx, indices in enumerate(layers):
        for idx in indices:
            layer_ids[idx] = layer_idx
    return layer_ids
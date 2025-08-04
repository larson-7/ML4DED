import numpy as np
from scipy.ndimage import median_filter, label

def compute_temporal_metrics(heights_mm, confidence_scores=None):
    """
    Compute comprehensive temporal stability metrics for part height.
    
    Args:
        heights_mm: Array of part heights in mm
        confidence_scores: Optional array of segmentation confidence scores
        
    Returns:
        Dictionary of temporal metrics
    """
    metrics = {}
    
    # 1. Frame-to-frame metrics
    if len(heights_mm) > 1:
        frame_diffs = np.diff(heights_mm)
        metrics['frame_to_frame_variance'] = np.var(frame_diffs)
        metrics['frame_to_frame_std'] = np.std(frame_diffs)
        metrics['mean_absolute_frame_diff'] = np.mean(np.abs(frame_diffs))
        
        # Jitter metric (sum of absolute differences)
        metrics['total_jitter'] = np.sum(np.abs(frame_diffs))
        
    # 2. Smoothness metric (second derivative)
    if len(heights_mm) > 2:
        second_derivative = np.diff(heights_mm, n=2)
        metrics['smoothness_metric'] = np.mean(np.abs(second_derivative))
        metrics['max_acceleration'] = np.max(np.abs(second_derivative))
    
    # 3.  height_regression
    if len(heights_mm) > 1:
        height_regression = np.sum(np.diff(heights_mm) < -0.1)  # Allow small measurement noise
        metrics['height_regressions'] = height_regression
        metrics['height_regression_rate'] = height_regression / (len(heights_mm) - 1)
    
    # 4. Overall statistics
    metrics['height_mean'] = np.mean(heights_mm)
    metrics['height_std'] = np.std(heights_mm)
    metrics['height_cv'] = metrics['height_std'] / metrics['height_mean'] if metrics['height_mean'] > 0 else float('inf')
    metrics['height_range'] = np.max(heights_mm) - np.min(heights_mm)
    
    # 5. Confidence-weighted metrics (if confidence scores provided)
    if confidence_scores is not None and len(confidence_scores) == len(heights_mm):
        metrics['mean_confidence'] = np.mean(confidence_scores)
        metrics['min_confidence'] = np.min(confidence_scores)
        # Confidence-weighted height variance
        weights = confidence_scores / np.sum(confidence_scores)
        weighted_mean = np.sum(heights_mm * weights)
        weighted_var = np.sum(weights * (heights_mm - weighted_mean)**2)
        metrics['confidence_weighted_variance'] = weighted_var
    
    return metrics

def analyze_layer_metrics(heights_mm, layer_ids, confidence_scores=None):
    """
    Compute per-layer and cross-layer metrics.
    
    Args:
        heights_mm: Array of part heights in mm
        layer_ids: Array of layer identifiers for each height measurement
        confidence_scores: Optional array of segmentation confidence scores
        
    Returns:
        Dictionary of layer metrics
    """
    layer_metrics = {}
    num_layers = np.max(layer_ids) + 1
    
    layer_means = []
    layer_stds = []
    layer_cvs = []
    
    for layer in range(num_layers):
        idxs = np.where(np.array(layer_ids) == layer)[0]
        if len(idxs) > 0:
            layer_heights = heights_mm[idxs]
            mean_h = np.mean(layer_heights)
            std_h = np.std(layer_heights)
            cv = std_h / mean_h if mean_h > 0 else float('inf')
            
            layer_means.append(mean_h)
            layer_stds.append(std_h)
            layer_cvs.append(cv)
            
            layer_metrics[f'layer_{layer+1}'] = {
                'mean_height': mean_h,
                'std_height': std_h,
                'cv': cv,
                'num_frames': len(idxs),
                'height_range': np.max(layer_heights) - np.min(layer_heights)
            }
            
            # Add confidence metrics if available
            if confidence_scores is not None and len(confidence_scores) > 0:
                # Convert idxs to list if needed and make sure indices are valid
                valid_indices = [i for i in idxs if i < len(confidence_scores)]
                if valid_indices:
                    layer_conf = [confidence_scores[i] for i in valid_indices]
                    layer_metrics[f'layer_{layer+1}']['mean_confidence'] = np.mean(layer_conf)
    
    # Cross-layer metrics
    if len(layer_means) > 1:
        layer_metrics['cross_layer'] = {
            'mean_layer_height': np.mean(layer_means),
            'std_layer_means': np.std(layer_means),
            'mean_cv': np.mean(layer_cvs),
            'layer_height_consistency': 1 - (np.std(layer_means) / np.mean(layer_means))
        }
        
        # Layer-to-layer height changes
        layer_height_diffs = np.diff(layer_means)
        layer_metrics['cross_layer']['mean_layer_increment'] = np.mean(layer_height_diffs)
        layer_metrics['cross_layer']['std_layer_increment'] = np.std(layer_height_diffs)
    
    return layer_metrics

def detect_anomalies(heights_mm, layer_ids):

    anomalies = {
        'sudden_jumps': [],
        'sudden_drops': [],
        'plateaus': [],
        'high_variance_regions': []
    }
    
    # Detect sudden changes
    if len(heights_mm) > 1:
        diffs = np.diff(heights_mm)
        jump_threshold = np.std(diffs) * 2  # 2 sigma threshold
        
        for i, diff in enumerate(diffs):
            if diff > jump_threshold:
                anomalies['sudden_jumps'].append({
                    'frame': i+1,
                    'magnitude': diff
                })
            elif diff < -0.2:  # Drops shouldn't happen in DED
                anomalies['sudden_drops'].append({
                    'frame': i+1,
                    'magnitude': diff
                })
    
    # Detect high variance regions
    window_size = 5
    if len(heights_mm) >= window_size:
        for i in range(len(heights_mm) - window_size + 1):
            window = heights_mm[i:i+window_size]
            if np.std(window) > 0.5:  # Threshold for high variance
                anomalies['high_variance_regions'].append({
                    'start_frame': i,
                    'end_frame': i+window_size-1,
                    'std': np.std(window)
                })
    
    return anomalies

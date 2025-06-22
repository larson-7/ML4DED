import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist
import cv2
from sklearn.metrics import silhouette_score
from scipy.ndimage import binary_erosion, binary_dilation


def inter_class_depth_separation(depth_map, seg_map, class_pairs=None):
    """
    Measure how well depth separates different object classes.
    Higher values = better separation between objects.
    """
    if class_pairs is None:
        # Test all pairs of classes
        classes = np.unique(seg_map)
        class_pairs = [(i, j) for i in classes for j in classes if i < j]

    separations = {}

    for class_a, class_b in class_pairs:
        mask_a = seg_map == class_a
        mask_b = seg_map == class_b

        if not (np.any(mask_a) and np.any(mask_b)):
            continue

        depths_a = depth_map[mask_a]
        depths_b = depth_map[mask_b]

        # Cohen's d (effect size) - measures separation between distributions
        pooled_std = np.sqrt(((len(depths_a) - 1) * np.var(depths_a) +
                              (len(depths_b) - 1) * np.var(depths_b)) /
                             (len(depths_a) + len(depths_b) - 2))
        cohens_d = abs(np.mean(depths_a) - np.mean(depths_b)) / (pooled_std + 1e-8)

        # Wasserstein distance (Earth Mover's Distance)
        wasserstein = stats.wasserstein_distance(depths_a, depths_b)

        # Overlap coefficient (lower = better separation)
        hist_a, bins = np.histogram(depths_a, bins=50, density=True)
        hist_b, _ = np.histogram(depths_b, bins=bins, density=True)
        overlap = np.sum(np.minimum(hist_a, hist_b)) * (bins[1] - bins[0])

        separations[(class_a, class_b)] = {
            'cohens_d': cohens_d,
            'wasserstein': wasserstein,
            'overlap': overlap,
            'mean_diff': abs(np.mean(depths_a) - np.mean(depths_b))
        }

    return separations


def depth_consistency_within_objects(depth_map, seg_map):
    """
    Measure depth consistency within each object.
    Lower values = more consistent depth within objects.
    """
    consistency_metrics = {}

    for class_id in np.unique(seg_map):
        mask = seg_map == class_id
        if np.sum(mask) < 10:  # Skip tiny objects
            continue

        depths = depth_map[mask]

        # Coefficient of variation (normalized standard deviation)
        cv = np.std(depths) / (np.mean(depths) + 1e-8)

        # Median Absolute Deviation (robust to outliers)
        mad = np.median(np.abs(depths - np.median(depths)))

        # Local smoothness - compare each pixel to its neighbors
        coords = np.where(mask)
        if len(coords[0]) > 100:  # Only for larger objects
            # Sample subset for efficiency
            idx = np.random.choice(len(coords[0]), min(500, len(coords[0])), replace=False)
            y_coords, x_coords = coords[0][idx], coords[1][idx]

            local_variations = []
            for i, (y, x) in enumerate(zip(y_coords, x_coords)):
                # Get 3x3 neighborhood
                y_min, y_max = max(0, y - 1), min(depth_map.shape[0], y + 2)
                x_min, x_max = max(0, x - 1), min(depth_map.shape[1], x + 2)

                neighborhood = depth_map[y_min:y_max, x_min:x_max]
                seg_neighborhood = seg_map[y_min:y_max, x_min:x_max]

                # Only consider pixels of the same class
                same_class_depths = neighborhood[seg_neighborhood == class_id]
                if len(same_class_depths) > 1:
                    local_var = np.std(same_class_depths)
                    local_variations.append(local_var)

            local_smoothness = np.mean(local_variations) if local_variations else 0
        else:
            local_smoothness = 0

        consistency_metrics[class_id] = {
            'coefficient_of_variation': cv,
            'median_abs_deviation': mad,
            'local_smoothness': local_smoothness
        }

    return consistency_metrics


def boundary_depth_contrast(depth_map, seg_map, boundary_width=5):
    """
    Measure depth contrast at object boundaries.
    Higher values = sharper depth transitions at boundaries.
    """
    boundary_contrasts = {}

    for class_id in np.unique(seg_map):
        if class_id == 0:  # Skip background
            continue

        mask = seg_map == class_id
        if np.sum(mask) < 50:
            continue

        # Get boundary pixels
        eroded = binary_erosion(mask, iterations=boundary_width // 2)
        dilated = binary_dilation(mask, iterations=boundary_width // 2)

        inner_boundary = mask & ~eroded  # Pixels just inside object
        outer_boundary = dilated & ~mask  # Pixels just outside object

        if not (np.any(inner_boundary) and np.any(outer_boundary)):
            continue

        inner_depths = depth_map[inner_boundary]
        outer_depths = depth_map[outer_boundary]

        # Average depth difference at boundaries
        mean_contrast = abs(np.mean(inner_depths) - np.mean(outer_depths))

        # Maximum local contrast
        max_contrast = 0
        inner_coords = np.where(inner_boundary)

        for i in range(min(100, len(inner_coords[0]))):  # Sample for efficiency
            y, x = inner_coords[0][i], inner_coords[1][i]

            # Find nearby outer boundary pixels
            y_min, y_max = max(0, y - boundary_width), min(depth_map.shape[0], y + boundary_width + 1)
            x_min, x_max = max(0, x - boundary_width), min(depth_map.shape[1], x + boundary_width + 1)

            local_outer = outer_boundary[y_min:y_max, x_min:x_max]
            if np.any(local_outer):
                local_outer_depths = depth_map[y_min:y_max, x_min:x_max][local_outer]
                local_contrast = abs(depth_map[y, x] - np.mean(local_outer_depths))
                max_contrast = max(max_contrast, local_contrast)

        boundary_contrasts[class_id] = {
            'mean_boundary_contrast': mean_contrast,
            'max_boundary_contrast': max_contrast
        }

    return boundary_contrasts


def small_object_visibility(depth_map, seg_map, size_threshold=5000):
    """
    Evaluate if small objects have distinguishable depth from their surroundings.
    """
    small_object_metrics = {}

    for class_id in np.unique(seg_map):
        if class_id == 0:  # Skip background
            continue

        mask = seg_map == class_id
        object_size = np.sum(mask)

        if object_size < size_threshold:  # Small object
            # Get surrounding context (dilated region minus object)
            context_mask = binary_dilation(mask, iterations=20) & ~mask

            if not np.any(context_mask):
                continue

            object_depths = depth_map[mask]
            context_depths = depth_map[context_mask]

            # Signal-to-noise ratio
            signal = abs(np.mean(object_depths) - np.mean(context_depths))
            noise = np.sqrt(np.var(object_depths) + np.var(context_depths))
            snr = signal / (noise + 1e-8)

            # Detectability score
            detectability = signal / (np.std(context_depths) + 1e-8)

            small_object_metrics[class_id] = {
                'object_size': object_size,
                'depth_snr': snr,
                'detectability': detectability,
                'depth_separation': signal
            }

    return small_object_metrics


def gradient_magnitude_analysis(depth_map, seg_map):
    """
    Analyze depth gradients to identify if details are washed out.
    """
    # Compute depth gradients
    grad_y, grad_x = np.gradient(depth_map)
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    gradient_stats = {}

    for class_id in np.unique(seg_map):
        mask = seg_map == class_id
        if np.sum(mask) < 10:
            continue

        class_gradients = gradient_magnitude[mask]

        # Gradient statistics
        gradient_stats[class_id] = {
            'mean_gradient': np.mean(class_gradients),
            'gradient_std': np.std(class_gradients),
            'gradient_percentiles': {
                '50th': np.percentile(class_gradients, 50),
                '90th': np.percentile(class_gradients, 90),
                '95th': np.percentile(class_gradients, 95)
            },
            'high_gradient_ratio': np.mean(class_gradients > np.percentile(gradient_magnitude, 75))
        }

    return gradient_stats, gradient_magnitude


def comprehensive_depth_evaluation(depth_map, seg_map, class_labels_enum=None):
    """
    Run all evaluation metrics and provide a comprehensive report.
    """

    print("\n--- Depth Metrics Panel ---")

    # 1. Inter-class separation
    print("1. OBJECT SEPARATION:")
    separations = inter_class_depth_separation(depth_map, seg_map)
    for (class_a, class_b), metrics in separations.items():
        name_a = class_labels_enum(class_a).name if class_labels_enum else f"Class {class_a}"
        name_b = class_labels_enum(class_b).name if class_labels_enum else f"Class {class_b}"
        print(f"   {name_a} vs {name_b}:")
        print(f"     Cohen's d: {metrics['cohens_d']:.3f} (>0.8 = good separation)")
        print(f"     Depth difference: {metrics['mean_diff']:.4f}")
        print(f"     Distribution overlap: {metrics['overlap']:.3f} (lower = better)")

    # 2. Within-object consistency
    print("\n2. WITHIN-OBJECT CONSISTENCY:")
    consistency = depth_consistency_within_objects(depth_map, seg_map)
    for class_id, metrics in consistency.items():
        name = class_labels_enum(class_id).name if class_labels_enum else f"Class {class_id}"
        print(f"   {name}:")
        print(f"     Coefficient of variation: {metrics['coefficient_of_variation']:.3f} (lower = more consistent)")
        print(f"     Local smoothness: {metrics['local_smoothness']:.4f}")

    # 3. Boundary contrast
    print("\n3. BOUNDARY CONTRAST:")
    boundaries = boundary_depth_contrast(depth_map, seg_map)
    for class_id, metrics in boundaries.items():
        name = class_labels_enum(class_id).name if class_labels_enum else f"Class {class_id}"
        print(f"   {name}:")
        print(f"     Mean boundary contrast: {metrics['mean_boundary_contrast']:.4f}")
        print(f"     Max boundary contrast: {metrics['max_boundary_contrast']:.4f}")

    # 4. Small object visibility
    print("\n4. SMALL OBJECT VISIBILITY:")
    small_objects = small_object_visibility(depth_map, seg_map)
    for class_id, metrics in small_objects.items():
        name = class_labels_enum(class_id).name if class_labels_enum else f"Class {class_id}"
        print(f"   {name} (size: {metrics['object_size']} pixels):")
        print(f"     Depth SNR: {metrics['depth_snr']:.3f} (>3.0 = good)")
        print(f"     Detectability: {metrics['detectability']:.3f}")

    # 5. Gradient analysis
    print("\n5. GRADIENT ANALYSIS (Detail Preservation):")
    gradients, _ = gradient_magnitude_analysis(depth_map, seg_map)
    for class_id, metrics in gradients.items():
        name = class_labels_enum(class_id).name if class_labels_enum else f"Class {class_id}"
        print(f"   {name}:")
        print(f"     Mean gradient: {metrics['mean_gradient']:.4f}")
        print(f"     High gradient ratio: {metrics['high_gradient_ratio']:.3f} (higher = more detail)")

    return {
        'separations': separations,
        'consistency': consistency,
        'boundaries': boundaries,
        'small_objects': small_objects,
        'gradients': gradients
    }
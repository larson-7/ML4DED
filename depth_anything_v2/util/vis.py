import numpy as np

def decode_segmap(image, nc=40):
    # NYUv2 40-class color palette (taken from official toolbox, can be customized)
    label_colors = np.array([
        (  0,   0,   0), (  0, 128, 128), (  0,   0, 128), (  0, 128,   0),
        (128,  64, 128), (128,   0, 128), (128, 128, 128), ( 64,   0, 128),
        (192,   0, 128), ( 64, 128, 128), (192, 128, 128), ( 64,   0,   0),
        (192,   0,   0), ( 64, 128,   0), (192, 128,   0), ( 64,  64,   0),
        (192,  64,   0), ( 64,   0,  64), (192,   0,  64), (  0, 192,   0),
        (128, 192,   0), (  0,  64, 128), (128,  64,   0), (  0, 192, 128),
        (128, 192, 128), (  0,  64,   0), (128,  64, 128), (  0, 192,  64),
        (128, 192,  64), (  0,  64,  64), (128,  64,  64), (192, 192,   0),
        ( 64, 192,   0), (192,  64,   0), ( 64, 192, 128), (192, 192, 128),
        ( 64,  64, 128), (192,  64, 128), ( 64, 192,  64), (192, 192,  64),
    ])
    # Ensure label_colors matches the number of classes
    assert label_colors.shape[0] >= nc, f"Color map only covers {label_colors.shape[0]} classes, need {nc}."

    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    for l in range(nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

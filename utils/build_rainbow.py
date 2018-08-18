import numpy as np

def build_rainbow(n, curve=None):
    rgb = []
    width = 2 * np.pi
    for i in range(3):
        offset = -i * width / 3
        cur = np.cos(np.linspace(offset, offset + width, n))
        rgb.append(cur)
    rainbow = (1 + np.vstack(rgb)) / 2
    if curve:
        rainbow = curve(rainbow)
    rainbow = np.minimum(rainbow * 256, 255).astype(int)
    return rainbow.T
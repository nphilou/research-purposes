import shutil
from io import BytesIO
from math import sqrt

import IPython.display
import PIL.Image
import numpy as np
from IPython.display import display, Image


def show_array(a, fmt='png', filename=None, retina=False, zoom=None):
    if len(a.shape) == 1:
        n = len(a)
        side = int(sqrt(n))
        if (side * side) == n:
            a = a.reshape(side, side)
        else:
            raise ValueError('input is one-dimensional', a.shape)
    a = np.uint8(np.clip(a, 0, 255))
    image_data = BytesIO()
    PIL.Image.fromarray(a).save(image_data, fmt)
    if filename is None:
        height, width = a.shape[:2]
        if zoom is not None:
            width *= zoom
            height *= zoom
        IPython.display.display(IPython.display.Image(data=image_data.getvalue(),
                                                      width=width,
                                                      height=height,
                                                      retina=retina))

        img = Image(image_data.getvalue(), width=width, height=height, retina=retina)
        display(img)
    else:
        with open(filename, 'wb') as f:
            image_data.seek(0)
            shutil.copyfileobj(image_data, f)

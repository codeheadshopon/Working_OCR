# -*- coding: utf-8 -*-

from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import random
import os
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
from sys import getdefaultencoding
import sys
import random
import matplotlib.pyplot as plt

d = getdefaultencoding()
if d != "utf-8":
    reload(sys)
    sys.setdefaultencoding("utf-8")


def speckle(img):
    severity = np.random.uniform(0, 0.0)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise
def imsave(fname, arr, vmin=None, vmax=None, cmap='gray', format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)

def paint_text(text, w, h, fontsize, rotate=False, ud=True, multi_fonts=False):
    newtext = ""
    banglachars = "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ঃৎং"
    # text = "নড়চড়"
    chars = []
    for i in range(0,len(banglachars),3):
        chars.append(banglachars[i:i+3])
    for i in range(0,len(text),3):
        ch=text[i:i+3]
        itsoke= 1
        for j in chars:
            if j==ch:
                itsoke = 0

    # text="অ"
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    import random

    FlagBlack = random.randint(0, 4)
    FlagBlack = 1
    with cairo.Context(surface) as context:
        if (FlagBlack == 2):
            context.set_source_rgb(0, 0, 0)  # White
        else:
            context.set_source_rgb(1, 0, 1)  # White

        context.paint()
        # this font list works in CentOS 7
        if multi_fonts:
            fonts = ['Solaimanlipi', 'Siyamrupali', 'kalpurush', 'Lohit', 'prothoma']
            context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                                     np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
        else:
            context.select_font_face('Solaimanlipi', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        import random

        context.set_font_size(fontsize)
        box = context.text_extents(text)
        border_w_h = (4, 4)
        if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
            while box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
                idx=len(text)-1
                for i in range(len(text)-1,0,-1):
                    # print(text[i])
                    if text[i]==" ":
                        idx=i
                        break
                text=text[0:idx]
                box = context.text_extents(text)
            box = context.text_extents(text)

            # raise IOError('Could not fit string into image. Max char count is too large for given image width.')

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate


        max_shift_x = w - box[2]
        max_shift_y = h - box[3] - border_w_h[1]
        top_left_x = np.random.randint(0, int(max_shift_x))
        if ud:
            rando= np.random.randint(0, int(max_shift_y))

            top_left_y =  rando
        else:
            if fontsize>40:
                top_left_y = h // 6
            elif fontsize>35:
                top_left_y = h // 4
            elif fontsize>30:
                top_left_y = h // 3
            else:
                top_left_y = h // 2


        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        if (FlagBlack == 2):
            context.set_source_rgb(1, 1, 1)
        else:
            context.set_source_rgb(0, 0, 0)

        # print(text)
        context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4)
    a = a[:, :, 0]  # grab single channel
    # imsave("test_2.jpg",a)
    # plt.imshow(a,cmap='gray')
    # plt.show()
    # a = speckle(a)

    # a = a.astype(np.float32) / 255
    # a = np.expand_dims(a, 0)
    # if rotate:
    #     a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
    a = speckle(a)


    return a

# paint_text("বাংলা ভাষা",1558,424,145)

# ঢাকা এদেশের রাজধানী
# -*- coding: utf-8 -*-

import os
import itertools
import codecs
import re
import datetime
import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
import pyrszimg

from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks
import wordgenerators_sequential as wg
from sys import getdefaultencoding
import sys
import random
import imagemaker as code
from PIL import Image
import matplotlib.pyplot as plt
d = getdefaultencoding()
if d != "utf-8":
    reload(sys)
    sys.setdefaultencoding("utf-8")
OUTPUT_DIR = 'image_ocr'

# character classes and matching regex filter
regex = r'^[a-z ]+$'
alphabet = u'abcdefghijklmnopqrstuvwxyz '

np.random.seed(55)


# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.6)
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


Dict = {'অ': 0,
        'আ': 1,
        'ই': 2,
        'ঈ': 3,
        'উ': 4,
        'ঊ': 5,
        'ঋ': 6,
        'এ': 7,
        'ঐ': 8,
        'ও': 9,
        'ঔ': 10,
        'ক': 11,
        'খ': 12,
        'গ': 13,
        'ঘ': 14,
        'ঙ': 15,
        'চ': 16,
        'ছ': 17,
        'জ': 18,
        'ঝ': 19,
        'ঞ': 20,
        'ট': 21,
        'ঠ': 22,
        'ড': 23,
        'ঢ': 24,
        'ণ': 25,
        'ত': 26,
        'থ': 27,
        'দ': 28,
        'ধ': 29,
        'ন': 30,
        'প': 31,
        'ফ': 32,
        'ব': 33,
        'ভ': 34,
        'ম': 35,
        'য': 36,
        'র': 37,
        'ল': 38,
        'শ': 39,
        'ষ': 40,
        'স': 41,
        'হ': 42,
        'ড়': 43,
        'ঢ়': 44,
        'য়': 45,
        'ৎ': 46,
        'ঃ': 47,
        'ং': 48
        }
# def paint_text(text, Flags, w, h, rotate=False, ud=True, multi_fonts=False):
#     background = Image.new('L', (564, 64), (0))
#     bg_w, bg_h = background.size
#     img_h = 64
#     splitwords = []
#     PrevInd = 0
#     # print(text)
#     # print(text)
#     Flags=[]
#     # basic=wg.basiWords()
#
#     for i in range(len(text)):
#         if text[i] == ' ':
#             if(text[PrevInd:i]!=""):
#                 splitwords.append(text[PrevInd:i])
#                 PrevInd = i + 1
#             # for i in range(PrevInd,i,3):
#         if i == len(text) - 1:
#             if(text[PrevInd:i+1]!=""):
#                 splitwords.append(text[PrevInd:i + 1])
#     #     print(splitwords[i])
#
#
#     Flag = 0
#     Start = 0
#
#     splitwords = splitwords[0:len(splitwords) - 1]
#
#     Size = random.randint(20,25)
#     Fl=2
#     for i in splitwords:
#         # Flag= random.randint(0,3)
#         if (Fl != 0):
#             text = i
#             img_1 = code.paint_text(text, 128, 64,Size)
#             im = Image.fromarray(np.uint8(img_1 * 255))
#             offset = (Start, (bg_h - img_h) / 2)
#             background.paste(im, offset)
#             Start += (128+2)
#             Flag = 1
#         else:
#             convertedarray = []
#             for j in range(0, len(i), 3):
#                 # print(i[j:j + 3])
#                 convertedarray.append(Dict[i[j:j + 3]] + 1)
#             # print(convertedarray)
#             for j in range(len(convertedarray)):
#                 charnumber = convertedarray[j]
#                 ImageName = 'Images/' + str(charnumber)
#                 for filename in os.listdir(ImageName):
#                     ImageName += "/"
#                     ImageName += filename
#                     break
#                 img = Image.open(ImageName)
#
#                 # img=Image.open('a.png')
#                 img = img.resize((Size+8, Size+8), Image.ANTIALIAS)
#                 # print((bg_h-img_h) /2)
#                 offset = (Start, img_h/3)
#                 background.paste(img, offset)
#
#                 Start += (Size+2)
#             Flag = 0
#         Fl+=1
#     a=np.asarray(background)
#     img=a
#     a = a.astype(np.float32) / 255
#     a = np.expand_dims(a, 0)
#     # a = speckle(a)
#     imsave('dataset/file_'+str(random.randint(0,199))+'.png',img)
#     return a
def paint_text(text, w, h,  rotate=False, ud=True, multi_fonts=False):
    newtext = ""
    import random
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
    w=random.randint(300,1900)
    h=random.randint(40,200)
    LargeWidth=0

    if(w>1000):
        LargeWidth=1
        if(h<100):
            h=random.randint(100,200)

    fontsize = random.randint(30, 40)

    if(LargeWidth==1):
        fontsize = random.randint(40,55)

    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    import random

    FlagBlack = random.randint(0, 4)
    # FlagBlack = 1
    with cairo.Context(surface) as context:
        if (FlagBlack == 2):
            context.set_source_rgb(0, 0, 0)  # White
        else:
            context.set_source_rgb(1, 1, 1)  # White

        context.paint()
        # this font list works in CentOS 7
        multi_fonts=True

        if multi_fonts:
            fonts = ['Solaimanlipi','Bangla','AponaLohit','Nikosh', 'Siyamrupali', 'kalpurush','AdorshoLipi','Likhan','Lohit Bengali','SutonnyBanglaOMJ','Sagar','Rupali','Mukti']
            context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                                     np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
        else:
           context.select_font_face('Mukti' , cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        import random

        context.set_font_size(fontsize)
        box = context.text_extents(text)
        border_w_h = (4, 4)
        if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):

            Flag = 0
            while box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
                fontsize -= 1
                if (fontsize == 0):
                    Flag = -1
                    break
                # print(fontsize)
                context.set_font_size(fontsize)
                box = context.text_extents(text)
            if Flag == -1:
                fontsize = 20
                text = "ক"
                context.set_font_size(fontsize)
                box = context.text_extents(text)

            # while box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
            #     idx=len(text)-1
            #     for i in range(len(text)-1,0,-1):
            #         # print(text[i])
            #         if text[i]==" ":
            #             idx=i
            #             break
            #     text=text[0:idx]
            #     box = context.text_extents(text)
            # # print(text)
            # text1="মঠ"
            # text2="যগ"
            # text3="যগ"
            # Fl=random.randint(0,3)
            # if(Fl==0):
            #     box = context.text_extents(text1)
            # elif(Fl==1):
            #     box = context.text_extents(text2)
            # else:
            #     box = context.text_extents(text3)


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
    import cv2
    vis2 = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    vis2 = cv2.resize(vis2, (564, 64))
    a=np.asarray(vis2)
    a = a[:, :, 0]  # grab single channel

    imsave('dataset/file_'+str(random.randint(0,1999))+'.png',a)

    # a = speckle(a)

    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    if rotate:
        a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
    a = speckle(a)


    return a

def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret

def FindOutPutShape():
    return 327  # Another Joint Charachter Khiyo Addeed
# Translation of characters to unique integer values

def text_to_labels(text):

    labeling = wg.labelingNewDataset(text)
    return labeling


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


# only a-z and space..probably not to difficult
# to expand to uppercase and symbols

def is_valid_str(in_str):
    search = re.compile(regex, re.UNICODE).search
    return bool(search(in_str))


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, monogram_file, bigram_file, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,type_t,
                 absolute_max_string_len=164):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.type_t=type_t
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return len(alphabet) + 1

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words
        tmp_string_list,self.flags = wg.newDataset(num_words,type_t)

        if len(tmp_string_list) != self.num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')


        self.string_list[::2] = tmp_string_list[:self.num_words // 2]
        self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

        for i, word in enumerate(self.string_list):
            wordlen = len(text_to_labels(word))
            self.Y_len[i] = wordlen

            self.Y_data[i, 0:wordlen] = text_to_labels(word)
            self.X_text.append(word)

        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(16000, 12, 1)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=False, ud=False, multi_fonts=False)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if 3 <= epoch < 6:
            self.paint_func = lambda text: paint_text(text,  self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=False)
        elif 6 <= epoch < 9:
            self.paint_func = lambda text: paint_text(text,self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=True)
        elif epoch >= 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=True, ud=True, multi_fonts=True)
        if epoch >= 21 and self.max_string_len < 12:
            self.build_word_list(32000, 24, 0.5)
# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):

    Total = wg.getTotalData()

    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        print(wg.decodeNewDataset(out_best))
        ret.append(wg.decodeNewDataset(out_best))
        # print(wg.decodeNewDataset(out_best))
    return ret


class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()


def train(run_name, start_epoch, stop_epoch, img_w,type_t,wf,in):
    # Input Parameters
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512
    minibatch_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    fdir = os.path.dirname(get_file('wordlists.tgz',
                                    origin='http://www.mythic-ai.com/datasets/wordlists.tgz', untar=True))

    img_gen = TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                 bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                 minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 2),
                                 val_split=words_per_epoch - val_words,
                                 type_t=type_t
                                 )
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(FindOutPutShape(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred)

    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd,metrics=['accuracy'])
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        # model.load_weights(weight_file)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())
    model.load_weights(wf)
    # history = model.fit_generator(generator=img_gen.next_train(),
    #                     steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
    #                     epochs=stop_epoch,
    #                     validation_data=img_gen.next_val(),
    #                     validation_steps=val_words // minibatch_size,
    #                     callbacks=[viz_cb, img_gen],
    #                     initial_epoch=start_epoch)
    #
    # # list all data in history
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('acc_vs_val_acc.png')
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('loss_vs_val_loss.png')
    #

    imgwide=564
    import cv2
    from PIL import Image
    img = Image.open(in)
    img = img.resize((imgwide, 64), Image.ANTIALIAS)

    # img = cv2.imread('testimg_9.png')
    # img = cv2.resize(img, (imgwide, 64))

    img = np.asarray(img)
    img = img[:, :, 0]  # grab single channel
    import matplotlib.pyplot as plt

    # plt.imshow(img, cmap='gray')
    # plt.show()

    img = img.astype(np.float32) / 255
    img = np.expand_dims(img, 0)

    data = np.reshape(img, (1, 64, imgwide))
    X_data = np.ones([1, imgwide, 64, 1])
    X_data[0, 0:imgwide, :, 0] = data[0, :, :].T

    decode_batch(test_func,X_data)
    #
if __name__ == '__main__':
    sys.argv[0] # prints python_script.py
    weightfile=sys.argv[1] # prints var1
    imagename=sys.argv[2]
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    # type_t="singleword"
    # train(run_name, 0, 20, 256,type_t)

    type_t="other"
    train(run_name, 0,1500, 564,type_t,weightfile,imagename)

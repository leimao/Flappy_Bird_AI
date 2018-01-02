# http://imageio.readthedocs.io/

import imageio
import os
import re
import cv2

DESTINATION_PATH = 'video_best_converted/'
SOURCE_PATH = 'video_best/'
GIF_NAME = 'flappy_bird_AI.gif'
COMPRESS_FACTOR = 1

# natural sort
_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]   

if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH)

filenames = sorted(os.listdir(SOURCE_PATH), key = natural_sort_key)

with imageio.get_writer(DESTINATION_PATH + GIF_NAME, mode = 'I') as writer:
    for filename in filenames:
        if filename.endswith('.png'):
            print('Processing ' + filename + '...')
            image = imageio.imread(SOURCE_PATH + filename)
            image = cv2.resize(image, (image.shape[1]/COMPRESS_FACTOR, image.shape[0]/COMPRESS_FACTOR))
            writer.append_data(image)
    print('Done')


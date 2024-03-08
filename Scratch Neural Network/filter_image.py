import argparse
import os

import numpy as np
from PIL import Image

from layers import Conv2d

FILTERS = {
    'gaussian': np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16,
    'sobel_x': np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]),
    'grad_x': np.array([
        [-1, 0, 1],
    ]),
    'grad_y': np.array([
        [-1],
        [0],
        [1],
    ]),
}

def main(img_fn, filter_type):
    img = np.array(Image.open(img_fn))
    img = img / 255.0
    filter = Conv2d(3, 3, [3, 3])
    for i in range(3):
        filter.params['weight'][i, i] = FILTERS[filter_type]
    output = filter.forward(img[None])[0]
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    if not os.path.exists('result'):
        os.makedirs('result')
    Image.fromarray(output).save(f'result/{filter_type}.jpg')
    print(f'Saved result/{filter_type}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=str, default='gaussian', choices=[
        'gaussian', 'grad_y', 'grad_x', 'sobel_x'
    ])
    parser.add_argument('--image', type=str, default='val_00000.jpg')
    args = parser.parse_args()
    main(args.image, args.filter)

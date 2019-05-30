from PIL import Image
import numpy as np
from numocr import cellocr, resultstr
import os


def shtocr(sess, x, y, keep, path, rows, cols, stroke=10, save=False):
    im = Image.open(path)
    w, h = im.size
    wcell = w / cols
    hcell = h / rows

    results = []
    for i in range(rows):
        for j in range(cols):
            newimg = im.crop((
                j * wcell + stroke,
                i * hcell + stroke,
                (j + 1) * wcell - stroke,
                (i + 1) * hcell - stroke))
            if save:
                newimg.save(os.path.join(path, r'..\splits\%(i)d-%(j)d.jpg' % {'i': i, 'j': j}))
            cur_result = resultstr(cellocr(newimg, sess, x, y, keep))
            # if not cur_result.any():
            #     continue
            if save:
                newimg.save(os.path.join(path, r'..\results\%(i)d-%(j)d-(%(s)s).jpg'
                                         % {'s': cur_result, 'i': i, 'j': j}))
            results.append(cur_result)
            # print(i, j, cur_result)

    return np.array(results).reshape((rows, cols))


def singleocr(sess, x, y, keep, path):
    im = Image.open(path)
    h = {}
    for i in range(100):
        single = resultstr(cellocr(im, sess, x, y, keep))
        if single not in h:
            h[single] = 0
        h[single] += 1

    max = 0
    most = ''
    for (k, v) in h.items():
        if v > max:
            max = v
            most = k
    return most, max

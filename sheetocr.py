from PIL import Image
import numpy as np
from numocr import cellocr, resultstr
import os
import cv2


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


def ptrans(orgpath, tarpath, points, tarw=1920, tarh=1080):
    im = cv2.imread(orgpath, cv2.IMREAD_COLOR)
    canvasp = np.float32([[0, 0], [tarw, 0], [tarw, tarh], [0, tarh]])
    pmatrix = cv2.getPerspectiveTransform(points, canvasp)
    pimg = cv2.warpPerspective(im, pmatrix, (tarw, tarh))
    cv2.imwrite(tarpath, pimg)

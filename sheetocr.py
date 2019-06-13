from PIL import Image
import numpy as np
from numocr import cellocr, resultstr
import os
import cv2

splitpath = r'D:\pic\splits'
resultspath = r'D:\pic\results'


def shtocr(sess, x, y, keep, path, rows, cols, splitn=None, stroke=10, save=False):
    im = Image.open(path)
    w, h = im.size
    wcell = w / cols
    hcell = h / rows

    imarray = np.array(im)
    # hlinewin = wcell / 4
    # vlinewin = hcell / 3

    # wipe hline
    for i in range(cols + 1):
        areamin = i * hcell - stroke
        areamax = i * hcell + stroke
        for indx in range(int(max(0, areamin)), int(min(areamax, h))):
            if (np.mean(imarray[indx, :]) < 150):
                imarray[indx, :] = 255

    # wipe vline
    for i in range(rows + 1):
        areamin = i * wcell - stroke
        areamax = i * wcell + stroke
        for indx in range(int(max(0, areamin)), int(min(areamax, w))):
            if (np.mean(imarray[:, indx]) < 150):
                imarray[:, indx] = 255

    im = Image.fromarray(imarray)
    im.save(os.path.join(resultspath, 'wipe.jpg'))

    results = []
    cutborder = 8
    if splitn is None:
        splitn = np.zeros(rows)

    for i in range(rows):
        for j in range(cols):
            newimg = im.crop((
                j * wcell + cutborder,
                i * hcell,
                (j + 1) * wcell - cutborder,
                (i + 1) * hcell))
            if save:
                newimg.save(os.path.join(splitpath, '%(i)d-%(j)d.jpg' % {'i': i, 'j': j}))
            cur_result = resultstr(cellocr(newimg, splitn[i], sess, x, y, keep))
            # if not cur_result.any():
            #     continue
            if save:
                newimg.save(os.path.join(resultspath, '%(i)d-%(j)d-(%(s)s).jpg'
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


def ptrans(orgpath, tarpath, points):
    im = cv2.imread(orgpath, cv2.IMREAD_GRAYSCALE)
    tarw = np.float32(1.1 * max(points[1][0] - points[0][0], points[3][0] - points[2][0]))
    tarh = np.float32(1.1 * max(points[3][1] - points[0][1], points[2][1] - points[1][1]))
    canvasp = np.float32([[0, 0], [tarw, 0], [tarw, tarh], [0, tarh]])
    pmatrix = cv2.getPerspectiveTransform(points, canvasp)
    pimg = cv2.warpPerspective(im, pmatrix, (tarw, tarh))
    cv2.imwrite(tarpath, pimg)

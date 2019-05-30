import numpy as np
from PIL import Image


def netoutput(images, sess, x, y, keep):
    result_test = sess.run(y, feed_dict={x: images, keep: 0.5})
    return np.argmax(result_test, axis=1)


def fig2images(im):
    tol = 0.9

    # im = Image.open(path)
    imraw = np.array(im)[:, :, 0]
    imsolid = np.where(imraw >= 128, 1, 0)

    hnum = imsolid.shape[0]
    vnum = imsolid.shape[1]
    vvec = np.mean(imsolid, axis=0)
    hvec = np.mean(imsolid, axis=1)

    vvec_valley = np.where(vvec > tol)[0]
    vvec_peak = np.where(vvec < max(0.8, np.min(vvec) + 0.1))[0]

    valleys = []
    valley_start = 0
    for i in range(len(vvec_valley)):
        if i - valley_start != vvec_valley[i] - vvec_valley[valley_start]:
            valleys.append(int((vvec_valley[valley_start] + vvec_valley[i - 1]) / 2))
            valley_start = i

    valleys.append(int((vvec_valley[valley_start] + vvec_valley[-1]) / 2))

    peaks = []
    peak_start = 0
    for i in range(len(vvec_peak)):
        if i - peak_start != vvec_peak[i] - vvec_peak[peak_start]:
            peaks.append(int((vvec_peak[peak_start] + vvec_peak[i - 1]) / 2))
            peak_start = i

    peaks.append(int((vvec_peak[peak_start] + vvec_peak[-1]) / 2))

    peaks.append(vnum)
    valleys = np.array(valleys)
    peaks = np.array(peaks)
    back = 0
    splits = []
    for front in peaks:
        choose = np.where((valleys >= back) & (valleys < front))[0]
        if len(choose) == 1:
            splits.append(valleys[choose[0]])
        if len(choose) > 1:
            middle = (front + back) / 2
            mindist = vnum
            bestvalley = -1
            for vl in valleys[choose]:
                curdist = abs(vl - middle)
                if curdist < mindist:
                    bestvalley = vl
                    mindist = curdist
            splits.append(bestvalley)
        back = front

    splitted_im = []
    regsize = 28

    for i in range(len(splits) - 1):
        # if splits[i + 1] - splits[i] < hnum / 10:
        #     continue
        curimg = im.crop((splits[i], 0, splits[i + 1], hnum))
        curimg.thumbnail((100, regsize))
        arrayed = np.array(curimg)[:, :, 0]
        factor = 255 / np.max(arrayed)
        arrayed = 255 - arrayed * factor
        curimg = Image.fromarray(arrayed)
        # curimg.show()

        # 求重心
        curw = arrayed.shape[1]
        sumxw = 0
        sumyw = 0
        sumw = 0
        for i in range(regsize):
            for j in range(curw):
                sumxw += j * arrayed[i][j]
                sumyw += i * arrayed[i][j]
                sumw += arrayed[i][j]
        xcenter = sumxw / sumw
        ycenter = sumyw / sumw
        # print(xcenter, ycenter)
        # leftpadding = int((regsize - curw) / 2)
        # rightpadding = regsize - curw - leftpadding
        padcurimg = curimg.crop((xcenter - regsize / 2,
                                 ycenter - regsize / 2,
                                 xcenter + regsize / 2,
                                 ycenter + regsize / 2))

        splitted_im.append(np.array(padcurimg))

    netimages = []
    for img in splitted_im:
        netimages.append(img.flatten())
    return np.array(netimages)


def cellocr(im, sess, x, y, keep):
    ims = fig2images(im)
    if not ims.any():
        return np.array([])
    return netoutput(ims, sess, x, y, keep)


def resultstr(arr):
    s = ''
    for i in arr:
        s += str(i)
    return s

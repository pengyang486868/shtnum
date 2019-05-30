import os
import netutil
from sheetocr import shtocr, singleocr
import numpy as np

path = r'D:\pic'
sess, x, y, keep = netutil.getnet(r'D:\pic\net2')

s, num = singleocr(sess, x, y, keep, os.path.join(path, 'simple.jpg'))
print(s, num)

# name = 'cut2.jpg'
# r = shtocr(sess, x, y, keep, os.path.join(path, name), rows=14, cols=12, stroke=12, save=True)
# np.savetxt(os.path.join(path, name + '-result.csv'), r, delimiter=',', fmt='%s')

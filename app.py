# coding: utf-8
from flask import Flask, jsonify
import os
import netutil
from sheetocr import shtocr, singleocr
import numpy as np

app = Flask(__name__)
path = r'D:\pic'
sess, x, y, keep = netutil.getnet(r'D:\pic\net2')


#
@app.route('/test')
def test():
    return jsonify({'a': 2, 'b': 3})


#
@app.route('/testsingle')
def testsingle():
    s, num = singleocr(sess, x, y, keep, os.path.join(path, 'simple.jpg'))
    print(s, num)
    return ""


#
@app.route('/testsheet')
def testsheet():
    name = 'cut2.jpg'
    r = shtocr(sess, x, y, keep, os.path.join(path, name), rows=14, cols=12, stroke=12, save=True)
    np.savetxt(os.path.join(path, name + '-result.csv'), r, delimiter=',', fmt='%s')
    return ""


# 启动程序
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=32454)

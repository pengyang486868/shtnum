# coding: utf-8
from flask import Flask, jsonify, request
import os
import netutil
from sheetocr import shtocr, singleocr, ptrans
import numpy as np
import uuid

app = Flask(__name__)
path = r'D:\pic'
sess, x, y, keep = netutil.getnet(r'D:\pic\net2')


#
@app.route('/test')
def test():
    httpargs = request.args
    return jsonify({'a': httpargs['a'], 'b': httpargs['b']})


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


@app.route('/testpt')
def testpt():
    name = 'pers.jpg'
    newname = 'pers-after.jpg'
    pts = np.float32([[101, 141], [516, 97], [467, 456], [125, 474]])
    ptrans(os.path.join(path, name), os.path.join(path, newname), pts, tarw=1920, tarh=1080)
    return ""


@app.route('/ocr')
def ocr():
    # 解析本api的参数们
    httpargs = request.args
    opath = httpargs['path']
    pts = np.float32([[httpargs['w1'], httpargs['h1']],
                      [httpargs['w2'], httpargs['h2']],
                      [httpargs['w3'], httpargs['h3']],
                      [httpargs['w4'], httpargs['h4']]])
    givenrows = int(httpargs['rows'])
    givencols = int(httpargs['cols'])
    givenstroke = int(httpargs['stroke'])

    newpath = os.path.join(path, 'temp', 'temp' + str(uuid.uuid1()) + '.jpg')
    ptrans(opath, newpath, pts, tarw=800, tarh=700)

    r = shtocr(sess, x, y, keep, newpath, rows=givenrows, cols=givencols, stroke=givenstroke, save=False)
    np.savetxt(opath + '-result.csv', r, delimiter=',', fmt='%s')
    return ""


# 启动程序
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=32454)

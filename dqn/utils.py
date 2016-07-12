import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf

def readXML(path, target_class):
    root = ET.parse(path).getroot()
    bnds = root.findall('object')
    mxs = 0.
    print path
    for obj in bnds:
        if obj.find('name').text != target_class:
            continue
        bnd = obj.find('bndbox')
        tup = float(bnd.find('xmin').text)
        tleft = float(bnd.find('ymin').text)
        tdown = float(bnd.find('xmax').text)
        tright = float(bnd.find('ymax').text)
        ts = (tdown - tup) * (tright - tleft)
        if ts > mxs:
            mxs = ts
            up = tup
            left = tleft
            down = tdown
            right = tright

    return np.array([up, left, down, right], dtype = int)

import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf

def readXML(path):
    root = ET.parse(path).getroot()
    bnd = root.findall('object')[0].find('bndbox')
    up = bnd.find('xmin').text
    if up.isdigit():
        up = int(up)
    else:
        up = int(float(up))
    left = bnd.find('ymin').text
    if left.isdigit():
        left = int(left)
    else:
        left = int(float(left))
    down = bnd.find('xmax').text
    if down.isdigit():
        down = int(down)
    else:
        down = int(float(down))
    right = bnd.find('ymax').text
    if right.isdigit():
        right = int(right)
    else:
        right = int(float(right))
    return np.array([up, left, down, right], dtype = int)

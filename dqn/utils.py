import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf

def readXML(path):
    root = ET.parse(path).getroot()
    bnd = root.findall('object')[0].find('bndbox')
    up = float(bnd.find('xmin').text)
    left = float(bnd.find('ymin').text)
    down = float(bnd.find('xmax').text)
    right = float(bnd.find('ymax').text)
    return np.array([up, left, down, right], dtype = int)

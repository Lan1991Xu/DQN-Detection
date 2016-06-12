import numpy as np
import xml.etree.ElementTree as ET

def readXML(path):
    root = ET.parse(path).getroot()
    bnd = root.findall('object')[0].find('bndbox')
    return np.array([int(bnd.find('xmin').text), int(bnd.find('ymin').text), int(bnd.find('xmax').text), int(bnd.find('ymax').text)], dtype = int)

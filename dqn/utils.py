import xml.etree.ElementTree as ET

def readXML(path):
    root = ET.parse(ET).getroot()
    bnd = root.findall('object')[0]
    return np.array([int(bnd.find('xmin').text), int(bnd.find('ymin').text), int(bnd.find('xmax').text), int(bnd.find('ymax').text)], dtype = int)

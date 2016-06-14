import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf

def readXML(path):
    root = ET.parse(path).getroot()
    bnd = root.findall('object')[0].find('bndbox')
    return np.array([int(bnd.find('xmin').text), int(bnd.find('ymin').text), int(bnd.find('xmax').text), int(bnd.find('ymax').text)], dtype = int)

def readImg(path, reader, sess):
    # DEBUG
    print path
    #
    path = tf.train.string_input_producer([path], shuffle = False, capacity = 1)
    
    _, value = reader.read(path)
    
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess = sess, coord = coord)

    sess.run(value) 
    # DEBUG
    # print value
    #
    img = tf.image.decode_jpeg(value, channels = 3).eval(session = sess)

    coord.request_stop()
    coord.join(thread)

    return img

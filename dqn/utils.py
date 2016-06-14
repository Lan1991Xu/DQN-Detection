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
    path = tf.train.string_input_producer([path])
    
    _, value = reader.read(path)
    
    coord = tf.train.Coordinator()
    try:
        thread = tf.train.start_queue_runners(sess = sess, coord = coord)
    except RuntimeError:
        pass

    sess.run(value) 
    # DEBUG
    print path.size().eval(session = sess)
    #
    img = tf.image.decode_jpeg(value, channels = 3).eval(session = sess)
    # imgRead = tf.image.decode_jpeg(value, channels = 3)
    # img = sess.run(imgRead)

    # DEBUG
    print img
    #

    # coord.request_stop()
    # coord.join(thread)

    # Debug
    #print img.shape
    ## Debug
    #exit()
    ##
    return img

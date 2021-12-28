from __future__ import print_function

import tensorflow as tf
'''
Keras+tensorflow conversion of 
https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
If you use this code please cite:
https://www.scirp.org/journal/PaperInformation.aspx?PaperID=84616
Konovalov, D.A., Hillcoat, S., Williams, G. , Birtles, R. A., Gardiner, N., and Curnock, M. I. (2018) 
Individual Minke Whale Recognition Using Deep Learning Convolutional Neural Networks. 
Journal of Geoscience and Environment Protection, 6, 25-36. doi: 10.4236/gep.2018.65003
'''


def FCN(input_shape=(256,256,3),   # (256,256,3) is for the tensorflow backend, (3,256,256) for Theano
                          num_classes=None,   # todo: CHANGE to your number of output classes
                          num_conv_filters=4096,  # in our papers we used 1024 and 512
                          use_bias=True,
                          weight_decay=0.,
                          last_activation='softmax'  # or e.g. 'sigmoid'
                          ):
    '''
    Keras+tensorflow conversion of
    https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
    If you use this code please cite:
    https://www.scirp.org/journal/PaperInformation.aspx?PaperID=84616
Konovalov, D.A., Hillcoat, S., Williams, G. , Birtles, R. A., Gardiner, N., and Curnock, M. I. (2018) 
    Individual Minke Whale Recognition Using Deep Learning Convolutional Neural Networks.
    Journal of Geoscience and Environment Protection, 6, 25-36. doi: 10.4236/gep.2018.65003
    '''
    
    wd = weight_decay
    kr = tf.keras.regularizers.l2
    in1 = tf.keras.Input(shape=input_shape)
    # ki = 'he_normal'
    ki = 'glorot_uniform'

    # padding
    y_pad = input_shape[0] % 32
    x_pad = input_shape[1] % 32
    assert y_pad == 0 and x_pad == 0

    
    base_model = tf.keras.applications.VGG16(include_top=False, input_tensor=in1, pooling=None)
    # base_model.summary()
    pool3 = base_model.layers[-9].output
    pool4 = base_model.layers[-5].output
    pool5 = base_model.layers[-1].output
    c6 = tf.keras.layers.Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6")(pool5)
    c7 = tf.keras.layers.Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7")(c6)

    # FCN-32 output
    fcn32_o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(32,32), strides=(32, 32), use_bias=False)(c7)
    o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(c7) # (16, 16, n)
    o = tf.keras.layers.Cropping2D(cropping=(1,1))(o) # (14, 14, n)

    o2 = pool4 # (14, 14, 512)
    o2 = tf.keras.layers.Conv2D(num_classes, (1,1), activation='relu', padding='same')(o2) # (14, 14, n)
    o = tf.keras.layers.Add()([o, o2]) # (14, 14, n)

    # FCN-16 output
    fcn16_o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(16,16), strides=(16,16), use_bias=False)(o)
    
    o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(4,4), strides=(2,2), use_bias=False)(o) # (30, 30, n)
    o = tf.keras.layers.Cropping2D(cropping=(1,1))(o) # (28, 28, n)
    
    o2 = pool3 # (28, 28, 256)
    o2 = tf.keras.layers.Conv2D(num_classes, (1,1), activation='relu', padding='same')(o2) # (28, 28, n)
    
    o = tf.keras.layers.Add()([o, o2]) # (28, 28, n)
    
    o = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(8,8), strides=(8,8), use_bias=False)(o) # (224, 224, n)
    # append a softmax to get the class probabilities

    model = tf.keras.Model(in1, o)

    return model

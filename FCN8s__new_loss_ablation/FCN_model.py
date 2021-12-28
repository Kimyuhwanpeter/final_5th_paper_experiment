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

    # NOTE no change from 16s
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
    # n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    relu6 = tf.keras.layers.Conv2D(num_conv_filters, 7,
               activation='relu',
               use_bias=use_bias,
               padding='same', name='fc6_relu6')(pool5)
    # n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    drop6 = tf.keras.layers.Dropout(0.5)(relu6)

    # n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    relu7 = tf.keras.layers.Conv2D(num_conv_filters, 1,
               activation='relu',
               use_bias=use_bias,
               name='fc7_relu7')(drop6)
    # n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    drop7 = tf.keras.layers.Dropout(0.5)(relu7)

    # n.score_fr = L.Convolution(n.drop7, num_output=21, kernel_size=1, pad=0,
    #     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    score_fr = tf.keras.layers.Conv2D(num_classes, 1,
                       use_bias=use_bias,
                       name='conv_fc3')(drop7)

    # UPSAMPLE 16
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn16s/net.py
    #  n.upscore2 = L.Deconvolution(n.score_fr,
    #     convolution_param=dict(num_output=21, kernel_size=4, stride=2, bias_term=False),
    #     param=[dict(lr_mult=0)])
    # NOTE no change from 16s
    upscore2 = tf.keras.layers.Conv2DTranspose(num_classes, 4,
                               strides=(2, 2),
                               padding='same',
                               use_bias=False,
                               name='upscore2')(score_fr)

    #  n.score_pool4 = L.Convolution(n.pool4, num_output=21, kernel_size=1, pad=0,
    #      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # NOTE no change from 16s
    score_pool4 = tf.keras.layers.Conv2D(num_classes, 1,
                         use_bias=use_bias)(pool4)

    # n.score_pool4c = crop(n.score_pool4, n.upscore2)
    # n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c, operation=P.Eltwise.SUM)
    # NOTE no change from 16s
    fuse_pool4 = tf.keras.layers.Add()([upscore2, score_pool4])

    # n.upscore_pool4 = L.Deconvolution(n.fuse_pool4,
    #     convolution_param=dict(num_output=21, kernel_size=4, stride=2, bias_term=False),
    #     param=[dict(lr_mult=0)])
    # NEW in 8s
    upscore_pool4 = tf.keras.layers.Conv2DTranspose(num_classes, 4,
                                    strides=(2, 2),
                                    padding='same',
                                    use_bias=False,
                                    name='upscore_pool4')(fuse_pool4)

    # n.score_pool3 = L.Convolution(n.pool3, num_output=21, kernel_size=1, pad=0,
    #     param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # n.score_pool3c = crop(n.score_pool3, n.upscore_pool4)
    # n.fuse_pool3 = L.Eltwise(n.upscore_pool4, n.score_pool3c, operation=P.Eltwise.SUM)
    # n.upscore8 = L.Deconvolution(n.fuse_pool3,
    #     convolution_param=dict(num_output=21, kernel_size=16, stride=8, bias_term=False),
    # NEW in 8s
    score_pool3 = tf.keras.layers.Conv2D(num_classes, 1, use_bias=use_bias)(pool3)
    fuse_pool3 = tf.keras.layers.Add()([upscore_pool4, score_pool3])
    upscore8 = tf.keras.layers.Conv2DTranspose(num_classes, 8,
                               strides=(8, 8),
                               padding='same',
                               use_bias=False,
                               name='upscore8')(fuse_pool3)

    # n.score = crop(n.upscore8, n.data)
    # n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=False, ignore_label=255))
    #score = tf.keras.layers.Activation(last_activation)(upscore8)

    #model = Model(inputs=[in1], outputs=[score])  # OLD
    model = tf.keras.Model(base_model.input, upscore8)

    return model

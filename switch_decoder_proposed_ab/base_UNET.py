# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_applications import get_submodules_from_kwargs
from _common_blocks import Conv2dBn
from backbones_factory import Backbones
from _utils import freeze_model, filter_keras_submodules

import tensorflow as tf
# https://github.com/qubvel/segmentation_models
def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }


# ---------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    #concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    concat_axis = 3

    def wrapper(input_tensor, skip=None):
        x = tf.keras.layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage_{}a_transpose'.format(stage)
    bn_name = 'decoder_stage_{}a_bn'.format(stage)
    relu_name = 'decoder_stage_{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage_{}b'.format(stage)
    concat_name = 'decoder_stage_{}_concat'.format(stage)
    

    concat_axis = bn_axis = 3 

    def layer(input_tensor, skip=None):

        x = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = tf.keras.layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = tf.keras.layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

# ---------------------------------------------------------------------
#  Unet Decoder
# ---------------------------------------------------------------------

def build_unet(
        backbone,
        decoder_block,
        decoder_block2,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output
    x_ = backbone.output
    ######################################################################
    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], tf.keras.layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # model head (define number of output classes)
    #x = tf.keras.layers.Conv2D(     # filters=classes,
    #    filters=classes,
    #    kernel_size=(3, 3),
    #    padding='same',
    #    use_bias=True,
    #    kernel_initializer='glorot_uniform',
    #    name='final_conv',
    #)(x)
    ######################################################################

    # extract skip connections # 이건 fix UNET일때 사용가능한것임
    skips2 = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in skip_connection_layers])
    if isinstance(backbone.layers[-1], tf.keras.layers.MaxPooling2D):
        x_ = Conv3x3BnReLU(512, use_batchnorm, name='center_block3')(x_)
        x_ = Conv3x3BnReLU(512, use_batchnorm, name='center_block4')(x_)

    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips2[i]
        else:
            skip = None

        x_ = decoder_block2(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x_, skip)

    ch_att = tf.keras.layers.GlobalAveragePooling2D()(x)
    ch_att = tf.reshape(ch_att, [-1, 1, 1, 16])
    x_ = tf.nn.sigmoid(ch_att) * x_

    # model head (define number of output classes)
    x = tf.keras.layers.Conv2D(     # filters=classes,
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)

    x_ = tf.keras.layers.Conv2D(     # filters=classes,
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv2',
    )(x_)

    x_ = tf.nn.sigmoid(x) * x_

    # create keras model instance
    model = tf.keras.Model(input_, x_)

    return model

def Unet(
        backbone_name='vgg16',
        input_shape=(None, None, 3),
        classes=1,
        weights=None,
        encoder_weights='imagenet',
        encoder_freeze=False,
        encoder_features='default',
        decoder_block_type='upsampling',
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
        **kwargs
):

    global backend, layers, models, keras_utils
    submodule_args = filter_keras_submodules(kwargs)
    backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

    if decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
    elif decoder_block_type == 'transpose':
        decoder_block = DecoderTransposeX2Block
    else:
        raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                         'Got: {}'.format(decoder_block_type))

    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    if encoder_features == 'default':
        encoder_features = Backbones.get_feature_layers(backbone_name, n=4)

    model = build_unet(
        backbone=backbone,
        decoder_block=DecoderUpsamplingX2Block,
        decoder_block2=DecoderTransposeX2Block,
        skip_connection_layers=encoder_features,
        decoder_filters=decoder_filters,
        classes=classes,
        n_upsample_blocks=len(decoder_filters),
        use_batchnorm=decoder_use_batchnorm,
    )


    # lock encoder weights for fine-tuning
    if encoder_freeze:
        freeze_model(backbone, **kwargs)

    # loading model weights
    if weights is not None:
        model.load_weights(weights)

    return model



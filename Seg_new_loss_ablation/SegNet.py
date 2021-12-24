# -*- coding:utf-8 -*-
from factory import Backbones
import tensorflow as tf


def MaxPooling(h):

    val, indx = tf.nn.max_pool_with_argmax(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    return val, indx, h.get_shape().as_list()

def UpSample2D(pool, indx, batch_size, output_shape):

    pool_ = tf.reshape(pool, [-1])
    batch_range = tf.reshape(tf.range(batch_size, dtype=indx.dtype), [tf.shape(pool)[0], 1, 1, 1])
    b = tf.ones_like(indx) * batch_range
    b = tf.reshape(b, [-1, 1])
    indx_ = tf.reshape(indx, [-1, 1])
    indx_ = tf.concat([b, indx_], 1)
    ret = tf.scatter_nd(indx_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
    ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])

    return ret

def SegNet_model(input_shape=(512, 512, 3), classes=3, batch_size=4):

    h = inputs = tf.keras.Input(input_shape, batch_size=batch_size)    # per_image_standliazation ?? ???Ѿ???
    
    backbone = Backbones.get_backbone(
        name="vgg16",
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )
   
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv1")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv2")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool1, poo1_indx1, shape_1 = MaxPooling(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv3")(pool1)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv4")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool2, poo1_indx2, shape_2 = MaxPooling(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv5")(pool2)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv6")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv7")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool3, poo1_indx3, shape_3 = MaxPooling(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv8")(pool3)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv9")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv10")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool4, poo1_indx4, shape_4 = MaxPooling(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv11")(pool4)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv12")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv13")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool5, poo1_indx5, shape_5 = MaxPooling(h)

    ######################################################################################################

    h = UpSample2D(pool5, poo1_indx5, batch_size, shape_5)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv14")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv15")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv16")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = UpSample2D(h, poo1_indx4, batch_size, shape_4)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv17")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv18")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv19")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = UpSample2D(h, poo1_indx3, batch_size, shape_3)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv20")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv21")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv22")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = UpSample2D(h, poo1_indx2, batch_size, shape_2)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv23")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv24")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = UpSample2D(h, poo1_indx1, batch_size, shape_1)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv25")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv26")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=classes, kernel_size=1, padding="same", name="outputs")(h)

    SegModel = tf.keras.Model(inputs=inputs, outputs=h)

    SegModel.get_layer("conv1").set_weights(backbone.get_layer("block1_conv1").get_weights())
    SegModel.get_layer("conv2").set_weights(backbone.get_layer("block1_conv2").get_weights())
    SegModel.get_layer("conv3").set_weights(backbone.get_layer("block2_conv1").get_weights())
    SegModel.get_layer("conv4").set_weights(backbone.get_layer("block2_conv2").get_weights())
    SegModel.get_layer("conv5").set_weights(backbone.get_layer("block3_conv1").get_weights())
    SegModel.get_layer("conv6").set_weights(backbone.get_layer("block3_conv2").get_weights())
    SegModel.get_layer("conv7").set_weights(backbone.get_layer("block3_conv3").get_weights())
    SegModel.get_layer("conv8").set_weights(backbone.get_layer("block4_conv1").get_weights())
    SegModel.get_layer("conv9").set_weights(backbone.get_layer("block4_conv2").get_weights())
    SegModel.get_layer("conv10").set_weights(backbone.get_layer("block4_conv3").get_weights())
    SegModel.get_layer("conv11").set_weights(backbone.get_layer("block5_conv1").get_weights())
    SegModel.get_layer("conv12").set_weights(backbone.get_layer("block5_conv2").get_weights())
    SegModel.get_layer("conv13").set_weights(backbone.get_layer("block5_conv3").get_weights())

    return SegModel
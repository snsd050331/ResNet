#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:06:02 2019

@author: john
"""

import tensorflow as tf
import numpy as np
import os, pdb
import cv2
import numpy as np
import random as rn
import tensorflow as tf
import threading
import time

global n_classes, layer_count 
n_classes = 50
layer_count = 0

def read_labeled_image_list(image_list_file, training_img_dir):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []

    for line in f:
        filename, label = line[:-1].split(' ')
        filename = training_img_dir+filename
        filenames.append(filename)
        labels.append(int(label))
        
    return filenames, labels
#################################################################################################################################################################
def read_images_from_disk(input_queue, size1=256):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    fn=input_queue[0]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    
    #example = tf.image.decode_png(file_contents, channels=3, name="dataset_image") # png fo rlfw
    example=tf.image.resize_images(example, [size1,size1])
    return example, label, fn
#################################################################################################################################################################
def setup_inputs(sess, filenames, training_img_dir, image_size=256, crop_size=224, isTest=False, batch_size=64):
    
    # Read each image file
    image_list, label_list = read_labeled_image_list(filenames, training_img_dir)

    images = tf.cast(image_list, tf.string)
    labels = tf.cast(label_list, tf.int64)
     # Makes an input queue
    if isTest is False:
        isShuffle = True
        numThr = 4
    else:
        isShuffle = False
        numThr = 1
        
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=isShuffle)
    image, y,fn = read_images_from_disk(input_queue)

    channels = 3
    image.set_shape([None, None, channels])
        
    # Crop and other random augmentations
    if isTest is False:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_hue(image,0.05)
        image = tf.image.random_saturation(image, .95, 1.05)
        image = tf.image.random_brightness(image, .05)
        image = tf.image.random_contrast(image, .95, 1.05)
    
        

    image = tf.random_crop(image, [crop_size, crop_size, 3])
    image = tf.cast(image, tf.float32)/255.0
    
    image, y,fn = tf.train.batch([image, y, fn], batch_size=batch_size, capacity=batch_size*3, num_threads=numThr, name='labels_and_images')

    tf.train.start_queue_runners(sess=sess)

    return image, y, fn, len(label_list)
#################################################################################################################################################################
def initializer(in_filters, out_filters,name='wb', ks=3):
    W = tf.get_variable(name+"W", [3,3, in_filters,out_filters], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable(name+"B", [out_filters], initializer=tf.truncated_normal_initializer())
    return W, b

def activation(x,name="activation"):
    return tf.nn.relu(x, name=name)
    
def conv2d(name, l_input, w, b, s, p):
    l_input = tf.nn.conv2d(l_input, w, strides=[1,s,s,1], padding=p, name=name)
    l_input = l_input+b

    return l_input

def max_pool(name, l_input, k, s):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='VALID', name=name)

def batchnorm(conv, isTraining, name='bn'):
    return tf.layers.batch_normalization(conv, momentum = 0.997, training=isTraining, name="bn"+name)
#################################################################################################################################################################
def _batch_norm(input_):
        """Batch normalization for a 4-D tensor"""
        assert len(input_.get_shape()) == 4
        filter_shape = input_.get_shape().as_list()
        mean, var = tf.nn.moments(input_, axes=[0, 1, 2])
        out_channels = filter_shape[3]
        offset = tf.Variable(tf.zeros([out_channels]))
        scale = tf.Variable(tf.ones([out_channels]))
        batch_norm = tf.nn.batch_normalization(input_, mean, var, offset, scale, 0.001)
        return batch_norm
def _conv(input, filter_shape, stride):
        """Convolutional layer"""
        return tf.nn.conv2d(input,
                            filter=init_tensor(filter_shape),
                            strides=[1, stride, stride, 1],
                            padding="SAME")
def init_tensor(shape):
        return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0))        
#################################################################################################################################################################
def residual_block(in_x, in_filters, out_filters, stride, tst, name):
    # first convolution layer
    global layer_count
  
    ##"How to have skip connection?"
    x=_batch_norm(in_x)
    x=activation(x)
    x=_conv(x, [3, 3, in_filters, out_filters], stride)
    x=_batch_norm(in_x)
    x=activation(x)
    x=_conv(x, [3, 3, out_filters, out_filters], stride)

    if in_filters != out_filters:        
        difference = out_filters - in_filters
        left_pad = difference // 2
        right_pad = difference - left_pad
        identity = tf.pad(in_x, [[0, 0], [0, 0], [0, 0], [left_pad, right_pad]])
        return x + identity
    else:
        return in_x + x
#################################################################################################################################################################
def ResNet(_X, tst):
    global n_classes
    w1 = tf.get_variable("firstW", [7,7,3, 64], initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("firstB", [64], initializer=tf.truncated_normal_initializer())
    
    x = conv2d('conv1', _X, w1, b1, 3, "VALID")
    x = batchnorm(x, tst, name='sbn')
    x= tf.nn.relu(x)
    
    filters_num = [64,128,256,512]
    block_num = [3,4,6,3]
    strides=[1,1,1,1]
    l_cnt = 1
    for i in range(len(filters_num)):
      for j in range(block_num[i]):
          x = residual_block(x, filters_num[i], filters_num[i], strides[i], tst, 'RB%d_%d'%(i,j))
          print('[L-%d] Build %dth residual block %d with %d channels' % (l_cnt,i, j, filters_num[i]))
          l_cnt +=1
          if ((j==block_num[i]-1) & (i<len(filters_num)-1)):
            x = batchnorm(x, tst, name='RB_bn%d_%d'%(i,j))
            w1, b1 = initializer(filters_num[i], filters_num[i+1], name='RB_pool%d_%d'%(i,j))
            x = conv2d('RB_pool%d_%d'%(i,j), x, w1, b1, 2, "VALID")
            x = activation(x)
            print('[L-%d] Build %dth connection layer %d from %d to %d channels' % (l_cnt, i, j, filters_num[i], filters_num[i+1]))
            l_cnt +=1

    wo, bo=initializer(filters_num[-1], n_classes, name='final_wb')
    x = conv2d('final', x, wo, bo, 1, "SAME")
    x = batchnorm(x, tst, name="final_bn1")
    x = activation(x, name='final_act')
    
    x = tf.reduce_mean(x, [1,2]) #b x7 x 7 x 50 ==> b x 1 x 1 x 50==>b x 50 ==> b x 50
    W = tf.get_variable("FinalW", [n_classes, n_classes], initializer=tf.truncated_normal_initializer())
    b = tf.get_variable("FinalB", [n_classes], initializer=tf.truncated_normal_initializer())
    
    out = tf.matmul(x, W) + b
                            

    return out
#################################################################################################################################################################
batch_size = 64
display_step = 80
learning_rate = tf.placeholder(tf.float32)      # Learning rate to be fed
lr = 1e-3              # Learning rate start
tst = tf.placeholder(tf.bool)

# Setup the tensorflow...
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

print("Preparing the training & validation data...")
train_data, train_labels, filelist1, glen1 = setup_inputs(sess, "train.txt", "", batch_size=batch_size)
val_data, val_labels, filelist2, tlen1 = setup_inputs(sess, "val.txt", "",isTest=True, batch_size=batch_size)

max_iter = glen1*400
print("Preparing the training model with learning rate = %.5f..." % (lr))


with tf.variable_scope("ResNet") as scope:
  pred = ResNet(train_data, True)
  scope.reuse_variables()
  valpred = ResNet(val_data, False)

with tf.name_scope('Loss_and_Accuracy'):
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    cost = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=pred)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
  correct_prediction = tf.equal(tf.argmax(pred, 1), train_labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  
  correct_prediction2 = tf.equal(tf.argmax(valpred,1), val_labels)
  accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
  
  tf.summary.scalar('Loss', cost)
  tf.summary.scalar('Training_Accuracy', accuracy)
  
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)
step = 0
writer = tf.summary.FileWriter("/tmp/log2", sess.graph)
summaries = tf.summary.merge_all()

print("We are going to train the ImageNet model based on ResNet!!!")
while (step * batch_size) < max_iter:
    epoch1=np.floor((step*batch_size)/glen1)
    if (((step*batch_size)%glen1 < batch_size) & (lr==1e-3) & (epoch1 >2)):
        lr /= 10

    sess.run(optimizer,  feed_dict={learning_rate: lr, tst: True})

    if (step % 15000==1) & (step>15000):
        save_path = saver.save(sess, "tf_resnet_model_iter" + str(step) + ".ckpt")
        print("Model saved in file at iteration %d: %s" % (step*batch_size,save_path))

    if step % display_step == 1:
        # calculate the loss
        
        loss, acc, summaries_string = sess.run([cost, accuracy, summaries], feed_dict={ tst: True})
        print("Iter=%d/epoch=%d, Loss=%.6f, Training Accuracy=%.6f, lr=%f" % (step*batch_size, epoch1 ,loss, acc, lr))
        writer.add_summary(summaries_string, step)
        
#         if step*batch_size==82048:
#         import pdb
#         pdb.set_trace()

  
    step += 1
print("Optimization Finished!")
save_path = saver.save(sess, "tf_resnet_model.ckpt")
print("Model saved in file: %s" % save_path)



# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import csv

import os

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

def listdir_search(dir_path, head_str):
    all_files = os.listdir(dir_path)
    searched_list = []
    for f in all_files:
        if len(f)>=len(head_str):
            if f[0:len(head_str)]==head_str:
                searched_list.append(f)
    return searched_list

slim = tf.contrib.slim



tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
    
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'self_label', '', 'self label for csv file')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

# csv_path=FLAGS.checkpoint_path+"."+FLAGS.self_label+FLAGS.dataset_split_name+".csv"
csv_path="/home/cl114/luke.csv"
tfrecord_files = listdir_search(FLAGS.dataset_dir, FLAGS.dataset_split_name)
tfrecord_files.sort()
tfrecord_number = len(tfrecord_files)
if tfrecord_number==0:
    print("No dataset found!")
    exit()

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    result_dict={}
    # eval based on differen resolution
    # write to csv file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["image_name", "prediction", "GT_label"]
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for tfrecord_file in tfrecord_files:
            eval_image_size=FLAGS.eval_image_size
            tf.reset_default_graph() 
            with tf.Graph().as_default():
                tf_global_step = slim.get_or_create_global_step()

                ######################
                # Select the dataset #
                ######################
                dataset = dataset_factory.get_dataset(
                    FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
                
                ####################
                # Select the model #
                ####################
                network_fn = nets_factory.get_network_fn(
                    FLAGS.model_name,
                    num_classes=(dataset.num_classes - FLAGS.labels_offset),
                    is_training=False)


                ################################
                # Create the image placeholder #
                ################################

                image_string = tf.placeholder(tf.string)
                image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3)

                #####################################
                # Select the preprocessing function #
                #####################################
                preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
                image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                    preprocessing_name,
                    is_training=False)

                processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
                processed_images  = tf.expand_dims(processed_image, 0)

                ####################
                # Define the model #
                ####################
                # logits, _ = network_fn(images)
                logits, _ = network_fn(processed_images)

                if FLAGS.quantize:
                    tf.contrib.quantize.create_eval_graph()

                if FLAGS.moving_average_decay:
                    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, tf_global_step)
                    variables_to_restore = variable_averages.variables_to_restore(slim.get_model_variables())
                    variables_to_restore[tf_global_step.op.name] = tf_global_step
                else:
                    variables_to_restore = slim.get_variables_to_restore()

                predictions = tf.argmax(logits, 1)


                if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
                else:
                    checkpoint_path = FLAGS.checkpoint_path

                tf.logging.info('Evaluating %s' % checkpoint_path)
                
                session_config = tf.ConfigProto()
                session_config.gpu_options.allow_growth = True

                init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)
                sess=tf.Session(config=session_config)
                init_fn(sess)

                image_name_list = []
                GT_label_list = []
                pred_list = []
                sample_number =0 
                print("Testing "+ os.path.join(FLAGS.dataset_dir, tfrecord_file)+" with "+str(eval_image_size)+" input")
                fls = tf.python_io.tf_record_iterator(os.path.join(FLAGS.dataset_dir, tfrecord_file))
                for fl in fls:
                    example = tf.train.Example()
                    example.ParseFromString(fl)
                    x = example.features.feature['image/encoded'].bytes_list.value[0]
                    image_name = example.features.feature['image/filename'].bytes_list.value[0]
                    GT_label = example.features.feature['image/class/label'].int64_list.value[0]
                    r_predictions,  = sess.run([predictions, ], feed_dict={image_string:x})

                    # Add the useful item
                    image_name_list.append(image_name)
                    GT_label_list.append(GT_label)
                    pred_list.append(r_predictions[0])
                    sample_number = sample_number+1
                print(str(sample_number)+" "+FLAGS.dataset_split_name+" samples found in"+os.path.join(FLAGS.dataset_dir, tfrecord_file))
                sess.close()
                result_dict["image_name"]=image_name_list
                result_dict["GT_label"]=GT_label_list
                result_dict["prediction"]=pred_list
                for i in range(len(result_dict["image_name"])):
                    writer.writerow({"image_name":str(result_dict["image_name"][i]), "prediction":int(result_dict["prediction"][i]), "GT_label":str(result_dict["GT_label"][i])})
        print("CSV file saved in: "+ csv_path)

if __name__ == '__main__':
  tf.app.run()

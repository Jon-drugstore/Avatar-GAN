import os
import cv2
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple
from models.module import generator_resnet_tiny, generator_unet
from utils.utils import save_images, imread, inverse_transform

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

class AvatarGAN:
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args

        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        
        if args.use_resnet:
            self.generator = generator_resnet_tiny
        else:
            self.generator = generator_unet
        
        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        print("build model...", end=" ", flush=True)
        self._build_model()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        print("done.", flush=True)

        if self.args.phase == "test":
            print("load pre-trained model from checkpoints...", flush=True)
            if self.load(args.checkpoint_dir):
                print("done.", flush=True)
            else:
                print("failed!!!", flush=True)

    def _build_model(self):
        # prepare test placeholders
        self.test_A = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                     name='test_A')

        # prepare testing
        self.testB = self.generator(self.test_A, self.options, reuse=False, name="generatorA2B")
        
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("./results/logs", self.sess.graph)

    def load(self, checkpoint_dir):
        print("load pre-trained model...", end=" ", flush=True)
        checkpoint_dir = os.path.join(checkpoint_dir, "combine3_%s" % self.image_size)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            #print_tensors_in_checkpoint_file(file_name=os.path.join(checkpoint_dir, ckpt_name), tensor_name='', all_tensors=False)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def infer(self, image, save_path, bright_diff=0, is_grayscale=True):
        # read image
        if isinstance(image, str):
            img = imread(image, is_grayscale=is_grayscale)
        else:
            img = image
        img = cv2.resize(img, dsize=(256, 256))
        img = np.reshape(img, newshape=(img.shape[0], img.shape[1], 1))
        gen_avatar = self.sess.run(self.testB, feed_dict={self.test_A: [img]})
        if save_path is not None:
            save_images(gen_avatar+bright_diff, size=[1, 1], image_path=save_path)
        gen_avatar = np.reshape(gen_avatar, newshape=list(gen_avatar.shape[1:-1]))
        gen_avatar = inverse_transform(gen_avatar)
        return gen_avatar


import os
import cv2
import time
import numpy as np
import tensorflow as tf
from glob import glob
from collections import namedtuple
from models.module import generator_resnet_tiny, discriminator, generator_unet, mae_criterion, sce_criterion, \
    abs_criterion
from utils.utils import load_train_data, load_test_data, ImagePool, save_images, imread, inverse_transform


class AvatarGAN:
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args

        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet_tiny
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        print("build model...", end=" ", flush=True)
        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=3)
        self.pool = ImagePool(args.max_size)
        print("done.", flush=True)

        if self.args.phase == "test":
            print("load pre-trained model from checkpoints...", flush=True)
            if self.load(args.checkpoint_dir):
                print("done.", flush=True)
            else:
                print("failed!!!", flush=True)

    def _build_model(self):
        # prepare train placeholders
        self.real_data = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim +
                                                     self.output_c_dim], name='real_A_and_B_images')
        self.real_data_cd = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim +
                                                        self.output_c_dim], name="real_C_and_D_images")

        self.fake_A_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                            name='fake_A_sample')

        self.fake_B_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.output_c_dim],
                                            name='fake_B_sample')

        self.fake_C_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                            name='fake_C_sample')

        self.fake_D_sample = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.output_c_dim],
                                            name='fake_D_sample')

        # prepare test placeholders
        self.test_A = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                     name='test_A')
        self.test_B = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.output_c_dim],
                                     name='test_B')
        self.test_C = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.input_c_dim],
                                     name='test_C')
        self.test_D = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.output_c_dim],
                                     name='test_D')

        self.real_A = self.real_data[:, :, :, 0: self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim: self.input_c_dim + self.output_c_dim]
        self.real_C = self.real_data_cd[:, :, :, 0: self.input_c_dim]
        self.real_D = self.real_data_cd[:, :, :, self.input_c_dim: self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.fake_D = self.generator(self.real_C, self.options, True, name="generatorA2B")
        self.fake_C_ = self.generator(self.fake_D, self.options, True, name="generatorB2A")
        self.fake_C = self.generator(self.real_D, self.options, True, name="generatorB2A")
        self.fake_D_ = self.generator(self.fake_C, self.options, True, name="generatorA2B")

        # compute g loss and d loss on A and B datasets
        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) + self.L1_lambda * abs_criterion(
            self.real_A, self.fake_A_) + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.L1_lambda * abs_criterion(
            self.real_A, self.fake_A_) + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.criterionGAN(
            self.DB_fake, tf.ones_like(self.DB_fake)) + self.L1_lambda * abs_criterion(
            self.real_A, self.fake_A_) + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        # compute g loss and d loss on C and D datasets
        self.DD_fake = self.discriminator(self.fake_D, self.options, reuse=False, name="discriminatorD")
        self.DC_fake = self.discriminator(self.fake_C, self.options, reuse=False, name="discriminatorC")
        self.DD_real = self.discriminator(self.real_D, self.options, reuse=True, name="discriminatorD")
        self.DC_real = self.discriminator(self.real_C, self.options, reuse=True, name="discriminatorC")
        self.DD_fake_sample = self.discriminator(self.fake_D_sample, self.options, reuse=True, name="discriminatorD")
        self.DC_fake_sample = self.discriminator(self.fake_C_sample, self.options, reuse=True, name="discriminatorC")

        self.g_loss_c2d = self.criterionGAN(self.DD_fake, tf.ones_like(self.DD_fake)) + self.L1_lambda * abs_criterion(
            self.real_C, self.fake_C_) + self.L1_lambda * abs_criterion(self.real_D, self.fake_D_)
        self.g_loss_d2c = self.criterionGAN(self.DC_fake, tf.ones_like(self.DC_fake)) + self.L1_lambda * abs_criterion(
            self.real_C, self.fake_C_) + self.L1_lambda * abs_criterion(self.real_D, self.fake_D_)
        self.g_loss_cd = self.criterionGAN(self.DC_fake, tf.ones_like(self.DC_fake)) + self.criterionGAN(
            self.DD_fake, tf.ones_like(self.DD_fake)) + self.L1_lambda * abs_criterion(
            self.real_C, self.fake_C_) + self.L1_lambda * abs_criterion(self.real_D, self.fake_D_)

        self.dd_loss_real = self.criterionGAN(self.DD_real, tf.ones_like(self.DD_real))
        self.dd_loss_fake = self.criterionGAN(self.DD_fake_sample, tf.zeros_like(self.DD_fake_sample))
        self.dd_loss = (self.dd_loss_real + self.dd_loss_fake) / 2
        self.dc_loss_real = self.criterionGAN(self.DC_real, tf.ones_like(self.DC_real))
        self.dc_loss_fake = self.criterionGAN(self.DC_fake_sample, tf.zeros_like(self.DC_fake_sample))
        self.dc_loss = (self.dc_loss_real + self.dc_loss_fake) / 2
        self.d_loss_cd = self.dc_loss + self.dd_loss

        # summarize two losses
        self.g_loss = self.args.alpha * self.g_loss + (1.0 - self.args.alpha) * self.g_loss_cd
        self.d_loss = self.args.alpha * self.d_loss + (1.0 - self.args.alpha) * self.d_loss_cd

        # add to summary
        self._add_summary()

        # prepare testing
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")
        self.testC = self.generator(self.test_C, self.options, True, name="generatorA2B")
        self.testD = self.generator(self.test_D, self.options, True, name="generatorB2A")

        # parameters
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        # training parameters
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.args.beta1).minimize(self.d_loss,
                                                                                       var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.args.beta1).minimize(self.g_loss,
                                                                                       var_list=self.g_vars)

        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("./results/logs", self.sess.graph)

    def _add_summary(self):
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge([self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum, self.db_loss_sum,
                                       self.db_loss_real_sum, self.db_loss_fake_sum, self.d_loss_sum])

    def save(self, checkpoint_dir, step):
        model_name = "avatargan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print("load pre-trained model...", end=" ", flush=True)
        checkpoint_dir = os.path.join(checkpoint_dir, "combine_%s" % self.image_size)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        data_a = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        data_b = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(data_a)
        np.random.shuffle(data_b)
        batch_files = list(zip(data_a[:self.batch_size], data_b[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)
        fake_a, fake_b = self.sess.run([self.fake_A, self.fake_B], feed_dict={self.real_data: sample_images})
        save_images(fake_a, [self.batch_size, 1], './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_b, [self.batch_size, 1], './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def train(self, args):

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print("done.", flush=True)
            else:
                print("failed!!!", flush=True)

        # load datasets
        data_a = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        data_b = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
        data_c = glob("./datasets/{}/*.*".format(self.args.dataset_dir + "/trainC"))
        data_d = glob("./datasets/{}/*.*".format(self.args.dataset_dir + "/trainD"))

        for epoch in range(args.epoch):
            # shuffle dataset
            np.random.shuffle(data_a)
            np.random.shuffle(data_b)
            batch_idxs = min(min(len(data_a), len(data_b)), args.train_size) // self.batch_size
            np.random.shuffle(data_c)
            np.random.shuffle(data_d)
            batch_idxs_cd = min(min(len(data_c), len(data_d)), args.train_size) // self.batch_size

            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)
            print(("Epoch: [%2d]  time: %4.4f" % (epoch, time.time() - start_time)))

            for idx in range(0, batch_idxs):
                if idx >= batch_idxs_cd:
                    idx_cd = idx % batch_idxs_cd
                else:
                    idx_cd = idx

                batch_files = list(zip(data_a[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       data_b[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in
                                batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                batch_files_cd = list(zip(data_c[idx_cd * self.batch_size:(idx_cd + 1) * self.batch_size],
                                          data_d[idx_cd * self.batch_size:(idx_cd + 1) * self.batch_size]))
                batch_images_cd = [load_train_data(batch_file_cd, args.load_size, args.fine_size) for batch_file_cd in
                                   batch_files_cd]
                batch_images_cd = np.array(batch_images_cd).astype(np.float32)

                # Update G network and record fake outputs
                fake_a, fake_b, fake_c, fake_d, _, summary_str = self.sess.run([self.fake_A, self.fake_B, self.fake_C,
                                                                                self.fake_D, self.g_optim, self.g_sum],
                                                                               feed_dict={
                                                                                   self.real_data: batch_images,
                                                                                   self.real_data_cd: batch_images_cd,
                                                                                   self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_a, fake_b] = self.pool([fake_a, fake_b])
                [fake_c, fake_d] = self.pool([fake_c, fake_d])

                # Update D network
                _, summary_str = self.sess.run([self.d_optim, self.d_sum], feed_dict={
                    self.real_data: batch_images,
                    self.real_data_cd: batch_images_cd,
                    self.fake_A_sample: fake_a,
                    self.fake_B_sample: fake_b,
                    self.fake_C_sample: fake_c,
                    self.fake_D_sample: fake_d,
                    self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

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

    def test(self, args):
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA_b'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            new_shape = list(sample_image.shape) + [1]
            sample_image = np.reshape(sample_image, newshape=new_shape)
            sample_image = sample_image[:, :, :, :self.input_c_dim]
            test_path = os.path.join(args.test_dir, args.dataset_dir)
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            image_path = os.path.join(args.test_dir, args.dataset_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                    '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                    '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()

from __future__ import division
import os
import time
from glob import glob
from collections import namedtuple

from module import *
from utils import *


class cyclegan(object):
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
            self.generator = generator_resnet
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

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=3)
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size, self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        '''-------------------------------------------- added placeholders ------------------------------------------'''
        self.real_data_cd = tf.placeholder(tf.float32,
                                           [None, self.image_size, self.image_size,
                                            self.input_c_dim + self.output_c_dim],
                                           name="real_C_and_D_images")
        '''----------------------------------------------------------------------------------------------------------'''

        self.real_A = self.real_data[:, :, :, 0: self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim: self.input_c_dim + self.output_c_dim]

        '''---------------------------------- added processes: split input images -----------------------------------'''
        self.real_C = self.real_data_cd[:, :, :, 0: self.input_c_dim]
        self.real_D = self.real_data_cd[:, :, :, self.input_c_dim: self.input_c_dim + self.output_c_dim]
        '''----------------------------------------------------------------------------------------------------------'''

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        '''---------------------- added processes: share generator to compute fake C and D images -------------------'''
        self.fake_D = self.generator(self.real_C, self.options, True, name="generatorA2B")  # C to fake D
        self.fake_C_ = self.generator(self.fake_D, self.options, True, name="generatorB2A")  # fake D to fake C
        self.fake_C = self.generator(self.real_D, self.options, True, name="generatorB2A")  # real D to fake C
        self.fake_D_ = self.generator(self.fake_C, self.options, True, name="generatorA2B")  # fake C to fake D
        '''----------------------------------------------------------------------------------------------------------'''

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")

        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) + self.L1_lambda * abs_criterion(
            self.real_A, self.fake_A_) + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.L1_lambda * abs_criterion(
            self.real_A, self.fake_A_) + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.criterionGAN(
            self.DB_fake, tf.ones_like(self.DB_fake)) + self.L1_lambda * abs_criterion(
            self.real_A, self.fake_A_) + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')

        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')

        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")

        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        '''---------------- added processed: individual discriminator for dataset C and D ---------------------------'''
        # if not share discriminator: change all the discriminator name in this block to "discriminator[C/D]"
        # and the set the first "discriminatorC" and "discriminatorD" to `reuse=False`
        self.DD_fake = self.discriminator(self.fake_D, self.options, reuse=False, name="discriminatorD")
        self.DC_fake = self.discriminator(self.fake_C, self.options, reuse=False, name="discriminatorC")

        self.g_loss_c2d = self.criterionGAN(self.DD_fake, tf.ones_like(self.DD_fake)) + self.L1_lambda * abs_criterion(
            self.real_C, self.fake_C_) + self.L1_lambda * abs_criterion(self.real_D, self.fake_D_)

        self.g_loss_d2c = self.criterionGAN(self.DC_fake, tf.ones_like(self.DC_fake)) + self.L1_lambda * abs_criterion(
            self.real_C, self.fake_C_) + self.L1_lambda * abs_criterion(self.real_D, self.fake_D_)

        self.g_loss_cd = self.criterionGAN(self.DC_fake, tf.ones_like(self.DC_fake)) + self.criterionGAN(
            self.DD_fake, tf.ones_like(self.DD_fake)) + self.L1_lambda * abs_criterion(
            self.real_C, self.fake_C_) + self.L1_lambda * abs_criterion(self.real_D, self.fake_D_)

        self.fake_C_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_C_sample')

        self.fake_D_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_D_sample')

        self.DD_real = self.discriminator(self.real_D, self.options, reuse=True, name="discriminatorD")
        self.DC_real = self.discriminator(self.real_C, self.options, reuse=True, name="discriminatorC")

        self.DD_fake_sample = self.discriminator(self.fake_D_sample, self.options, reuse=True, name="discriminatorD")
        self.DC_fake_sample = self.discriminator(self.fake_C_sample, self.options, reuse=True, name="discriminatorC")

        self.dd_loss_real = self.criterionGAN(self.DD_real, tf.ones_like(self.DD_real))
        self.dd_loss_fake = self.criterionGAN(self.DD_fake_sample, tf.zeros_like(self.DD_fake_sample))
        self.dd_loss = (self.dd_loss_real + self.dd_loss_fake) / 2
        self.dc_loss_real = self.criterionGAN(self.DC_real, tf.ones_like(self.DC_real))
        self.dc_loss_fake = self.criterionGAN(self.DC_fake_sample, tf.zeros_like(self.DC_fake_sample))
        self.dc_loss = (self.dc_loss_real + self.dc_loss_fake) / 2
        self.d_loss_cd = self.dc_loss + self.dd_loss
        '''----------------------------------------------------------------------------------------------------------'''

        ''' ------------------------------------------- merge two losses ------------------------------------------- '''
        self.g_loss = self.args.alpha * self.g_loss + (1.0 - self.args.alpha) * self.g_loss_cd
        self.d_loss = self.d_loss + self.d_loss_cd
        '''----------------------------------------------------------------------------------------------------------'''

        # add loss to tf_summary
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

        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        '''Note: Here we do not add the augmented C and D related losses into tf summary'''

        # prepare test placeholders
        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        # test reconstructed results
        self.recB = self.generator(self.testA, self.options, True, name="generatorA2B")
        self.recA = self.generator(self.testB, self.options, True, name="generatorB2A")
        ''' -------------------------------- prepare test placeholders for C and D ----------------------------------'''
        self.test_C = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_C')
        self.test_D = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_D')
        self.testC = self.generator(self.test_C, self.options, True, name="generatorA2B")
        self.testD = self.generator(self.test_D, self.options, True, name="generatorB2A")
        '''----------------------------------------------------------------------------------------------------------'''

        # here is used to print variable names
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars:
            print(var.name)

        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.args.beta1).minimize(self.d_loss,
                                                                                       var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.args.beta1).minimize(self.g_loss,
                                                                                       var_list=self.g_vars)

        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    def train(self, args):

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        # load datasets
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))

        ''' ---------------------------- add C and D datasets -------------------------------------------------------'''
        dataC = glob("./datasets/{}/*.*".format(self.args.dataset_dir + "/trainC"))
        dataD = glob("./datasets/{}/*.*".format(self.args.dataset_dir + "/trainD"))
        ''' -------------------------------------------------------------------------------------------------------- '''

        for epoch in range(args.epoch):
            # shuffle dataset
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            '''------------------------------------------------------------------------------------------------------'''
            np.random.shuffle(dataC)
            np.random.shuffle(dataD)
            batch_idxs_cd = min(min(len(dataC), len(dataD)), args.train_size) // self.batch_size
            '''------------------------------------------------------------------------------------------------------'''

            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch - epoch) / (args.epoch - args.epoch_step)
            print(("Epoch: [%2d]  time: %4.4f" % (epoch, time.time() - start_time)))

            for idx in range(0, batch_idxs):
                ''' ---------------------------------- assign index for C and D dataset -----------------------------'''
                if idx >= batch_idxs_cd:
                    idx_cd = idx % batch_idxs_cd
                else:
                    idx_cd = idx
                '''--------------------------------------------------------------------------------------------------'''

                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in
                                batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                ''' ------------------------------- load C and D datasets -------------------------------------------'''
                batch_files_cd = list(zip(dataC[idx_cd * self.batch_size:(idx_cd + 1) * self.batch_size],
                                          dataD[idx_cd * self.batch_size:(idx_cd + 1) * self.batch_size]))
                batch_images_cd = [load_train_data(batch_file_cd, args.load_size, args.fine_size) for batch_file_cd in
                                   batch_files_cd]
                batch_images_cd = np.array(batch_images_cd).astype(np.float32)
                '''--------------------------------------------------------------------------------------------------'''

                # Update G network and record fake outputs
                '''fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])'''

                fake_A, fake_B, fake_C, fake_D, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.fake_C, self.fake_D, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.real_data_cd: batch_images_cd, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])
                [fake_C, fake_D] = self.pool([fake_C, fake_D])

                # Update D network
                '''_, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)'''

                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.real_data_cd: batch_images_cd,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.fake_C_sample: fake_C,
                               self.fake_D_sample: fake_D,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                # print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        # step = 1002
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        #model_dir = "combine3_%s" % (self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

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


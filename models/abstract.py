import os
import numpy as np
import tensorflow as tf
from utils import various as uv
from matplotlib import pyplot as plt

"""
    tensorboard --logdir=saved/netname
"""

class AbstractNet(object):
    """ Common interface """
    def __init__(self, netname, params = None):
        """ First try """
        self.netname = netname
        self.savepath = 'saved/' + netname + '.ckpt'
        self.boardpath = 'saved/{}'.format(self.netname)

        # Parameters preprocessing
        self.params = self.consume_params(params)

        # Training parameters
        self.lrs        = [0.1, 0.003, 0.0005, 1e-4]
        self.epochs     = 1000
        self.batch_size = 256

        # Meta factors
        self.std = 0.5

        # Training and validation loss
        self.accurs = []
        self.losses = []
        self.v_losses = []

    def print_ops(self):
        """ Viewer """
        # Start tensorflowing
        with tf.Session(graph = self.graph) as sess:
            # Load weights and biases
            saver = tf.train.Saver()
            saver.restore(sess, self.savepath)
            ops = self.graph.get_operations()

        return ops

    def get_tensor(self, name):
        """ Value by name """
        with tf.Session(graph = self.graph) as sess:
            # Load weights and biases
            saver = tf.train.Saver()
            saver.restore(sess, self.savepath)

            tensor = self.graph.get_tensor_by_name(name)
            val = sess.run(tensor)

        return val

    def set_batch_size(self, batch_size):
        """ Setter """
        self.batch_size = batch_size

    def set_learning_rates(self, rates):
        """ Feed me list """
        self.lrs  = rates

    def consume_params(self, params):
        """ Separate training from inference """
        # Re-use parameters
        if not params:
            # Load from below
            params = np.load(self.savepath + '.npy')
            # We are loading so boardpath must exist
        else:
            # Save name-specific set of parameters
            np.save(self.savepath, params)

            # Prepare a place for the tensorboard files
            if not os.path.exists(self.boardpath):
                os.makedirs(self.boardpath)
            else:
                # Remove everythin
                uv.clean_path(self.boardpath)

        return params

    def build_net(self):
        """ Second round constructor """
        # Start building the graph
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.keep_prob = tf.placeholder(tf.float32)
            self.learning_rate = tf.Variable(0.0,
                                        name = 'l_rate',
                                        trainable=False)
            self.make_graph()
            self.prepare_board()
            self.prepare_training()

    def set_epochs(self, howmany):
        """ Setter """
        self.epochs = howmany

    def prepare_training(self):
        """ Make tf.op for that """
        opt = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = opt.minimize(self.loss)

    def train(self, dataset):
        """ Hard """
        # Prepare tensorflow environment
        with tf.Session(graph = self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Iteration iterator
            it = 0
            for en, l_rate in enumerate(self.lrs):
                # Varying learning rate mechanism
                sess.run(tf.assign(self.learning_rate, l_rate))

                # Run the epochs
                for et in range(self.epochs):
                    # Load batch
                    bx, by = dataset.next_batch(self.batch_size)

                    fd = { self._input  : bx,
                           self._output   : by }

                    ops = [self.loss,
                           self.train_op,
                           self.train_summary]

                    loss, _, ss = sess.run(ops,
                                           feed_dict = fd)

                    # Tensorboard
                    self.t_writer.add_summary(ss, it)
                    it += 1

                    # Conventional debugging
                    self.losses.append(loss)
                    if et % 10 == 0:
                        # Prepare validation data/ops (on a larger batch)
                        vx, vy = dataset.validation_batch(1000)
                        vdic = { self._input : vx,
                                 self._output  : vy }
                        vops = [self.loss, self.valid_summary]

                        # Validation run
                        vloss, vsum = sess.run(vops, vdic)
                        self.t_writer.add_summary(vsum, it)
                        dbg = 'Eon: {}, Epoch: {}, loss: {}, valid: {}'
                        print dbg.format(en, et, loss, vloss)

                    if et % 100 == 0:
                        self.produce_plots()

            # Save results
            saver = tf.train.Saver()
            saver.save(sess, self.savepath)
            print 'Trained model saved to: ', self.savepath

    def produce_plots(self):
        """ Make some plots """
        # Learning curve
        savepath = 'tmp/{}.png'.format(self.netname)
        plt.clf()
        plt.plot(self.losses,
                 '--o',
                 lw = 2,
                 ms = 10,
                 alpha = 0.8)

        # Loss with 3 significant digits pls
        title = '{0} - {1:.3g}'.format(self.netname,
                                       self.losses[-1])
        plt.title(title)
        plt.savefig(savepath)
        plt.clf()

    def prepare_board(self):
        """ Tensorboard """
        # Try summaries
        with tf.name_scope('summaries'):
            self.train_summary = tf.summary.scalar('training_loss', self.loss)
            self.valid_summary = tf.summary.scalar('validation_loss', self.loss)

        self.t_writer = tf.summary.FileWriter(self.boardpath,
                                              self.graph)

    def process(self, signals):
        """ Analyze one batch """
        # Start tensorflowing
        with tf.Session(graph = self.graph) as sess:
            # Load weights and biases
            saver = tf.train.Saver()
            saver.restore(sess, self.savepath)

            # Feed the graph
            fd = { self._input : signals }

            score = sess.run(self.inference, fd)

        return score

    def consume(self, signals, reality):
        """ Calculate loss """
        # Start tensorflowing
        with tf.Session(graph = self.graph) as sess:
            # Load weights and biases
            saver = tf.train.Saver()
            saver.restore(sess, self.savepath)

            # Feed the graph
            fd = { self._input : signals,
                   self._output : reality }

            score = sess.run(self.loss, fd)

        return score

    def make_graph():
        """ Please implement me """
        print 'Please implement the make_graph() method'

class ConvEncoder(AbstractNet):
    """ Simplest class to solve the problem """
    def __init__(self, netname, params = None):
        """ Constructor """
        # Init the parent
        AbstractNet.__init__(self, netname, params)

        # lrs = [0.5, 0.003, 0.0005, 1e-4]
        lrs = [0.003, 0.0007]
        self.set_learning_rates(lrs)

        # Extract the parameters
        self.n_input = self.params[0]
        self.n_output = self.n_input

        self.stddev = 0.5
        self.batch_size = 128
        self.set_epochs(1500)

        # Make the graphs
        self.build_net()

    def conv(self, values, filters, width, stride):
        """ Wrapper for a full conv/pool layer """
        # Deduce the required filter shapes
        v_shape = values.get_shape().as_list()
        in_filters = v_shape[2]

        # Prepare weights for training
        w_shape = [width, in_filters , filters]
        weights = tf.random_normal(w_shape, stddev = self.stddev)
        weights = tf.Variable(weights, name = 'weights')

        # Convolve
        out = tf.nn.conv1d(values,
                           weights,
                           stride = stride,
                           padding = 'VALID')

        b_shape = [filters]
        bias = tf.random_normal(b_shape, stddev = self.stddev)
        bias = tf.Variable(bias, name = 'bias')

        out += bias

        out = tf.nn.tanh(out)

        print 'Conv:', out.get_shape().as_list()

        return out

    def convt(self, values, filters, width, stretch):
        """ Transposed convolution """
        # Add dimension to fit the conv2d_tranpose
        # We want to use NHWC format and have height = 1
        # So we add a dimension after the first existing one
        values = tf.expand_dims(values, 2)

        # Extract the number of input channels
        in_shape = values.get_shape().as_list()
        in_filters = in_shape[3]

        # Prepare the weights to be learned
        w_shape = [1, width, filters, in_filters]
        weights = tf.random_normal(w_shape)
        weights = tf.Variable(weights, name = 'weights')

        # Infere the output shape
        output_shape = in_shape[:]
        output_shape[3] = filters
        output_shape[1] *= stretch
        # output_shape[0] = -1

        out = tf.nn.conv2d_transpose(values,
                                     weights,
                                     output_shape,
                                     strides = [1, stretch, 1, 1],
                                     padding = 'SAME')

        # Get rid of the added dimension
        out = tf.squeeze(out, 2)

        b_shape = [filters]
        bias = tf.random_normal(b_shape, stddev = self.stddev)
        bias = tf.Variable(bias, name = 'bias')

        out += bias

        print 'DeConv:', out.get_shape().as_list()

        return out
                                     
    def full_layer(self, values, out_width):
        """ Matmul and bias add """
        # Prepare weights
        in_width = values.get_shape().as_list()[1]
        w1_shape = [in_width, out_width]
        w1 = tf.random_normal(w1_shape, stddev = self.std)
        w1 = tf.Variable(w1)

        # Propagate
        out = tf.matmul(values, w1)

        # Make the bias
        b1 = tf.random_normal(shape = [out_width], stddev = self.std)
        b1 = tf.Variable(b1)
        out += b1

        return out

    def make_graph(self):
        """ Construct the pipeline """
        # Connection with the input data
        in_shape = [None, self.n_input]
        self._input = tf.placeholder(tf.float32, in_shape)
        print 'Input:', in_shape

        # Shrink down
        with tf.name_scope('conv1'):
            # Add the channels dimension
            c1 = tf.expand_dims(self._input, -1)
            c1 = self.conv(c1,
                           filters = 32,
                           width = 32,
                           stride = 1)

        with tf.name_scope('conv2'):
            c2 = self.conv(c1,
                           filters = 64,
                           width = 64,
                           stride = 1)

        with tf.name_scope('conv3'):
            c3 = self.conv(c2,
                           filters = 64,
                           width = 64,
                           stride = 1)

        with tf.name_scope('conv4'):
            c4 = self.conv(c3,
                           filters = 32,
                           width = 32,
                           stride = 1)

        with tf.name_scope('conv5'):
            c5 = self.conv(c4,
                           filters = 16,
                           width = 32,
                           stride = 1)

        # Stretch up (with transposed convolutions)
        with tf.name_scope('softmax1'):
            c5_shape = c5.get_shape().as_list()
            r1_width = c5_shape[1] * c5_shape[2]
            r1 = tf.reshape(c5, [-1, r1_width])
            r1 = tf.nn.softmax(r1)

        with tf.name_scope('full_softmax'):
            r2 = self.full_layer(r1, self.n_input + 127)
            r2 = tf.nn.softmax(r2)

        with tf.name_scope('conv_out'):
            r2 = tf.expand_dims(r2, -1)
            r2 = self.conv(r2,
                           filters = 1,
                           width = 128,
                           stride = 1)

        print 'Stretched:', r2.get_shape().as_list()

        # Connect to the ground truth
        with tf.name_scope('inference'):
            self.inference = tf.squeeze(r2, 2)
            self._output = tf.placeholder(tf.float32, in_shape)

            # Make the loss op
            diff = self._output - self.inference
            power = tf.pow(diff, 2)
            self.loss = tf.reduce_mean(power)

class FIREncoder(ConvEncoder):
    """ Simplified version """
    def make_graph(self):
        """ Construct the pipeline """
        # Connection with the input data
        in_shape = [None, self.n_input]
        self._input = tf.placeholder(tf.float32, in_shape)
        print 'Input:', in_shape

        # Shrink down
        with tf.name_scope('conv1'):
            # Add the channels dimension
            c1 = tf.expand_dims(self._input, -1)
            c1 = self.conv(c1,
                           filters = 4,
                           width = 128,
                           stride = 1)

        with tf.name_scope('conv2'):
            c2 = self.conv(c1,
                           filters = 4,
                           width = 64,
                           stride = 1)

        with tf.name_scope('conv3'):
            c3 = self.conv(c2,
                           filters = 4,
                           width = 32,
                           stride = 1)

        with tf.name_scope('shuffle'):
            c3_shape = c3.get_shape().as_list()
            r1_width = c3_shape[1] * c3_shape[2]

            r1 = tf.reshape(c3, [-1, r1_width])
            r1 = self.full_layer(r1, self.n_input + 127)

        with tf.name_scope('conv_out'):
            r2 = tf.expand_dims(r1, -1)
            r2 = self.conv(r2,
                           filters = 1,
                           width = 128,
                           stride = 1)

        # Connect to the ground truth
        with tf.name_scope('inference'):
            self.inference = tf.squeeze(r2, 2)
            self._output = tf.placeholder(tf.float32, in_shape)

            # Make the loss op
            diff = self._output - self.inference
            power = tf.pow(diff, 2)
            self.loss = tf.reduce_mean(power)


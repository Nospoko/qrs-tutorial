import os
import numpy as np
import tensorflow as tf
from utils import various as uv

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
        self.lrs        = [0.001, 0.0003, 0.00005, 1e-5]
        self.eons       = len(self.lrs)
        self.epochs     = 100
        self.batch_size = 256

        # Meta factors
        self.std = 0.5

        # Training and validation loss
        self.accurs = []
        self.losses = []
        self.v_losses = []

    def set_learning_rates(self, rates):
        """ Feed me list """
        self.lrs  = rates
        self.eons = len(rates)

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

    def full_layer(self, putin, out_width):
        """ Matmul and bias add """
        # Prepare weights
        in_width = putin.get_shape().as_list()[1]
        w1_shape = [in_width, out_width]
        w1 = tf.random_normal(w1_shape, stddev = self.std)
        w1 = tf.Variable(w1)

        # Propagate
        out = tf.matmul(putin, w1)

        # Make the bias
        # b1 = 0.1 * tf.ones(shape = [out_width])
        b1 = tf.random_normal(shape = [out_width])
        b1 = tf.Variable(b1)
        out += b1

        return out

    def conv_layer(self, putin, f = 1, s = 1):
        """ Name it, use f filters """
        # Input size deduction
        in_layers = putin.get_shape().as_list()[3]

        # Prepare first conv layer
        w1_shape = [self.f_n_size,
                    self.f_m_size,
                    in_layers,
                    f]

        wc1 = tf.random_normal(w1_shape, stddev = self.std)
        wc1 = tf.Variable(wc1, name = 'filters')
        c1 = tf.nn.conv2d(putin,
                          wc1,
                          strides = [1, 1, 1, 1],
                          padding = 'SAME')

        bc1 = tf.constant(0.1, shape = [f])
        bc1 = tf.Variable(bc1, name = 'biases')

        c1 += bc1

        c1 = tf.nn.tanh(c1)

        c1 = tf.nn.max_pool(c1,
                            ksize = [1, 1, s, 1],
                            strides = [1, 1, s, 1],
                            padding = 'SAME')
        return c1

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

            for l_rate, en in zip(self.lrs, range(self.eons)):
                # Varying learning rate mechanism
                sess.run(tf.assign(self.learning_rate, l_rate))

                # Run the epochs
                for et in range(self.epochs):

                    bx, by = dataset.next_batch(self.batch_size)
                    # FIXME Clean this up

                    fd = { self.putin  : bx,
                           self.real   : by }

                    ops = [self.loss,
                           self.train_op,
                           self.summaries]

                    loss, _,  ss = sess.run(ops,
                                            feed_dict = fd)

                    # Tensorboard
                    self.t_writer.add_summary(ss, it)
                    it += 1

                    # Conventional debugging
                    self.losses.append(loss)
                    dbg = 'Eon: {}, Epoch: {}, loss: {}'
                    if et % 60 == 0:
                        print dbg.format(en, et, loss)
                    if et % 100 == 0:
                        self.validate()
                        # print '! Learning rate:', l_rate

            # Save results
            saver = tf.train.Saver()
            saver.save(sess, self.savepath)
            print 'Trained model saved to: ', self.savepath

    def validate(self):
        """ Make some plots """
        # Learning curve
        savepath = 'tmp/{}.png'.format(self.netname)
        plt.clf()
        plt.plot(self.losses[-500:],
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
            tf.summary.scalar('loss', self.loss)

        self.summaries = tf.summary.merge_all()

        self.t_writer = tf.summary.FileWriter(self.boardpath,
                                              self.graph)

    def process(self, batch):
        """ Analyze one batch """
        # Start tensorflowing
        with tf.Session(graph = self.graph) as sess:
            # Load weights and biases
            saver = tf.train.Saver()
            saver.restore(sess, self.savepath)

            # Feed the graph
            fd = { self.putin : batch }

            score = sess.run(self.inference, fd)

        return score

class FirstTry(AbstractNet):
    """ Simplest class to solve the problem """
    def __init__(self, netname, params = None):
        """ Constructor """
        # Init the parent
        AbstractNet.__init__(self, netname, params)

        # Extract the parameters
        self.n_input = self.params[0]
        self.n_output = self.params[1]

        self.stddev = 0.1

        # Make the graphs
        self.build_net()

    def conv(self, value, filters, width, stride):
        """ Wrapper for a full conv/pool layer """
        # Deduce the required filter shapes
        v_shape = value.get_shape().as_list()
        in_width = v_shape[2]

        # Prepare weights for training
        w_shape = [width, in_width, filters]
        weights = tf.random_normal(w_shape, stddev = self.stddev)
        weights = tf.Variable(weights, name = 'weights')

        # Convolve
        out = tf.nn.conv1d(value,
                           weights,
                           stride = stride,
                           padding = 'SAME')

        b_shape = [filters]
        bias = tf.random_normal(b_shape, stddev = self.stddev)
        bias = tf.Variable(bias, name = 'bias')

        out += bias

        print 'Conv:', out.get_shape().as_list()

        return out

    def make_graph(self):
        """ Construct the pipeline """
        # Connection with the input data
        in_shape = [None, self.n_input]
        self.putin = tf.placeholder(tf.float32, in_shape)
        print 'Input:', in_shape

        # Shrink down
        with tf.name_scope('conv1'):
            # Add the channels dimension
            c1 = tf.expand_dims(self.putin, -1)
            c1 = self.conv(c1,
                           filters = 8,
                           width = 13,
                           stride = 2)

        with tf.name_scope('conv2'):
            c2 = self.conv(c1,
                           filters = 8,
                           width = 13,
                           stride = 2)

        with tf.name_scope('conv3'):
            c3 = self.conv(c2,
                           filters = 8,
                           width = 13,
                           stride = 4)

        print 'Shrinked:', c3.get_shape().as_list()

        self.loss = tf.reduce_sum(c3)

        return

        # Stretch up (with transposed convolutions)
        with tf.name_scope('deconv1'):
            d1 = uo.convt(c3,
                          filters = 8,
                          stretch = 2)

        with tf.name_scope('deconv2'):
            d2 = uo.convt(d1,
                          filters = 8,
                          stretch = 2)

        with tf.name_scope('deconv3'):
            d3 = uo.convt(d2,
                          filters = 8,
                          stretch = 4)

        print 'Stretched:', d3.get_shape().as_list()

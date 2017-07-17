def train(self, dataset):
    # Prepare tensorflow environment
    with tf.Session(graph = self.graph) as sess:
        sess.run(tf.global_variables_initializer())

        # Iteration iterator
        it = 0
        for en, l_rate in enumerate(self.learning_rates):
            # Varying learning rate mechanism
            sess.run(tf.assign(self.learning_rate, l_rate))

            # Run the epochs
            for epoch in range(self.epochs):
                # Load batch
                bx, by = dataset.next_batch(self.batch_size)

                # Feed the placeholders
                fd = { self._input  : bx, self._output : by }
                ops = [self.loss, self.train_op, self.train_summary]

                loss, _, ss = sess.run(ops, feed_dict = fd)

                # Tensorboard
                self.t_writer.add_summary(ss, it)
                it += 1

        # Save results
        saver = tf.train.Saver()
        saver.save(sess, self.savepath)
        print 'Trained model saved to: ', self.savepath

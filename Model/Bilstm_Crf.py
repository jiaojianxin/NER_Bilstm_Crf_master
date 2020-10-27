from Parameters import Parameters as pm
from Public_fuc import batch_iter, process_seq, get_bert_vec

import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode


class LSTM_CRF(object):
    """模型类"""
    def __init__(self):
        self.input_x = tf.placeholder(tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, None], name='input_y')
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name='sequence_length')
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.Model()

    def Model(self):
        # tf.device定义模型运行的具体设备，tf.name_scope定义对象属于哪个区域
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
        # with tf.device(None), tf.name_scope('embedding'):
            # embedding_ = tf.Variable(tf.truncated_normal([pm.vocab_size, pm.embedding_size], -0.25, 0.25), name='w')
            # 替换词语向量
            embedding_ = get_bert_vec(pm.word_vec_path)
            # 在嵌入的张量中寻找id
            embedding = tf.nn.embedding_lookup(embedding_, self.input_x)
            # 将张量正则化处理防止过拟合
            self.embedding = tf.nn.dropout(embedding, pm.keep_pro)

        with tf.name_scope('biLSTM'):
            # 定义双向LSTM网络, tf.nn.rnn_cell.LSTMCell与tf.contrib.rnn.LSTMCell一样
            # cell_fw = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim)
            # cell_bw = tf.nn.rnn_cell.LSTMCell(pm.hidden_dim)
            cell_fw = tf.contrib.rnn.LSTMCell(pm.hidden_dim)
            cell_bw = tf.contrib.rnn.LSTMCell(pm.hidden_dim)

            # 创建双向递归神经网络的动态版本
            outputs, outstats = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=self.embedding,
                                                                sequence_length=self.seq_length, dtype=tf.float32)
            # 将双向神经网络拼接
            outputs = tf.concat(outputs, 2)

        with tf.name_scope('output'):
            s = tf.shape(outputs)
            # output = tf.reshape(outputs, [-1, 2 * pm.hidden_dim])
            # dense1 = tf.layers.dense(inputs=output, units=512, activation=tf.nn.relu,
            #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
            # dense2 = tf.layers.dense(inputs=dense1, units=256, activation=tf.nn.relu)
            # output = tf.layers.dense(inputs=dense2, units=pm.num_tags, activation=None)

            # 全连接层，最后输出维度等于pm.num_tags
            output = tf.reshape(outputs, [-1, 2 * pm.hidden_dim])
            output = tf.layers.dense(output, pm.num_tags)
            # TODO 高级tf.nn.dropout，防止过拟合，正则化张量keep_pro元素保留概率
            output = tf.contrib.layers.dropout(output, pm.keep_pro)
            self.logits = tf.reshape(output, [-1, s[1], pm.num_tags])

        with tf.name_scope('crf'):
            # log_likelihood是对数似然函数，transition_params是转移概率矩阵
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits, tag_indices=self.input_y,
                                                                        sequence_lengths=self.seq_length)

        with tf.name_scope('loss'):
            # tf.reduce_mean 主要用作降维或者计算tensor（图像）的平均值。keep_dims：是否降维度Flase降维
            self.loss = tf.reduce_mean(-log_likelihood, keepdims=False)  # 最大似然取负，使用梯度下降

        with tf.name_scope('optimizer'):
            # tf.train.AdamOptimizer寻找全局最优解的优化算法，引入二次方梯度校正
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)  # AdamOptimizer --> adam优化器
            # TODO 梯度剪裁
            gradients, variable = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            self.optimizer = optimizer.apply_gradients(zip(gradients, variable), global_step=self.global_step)
            # self.optimizer = optimizer.apply_gradients(zip(gradients, variable))

    def feed_data(self, x_batch, y_batch, seq_length, keep_pro):
        feed_dict = {self.input_x: x_batch,
                     self.input_y: y_batch,
                     self.seq_length: seq_length,
                     self.keep_pro: keep_pro
                     }
        return feed_dict

    def test(self, sess, x, y):
        batch_test = batch_iter(x, y, batch_size=pm.batch_size)
        for x_batch, y_batch in batch_test:
            x_pad, seq_length_x = process_seq(x_batch)
            y_pad, seq_length_y = process_seq(y_batch)
            feed_dict = self.feed_data(x_pad, y_pad, seq_length_x, 1.0)
            loss = sess.run(self.loss, feed_dict=feed_dict)
            # TODO 返回loss位置需要确定
            return loss

    def predict(self, sess, seqs):
        seq_pad, seq_length = process_seq(seqs)
        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict={self.input_x: seq_pad,
                                                                                               self.seq_length: seq_length,
                                                                                               self.keep_pro: 1.0})
        label_ = []
        for logit, length in zip(logits, seq_length):
            # logit 每个子句的输出值，length子句的真实长度，logit[:length]的真实输出值
            # 调用维特比算法求最优标注序列
            viterbi_seq, _ = viterbi_decode(logit[:length], transition_params)
            label_.append(viterbi_seq)
        return label_

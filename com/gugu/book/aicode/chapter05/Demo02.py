# import os
# from datetime import time
#
# import data_helpers
# import tensorflow as tf
#
# from com.gugu.book.aicode.chapter05.WideAndDeepModel import WideAndDeepModel
#
# # 模型训练数据路径
# tf.compat.v1.flags.DEFINE_string('train_dir', 'zutao2.csv', 'path of train data')
# tf.compat.v1.flags.DEFINE_integer('wide_length', 133, 'path of train data  ')
# tf.compat.v1.flags.DEFINE_integer('deep_length', 133, 'path of train data  ')
# tf.compat.v1.flags.DEFINE_integer('deep_last_layer_len', 128, 'path of train data  ')
# tf.compat.v1.flags.DEFINE_integer('softmax_label', 1001, 'path of train data  ')
# # 设定模型训练参数
# tf.compat.v1.flags.DEFINE_integer('batch_size', 32, 'batch size')
# tf.compat.v1.flags.DEFINE_integer('num_epochs', 5, 'Number of training epochs')
# tf.compat.v1.flags.DEFINE_integer('display_every', 100, 'Number of iterations to display training info.')
# tf.compat.v1.flags.DEFINE_float('learning_rate', 1e-3, 'Which learning rate to start with.')
# tf.compat.v1.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store.')
# tf.compat.v1.flags.DEFINE_integer('checkpoint_every', 500, 'save model after this many steps.')
#
# tf.compat.v1.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
# tf.compat.v1.flags.DEFINE_boolean('log_device_placement', False, 'log placement of ops on devices')
# FLAGS = tf.compat.v1.flags.FLAGS
#
#
# def train():
#     FLAGS = tf.compat.v1.flags.FLAGS
#
#     with tf.device('/cpu:0'):
#         x, y = data_helpers.load_data_and_labels(FLAGS.train_dir)
#     print('-' * 120)
#     print(x.shape)
#     print('-' * 120)
#
#     with tf.Graph().as_default():
#         session_conf = tf.ConfigProto(
#             allow_soft_placement=FLAGS.allow_soft_placement,
#             log_device_placement=FLAGS.log_device_placement
#         )
#         sess = tf.session(config=session_conf)
#
#         with sess.as_default():
#             model = WideAndDeepModel(
#                 wide_length=FLAGS.wide_length,
#                 deep_length=FLAGS.deep_length,
#                 deep_last_layer_len=FLAGS.deep_last_layer_len,
#                 softmax_label=FLAGS.softmax_label
#             )
#
#             global_step = tf.Variable(0, name='global_step', trainable=False)
#             train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)
#
#             timestamp = str(int(time.time()))
#             out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
#             checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
#             checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
#             if not os.path.exists(checkpoint_dir):
#                 os.mkdir(checkpoint_dir)
#             saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGES.num_checkpoints)
#             # 初始化所有变量
#             sess.run(tf.global_variables_initializater())
#
#             batches = data_helpers.batch_iter(
#                 list(zip(x, y)),
#                 FLAGES.batch_size,
#                 FLAGES.num_epochs
#             )
#
#             for batch in batches:
#                 x_batch, y_batch = zip(*batch)
#
#                 feed_dict = {
#                     model.input_wide_part: x_batch,
#                     model.input_deep_part: y_batch,
#                     model.input_y: y_batch
#                 }
#
#                 _, step, loss, accuracy = sess.run([
#                     train_op, global_step, model.loss, model.accuracy
#                 ], feed_dict)
#                 if step % FLAGES.display_every == 0:
#                     time_str = datetime.datetime.now().isoformat()
#                     print("{}：step {}，loss {:g} , acc {:g}".format(time_str, step, loss, accuracy))
#
#                 if step % FLAGES.checkpoint_every == 0:
#                     path = saver.save(sess, checkpoint_prefix, golbal_step=step)
#                     print("Saved model checkpoint to {} \n".format(path))
#
#         save_path = saver.save(sess, checkpoint_prefix)
#
#
# def main(_):
#     train()
#
#
# if __name__ == '__main__':
#     tf.compat.v1.app.run()

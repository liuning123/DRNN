import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time
import datetime
import os
from drnn import DRNN
from tensorflow.contrib import learn
import data_helpers


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("pre_training_word_embedding_file", "./data/glove.6B.300d.txt", "pre_trained word embeddings")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("hidden_dim", 300, "Dimensionality of  rnn output  (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("num_k", 3, "the number of k")
tf.flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [50]")
tf.flags.DEFINE_float("lr_rate", 1e-3, "learninig rate....")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.2, "per_process_gpu_memory_fraction")
tf.flags.DEFINE_string("visible_devices", '1', "CUDA_VISIBLE_DEVICES")

FLAGS = tf.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.visible_devices  # 使用 GPU 1

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev, max_document_length


def get_matrix(vocab_processor):
    embedding_index = {}
    f = open(FLAGS.pre_training_word_embedding_file, 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coef = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coef
    f.close()

    vocab_size = len(vocab_processor.vocabulary_)
    embedding_matrix = np.zeros([vocab_size, 300])
    for word, i in vocab_processor.vocabulary_._mapping.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, vocab_size


def train(x_train, y_train,x_dev, y_dev,  embedding_matrix, vocab_size, sequence_length):
    with tf.Graph().as_default():
        # ------------liuning 2018/07/06--------------
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement,
          gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        num_classes = 2
        num_k = int(FLAGS.num_k)
        embedding_dim = FLAGS.embedding_dim
        hidden_dim = FLAGS.hidden_dim
        batch_size = FLAGS.batch_size
        with sess.as_default():
            drnn = DRNN(sequence_length, num_classes, vocab_size, embedding_dim, hidden_dim, num_k, batch_size)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.lr_rate)
            grads_and_vars = optimizer.compute_gradients(drnn.loss)
            clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], FLAGS.max_grad_norm), gv[1]) \
                                      for gv in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", drnn.loss)
            acc_summary = tf.summary.scalar("accuracy", drnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(drnn.embedding_init, feed_dict={drnn.embedding_placeholder: embedding_matrix})

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  drnn.input_x: x_batch,
                  drnn.input_y: y_batch,
                  drnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                  drnn.is_training: True
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, drnn.loss, drnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  drnn.input_x: x_batch,
                  drnn.input_y: y_batch,
                  drnn.dropout_keep_prob: 1.0,
                  drnn.is_training: False
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, drnn.loss, drnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, sequence_length = preprocess()
    embedding_matrix, vocab_size = get_matrix(vocab_processor)
    del vocab_processor
    train(x_train, y_train, x_dev, y_dev, embedding_matrix, vocab_size, sequence_length)

if __name__ == '__main__':
    tf.app.run()
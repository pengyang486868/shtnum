import tensorflow as tf


def getnet(modelpath):
    sess = tf.Session()
    model_file = tf.train.latest_checkpoint(modelpath)
    saver = tf.train.import_meta_graph(model_file + '.meta')

    saver.restore(sess, model_file)
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    keep = graph.get_tensor_by_name("keep:0")
    return sess, x, y, keep

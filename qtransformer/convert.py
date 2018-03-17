import tensorflow as tf
import numpy as np
import os
import itertools

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

def quantize_weights(ws, num_codes=256, sharded=False):
    if sharded:
        ws_full = np.vstack(ws).flatten()
    else:
        ws_full = ws.flatten()

    # Indexes are the bin that each weight falls in
    bins = np.linspace(0, 100, num_codes + 1)
    pers = np.percentile(ws_full, bins)
    if sharded:
        idxs = [np.digitize(w, pers) - 1 for w in ws]
    else:
        idxs = np.digitize(ws, pers) - 1

    # Quantized weights are the center of each bin
    codes = np.percentile(ws_full, bins[:-1] + np.diff(bins)/2)

    return idxs, codes

def quantize_scope(reader, scope, weights_name, idxs_name, codebook_name,
        num_codes=256, sharded=False, num_shards=1):
    if sharded:
        ws = [reader.get_tensor(scope + '/' + weights_name + '_{}'.format(i))
            for i in range(num_shards)]
    else:
        ws = reader.get_tensor(scope + '/' + weights_name)

    idxs, codes = quantize_weights(ws, num_codes=num_codes, sharded=sharded)

    initializers = []
    with tf.variable_scope(scope, reuse=True):
        initializers.append(tf.assign(tf.get_variable(codebook_name), codes).op)

        if sharded:
            for i in range(num_shards):
                initializers.append(tf.assign(
                    tf.get_variable(idxs_name + '_{}'.format(i), dtype=tf.int32), idxs[i]).op)
        else:
            initializers.append(tf.assign(
                tf.get_variable(idxs_name, dtype=tf.int32), idxs).op)

    return initializers

def make_quant_initializers_from_checkpoint(graph, ckpt, hparams, mode=tf.estimator.ModeKeys.TRAIN):
    with graph.as_default():
        # Build the graph for the problem
        # Create the input
        problem = hparams.problem_instances[0]
        features, labels = problem.input_fn(mode, hparams)

        # Create the model
        model_cls = registry.model('transformer')
        estimator_spec = model_cls.estimator_model_fn(hparams, features, labels, mode)

        # Grab the weights of the embedding variables directly from the checkpoint
        ckpt_reader = tf.train.NewCheckpointReader(ckpt)

        # Grab the number of codes from the hparams
        num_codes = hparams.quantize_codes

        embedding_initializers = quantize_scope(ckpt_reader,
            scope='transformer/symbol_modality_33708_512/shared',
            weights_name='weights', idxs_name='weight_idxs', codebook_name='codebook',
            num_codes=num_codes, sharded=True, num_shards=16)

        ffn_initializers = []
        for layer_name, layer_idx, layer_op in itertools.product(
                ['encoder', 'decoder'], range(6), ['ffn/conv1', 'ffn/conv2']):
            scope = 'transformer/body/{name}/layer_{idx}/{op}'.format(
                    name=layer_name, idx=layer_idx, op=layer_op)
            ffn_initializers.append(
                quantize_scope(ckpt_reader, scope=scope,
                    weights_name='kernel', idxs_name='kernel_idx', codebook_name='codebook',
                    num_codes=num_codes, sharded=False))

        encoder_attn_initializers = []
        for layer_name, layer_idx, attn_type, transform in itertools.product(
                ['encoder'], range(6), ['self_attention'], ['q', 'k', 'v', 'output_transform']):
            scope = ('transformer/body/{layer_name}/layer_{idx}/{attn_type}/'
                + 'multihead_attention/{transform}').format(
                    layer_name=layer_name, idx=layer_idx, attn_type=attn_type, transform=transform)
            encoder_attn_initializers.append(
                quantize_scope(ckpt_reader, scope=scope,
                    weights_name='kernel', idxs_name='kernel_idx', codebook_name='codebook',
                    num_codes=num_codes, sharded=False))

        decoder_attn_initializers = []
        for layer_name, layer_idx, attn_type, transform in itertools.product(
                ['decoder'], range(6), ['self_attention', 'encdec_attention'],
                ['q', 'k', 'v', 'output_transform']):
            scope = ('transformer/body/{layer_name}/layer_{idx}/{attn_type}/'
                + 'multihead_attention/{transform}').format(
                    layer_name=layer_name, idx=layer_idx, attn_type=attn_type, transform=transform)
            decoder_attn_initializers.append(
                quantize_scope(ckpt_reader, scope=scope,
                    weights_name='kernel', idxs_name='kernel_idx', codebook_name='codebook',
                    num_codes=num_codes, sharded=False))

        initializers = tf.group(
                embedding_initializers, ffn_initializers, encoder_attn_initializers,
                decoder_attn_initializers)

    return initializers

def make_quant_saver_from_graph(graph):
    quant_vars = set(graph.get_collection('idxs') + graph.get_collection('codebooks'))
    graph_vars = set(graph.get_collection(tf.GraphKeys.VARIABLES))
    train_vars = set(
            graph.get_collection(tf.GraphKeys.VARIABLES, scope='training') +
            graph.get_collection(tf.GraphKeys.VARIABLES, scope='losses_avg'))

    return tf.train.Saver(graph_vars - quant_vars - train_vars)

def quantize_ckpt(old_ckpt_dir, new_ckpt_dir, hparams):
    graph = tf.Graph()
    old_ckpt = tf.train.latest_checkpoint(old_ckpt_dir)

    # Initialize the quantized ops from the checkpoint
    initializers = make_quant_initializers_from_checkpoint(
            graph, old_ckpt, hparams=hparams)

    # Restore the rest of the variables directly from the checkpoint using tf.train.Saver
    saver = make_quant_saver_from_graph(graph)

    # Create a new saver that checkpoints the initialized session
    new_saver = tf.train.Saver(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    # Initialize variables in a session
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, old_ckpt)
        sess.run(initializers)

        # Save the initialized session
        new_saver.save(sess, os.path.join(new_ckpt_dir, 'quantized'), global_step=tf.train.get_global_step(graph))

    return graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('base_dir', 'test_data/transformer', 'Base directory path')
tf.app.flags.DEFINE_string('data_dir', 'test_data/transformer/data', 'Data directory path')
tf.app.flags.DEFINE_string('ckpt_dir', 'test_data/transformer/pretrained', 'Pretrained checkpoint directory path')
tf.app.flags.DEFINE_string('qout_dir', 'test_data/transformer/quantized', 'Output directory for the quantized models')

def main(_):
    base_dir = FLAGS.base_dir
    data_dir = FLAGS.data_dir
    ckpt_dir = FLAGS.ckpt_dir

    problem_name = "translate_ende_wmt32k"
    hparams_set = "transformer_base"
    hparams = trainer_lib.create_hparams(
            hparams_set,
            data_dir=data_dir,
            problem_name=problem_name)

    for num_codes in [2**4, 2**5, 2**6, 2**7, 2**8]:
        hparams
        new_ckpt_dir = os.path.join(FLAGS.qout_dir, 'quantized_{}'.format(num_codes))
        _ = quantize_ckpt(ckpt_dir, new_ckpt_dir, hparams)

if __name__ == '__main__':
    tf.app.run(main)

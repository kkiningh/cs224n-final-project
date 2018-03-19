import tensorflow as tf
import numpy as np
import os
import itertools
import convert_lib

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'test_data/transformer/data',
    'Data directory path')
tf.app.flags.DEFINE_string('ckpt_dir', 'checkpoints',
    'Pretrained checkpoint directory path')
tf.app.flags.DEFINE_string('problem_name', 'translate_ende_wmt32k', 'T2T problem name')
tf.app.flags.DEFINE_string('hparams_set', 'transformer_base', 'Hyperparameters to use for model')
tf.app.flags.DEFINE_integer('threshold_percentile', 40, 'Pruning threshold')

def main(_):
    data_dir = os.path.expanduser(FLAGS.data_dir)
    ckpt_dir = FLAGS.ckpt_dir
    percent  = float(FLAGS.threshold_percentile) / 100
    new_ckpt = os.path.join(ckpt_dir, 'pruned/pruned_{}'.format(percent))

    hparams = trainer_lib.create_hparams(
            hparams_set=FLAGS.hparams_set,
            data_dir=data_dir,
            problem_name=FLAGS.problem_name)

    convert_lib.prune_checkpoint(hparams, ckpt_dir=ckpt_dir,
            threshold_percentile=percent, new_ckpt=new_ckpt)

if __name__ == '__main__':
    tf.app.run(main)

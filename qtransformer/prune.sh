DATA_DIR=~stanley1jacob/t2t_data
PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base

TRAIN_DIR=./checkpoints/pruned

t2t-trainer \
	--data_dir=$DATA_DIR \
	--problems=$PROBLEM \
	--model=$MODEL \
	--hparams_set=$HPARAMS \
	--output_dir=$TRAIN_DIR \
	--train_steps=260000 \
	--local_eval_frequency=4000

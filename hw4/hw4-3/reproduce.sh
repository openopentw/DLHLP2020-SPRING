downloaded_model=$1
test_path=$2
output=$3

mv ${downloaded_model} ./model/pytorch_model.bin

python ./examples/run_squad.py \
	--model_type bert \
	--do_eval \
	--model_name_or_path ./model \
	--tokenizer_name bert-base-chinese \
	--train_file Data/hw4-3_train.json \
	--predict_file ${test_path} \
	--max_seq_length 384 \
	--per_gpu_eval_batch_size 100 \
	--output_dir ./model
	# --fp16

python3 ./process_ans.py \
	./model/predictions_.json \
	${output}

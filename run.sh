#!/bin/bash
#
set -xue

export CUDA_VISIBLE_DEVICES=1
parlai train \
    -m nar/nar -t multiwoz_dst -eps 10 -bs 2 -opt sgd -lr 5e-4 \
    --eval-batchsize 1 \
    --warmup_updates 100 \
    --warmup_rate 1e-5 \
    --log-every-n-secs 100 \
    --validation-every-n-epochs 10 \
    --model-file experiment/nar/model \
    --validation-metric 'joint goal acc' \
    --validation-metric-mode max \
    --fp16 false 


parlai eval \
    -dt test \
    -m nar/nar -t multiwoz_dst -bs 1 \
    -mf experiment/nar/model \
    --log-every-n-secs 100 \
    --report-filename experiment/nar/model.report \
    --world-logs True

# # export CUDA_VISIBLE_DEVICES=1
# parlai train \
#     -m bart/bart -t multiwoz_dst -eps 11.2 -bs 2 -opt sgd -lr 5e-4 \
#     --eval-batchsize 2 \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file experiment/bart/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --fp16 false 

# parlai eval \
#     -dt test \
#     -m bart/bart -t multiwoz_dst -bs 1 \
#     -mf experiment/bart/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/bart/model.report \
#     --world-logs True

# parlai train \
#     -m hugging_face/gpt2 -t multiwoz_dst -eps 1.2 -bs 2 -opt sgd -lr 5e-4 \
#     --eval-batchsize 8 \
#     --warmup_updates 10 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file experiment/gen_gpt2/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --fp16 false \
#     --just_test True

# cd experiment/gen_gpt2/
# cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
# cd ../../
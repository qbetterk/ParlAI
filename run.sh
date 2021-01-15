#!/bin/bash
#
set -xue

parlai multiprocessing_train \
    -m hugging_face/gpt2 -t multiwoz_dst -eps 20.2 -bs 2 -opt adam -lr 5e-4 \
    --eval-batchsize 8 \
    --warmup_updates 100 \
    --warmup_rate 1e-5 \
    --log-every-n-secs 100 \
    --validation-every-n-epochs 1 \
    --model-file experiment/gen_gpt2/model \
    --validation-metric 'joint goal acc' \
    --validation-metric-mode max \
    --add-special-tokens True

parlai multiprocessing_eval \
    -dt test \
    -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
    -mf experiment/gen_gpt2/model \
    --log-every-n-secs 100 \
    --report-filename experiment/gen_gpt2/model.report \
    --save-world-logs True


cd experiment/gen_gpt2/
cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
cd ../../
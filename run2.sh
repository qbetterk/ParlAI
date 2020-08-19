#!/bin/bash
#
set -xue


parlai multiprocessing_train \
    -m hugging_face/gpt2 -t multiwozdst -eps 30 -bs 8 -opt adam -lr 5e-4 \
    --eval-batchsize 1 \
    --fp16 true \
    --warmup_updates 100 \
    --warmup_rate 1e-5 \
    --log-every-n-secs 100 \
    --validation-every-n-epochs 2 \
    --model-file experiment/gpt2_dst/model \
    --validation-metric 'joint goal acc' \
    --validation-metric-mode max \
    --add-special-tokens True


parlai multiprocessing_eval \
    -dt test \
    -m hugging_face/gpt2 -t multiwozdst -bs 1 \
    --fp16 true \
    -mf experiment/gpt2_dst_test/model \
    --log-every-n-secs 10 \
    --report-filename experiment/gpt2_dst/model.report \
    --save-world-logs True
#!/bin/bash
#
set -xue

parlai train \
    -m nar/nar -t multiwoz_dst -eps 2.2 -bs 1 -opt sgd -lr 5e-4 \
    --eval-batchsize 8 \
    --warmup_updates 100 \
    --warmup_rate 1e-5 \
    --log-every-n-secs 100 \
    --validation-every-n-epochs 1 \
    --model-file experiment/nar/model \
    --validation-metric 'joint goal acc' \
    --validation-metric-mode max \
    --fp16 false

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#     -mf experiment/gen_gpt2/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2/model.report \
#     --world-logs True


# cd experiment/gen_gpt2/
# cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
# cd ../../
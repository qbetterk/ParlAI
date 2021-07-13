#!/bin/bash
#
set -xue


# srun --gres=gpu:2 --partition=a100 --time=72:00:00 --nodes 1 --pty /bin/bash -l

# # # # # # # just for test code

# parlai display_data -t disambiguation --level 1 --force_gen True

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t disambiguation -eps 20.1 -bs 32 -opt adam -lr 5e-4 \
#     --eval-batchsize 8 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /fsx/kunqian/parlai/disambiguation/level1/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max


# mkdir -p experiment/level1/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t disambiguation -bs 8 \
#     --fp16 true \
#     -mf /fsx/kunqian/parlai/disambiguation/level1/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/level1/model.report \
#     --world-logs experiment/level1/world_log.jsonl

# cd experiment/level1/
# cp world_log_0.jsonl result_test.jsonl
# cd ../../

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t disambiguation -eps 20.1 -bs 64 -opt adam -lr 5e-4 \
#     --eval-batchsize 8 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /fsx/kunqian/parlai/disambiguation/level2/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --level 2


# mkdir -p experiment/level2/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t disambiguation -bs 8 \
#     --fp16 true \
#     -mf /fsx/kunqian/parlai/disambiguation/level2/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/level2/model.report \
#     --world-logs experiment/level2/world_log.jsonl \
#     --level 12

# cd experiment/level2/
# cp world_log_0.jsonl result_test.jsonl
# cd ../../


# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t disambiguation -eps 20.1 -bs 64 -opt adam -lr 5e-4 \
#     --eval-batchsize 8 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /fsx/kunqian/parlai/disambiguation/level12/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --level 12



# mkdir -p experiment/level12/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t disambiguation -bs 8 \
#     --fp16 true \
#     -mf /fsx/kunqian/parlai/disambiguation/level12/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/level12/model.report \
#     --world-logs experiment/level12/world_log.jsonl \
#     --level 12

# cd experiment/level12/
# cp world_log_0.jsonl result_test.jsonl
# cd ../../

# jun 29 level 3 /123
# level=123
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t disambiguation -eps 20.1 -bs 64 -opt adam -lr 1e-4 \
#     --eval-batchsize 32 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /fsx/kunqian/parlai/disambiguation/level${level}/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --level ${level}



# mkdir -p experiment/level${level}/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t disambiguation -bs 8 \
#     --fp16 true \
#     -mf /fsx/kunqian/parlai/disambiguation/level${level}/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/level${level}/model.report \
#     --world-logs experiment/level${level}/world_log.jsonl \
#     --level ${level}

# cd experiment/level${level}/
# cp world_log_0.jsonl result_test.jsonl
# cd ../../

# jun 30 multiwoz_dst / multi-123

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_dst -eps 20.1 -bs 16 -opt adam -lr 1e-4 \
#     --eval-batchsize 4 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /fsx/kunqian/parlai/multiwoz_dst/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max 

level=123
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t disambiguation -eps 20.1 -bs 64 -opt adam -lr 5e-4 \
#     --eval-batchsize 32 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /fsx/kunqian/parlai/disambiguation/level${level}_multi/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --level ${level} --multi_opt True

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t disambiguation -eps 20.1 -bs 64 -opt adam -lr 5e-4 \
#     --eval-batchsize 32 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /fsx/kunqian/parlai/disambiguation/level${level}_err/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --level ${level} --multi_opt False --import_err True

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t disambiguation -eps 20.1 -bs 64 -opt adam -lr 5e-4 \
#     --eval-batchsize 32 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /fsx/kunqian/parlai/disambiguation/level${level}_multi_err/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --level ${level} --multi_opt True --import_err True

# # mkdir -p experiment/level${level}/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t disambiguation -bs 8 \
#     --fp16 true \
#     -mf /fsx/kunqian/parlai/disambiguation/level${level}_multi/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/level${level}_multi/model.report \
#     --world-logs experiment/level${level}_multi/world_log.jsonl \
#     --level ${level}

# # cd experiment/level${level}/
# # cp world_log_0.jsonl result_test.jsonl
# # cd ../../

level=3
# mkdir -p experiment/dst_test/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t disambiguation -bs 8 \
#     --fp16 true \
#     -mf /fsx/kunqian/parlai/multiwoz_dst/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/dst_test/model_level${level}_multi.report \
#     --world-logs experiment/dst_test/world_log_level${level}_multi.jsonl \
#     --level ${level} --multi_opt True 
    
parlai multiprocessing_eval \
    -dt test \
    -m hugging_face/gpt2 -t disambiguation -bs 8 \
    --fp16 true \
    -mf /fsx/kunqian/parlai/multiwoz_dst/model \
    --log-every-n-secs 100 \
    --report-filename experiment/dst_test/model_level${level}.report \
    --world-logs experiment/dst_test/world_log_level${level}.jsonl \
    --level ${level} --multi_opt False --import_err False


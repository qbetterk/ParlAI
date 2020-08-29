#!/bin/bash
#
set -xue


# # # # # # # just for test code

parlai display_data -t multiwozdst_cor \
                --add_err True \
                # --data_name 'dials_nodict_bs8.json' \
                # --err_data_path 'experiment/gen_gpt2_nodict/result_decode_all_bs8.jsonl'

# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 10 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen_errbs8_foex/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs1_foextra.json' \
#     --add-special-tokens True \
#     --just_test True

# for i in `seq 1 10`; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen_errbs8_foex/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen_errbs8_foex/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_gen_errbs8_foex/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# parlai train -df dictfile -m hugging_face/gpt2 -t multiwozdst_cor \
#             -eps 0.5 -bs 1 -opt adam -lr 1e-3 \
#             --fp16 true --display-examples True --just_test True

# parlai train -df dictfile -m hugging_face/gpt2 -t multiwozdst \
#             -eps 0.5 -bs 1 -opt adam -lr 1e-3 \
#             --fp16 true --just_test True --display-examples True

# parlai train -df dictfile -m hugging_face/gpt2 -t google_sgd_dst \
#             -eps 1.0 -bs 1 -opt adam -lr 1e-3 \
#             --fp16 true --no-cuda \
#             -dt test
# parlai eval -df dictfile -m hugging_face/gpt2 -t google_sgd_dst \
#             --fp16 true --no-cuda \

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst -eps 3 -bs 8 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file experiment/test/model \
#     --just_test True

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 5 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 8 \
#     --fp16 true \
#     --warmup_updates 10 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_test/model \
#     --split_decoded_data False \
#     --data-name 'dials_owndict_bs8.json'\
#     --just_test True
    # --validation-metric 'joint goal acc' \
    # --validation-metric-mode max \
    
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_test/model \
#     --log-every-n-secs 10 \
#     --report-filename experiment/gpt2_dst_test/model.report \
#     --save-world-logs True \
#     --just-test True
#     # --display-examples True \

# parlai display_data -t multiwozdst


# # # # # script for experiment

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst -eps 30 -bs 8 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 2 \
#     --model-file experiment/gpt2_dst_specialtoken/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True


# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_test/model \
#     --log-every-n-secs 10 \
#     --report-filename experiment/gpt2_dst_test/model.report \
#     --save-world-logs True \
#     --just-test False
#     # --display-examples True \

# parlai eval \
#     -dt test \
#     -df dict/dictfile_multiwoz -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_nodict/model \
#     --log-every-n-secs 10 \
#     --report-filename experiment/gpt2_dst_nodict/model.report \
#     --save-world-logs True


# parlai multiprocessing_eval \
#     -dt test \
#     -df dict/dictfile_multiwozdst -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_owndict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gpt2_dst_owndict/model.report \
#     --save-world-logs True

# cd experiment/gpt2_dst_owndict/
# mv model.report model.report_test_bs1_4gpu
# mv model_multiwozdst_0_replies.jsonl result_test_bs1_4gpu.jsonl
# rm model_multiwozdst_*
# cd ../../

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 32 \
#     --fp16 true \
#     -mf experiment/gpt2_dst/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gpt2_dst_nodict/model.report \
#     --save-world-logs True \



# # # # # # explore decoding with diff batch size & diff # of gpus & dict


# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 4 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gpt2_dst_nodict/model.report \
#     --save-world-logs True

# cd experiment/gpt2_dst_nodict/
# mv model.report model.report_bs4_4gpu
# mv model_multiwozdst_0_replies.jsonl result_test_bs4_4gpu.json
# rm model_multiwozdst_*
# cd ../../

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 8 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gpt2_dst_nodict/model.report \
#     --save-world-logs True

# cd experiment/gpt2_dst_nodict/
# mv model.report model.report_bs8_4gpu
# mv model_multiwozdst_0_replies.jsonl result_test_bs8_4gpu.json
# rm model_multiwozdst_*
# cd ../../

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 16 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gpt2_dst_nodict/model.report \
#     --save-world-logs True

# cd experiment/gpt2_dst_nodict/
# mv model.report model.report_bs16_4gpu
# mv model_multiwozdst_0_replies.jsonl result_test_bs16_4gpu.json
# rm model_multiwozdst_*
# cd ../../

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gpt2_dst_nodict/model.report \
#     --save-world-logs True

# cd experiment/gpt2_dst_nodict/
# mv model.report model.report_bs1_4gpu
# mv model_multiwozdst_0_replies.jsonl result_test_bs1_4gpu.json
# rm model_multiwozdst_*
# cd ../../


# # # # # # batch 1 for different # of gpu
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gpt2_dst_nodict/model.report \
#     --save-world-logs True

# cd experiment/gpt2_dst_nodict/
# mv model.report model.report_bs1_2gpu
# mv model_multiwozdst_0_replies.jsonl result_test_bs1_2gpu.json
# rm model_multiwozdst_*
# cd ../../



# # # # # # # # correction model

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gpt2_dst_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gpt2_dst_nodict/model.report \
#     --save-world-logs True \
#     --decode-all True

# cp experiment/gpt2_dst_owndict/model_multiwozdst_0_replies.jsonl experiment/gpt2_dst_owndict/result_decode_all.json

# # parlai display_data -t multiwozdst_cor

# # parlai build_dict -t multiwozdst_cor --dict-file dict/dictfile_multiwozdst_cor



# # # # correction model baseline fresh
# parlai multiprocessing_train \
#     -df dict/dictfile_multiwozdst_cor \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 30 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True
#     --model-file experiment/cor_gpt2/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf experiment/cor_gpt2/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2/model.report \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs8.json'

# # # # # correction model based on gen model
# parlai multiprocessing_train \
#     -df dict/dictfile_multiwozdst_cor \
#     --init-model experiment/gpt2_dst_owndict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 30 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file experiment/cor_gpt2_gen/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf experiment/cor_gpt2_gen/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_gen/model.report \
#     --save-world-logs True

# # # # repeat turns with one or two errs (default: 80% for 5 times)
# parlai multiprocessing_train \
#     -df dict/dictfile_multiwozdst_cor \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 30 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 32 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 200 \
#     --validation-every-n-epochs 2 \
#     --model-file experiment/cor_gpt2_re/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --repeat-minor-err True

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf experiment/cor_gpt2_re/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_re/model.report \
#     --save-world-logs True




# # # # # use bart for correction model
# parlai multiprocessing_train \
#     -m bart -t multiwozdst_cor -eps 30 -bs 2 -opt adam -lr 5e-4 \
#     --init-model zoo:bart/bart_large/model \
#     --eval-batchsize 8 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 2 \
#     --model-file experiment/cor_bart/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max

# # srun --gpus-per-node=4 --partition=dev --time=72:00:00 --cpus-per-task 10 --pty /bin/bash -l



# # # Using bs=1 when decodeing all data

# # # # for gen_gpt2_owndict 

# parlai multiprocessing_train \
#     -df dict/dictfile_multiwozdst_cor \
#     --init-model experiment/gpt2_dst_owndict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_owd_gen_errbs1/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --split_decoded_data False \
#     --data-name 'dials_owndict_bs1.json'\
#     --save-every-n-secs 500

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf experiment/cor_gpt2_owd_gen_errbs1/model.checkpoint \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_owd_gen_errbs1/model.report \
#     --save-world-logs True \
#     --data-name 'dials_owndict_bs1.json'

# parlai multiprocessing_train \
#     -df dict/dictfile_multiwozdst_cor \
#     --init-model experiment/gpt2_dst_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen_errbs1/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --err_data_path 'experiment/gpt2_dst_nodict/result_decode_all_bs1.json' \
#     --split_decoded_data False \
#     --data-name 'dials_nodict_bs1.json' \
#     --save-every-n-secs 1000


# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf experiment/cor_gpt2_nod_gen_errbs1/model.checkpoint \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_gen_errbs1/model.report \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs1.json'





# # # # Aug 21 rerun and save at each epoch , check overfiting

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst -eps 20 -bs 8 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file experiment/gen_gpt2_nodict/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True

# # # epoch 8 is the best
# # # # evaluate for each epoch
# for i in `seq 1 10`; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#         --fp16 true \
#         -mf experiment/gen_gpt2_nodict/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/gen_gpt2_nodict/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/gen_gpt2_nodict/
#     cp model_multiwozdst_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done


# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_decode_all \
#     --save-world-logs True \
#     --decode-all True

# cd experiment/gen_gpt2_nodict/
# cp model_multiwozdst_0_replies.jsonl result_decode_all.jsonl
# cd ../../

# i=10
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gen_gpt2_nodict/model.checkpoint_ep${i} \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_decode_all_ep${i} \
#     --save-world-logs True \
#     --decode-all True

# cd experiment/gen_gpt2_nodict/
# cp model_multiwozdst_0_replies.jsonl result_decode_all_ep${i}.jsonl
# cd ../../

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 16 \
#     --fp16 true \
#     -mf experiment/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_decode_all_bs8 \
#     --save-world-logs True \
#     --decode-all True

# cd experiment/gen_gpt2_nodict/
# cp model_multiwozdst_0_replies.jsonl result_decode_all_bs8.jsonl
# cd ../../

# # # pre-built dict
# parlai multiprocessing_train \
#     -df dict/dictfile_multiwozdst \
#     -m hugging_face/gpt2 -t multiwozdst -eps 20 -bs 8 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file experiment/gen_gpt2_owndict/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gen_gpt2_owndict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_owndict/model.report_test_ep11 \
#     --save-world-logs True

# cd experiment/gen_gpt2_owndict/
# cp model_multiwozdst_0_replies.jsonl result_test_ep11.jsonl
# cd ../../

# # # no special token
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst -eps 20 -bs 8 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file experiment/gen_gpt2_nodict_noseptok/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max


# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 1 \
#     --fp16 true \
#     -mf experiment/gen_gpt2_nodict_noseptok/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict_noseptok/model.report_test_ep11 \
#     --save-world-logs True

# cd experiment/gen_gpt2_nodict_noseptok/
# cp model_multiwozdst_0_replies.jsonl result_test_ep11.jsonl
# cd ../../

# parlai display_data -t multiwozdst_cor \
#                 --split_decoded_data True \
#                 --data_name 'dials_nodict_bs8.json' \
#                 --err_data_path 'experiment/gen_gpt2_nodict/result_decode_all_bs8.jsonl'

# # # Aug 24
# # # # correction model
# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen_errbs1/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs1.json' \
#     --add-special-tokens True

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf experiment/cor_gpt2_nod_gen_errbs1/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_gen_errbs1/model.report \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs1.json'

# cd experiment/cor_gpt2_nod_gen_errbs1/
# cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
# rm model_multiwozdst_cor_*
# cd ../../


# # # train cor from scratch
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_fresh_errbs1/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs1.json' \
#     --add-special-tokens True

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf experiment/cor_gpt2_nod_fresh_errbs1/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_errbs1/model.report \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs1.json'

# cd experiment/cor_gpt2_nod_fresh_errbs1/
# cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
# rm model_multiwozdst_*
# cd ../../




# # # # # train with data filter out extra err

# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen_errbs1_foex/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs1_foextra.json' \
#     --add-special-tokens True

# for i in 8 9; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen_errbs1_foex/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen_errbs1_foex/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_gen_errbs1_foex/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # train with data filter out miss err

# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen_errbs1_fomi/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs1_fomiss.json' \
#     --add-special-tokens True 

# for i in 9; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen_errbs1_fomi/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen_errbs1_fomi/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_gen_errbs1_fomi/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done


# # # Aug 25
# # # # train from checkpoint4
# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model.checkpoint_ep4 \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 5.4 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen4_errbs1/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs1.json' \
#     --add-special-tokens True

# for i in `seq 6 10`; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen4_errbs1/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen4_errbs1/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_gen4_errbs1/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # train with data bs8
# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen_errbs8/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True

# for i in 9; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen_errbs8/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen_errbs8/model.report_ep${i} \
#         --save-world-logs True 
    
#     cd experiment/cor_gpt2_nod_gen_errbs8/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done


# # # # # train with data filter out extra err

# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen_errbs8_foex/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8_foextra.json' \
#     --add-special-tokens True 

# for i in `seq 2 2 10`; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen_errbs8_foex/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen_errbs8_foex/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_gen_errbs8_foex/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # train with data filter out miss err

# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file experiment/cor_gpt2_nod_gen_errbs8_fomi/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8_fomiss.json' \
#     --add-special-tokens True 

# for i in `seq 2 2 9`; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen_errbs8_fomi/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen_errbs8_fomi/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_gen_errbs8_fomi/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # repeat turns with one or two errs (default: 80% for 5 times)
# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file experiment/cor_gpt2_nod_gen_errbs8_re/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --repeat-minor-err True

# for i in `seq 6 2 10`; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen_errbs8_re/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen_errbs8_re/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_gen_errbs8_re/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done
# # # # repeat turns with one or two errs (80% for 10 times)
# parlai multiprocessing_train \
#     --init-model experiment/gen_gpt2_nodict/model \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 10.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file experiment/cor_gpt2_nod_gen_errbs8_re_8010/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --repeat-minor-err True

# for i in 9; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf experiment/cor_gpt2_nod_gen_errbs8_re_8010/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_gen_errbs8_re_8010/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_gen_errbs8_re_8010/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # checkout to commit on aug 3 to check overfitting
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst -eps 20 -bs 8 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file experiment/gen_gpt2_nodict_aug3/model \
#     --validation-metric 'ppl' \
#     --validation-metric-mode min \
#     --add-special-tokens True \
#     --skip-generation True






















# # #####################3 google sgd ##############################

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t google_sgd_dst -eps 10 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --save-after-valid True \
#     --model-file sgd_experiment/gen_gpt2/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True


# for i in 2; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t google_sgd_dst -bs 1 \
#         --fp16 true \
#         -mf sgd_experiment/gen_gpt2/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename sgd_experiment/gen_gpt2/model.report_test_ep${i} \
#         --save-world-logs True

#         cd sgd_experiment/gen_gpt2/
#         cp model_google_sgd_dst_0_replies.jsonl result_test_ep${i}.jsonl
#         rm model_google_sgd_dst_*
#         cd ../../
# done
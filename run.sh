#!/bin/bash
#
set -xue


# # # # # # # just for test code

# parlai display_data -t multiwozdst_cor_err \
#                 --add_err True
# parlai display_data -t multiwozdst_cor_err
# parlai display_data -t multiwozdst_cor_gt \
#                 --split_decoded_data True \
#                 --data_name 'dials_nodict_bs8.json' \
#                 --err_data_path 'experiment/gen_gpt2_nodict/result_decode_all_bs8.jsonl'
# parlai display_data -t multiwozdst_cor \
#                 --split_decoded_data True \
#                 --data_name 'dials_nodict_bs1.json' \
#                 --err_data_path 'experiment/gen_gpt2_nodict/result_decode_all.jsonl'
# parlai display_data -t multiwozdst_cor_err \
#                 --data_name 'dials_nodict_bs8_kpextra.json'
# parlai display_data -t multiwoz_dst \
#                 --create_aug_data True \
#                 --data_aug True 
# parlai display_data -t multiwoz_dst

# parlai train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 3.3 -bs 1 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 10 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/test/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True \
#     --just_test True
# rm /checkpoint/kunqian/parlai/test/*
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_dom -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 10 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 10 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/test/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
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
#     -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#     --fp16 true \
#     -mf experiment/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report \
#     --save-world-logs True
# cd experiment/gen_gpt2_nodict/
# cp model_multiwozdst_0_replies.jsonl result_test.jsonl
# cd ../../

# parlai multiprocessing_eval \
#     -dt valid \
#     -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_valid \
#     --save-world-logs True 
# cd experiment/gen_gpt2_nodict/
# cp model_multiwoz_dst_0_replies.jsonl result_valid.jsonl
# cd ../../
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_bs1_bm5 \
#     --save-world-logs True \
#     --inference beam \
#     --beam-size 5
# cd experiment/gen_gpt2_nodict/
# cp model_multiwoz_dst_0_replies.jsonl result_test_bs1_bm5.jsonl
# cd ../../

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst -bs 8 \
#     --fp16 true \
#     -mf experiment/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_decode_all_bs8 \
#     --save-world-logs True \
#     --decode-all True
# cd experiment/gen_gpt2_nodict/
# cp model_multiwozdst_0_replies.jsonl result_decode_all_bs8.jsonl
# cd ../../

# # # different decode method
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_dst -bs 8 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_decode_all_bs8_top5 \
#     --save-world-logs True \
#     --decode-all True \
#     --inference topk \
#     --topk 5
# cd experiment/gen_gpt2_nodict/
# cp model_multiwoz_dst_0_replies.jsonl result_decode_all_bs8_top5.jsonl
# cd ../../
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_dst -bs 8 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_decode_all_bs8_beam5 \
#     --save-world-logs True \
#     --decode-all True \
#     --inference beam \
#     --beam-size 5
# cd experiment/gen_gpt2_nodict/
# cp model_multiwoz_dst_0_replies.jsonl result_decode_all_bs8_beam5.jsonl
# cd ../../
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_dst -bs 8 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nodict/model.report_decode_all_bs8_topp08 \
#     --save-world-logs True \
#     --decode-all True \
#     --inference nucleus \
#     --topp 0.8
# cd experiment/gen_gpt2_nodict/
# cp model_multiwoz_dst_0_replies.jsonl result_decode_all_bs8_topp08.jsonl
# cd ../../

# for sd in 0 1 2; do
#     parlai multiprocessing_train \
#         -m hugging_face/gpt2 -t multiwoz_dst -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#         --eval-batchsize 1 \
#         --fp16 true \
#         --warmup_updates 100 \
#         --warmup_rate 1e-5 \
#         --log-every-n-secs 100 \
#         --validation-every-n-epochs 1 \
#         --save-after-valid True \
#         --model-file /checkpoint/kunqian/parlai/gen_gpt2_nodict_sd${sd}/model \
#         --validation-metric 'joint goal acc' \
#         --validation-metric-mode max \
#         --add-special-tokens True \
#         --validation-patience 5 \
#         --rand-seed ${sd}

#     # mkdir -p experiment/gen_gpt2_nodict_sd${sd}/
#     # parlai multiprocessing_eval \
#     #     -dt test \
#     #     -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#     #     --fp16 true \
#     #     -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict_sd${sd}/model \
#     #     --log-every-n-secs 100 \
#     #     --report-filename experiment/gen_gpt2_nodict_sd${sd}/model.report \
#     #     --save-world-logs True

#     # cd experiment/gen_gpt2_nodict_sd${sd}/
#     # cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
#     # cd ../../
# done

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

# for sd in 0 1 2; do
#     # parlai multiprocessing_train \
#     #     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     #     --eval-batchsize 1 \
#     #     --fp16 true \
#     #     --warmup_updates 100 \
#     #     --warmup_rate 1e-5 \
#     #     --log-every-n-secs 100 \
#     #     --validation-every-n-epochs 1 \
#     #     --save-after-valid True \
#     #     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_bs1_sd${sd}/model \
#     #     --validation-metric 'joint goal acc' \
#     #     --validation-metric-mode max \
#     #     --data-name 'dials_nodict_bs1.json' \
#     #     --add-special-tokens True \
#     #     --validation-patience 5 \
#     #     --rand-seed ${sd}
#     mkdir -p experiment/cor_gpt2_nod_fresh_bs1_sd${sd}
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_bs1_sd${sd}/model \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_bs1_sd${sd}/model.report \
#         --save-world-logs True 
    
#     cd experiment/cor_gpt2_nod_fresh_bs1_sd${sd}/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done
# # # # # # train with data bs8
# for sd in 0 1 2; do
#     parlai multiprocessing_train \
#         -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#         --eval-batchsize 1 \
#         --fp16 true \
#         --warmup_updates 100 \
#         --warmup_rate 1e-5 \
#         --log-every-n-secs 100 \
#         --validation-every-n-epochs 1 \
#         --save-after-valid True \
#         --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_bs8_sd${sd}/model \
#         --validation-metric 'joint goal acc' \
#         --validation-metric-mode max \
#         --data-name 'dials_nodict_bs8.json' \
#         --add-special-tokens True \
#         --validation-patience 5
#         # mkdir -p experiment/cor_gpt2_nod_fresh_bs8_sd${sd}
#         # parlai multiprocessing_eval \
#         #     -dt test \
#         #     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         #     --fp16 true \
#         #     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_bs8_sd${sd}/model \
#         #     --log-every-n-secs 100 \
#         #     --report-filename experiment/cor_gpt2_nod_fresh_bs8_sd${sd}/model.report \
#         #     --save-world-logs True 
        
#         # cd experiment/cor_gpt2_nod_fresh_bs8_sd${sd}/
#         # cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
#         # rm model_multiwozdst_*
#         # cd ../../
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

# # # # train with random err (random distributed)
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# for i in 9; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_fresh_randerr/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # iterative correction
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_randerr/model.report_iter1 \
#     --save-world-logs True \
#     --skip-generation False \
#     --generated-test-result-path './experiment/cor_gpt2_nod_fresh_randerr/result_test_ep11.jsonl'

# cd experiment/cor_gpt2_nod_fresh_randerr/
# cp model_multiwozdst_cor_0_replies.jsonl result_test_ep11_iter1.jsonl
# rm model_multiwozdst_*
# cd ../../


# # # # val with ppl at each epoch
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_skipgen/model \
#     --validation-metric 'ppl' \
#     --validation-metric-mode min \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True \
#     --skip-generation True

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_skipgen/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_randerr_skipgen/model.report \
#     --save-world-logs True \
#     --skip-generation False

# cd experiment/cor_gpt2_nod_fresh_randerr_skipgen/
# cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
# rm model_multiwozdst_*
# cd ../../


# # # # val with acc at each epoch
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_valacc/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True

# # # mkdir -p experiment/cor_gpt2_nod_fresh_randerr_valacc/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_valacc/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_randerr_valacc/model.report_greedy \
#     --save-world-logs True 

# cd experiment/cor_gpt2_nod_fresh_randerr_valacc/
# cp model_multiwozdst_cor_0_replies.jsonl result_test_greedy.jsonl
# rm model_multiwozdst_*
# cd ../../

# # # # import err with distribution
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 30.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_disterr/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True

# for i in `seq 21 30`; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_disterr/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_disterr/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_fresh_disterr/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # import err with distribution with val each epoch
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_disterr_valacc/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True

# mkdir -p experiment/cor_gpt2_nod_fresh_disterr_valacc/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_disterr_valacc/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_disterr_valacc/model.report \
#     --save-world-logs True \
#     --skip-generation False

# cd experiment/cor_gpt2_nod_fresh_disterr_valacc/
# cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
# rm model_multiwozdst_*
# cd ../../


# # # # # train with random err (random distributed) err num range 3 for each
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_num3/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# # mkdir -p experiment/cor_gpt2_nod_fresh_randerr_num3/
# for i in `seq 11 20`; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_num3/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr_num3/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_fresh_randerr_num3/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # # train with random err (random distributed) err num range 3 for each and 1:10 for wi/wo err
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_num3_wi10/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# # mkdir -p experiment/cor_gpt2_nod_fresh_randerr_num3_wi10/
# for i in 16; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_num3_wi10/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr_num3_wi10/model.report_ep${i}_bs1 \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_fresh_randerr_num3_wi10/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}_bs1.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # # train with random err (random distributed) 3 (wo:wi=1:3) 0 (match err) 3 (extra) 0 (miss)
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr3030/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# mkdir -p experiment/cor_gpt2_nod_fresh_randerr3030/
# for i in 15; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr3030/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr3030/model.report_ep${i}_bs1 \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_fresh_randerr3030/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}_bs1.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # # train with random err (random distributed) 3 (wo:wi=1:3) 0 (match err) 0 (extra) 3 (miss)
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr3003/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# # mkdir -p experiment/cor_gpt2_nod_fresh_randerr3003/
# for i in 10; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr3003/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr3003/model.report_ep${i}_bs1 \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_fresh_randerr3003/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}_bs1.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # # train with random err (random distributed) 03 (wo:wi=0:3) 0 (match err) 3 (extra) 0 (miss)
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_03030/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# mkdir -p experiment/cor_gpt2_nod_fresh_randerr_03030/
# for i in 9 ; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_03030/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr_03030/model.report_ep${i} \
#         --save-world-logs True
    
#     cd experiment/cor_gpt2_nod_fresh_randerr_03030/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # iterative correction
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_disterr_valacc/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_disterr_valacc/model.report_iter1 \
#     --save-world-logs True \
#     --skip-generation False \
#     --generated-test-result-path './experiment/cor_gpt2_nod_fresh_disterr_valacc/result_test.jsonl'

# cd experiment/cor_gpt2_nod_fresh_disterr_valacc/
# cp model_multiwozdst_cor_0_replies.jsonl result_test_iter1.jsonl
# rm model_multiwozdst_*
# cd ../../

# # # # iterative correction
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_valacc/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_randerr_valacc/model.report_iter1 \
#     --save-world-logs True \
#     --skip-generation False \
#     --generated-test-result-path './experiment/cor_gpt2_nod_fresh_randerr_valacc/result_test.jsonl'

# cd experiment/cor_gpt2_nod_fresh_randerr_valacc/
# cp model_multiwozdst_cor_0_replies.jsonl result_test_iter1.jsonl
# rm model_multiwozdst_*
# cd ../../

# # # # # train with random err (random distributed) for different seed
# for sd in 2; do
#     # parlai multiprocessing_train \
#     #     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     #     --eval-batchsize 1 \
#     #     --fp16 true \
#     #     --warmup_updates 100 \
#     #     --warmup_rate 1e-5 \
#     #     --log-every-n-secs 100 \
#     #     --validation-every-n-epochs -1 \
#     #     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_sd${sd}/model \
#     #     --validation-metric 'joint goal acc' \
#     #     --validation-metric-mode max \
#     #     --data-name 'dials_nodict_bs8.json' \
#     #     --add-special-tokens True \
#     #     --add_err True 

#     # # mkdir -p experiment/cor_gpt2_nod_fresh_randerr_sd${sd}/
#     for i in `seq 7 20`; do
#         parlai multiprocessing_eval \
#             -dt test \
#             -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#             --fp16 true \
#             -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_sd${sd}/model.checkpoint_ep${i} \
#             --log-every-n-secs 100 \
#             --report-filename experiment/cor_gpt2_nod_fresh_randerr_sd${sd}/model.report_ep${i}_bs1 \
#             --save-world-logs True

#         cd experiment/cor_gpt2_nod_fresh_randerr_sd${sd}/
#         cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}_bs1.jsonl
#         rm model_multiwozdst_*
#         cd ../../
#     done
# done
# # # # # train with random err valacc (random distributed) for different seed
# for sd in 0; do
#     # parlai multiprocessing_train \
#     #     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     #     --eval-batchsize 1 \
#     #     --fp16 true \
#     #     --warmup_updates 100 \
#     #     --warmup_rate 1e-5 \
#     #     --log-every-n-secs 100 \
#     #     --validation-every-n-epochs 1 \
#     #     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_valacc_sd${sd}_dtv/model \
#     #     --validation-metric 'joint goal acc' \
#     #     --validation-metric-mode max \
#     #     --data-name 'dials_nodict_bs8.json' \
#     #     --add-special-tokens True \
#     #     --add_err True \
#     #     --rand_seed ${sd}

#     mkdir -p experiment/cor_gpt2_nod_fresh_randerr_valacc_sd${sd}_sam/
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_valacc_sd${sd}_sam/model \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr_valacc_sd${sd}_sam/model.report_bs1 \
#         --save-world-logs True

#     cd experiment/cor_gpt2_nod_fresh_randerr_valacc_sd${sd}_sam/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_bs1.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done


# # # # # train with random err (random distributed) 9999 3 13 3 (at least one extra err)
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 30.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_inf3133/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# # mkdir -p experiment/cor_gpt2_nod_fresh_randerr_inf3133/
# for i in 11; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_inf3133/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr_inf3133/model.report_ep${i}_bs1 \
#         --save-world-logs True

#     cd experiment/cor_gpt2_nod_fresh_randerr_inf3133/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}_bs1.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # # # train with random err (random distributed) 03 0 13 13 (0:3, no match, extra/miss 1-3)
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_0301313/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# # # # mkdir -p experiment/cor_gpt2_nod_fresh_randerr_0301313/
# for i in 5 6 7 8 9 10 11 12; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_0301313/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr_0301313/model.report_ep${i}_bs1 \
#         --save-world-logs True

#     cd experiment/cor_gpt2_nod_fresh_randerr_0301313/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test_ep${i}_bs1.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # # # #  new err for each epoch  randerr3222
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_new_comma/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# mkdir -p experiment/cor_gpt2_nod_fresh_randerr_new_comma/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_new_comma/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_randerr_new_comma/model.report \
#     --save-world-logs True

# cd experiment/cor_gpt2_nod_fresh_randerr_new_comma/
# cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
# rm model_multiwozdst_*
# cd ../../

# # # # # # # # #  new err for each epoch  randerr3222 different seed
# for sd in 3; do
#     # parlai multiprocessing_train \
#     #     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     #     --eval-batchsize 1 \
#     #     --fp16 true \
#     #     --warmup_updates 100 \
#     #     --warmup_rate 1e-5 \
#     #     --log-every-n-secs 100 \
#     #     --validation-every-n-epochs 1 \
#     #     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_new_sd${sd}/model \
#     #     --validation-metric 'joint goal acc' \
#     #     --validation-metric-mode max \
#     #     --data-name 'dials_nodict_bs8.json' \
#     #     --add-special-tokens True \
#     #     --add_err True \
#     #     --rand_seed ${sd}
#     mkdir -p experiment/cor_gpt2_nod_fresh_randerr_new_sd${sd}/
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_new_sd${sd}/model \
#         --log-every-n-secs 100 \
#         --report-filename experiment/cor_gpt2_nod_fresh_randerr_new_sd${sd}/model.report \
#         --save-world-logs True
#     cd experiment/cor_gpt2_nod_fresh_randerr_new_sd${sd}/
#     cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
#     rm model_multiwozdst_*
#     cd ../../
# done

# # # # #  new err for each epoch  randerr03344
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_new03344/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True 

# # mkdir -p experiment/cor_gpt2_nod_fresh_randerr_new03344/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_randerr_new03344/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_randerr_new03344/model.report \
#     --save-world-logs True

# cd experiment/cor_gpt2_nod_fresh_randerr_new03344/
# cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
# rm model_multiwozdst_*
# cd ../../






# # # # # # # # # # predict only errs
# # predict only miss errs
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_kpmiss/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8_kpmiss.json' \
#     --add-special-tokens True 

# mkdir -p experiment/cor_gpt2_nod_fresh_kpmiss/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_kpmiss/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_kpmiss/model.report_bs8 \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs8_kpmiss.json' \
#     --add-special-tokens True 
# cd experiment/cor_gpt2_nod_fresh_kpmiss/
# cp model_multiwozdst_cor_err_0_replies.jsonl result_test_bs8.jsonl
# rm model_multiwozdst_cor_err_*
# cd ../../

# # predict only extra errs
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_kpextra/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8_kpextra.json' \
#     --add-special-tokens True 
# mkdir -p experiment/cor_gpt2_nod_fresh_kpextra/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_kpextra/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_kpextra/model.report_bs8 \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs8_kpextra.json' \
#     --add-special-tokens True 
# cd experiment/cor_gpt2_nod_fresh_kpextra/
# cp model_multiwozdst_cor_err_0_replies.jsonl result_test_bs8.jsonl
# rm model_multiwozdst_cor_err_*
# cd ../../

# filter out miss err and predict only extra errs
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_fom_kpe/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8_fomiss_kpextra.json' \
#     --add-special-tokens True 
# mkdir -p experiment/cor_gpt2_nod_fresh_fom_kpe/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_fom_kpe/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_fom_kpe/model.report_bs8 \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs8_fomiss_kpextra.json'
# cd experiment/cor_gpt2_nod_fresh_fom_kpe/
# cp model_multiwozdst_cor_err_0_replies.jsonl result_test_bs8.jsonl
# rm model_multiwozdst_cor_err_*
# cd ../../

# # # filter out extra err and predict only miss errs
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_foe_kpm/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8_foextra_kpmiss.json' \
#     --add-special-tokens True 

# # predict only miss and extra errs
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_kperr/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8_kperr.json' \
#     --add-special-tokens True 
# mkdir -p experiment/cor_gpt2_nod_fresh_kperr/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_kperr/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_kperr/model.report_bs8 \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs8_kperr.json' \
#     --add-special-tokens True 
# cd experiment/cor_gpt2_nod_fresh_kperr/
# cp model_multiwozdst_cor_err_0_replies.jsonl result_test_bs8.jsonl
# rm model_multiwozdst_cor_err_*
# cd ../../

# # # with manual err of uniform distribution
# # predict only miss and extra errs, 0 or 1 miss / extra err per turn
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_rand1101/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --validation-patience 5 \
#     --add_err True
# mkdir -p experiment/cor_gpt2_nod_fresh_rand1101/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_rand1101/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_rand1101/model.report_bs8 \
#     --data-name 'dials_nodict_bs1.json' \
#     --save-world-logs True
# cd experiment/cor_gpt2_nod_fresh_rand1101/
# cp model_multiwozdst_cor_err_0_replies.jsonl result_test_bs8.jsonl
# rm model_multiwozdst_cor_err_*
# cd ../../

# # # predict only extra errs, 0 or 1 extra err per turn
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_err_gpt2_nod_fresh_rand20/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True
# mkdir -p experiment/cor_gpt2_nod_fresh_rand20/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_rand20/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_rand20/model.report_bs8 \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs1.json'
# cd experiment/cor_err_gpt2_nod_fresh_rand20/
# cp model_multiwozdst_cor_err_0_replies.jsonl result_test_bs1.jsonl
# rm model_multiwozdst_cor_err_*
# cd ../../

# # # predict only miss errs, 0 or 1 miss err per turn
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/cor_err_gpt2_nod_fresh_rand012/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --add_err True
# mkdir -p experiment/cor_err_gpt2_nod_fresh_rand012/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor_err -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_err_gpt2_nod_fresh_rand012/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_err_gpt2_nod_fresh_rand012/model.report_bs8 \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs1.json'
# cd experiment/cor_err_gpt2_nod_fresh_rand012/
# cp model_multiwozdst_cor_err_0_replies.jsonl result_test_bs1.jsonl
# rm model_multiwozdst_cor_err_*
# cd ../../












# # # # # # # # # # # # use ground truth for correction
# for sd in 2; do
#     parlai multiprocessing_train \
#         -m hugging_face/gpt2 -t multiwozdst_cor_gt -eps 10.3 -bs 4 -opt adam -lr 5e-4 \
#         --eval-batchsize 1 \
#         --fp16 true \
#         --warmup_updates 100 \
#         --warmup_rate 1e-5 \
#         --log-every-n-secs 100 \
#         --validation-every-n-epochs 1 \
#         --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_gt_sd${sd}/model \
#         --validation-metric 'joint goal acc' \
#         --validation-metric-mode max \
#         --data-name 'dials_nodict_bs8.json' \
#         --add-special-tokens True \
#         --validation-patience 5 \
#         --rand-seed ${sd}
#     # mkdir -p experiment/cor_gpt2_nod_fresh_gt_sd${sd}/
#     # parlai multiprocessing_eval \
#     #     -dt test \
#     #     -m hugging_face/gpt2 -t multiwozdst_cor_gt -bs 1 \
#     #     --fp16 true \
#     #     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_gt_sd${sd}/model \
#     #     --log-every-n-secs 100 \
#     #     --report-filename experiment/cor_gpt2_nod_fresh_gt_sd${sd}/model.report \
#     #     --save-world-logs True \
#     #     --data-name 'dials_nodict_bs1.json'
#     # cd experiment/cor_gpt2_nod_fresh_gt_sd${sd}/
#     # cp model_multiwozdst_cor_gt_0_replies.jsonl result_test.jsonl
#     # rm model_multiwozdst_cor_gt_*
#     # cd ../../
# done













# # # train cor from scratch with focal loss
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozdst_cor -eps 20 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --save-after-valid True \
#     --model-file /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_fl/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True \
#     --validation-patience 5
# mkdir -p experiment/cor_gpt2_nod_fresh_fl/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwozdst_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/cor_gpt2_nod_fresh_fl/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/cor_gpt2_nod_fresh_fl/model.report \
#     --save-world-logs True \
#     --data-name 'dials_nodict_bs1.json'
# cd experiment/cor_gpt2_nod_fresh_fl/
# cp model_multiwozdst_cor_0_replies.jsonl result_test.jsonl
# rm model_multiwozdst_*
# cd ../../
















# # ######################3 2 stage: 1 for slot type 1 for slot value ##############################
# # # # # # # # # stage 1 predict slot type
# # parlai multiprocessing_train \
# #     -m hugging_face/gpt2 -t multiwoztype -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
# #     --eval-batchsize 1 \
# #     --fp16 true \
# #     --warmup_updates 100 \
# #     --warmup_rate 1e-5 \
# #     --log-every-n-secs 100 \
# #     --validation-every-n-epochs -1 \
# #     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_type/model \
# #     --validation-metric 'joint goal acc' \
# #     --validation-metric-mode max \
# #     --add-special-tokens True 

# # # mkdir -p experiment/gen_gpt2_nod_type/
# # for i in 10; do
# #     parlai multiprocessing_eval \
# #         -dt test \
# #         -m hugging_face/gpt2 -t multiwoz_type -bs 1 \
# #         --fp16 true \
# #         -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_type/model.checkpoint_ep${i} \
# #         --log-every-n-secs 100 \
# #         --report-filename experiment/gen_gpt2_nod_type/model.report_ep${i} \
# #         --save-world-logs True

# #     cd experiment/gen_gpt2_nod_type/
# #     cp model_multiwoztype_0_replies.jsonl result_test_ep${i}.jsonl
# #     rm model_multiwoztype_*
# #     cd ../../
# # done

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_type -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_type_valacc/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True 

# mkdir -p experiment/gen_gpt2_nod_type_valacc/

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_type -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_type_valacc/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_type_valacc/model.report \
#     --save-world-logs True 

# cd experiment/gen_gpt2_nod_type_valacc/
# cp model_multiwoz_type_0_replies.jsonl result_test.jsonl
# rm model_multiwoz_type_*
# cd ../../

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_type -bs 8 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_type_valacc/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_type_valacc/model.report_decodel_all_bs8 \
#     --save-world-logs True \
#     --decode-all True

# cd experiment/gen_gpt2_nod_type_valacc/
# cp model_multiwoz_type_0_replies.jsonl result_decode_all_bs8.jsonl
# rm model_multiwoz_type_*
# cd ../../

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_type -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_type_valacc/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_type_valacc/model.report_decodel_all_bs1 \
#     --save-world-logs True \
#     --decode-all True
# cd experiment/gen_gpt2_nod_type_valacc/
# cp model_multiwoz_type_0_replies.jsonl result_decode_all_bs1.jsonl
# rm model_multiwoz_type_*
# cd ../../

# # # # # # # # # #  correct slot type 
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_type_cor -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_type_cor/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --data-name 'dials_nodict_bs8.json' \
#     --add-special-tokens True 

# # mkdir -p experiment/gen_gpt2_nod_type_cor/

# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_type_cor -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_type_cor/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_type_cor/model.report \
#     --save-world-logs True 
# cd experiment/gen_gpt2_nod_type_cor/
# cp model_multiwoz_type_cor_0_replies.jsonl result_test.jsonl
# rm model_multiwoz_type_cor_*
# cd ../../

# # # # # # # # # stage 2
#  parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwozvalue -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_val/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True 

# # # mkdir -p experiment/gen_gpt2_nod_val/
# for i in 20; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozvalue -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_val/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/gen_gpt2_nod_val/model.report_ep${i} \
#         --save-world-logs True

#     cd experiment/gen_gpt2_nod_val/
#     cp model_multiwozvalue_0_replies.jsonl result_test_ep${i}.jsonl
#     rm model_multiwozvalue_*
#     cd ../../
# done

# # # # # # # # # # stage 1+2
# for i in 5; do
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwozvalue -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_val/model.checkpoint_ep${i} \
#         --log-every-n-secs 100 \
#         --report-filename experiment/gen_gpt2_nod_val/model.report_ep${i}_type5 \
#         --save-world-logs True \
#         --generated_type_result_path './experiment/gen_gpt2_nod_type/result_test_ep5.jsonl'

#     cd experiment/gen_gpt2_nod_val/
#     cp model_multiwozvalue_0_replies.jsonl result_test_ep${i}_type5.jsonl
#     rm model_multiwozvalue_*
#     cd ../../
# done


# # # # # # # # # stage 0 predict only domain
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_dom -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_dom/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True

# # # # # # # # # # stage 2: dom + all slot type --->  values
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_value_alltype -eps 10.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs -1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_val_alltype/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_value_alltype -eps 10.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_val_alltype_valacc/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True

# mkdir -p experiment/gen_gpt2_nod_val_alltype_valacc/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_value_alltype -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_val_alltype_valacc/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_val_alltype_valacc/model.report \
#     --save-world-logs True

# cd experiment/gen_gpt2_nod_val_alltype_valacc/
# cp model_multiwoz_value_alltype_0_replies.jsonl result_test.jsonl
# rm model_multiwoz_value_alltype_*
# cd ../../

# # # # # # # # # # stage 2: dom + all slot type --->  values % sep, one slot value each turn
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_value_alltype_sep -eps 10.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_val_alltype_sep/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True

# mkdir -p experiment/gen_gpt2_nod_val_alltype_sep/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_value_alltype_sep -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_val_alltype_sep/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_val_alltype_sep/model.report \
#     --save-world-logs True

# cd experiment/gen_gpt2_nod_val_alltype_sep/
# cp model_multiwoz_value_alltype_sep_0_replies.jsonl result_test.jsonl
# rm model_multiwoz_value_alltype_sep_*
# cd ../../

# # # # # # # # # # stage 2: dom + all slot type --->  values % sep, one slot value each turn
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_value_alltype_seq -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_val_alltype_seq/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True

# mkdir -p experiment/gen_gpt2_nod_val_alltype_seq/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_value_alltype_seq -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_val_alltype_seq/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_val_alltype_seq/model.report \
#     --save-world-logs True

# cd experiment/gen_gpt2_nod_val_alltype_seq/
# cp model_multiwoz_value_alltype_seq_0_replies.jsonl result_test.jsonl
# rm model_multiwoz_value_alltype_seq_*
# cd ../../








# # # # # # #  translate dst into natural language with templates
# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_nldst -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_nldst/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True 

# mkdir -p experiment/gen_gpt2_nod_nldst/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_nldst -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_nldst/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_nldst/model.report \
#     --save-world-logs True

# cd experiment/gen_gpt2_nod_nldst/
# cp model_multiwoz_nldst_0_replies.jsonl result_test.jsonl
# rm model_multiwoz_nldst_*
# cd ../../

# parlai multiprocessing_train \
#     -m hugging_face/gpt2 -t multiwoz_nldst -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     --eval-batchsize 1 \
#     --fp16 true \
#     --warmup_updates 100 \
#     --warmup_rate 1e-5 \
#     --log-every-n-secs 100 \
#     --validation-every-n-epochs 1 \
#     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nod_nldst_tem2/model \
#     --validation-metric 'joint goal acc' \
#     --validation-metric-mode max \
#     --add-special-tokens True 

# mkdir -p experiment/gen_gpt2_nod_nldst_tem2/
# parlai multiprocessing_eval \
#     -dt test \
#     -m hugging_face/gpt2 -t multiwoz_nldst -bs 1 \
#     --fp16 true \
#     -mf /checkpoint/kunqian/parlai/gen_gpt2_nod_nldst_tem2/model \
#     --log-every-n-secs 100 \
#     --report-filename experiment/gen_gpt2_nod_nldst_tem2/model.report \
#     --save-world-logs True

# cd experiment/gen_gpt2_nod_nldst_tem2/
# cp model_multiwoz_nldst_0_replies.jsonl result_test.jsonl
# rm model_multiwoz_nldst_*
# cd ../../







# # # # # # Sep 23 data augmentation for gen model (orgi + change x3)

# for sd in 2; do
#     # parlai multiprocessing_train \
#     #     -m hugging_face/gpt2 -t multiwoz_dst -eps 20.3 -bs 8 -opt adam -lr 5e-4 \
#     #     --eval-batchsize 1 \
#     #     --fp16 true \
#     #     --warmup_updates 100 \
#     #     --warmup_rate 1e-5 \
#     #     --log-every-n-secs 100 \
#     #     --validation-every-n-epochs 1 \
#     #     --save-after-valid True \
#     #     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nodict_aug_sd${sd}/model \
#     #     --validation-metric 'joint goal acc' \
#     #     --validation-metric-mode max \
#     #     --add-special-tokens True \
#     #     --validation-patience 5 \
#     #     --data_aug True \
#     #     --rand-seed ${sd}

#     mkdir -p experiment/gen_gpt2_nodict_aug_sd${sd}/
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict_aug_sd${sd}/model \
#         --log-every-n-secs 100 \
#         --report-filename experiment/gen_gpt2_nodict_aug_sd${sd}/model.report \
#         --save-world-logs True

#     cd experiment/gen_gpt2_nodict_aug_sd${sd}/
#     cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
#     cd ../../
# done



# # # # # # # # Sep 27 data augmentation for gen model (orgi + change+unchange x3)

# for sd in 2; do
#     # parlai multiprocessing_train \
#     #     -m hugging_face/gpt2 -t multiwoz_dst -eps 20.3 -bs 8 -opt adam -lr 5e-4 \
#     #     --eval-batchsize 1 \
#     #     --fp16 true \
#     #     --warmup_updates 100 \
#     #     --warmup_rate 1e-5 \
#     #     --log-every-n-secs 100 \
#     #     --validation-every-n-epochs 1 \
#     #     --save-after-valid True \
#     #     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nodict_aug_all_sd${sd}/model \
#     #     --validation-metric 'joint goal acc' \
#     #     --validation-metric-mode max \
#     #     --add-special-tokens True \
#     #     --validation-patience 5 \
#     #     --data_aug True \
#     #     --rand-seed ${sd}

#     mkdir -p experiment/gen_gpt2_nodict_aug_all_sd${sd}/
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict_aug_all_sd${sd}/model \
#         --log-every-n-secs 100 \
#         --report-filename experiment/gen_gpt2_nodict_aug_all_sd${sd}/model.report \
#         --save-world-logs True

#     cd experiment/gen_gpt2_nodict_aug_all_sd${sd}/
#     cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
#     cd ../../
# done

# # # # # # Sep 27 multiwoz 2.2

# for sd in 0; do
#     # parlai multiprocessing_train \
#     #     -m hugging_face/gpt2 -t multiwoz_dst -eps 20.3 -bs 4 -opt adam -lr 5e-4 \
#     #     --eval-batchsize 1 \
#     #     --fp16 true \
#     #     --warmup_updates 100 \
#     #     --warmup_rate 1e-5 \
#     #     --log-every-n-secs 100 \
#     #     --validation-every-n-epochs 1 \
#     #     --save-after-valid True \
#     #     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nodict_m22_sd${sd}/model \
#     #     --validation-metric 'joint goal acc' \
#     #     --validation-metric-mode max \
#     #     --add-special-tokens True \
#     #     --validation-patience 5 \
#     #     --data_version 2.2 \
#     #     --rand-seed ${sd}

#     mkdir -p experiment/gen_gpt2_nodict_m22_sd${sd}/
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict_m22_sd${sd}/model \
#         --log-every-n-secs 100 \
#         --report-filename experiment/gen_gpt2_nodict_m22_sd${sd}/model.report \
#         --save-world-logs True \
#         --data_version 2.2

#     cd experiment/gen_gpt2_nodict_m22_sd${sd}/
#     cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
#     cd ../../
# done




# # # #####################data with scrambled slot value ##############################
for sd in 0; do
    # parlai multiprocessing_train \
    #     -m hugging_face/gpt2 -t multiwoz_dst -eps 20.3 -bs 8 -opt adam -lr 5e-4 \
    #     --eval-batchsize 1 \
    #     --fp16 true \
    #     --warmup_updates 100 \
    #     --warmup_rate 1e-5 \
    #     --log-every-n-secs 100 \
    #     --validation-every-n-epochs 1 \
    #     --save-after-valid True \
    #     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nodict_scr_sd${sd}/model \
    #     --validation-metric 'joint goal acc' \
    #     --validation-metric-mode max \
    #     --add-special-tokens True \
    #     --validation-patience 5 \
    #     --data_scr True \
    #     --rand-seed ${sd}

    mkdir -p experiment/gen_gpt2_nodict_scr_sd${sd}/
    parlai multiprocessing_eval \
        -dt test \
        -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
        --fp16 true \
        -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict_scr_sd${sd}/model \
        --log-every-n-secs 100 \
        --report-filename experiment/gen_gpt2_nodict_scr_sd${sd}/model.report \
        --save-world-logs True 

    cd experiment/gen_gpt2_nodict_scr_sd${sd}/
    cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
    cd ../../
done
# for sd in 0; do
#     # parlai multiprocessing_train \
#     #     -m hugging_face/gpt2 -t multiwoz_dst -eps 20.3 -bs 8 -opt adam -lr 5e-4 \
#     #     --eval-batchsize 1 \
#     #     --fp16 true \
#     #     --warmup_updates 100 \
#     #     --warmup_rate 1e-5 \
#     #     --log-every-n-secs 100 \
#     #     --validation-every-n-epochs 1 \
#     #     --save-after-valid True \
#     #     --model-file /checkpoint/kunqian/parlai/gen_gpt2_nodict_scr_all_sd${sd}/model \
#     #     --validation-metric 'joint goal acc' \
#     #     --validation-metric-mode max \
#     #     --add-special-tokens True \
#     #     --validation-patience 5 \
#     #     --data_scr True \
#     #     --rand-seed ${sd}

#     mkdir -p experiment/gen_gpt2_nodict_scr_all_sd${sd}/
#     parlai multiprocessing_eval \
#         -dt test \
#         -m hugging_face/gpt2 -t multiwoz_dst -bs 1 \
#         --fp16 true \
#         -mf /checkpoint/kunqian/parlai/gen_gpt2_nodict_scr_all_sd${sd}/model \
#         --log-every-n-secs 100 \
#         --report-filename experiment/gen_gpt2_nodict_scr_all_sd${sd}/model.report \
#         --save-world-logs True 

#     cd experiment/gen_gpt2_nodict_scr_all_sd${sd}/
#     cp model_multiwoz_dst_0_replies.jsonl result_test.jsonl
#     cd ../../
# done




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
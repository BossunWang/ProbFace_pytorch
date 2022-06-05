mkdir "log_prob_face_masked"
export OMP_NUM_THREADS=4
python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 \
--master_addr="127.0.0.1" --master_port=1324 \
  train.py \
--data_root "/media/glory/Transcend/Dataset/Face_Recognition_Dataset/WebFace260M/WebFace260M" \
--train_file "data/data.pkl" \
--lr 0.01 \
--backbone_type "ir_101" \
--out_dir "prob_face_ir_101_masked" \
--device "0,1" \
--epoches 1 \
--step "60000" \
--print_freq 1000  \
--save_freq 10000 \
--batch_size 32 \
--sample_size 8 \
--momentum 0.9 \
--triplet_margin 3.0 \
--masked_ratio 1.0 \
--discriminate_loss_weight 0.0001 \
--output_constraint_loss_weight 0.1 \
--log_dir "log_prob_face_masked" \
--tensorboardx_logdir "log_prob_face_masked/prob_face_ir_101_masked" \
2>&1 | tee log_prob_face_masked/log.log

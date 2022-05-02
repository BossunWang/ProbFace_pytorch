mkdir "log_prob_face"
export OMP_NUM_THREADS=4
python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 \
--master_addr="127.0.0.1" --master_port=1324 \
  train.py \
--data_root "/media/glory/Transcend/Dataset/Face_Recognition_Dataset/WebFace260M/WebFace260M" \
--train_file "data/data.pkl" \
--lr 0.01 \
--backbone_type "ir_101" \
--out_dir "prob_face_ir_101" \
--device "1" \
--epoches 20 \
--step "10,13,16" \
--print_freq 10000  \
--save_freq 100000 \
--batch_size 16 \
--sample_size 16 \
--momentum 0.9 \
--triplet_margin 3.0 \
--sync_bn \
--log_dir "log_prob_face" \
--tensorboardx_logdir "log_prob_face/prob_face_ir_101" \
2>&1 | tee log_prob_face/log.log

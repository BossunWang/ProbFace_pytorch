#CUDA_VISIBLE_DEVICES=1 python -u IJB_11_uc.py \
#  --model_type IR_101 \
#  --model_path ../pretrained/adaface_ir101_webface12m.ckpt \
#  --uc_model_path ../prob_face_ir_101/Epoch_0_batch_59999.pt \
#  --gpu 0 \
#  --image-path ../../face_dataset/IJB-C/IJB/IJB_release/IJBC \
#  --job ir101_adaface_uc \
#  --result-dir "ir101_adaface_uc" \
#2>&1 | tee ir101_adaface_uc.log

CUDA_VISIBLE_DEVICES=1 python -u IJB_11_uc.py \
  --model_type IR_101 \
  --model_path ../pretrained/adaface_ir101_webface12m.ckpt \
  --uc_model_path ../prob_face_ir_101_masked/Epoch_0_batch_59999.pt \
  --gpu 0 \
  --image-path ../../face_dataset/IJB-C/IJB/IJB_release/IJBC \
  --job ir101_adaface_uc_masked \
  --result-dir "ir101_adaface_uc_masked" \
2>&1 | tee ir101_adaface_uc_masked.log
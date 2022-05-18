CUDA_VISIBLE_DEVICES=1 python -u IJB_11.py \
  --model_type IR_101 \
  --model_path ../pretrained/adaface_ir101_webface12m.ckpt \
  --gpu 0 \
  --image-path ../../face_dataset/IJB-C/IJB/IJB_release/IJBC \
  --job ir101_adaface \
  --result-dir "ir101_adaface" \
2>&1 | tee ir101_adaface.log

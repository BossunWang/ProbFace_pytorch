# ProbFace_pytorch
implement ProbFace using pytorch from https://github.com/KaenChan/ProbFace

# webface 206M IDs
* IJBC 1e-4: 97.71
  * --batch_size 32 \
  --sample_size 4 \
  --triplet_margin 3.0 \
  --masked_ratio 0.0 \
  --discriminate_loss_weight 0.0001 \
  --output_constraint_loss_weight 0.1
  * sigma_sq_max, sigma_sq_min: 0.70~0.30
  * triplet loss: 2.0~2.1

# webface 206M IDs with masked
* IJBC 1e-4: 97.66
  * --batch_size 32 \
  --sample_size 4 \
  --triplet_margin 3.0 \
  --masked_ratio 0.5 \
  --discriminate_loss_weight 0.001 \
  --output_constraint_loss_weight 1.0
  * sigma_sq_max, sigma_sq_min: 0.60~0.40
  * triplet loss: 1.8~1.9

# webface 206M IDs with masked
* IJBC 1e-4: 97.58
  * --batch_size 32 \
  --sample_size 4 \
  --triplet_margin 5.0 \
  --masked_ratio 0.5 \
  --discriminate_loss_weight 0.001 \
  --output_constraint_loss_weight 1.0
  * sigma_sq_max, sigma_sq_min: 0.60~0.40
  * triplet loss: 2.2~2.3
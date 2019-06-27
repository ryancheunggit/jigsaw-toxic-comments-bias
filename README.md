# jigsaw-toxic-comments-bias

To get in silver zone with one week work. 

```bash
python train_bert.py \
    --gpu 0 \
    --model bert-base-uncased \
    --grad_accum 4 \
    --initial_lr 2e-5 \
    --warmup 0.03 \
    --layer_wise_decay .95 \
    --valid_per_steps 5075 \
    --epochs 2 \
    --lr_schedule cosine  
    
python train_bert.py \
    --gpu 0 \
    --model bert-large-uncased-whole-word-masking \
    --grad_accum 2 \
    --initial_lr 2e-5 \
    --warmup 0.03 \
    --layer_wise_decay .95 \
    --valid_per_steps 5075 \
    --epochs 2 \
    --lr_schedule cosine
    
python train_bert.py \
    --gpu 0 \
    --model bert-large-uncased \
    --grad_accum 2 \
    --initial_lr 2e-5 \
    --warmup 0.03 \
    --layer_wise_decay .95 \
    --valid_per_steps 5075 \
    --epochs 2 \
    --lr_schedule cosine
```

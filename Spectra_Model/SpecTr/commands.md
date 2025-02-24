```bash
CUDA_VISIBLE_DEVICES=0 python code/train_main.py \
-r /mnt/Windows/cv_projects/SpecTr/HyperBlood \
-dd /mnt/Windows/cv_projects/SpecTr/four_folds_blood.json \
-dh data \
-dm anno \
-me 'npz' \
-et 'float' \
-sn 128 -cut 100 -e 1 \
-b 1 -c 9
```
brain

```bash
CUDA_VISIBLE_DEVICES=0 python code/train_main.py \
-r /mnt/Windows/cv_projects/Brain \
-dd /mnt/Windows/cv_projects/SpecTr/brain_small.json \
-sn 300 -cut 100 -e 100 \
-b 1 -c 5 --dataset brain -et ''
```
dental

```bash
CUDA_VISIBLE_DEVICES=0 python code/train_main.py \
-r /mnt/Windows/cv_projects/dental \
-dd /mnt/Windows/cv_projects/SpecTr/dental_small.json \
-sn 51 -cut 100 -e 100 \
-b 1 -c 36 --dataset dental
```
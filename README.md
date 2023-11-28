# EsmTherm
 
### Installation
```angular2html
conda create -n esmtherm python=3.10 --file=environment.yml
conda activate esmtherm
pip install -e .

# To download the model checkpoints
gdown https://drive.google.com/drive/u/1/folders/1z3_IbeD8oORmLndqCCiqYRCxmbKNbWXb -O output_dir/checkpoint-best --folder
```

### Dataset Preparation
Download the supplementary materials from [Tsuboyama et al. (2023)](https://www.nature.com/articles/s41586-023-06328-6),
extract K50_dG_Dataset1_Dataset2.csv and place it under `data` directory.

```angular2html
# analyze and filter dataset
python prebuild_dataset.py

# create training dataset
python build_dataset.py \
    --dataset_dir datasets/dataset \
    --csv datasets/analysis/filtered_data.csv \
    --split_csv datasets/wildtype_split.csv
```

### Training and Evaluation
```angular2html
# training
python train.py \
    --dataset_dir datasets/dataset \
    --output_dir output_dir \
    --model_name facebook/esm2_t12_35M_UR50D
```

```
# evaluation
python evaluate.py \
    --model_name_or_path output_dir/checkpoint-best \
    --input_csv _your_input_csv_ \
    --output_csv __your_output_csv_
```

More instructions can be found with `--help` flag.

# IM2LATEX Model in Detect, Attend and Parse

The code is edited based on https://github.com/luopeixiang/im2latex. We added the historical context vector in the calculation of attention mechanism to improve the effect of recognition.

## Getting Started

Install dependency:
```shell
pip install -r requirement.txt
```

Downloading IM2LATEX-100K dataset:
```shell
cd dataset
wget http://lstm.seas.harvard.edu/latex/data/im2latex_validate_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_train_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_test_filter.lst
wget http://lstm.seas.harvard.edu/latex/data/im2latex_formulas.norm.lst
wget http://lstm.seas.harvard.edu/latex/data/formula_images_processed.tar.gz
tar -zxvf formula_images_processed.tar.gz
```

Creating .pkl file for all sets:
```shell
python preprocess.py
```

Create vocabulary for decoding:
```shell
python build_vocab.py
```

Training:
```shell
python train.py --data_path=./dataset --save_dir=./ckpts --dropout=0.2 --add_position_features --epoches=50 --max_len=150 --batch_size=16
```

Training from checkpoint:
```shell
python train.py --data_path=./dataset --save_dir=./ckpts --dropout=0.2 --add_position_features --epoches=50 --max_len=150 --batch_size=16 --from_check_point
```

Testing:
```shell
python evaluate.py --split=test --model_path=./ckpts/best_ckpt.pt --data_path=./dataset --batch_size=32 --beam_size=4
```

Predicting from image:
```shell
python predict.py --data_path=./images --model_path=./ckpts/best_ckpt.pt --beam_size=4 --batch_size=1
```

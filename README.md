# Anime Character Classification
Code for training anime character classification models on the [DAF:re dataset](https://arxiv.org/abs/2101.08674). A fine-tuned BEiT-b/16 model achieves a test accuracy of 94.84\%.


## Requirements
- Python 3.8+
- `pip install -r requirements.txt`

## Usage
### Training
- [`config/`](configs/) contains the configuration files used to produce the best model and can be run with:
```
python train.py --accelerator gpu --devices 1 --precision 16 --config path/to/config
```
- To get a list of all arguments run `python train.py --help`

##### Training examples
- In all examples `...` represents default options such as `--accelerator gpu --devices 1 --precision 16 --data.root data/dafre --max_step 50000 --val_check_interval 2000`.
<details><summary>Fine-tune a classification layer</summary>

```
python train.py ... --model.linear_prob true
```

</details>

<details><summary>Fine-tune the entire model initialize with a trained classifier (or entire model)</summary>

```
python train.py ... --model.weights /path/to/linear/checkpoint
```

</details>

<details><summary>Apply data augmentations</summary>

```
python train.py ...  --data.erase_prob 0.25 --data.use_trivial_aug true --data.min_scale 0.8
```

</details>

<details><summary>Apply regularization</summary>

```
python train.py ...  --model.mixup_alpha 1 --model.cutmix_alpha 1 --model.label_smoothing 0.1
```

</details>

<details><summary>Train with class-balanced softmax loss</summary>

```
python train.py ...  --model.loss_type balanced-sm --model.samples_per_class_file  samples_per_class.pkl
```

</details>

<details><summary>Train with class-balanced data sampling</summary>

```
python train.py ...  --data.use_balanced_sampler true
```

</details>

### Evaluate
To evaluate a trained model on the test set run:
```
python test.py --accelerator gpu --devices 1 --precision 16 --checkpoint path/to/checkpoint
```
- __Note__: Make sure the `--precision` argument is set to the same level as used during training.



## Results

| Model     | Top-1 Val Acc | Top-5 Val Acc | Top-1 Test Acc| Top-5 Test Acc| Configs | 
|:---------:|:-------------:|:-------------:|:-------------:|:-------------:|:------:|
| BEiT-b/16 | 95.26         | 98.38         | 94.84         | 98.30         | [1](configs/dafre-linear.yaml)  [2](configs/dafre-ft.yaml) [3](configs/dafre-balanced-linear.yaml) |


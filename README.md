# DAF:Re
Code for fine-tuning ViT models on various classification datasets.




## Requirements
- Python 3.8+
- `pip install -r requirements.txt`

## Usage
### Training
- To fine-tune a ViT-B/16 model on CIFAR-100 run:
```
python train.py --accelerator gpu --devices 1 --precision 16 --max_steps 5000 --model.lr 0.01
--model.warmup_steps 500 --val_check_interval 250 --data.batch_size 128 --data.dataset cifar100
```
- [`config/`](configs/) contains example configuration files which can be run with:
```
python train.py --accelerator gpu --devices 1 --precision 16 --config path/to/config
```
- To get a list of all arguments run `python train.py --help`


### Evaluate
To evaluate a trained model on its test set run:
```
python test.py --accelerator gpu --devices 1 --precision 16 --checkpoint path/to/checkpoint
```
- __Note__: Make sure the `--precision` argument is set to the same level as used during training.





## Results
All results are from fine-tuned ViT-B/16 models which were pretrained on ImageNet-21k.

| Dataset            | Total Steps | Warm Up Steps | Learning Rate | Accuracy | Config                         | 
|:------------------:|:-----------:|:-------------:|:-------------:|:--------:|:------------------------------:|
| CIFAR-10           | 5000        | 500           | 0.01          | 99.00    | [Link](configs/cifar10.yaml)   |
| CIFAR-100          | 5000        | 500           | 0.01          | 92.89    | [Link](configs/cifar100.yaml)  |
| Oxford Flowers-102 | 1000        | 100           | 0.03          | 99.02    | [Link](configs/flowers102.yaml)|
| Oxford-IIIT Pets   | 2000        | 200           | 0.01          | 93.68    | [Link](configs/pets37.yaml)    |
| Food-101           | 5000        | 500           | 0.03          | 90.67    | [Link](configs/food101.yaml)   |


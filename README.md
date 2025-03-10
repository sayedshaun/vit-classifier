# Image Classification with Vision Transformer
This repository contains the code for the Image Classifier with Vision Transformer (ViT).

## Project Structure
```
vit-image-classifier/
├── data/
│   └── directory_of_images/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   ├── class_3/
│   │   └── ...
│   └── src/
│       ├── data.py
│       ├── model.py
│       ├── metrics.py
│       ├── trainer.py
│       └── config.py
├── pipeline.py
├── train.sh
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```    

## Setup
To set up the environment, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
To train the model, run the following command:

```bash
bash train.sh
```
Provide the following arguments in the bash script:
```bash
data_dir=dataset
image_size=224
patch_size=16
color_channels=3
hidden_size=128
num_heads=4
num_layers=4
norm_epsilon=1e-5
dropout=0.1
batch_size=32
epochs=10
learning_rate=1e-3
weight_decay=1e-4
gradient_accumulation_steps=1
gradient_clipping=0.0
precision=32
log_and_eval_step=10
save_steps=10
num_workers=0
seed=42
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
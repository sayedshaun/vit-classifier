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
├── predict.py
├── infer.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```    

## Setup
To set up the environment, run the following command:

For stable release
```bash
git clone https://github.com/sayedshaun/vit-classifier.git
cd vit-classifier
```

For development release
```bash
git clone -b dev https://github.com/sayedshaun/vit-classifier.git
cd vit-classifier
```

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
learning_rate=1e-4
weight_decay=0
gradient_accumulation_steps=1
gradient_clipping=0.0
precision="fp16"
device="cuda"
log_and_eval_step=10
save_steps=10
num_workers=0
seed=42
save_directory="my_checkpoint"
```

## Test report
Although auto saved, the test report after completion of training if set `--do_predict` flag. but if you want to test the model explicitly, run the following command:

```bash
python predict.py --data_dir "my_dataset" --saved_dir "my_checkpoint"
```

## Inference
It's import to see actual results, run the following command:

```bash
python infer.py --image_path "my_image.png" --saved_dir "my_checkpoint"
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
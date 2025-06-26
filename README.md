# Multi-Defendant-Legal-Judgment-Prediction
Multi-Defendant Legal Judgment Prediction for FNLP class
## Installation
```bash
git clone https://github.com/daihangpku/Multi-Defendant-Legal-Judgment-Prediction.git
cd Multi-Defendant-Legal-Judgment-Prediction
conda create -n law python=3.10 -y
conda activate law
# replace xxx with 118/124/126
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cuxxx
pip install -r requirements.txt
```
## Quick Start
### Subtask1
For data preprocessing
```bash
python preprocess/subtask1/preprocess.py
```
For training
```bash
python scripts/subtask1/train.py
```
For evaluation/inference
```bash
python scripts/subtask1/inference.py --ckpt_path your/ckpt_path
```

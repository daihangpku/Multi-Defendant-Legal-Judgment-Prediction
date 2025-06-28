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
You can download the preprocessed data from 
https://disk.pku.edu.cn/link/AAECF2C690C356451898F82534291EBBEE
文件夹名：data
有效期限：2025-07-28 18:17
and put it as ./data

You can download the ckpt from 

and put it as ./checkpoints
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
python scripts/subtask1/inference.py --ckpt_path checkpoints/subtask1_20250628_135703/80.pt
```

### Subtask2
For data preprocessing
```bash
python preprocess/subtask2/preprocess.py
```
For training
```bash
python scripts/subtask2/train.py
```
For evaluation/inference

```bash
python scripts/subtask2/inference.py --ckpt_path checkpoints/subtask2_20250626_205705/100.pt
```

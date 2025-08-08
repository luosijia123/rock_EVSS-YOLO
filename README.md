### 1. Install Dependencies
    pip install -r requirements.txt
### 2. Download Dataset
    DOI: https://doi.org/10.34740/kaggle/dsv/12706253
### Split Training and Validation Sets
    python split_train_val.py
###  4. Train Model
    //You can choose between object detection or image classification tasks.
    python train.py
### 5. Make Predictions
    python detect.py
### 6. Generate Heatmap
    python Heatmap.py

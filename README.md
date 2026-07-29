from os import mkdir

### Install
You can follow the steps to prepare the environment:
```
conda create -n GRASC python=3.8 -y

source activate GRASC

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install tqdm

pip install -U openmim

mim install mmengine==0.7.3

mim install mmcv==2.0.0

mkdir data
mkdir cache
mkdir temp_cheakpoints
pip install -v -e .
chmod -R 777 ./tools/dist_train.sh
```

### Dataset Prepare
Download the VOC2007 dataset and note the root directory path of your VOC2007 folder.
Open replace.py and set the variable: new_string = "your_VOC2007_directory_path"
```python
# Run
python replace.py
```
Next, configure the pattern variable in pascal_voc_split.py to either '10+10' or '10+5', depending on your desired experimental setting. Then, execute the script:
```python
# run
python pascol_voc_split.py
```

### Train
```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.

# you can train VOC(10-10) [Task0, Task1]
# This is Task0 for VOC(10-10)
sh tools/10+10/task0.sh
# This is Task1 for VOC(10-10)
sh tools/10+10/task1.sh

# you can train VOC(10-5) [Task0, Task1, Task2]
# This is Task1 for VOC(10-5)
sh tools/10+5+5/task1.sh
# This is Task2 for VOC(10-5)
sh tools/10+5+5/task2.sh
```



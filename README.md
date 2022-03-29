# LPKT_tensorflow_version

Source code and data set for the paper Learning Process-consistent Knowledge Tracing.

The code is the implementation of LPKT model, and the data set is the public data set [ASSIST2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect).

If this code helps with your studies, please kindly cite the following publication:
```
@inproceedings{10.1145/3447548.3467237,
author = {Shen, Shuanghong and Liu, Qi and Chen, Enhong and Huang, Zhenya and Huang, Wei and Yin, Yu and Su, Yu and Wang, Shijin},
title = {Learning Process-Consistent Knowledge Tracing},
year = {2021},
isbn = {9781450383325},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3447548.3467237},
doi = {10.1145/3447548.3467237},
pages = {1452–1460},
numpages = {9},
location = {Virtual Event, Singapore},
series = {KDD '21}
}
```

## Dependencies:

- python >= 3.7
- tesorflow-gpu >= 2.0 
- numpy
- tqdm
- utils
- pandas
- sklearn


## Usage

First, download the data file: [2012-2013-data-with-predictions-4-final.csv](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect), then put it in the folder 'data/' 

Then, run data_pre.py to preprocess the data set, and run data_save.py {sequence length} to divide the original data set into train set, validation set and test set. 

`python data_pre.py`


`python data_save.py 100`

Train the model:

`python train_lpkt.py {fold}`

For example:

`python train_lpkt.py 1`  or `python train_lpkt.py 2`

Test the trained the model on the test set:

`python test.py {model_name}`


## Correction

There is a mistake in the KDD conference paper. Figure. 4 should be results on dataset ASSIST2012

The experimental results would be better than our original paper, as we have optimized the data proprocessing

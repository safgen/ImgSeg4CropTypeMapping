# Image Segmentation for Crop-type Mapping
(some scripts are upload pending)

Corn and Soybean Mapping model at 30 m resolution from Landsat 8/ Sentinel 2

**train/train.py** -  trains the model. Make sure the data directory/folder are set correctly. The training data can be exported as _TFRecord_ from Earth Engine. The model expects data as 256x256 samples with the bands and the labels stacked together as channels. (export code in python through earth engine api TBU. Refer to this [notebook](https://github.com/safgen/CroplandMappingCDL/blob/main/DataPrep.ipynb) for reference)



**requirements.txt** - python packages required to run this repo.

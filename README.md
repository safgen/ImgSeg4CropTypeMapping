# Image Segmentation for Crop-type Mapping


Corn and Soybean Mapping model at 30 m resolution from Landsat 8/ Sentinel 2 as presented in [In-Season Wall-to-Wall Crop-Type Mapping Using Ensemble of Image Segmentation Models](https://ieeexplore.ieee.org/document/10323532)


### Model Traininng
**train/train.py** -  trains the model. Make sure the data directory/folder are set correctly. The training data can be exported as _TFRecord_ from Earth Engine. The model expects data as 256x256 samples with the bands and the labels stacked together as channels. ( Refer to this [notebook](https://github.com/safgen/CroplandMappingCDL/blob/main/DataPrep.ipynb) for reference)

### Model Inference for Map Generation
To be updated soon



### Citation
If you find this code helpful, please cite the followinbg work:

<p>@ARTICLE{10323532,<br>
  author={Zaheer, Sheir A. and Ryu, Youngryel and Lee, Junghee and Zhong, Zilong and Lee, Kyungdo},<br>
  journal={IEEE Transactions on Geoscience and Remote Sensing}, <br>
  title={In-Season Wall-to-Wall Crop-Type Mapping Using Ensemble of Image Segmentation Models}, <br>
  year={2023},<br>
  volume={61},<br>
  number={},<br>
  pages={1-11},<br>
  doi={10.1109/TGRS.2023.3335214}}

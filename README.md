# SENet-ResNet34-DCN
Datasets and code
Classification of soybean seeds based on RGB reconstruction of hyperspectral images



Firstly, hyperspectral images of seven varieties of soybean, H1, H2, H3, H4, H5, H6 and H7, were collected by hyperspectral imager, and by using the principle of the three base colours, the R, G and B bands which have more characteristic information are selected to reconstruct the images with different texture and colour characteristics to generate a new dataset for seed segmentation, and finally, a comparison is made with the classification effect of the seven models.The experimental results in ResNet34 show that the classification accuracy of the dataset before and after RGB reconstruction increases from 88.87% to 91.75%, demonstrating that RGB image reconstruction can strengthen image features; ResNet18, ResNet34, ResNet50, ResNet101, CBAM-ResNet34, SENet-ResNet34, and SENet-ResNet34-DCN models have classification accuracies of 72.25%, 91.75%, 89%, 88.48%, 92.28%, 92.80%, and 94.24%, respectively.SENet-ResNet34-DCN achieves the greatest classification accuracy results, with a model loss of roughly 0.3. The proposed SENet-ResNet34-DCN model is the most effective at classifying soybean seeds.

Description of the data and file structure

The single-band grayscale map is used to derive the three sets of R, G, and B feature bands with the highest classification accuracy using the ResNet34 algorithm, and then the RGB pseudo-color image is reconstructed to construct a new dataset, which is subsequently segmented, and each seed is extracted from the image, and then recognized and compared with the original dataset using ResNet34 to validate the validity of the hyperspectral image reconstruction method.The classification accuracy of the original dataset 188-120-60 is 88.87%, with a 2% increase across all nine combinations. The classification accuracy of the 188-83-41 dataset is 91.75%, an improvement of 2.88% over the original dataset, and the 188-83-41 dataset is referred to as the new dataset and applied to the following algorithm for soybean seed classification.



Code/Software

RGB reconstruction code for hyperspectral images

import spectral
from spectral import imshow
import numpy as np
from PIL import Image

# 读取 bil 格式高光谱数据
img = spectral.open_image('D:/makj_work/aaa/H8_1.bil.hdr')

# 选择第 10，20，30 个波段
bands = [img.read_band(i) for i in [188,83,41]]
print(bands)

# 将选择的三个波段重构为 RGB 格式
rgb = np.stack(bands, axis=-1)
rgb = np.clip(rgb / np.percentile(rgb, 99.1), 0, 1) # 像素压缩

# 将重构的 RGB 格式转化为图片并保存
im = Image.fromarray((rgb*255).astype(np.uint8))
im.save('h41-197-83-41.tif')
# # imshow(im)


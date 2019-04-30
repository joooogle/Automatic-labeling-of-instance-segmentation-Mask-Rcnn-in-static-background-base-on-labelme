import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data,segmentation,measure,morphology,color
import cv2

back_img=np.load("E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6.npy")#TODO：读取背景矩阵
img_number=304
#读取所有图片
all_img = [np.array(Image.open('E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\'+str(i+1)+'.png', 'r').convert('L')) for i in range(img_number)]
# 帧的宽高
h = all_img[0].shape[0]
v = all_img[0].shape[1]

for idex,i in enumerate(all_img):

    front_img = np.array(i - back_img)
    front_img = np.abs(front_img)
    print(i.shape)

    # 前景二值化 设定阈值将前景像素值化为0或1
    threshold_level = 100
    threshold = np.full((h, v), threshold_level)
    front_img = np.array(front_img > threshold, dtype=bool)
    dst = morphology.remove_small_objects(front_img, min_size=5000, connectivity=1,in_place=True)#TODO：把小于min_size的连通域去除
    front_img = dst*255
    front_img = front_img.astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    front_img = cv2.morphologyEx(front_img, cv2.MORPH_CLOSE, kernel)  # TODO:cv2.MORPH_CLOSE闭运算,膨胀腐蚀去毛刺
    #保存二值矩阵
    np.save('E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\front_img\\'+str(idex+1)+'.npy', front_img)

    front_img = np.fmin(front_img, i)# 在原帧上抠图 得到真实的前景
    print(front_img.shape)
    # 保存抠图图片
    print(idex)
    Image.fromarray(front_img).convert('RGB').save('E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\front_img\\' + str(idex + 1) + '.jpg')
    if idex==1000:
        break



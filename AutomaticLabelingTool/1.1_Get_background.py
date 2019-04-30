# coding: utf-8
'''
用两张图片拼接得到背景的程序 实际效果不理想
'''
import cv2
import numpy as np
from PIL import Image

img1=np.array(Image.open('E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\1.png', 'r').convert('L'))#TODO：打开图片 转成灰度图
img2=np.array(Image.open('E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\56.png', 'r').convert('L'))
print(img1.shape)
imgtemp=np.zeros(img1.shape)
img2[:,0:1000]=img1[:,0:1000]#TODO：把img2的左半边换成img1的左半边
cv2.imshow("sa",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\background6_2.png',img2)
np.save("E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6_2.npy",img2)
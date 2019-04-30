import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

cap = cv2.VideoCapture("E:\\deeplabcut\\test2-jzh-2019-04-22\\videos\\6.mp4")
back_img = np.zeros((int(cap.get(4)),int(cap.get(3))))
nframes=int(cap.get(7)//2)
print(nframes)
counter=0
for i in range(nframes):
    ret, frame = cap.read()
    if ret:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sframe=np.array(frame)
        back_img+=sframe
        counter+=1
        print(counter)
    else:
        break
back_img/=counter#TODO：平均背景法
np.save("E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6.npy",back_img)#TODO：存背景矩阵
cv2.imwrite("E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\background6.png", back_img)
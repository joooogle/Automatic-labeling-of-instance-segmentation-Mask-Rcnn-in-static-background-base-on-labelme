'''
TODO:简单等间隔取帧程序
'''
import numpy as np
import cv2
counter = 0
count=1
cap = cv2.VideoCapture("E:\\deeplabcut\\test2-jzh-2019-04-22\\videos\\2.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

print(cap.get(5))
nframes=cap.get(7)//2 #读取的帧数比实际大两倍
print(nframes)
a=nframes//300  #TODO：300:要取的帧数
while(cap.isOpened()):
    # Capture frame-by-frame

    ret, frame = cap.read()

    if counter%a==0 :
        cv2.imwrite('E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\7red\\'+str(count)+'.png',frame)
        count+=1

    counter += 1
    print(counter)
    cv2.waitKey(1)
    if not ret:
        break
cap.release()
cv2.destroyAllWindows()

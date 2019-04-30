import os
import natsort
labelme_json = "C:\\Users\\213\AppData\Local\conda\conda\envs\labelme\Scripts\labelme_json_to_dataset.exe" #labelme_json_to_dataset.exe 程序路径
file_path = "E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\front_img\\automatic_label\\"   # 处理文件所在路径
dir_info = os.listdir(file_path)
dir_info = natsort.natsorted(dir_info)
"""循环处理‘.json’文件"""
os.system('cd ‪C:\\Users\\213\\AppData\\Local\\conda\\conda\\envs\\labelme\\Scripts\\')
for file_name in dir_info:
    file_name = os.path.join(file_path + "\\" + file_name)
    #
    os.system(labelme_json + " " + file_name)
# -*- coding:utf-8 -*-

'''
仿照labelme的json文件写入自己的数据
'''
import cv2
import json
from base64 import b64encode
from json import dumps
import os
import glob
import numpy as np
from skimage import data,segmentation,measure,morphology,color

# 参考labelme的json格式重新生成json文件，
# 便可以使用labelme的接口解析数据
class img_to_json(object):
    """
        这个类是用来将图像数据转化成json文件的，方便下一步的处理。主要是为了获取
        图像的字符串信息
    """
    def __init__(self, process_img_path,
                 img_type=".png",
                 out_file_path="",
                 out_file_type=".json"):
        """
        :param process_img_path: 待处理图片的路径
        :param img_type: 待处理图片的类型
        :param out_file_path: 输出文件的路径
        :param out_file_type: 输出文件的类型
        使用glob从指定路径中获取所有的img_type的图片
        """
        self.process_img = [process_img_path]
        self.out_file = out_file_path
        self.out_file_type = out_file_type
        self.img_type = img_type

    def en_decode(self):
        """
        对获取的图像数据进行编码，解码后并存储到指定文件，保存为json文件
        :return: null
        """
        print('-' * 30)
        print("运行 Encode--->Decode\nStart process.....\nPlease wait a moment")
        print('-' * 30)
        """
        Start process.....   Please wait a moment
        """
        """filepath, shotname, extension, tempfilename:目标文件所在路径，文件名，文件后缀,文件名+文件后缀"""
        def capture_file_info(filename):
            (filepath, tempfilename) = os.path.split(filename)
            (shotname, extension) = os.path.splitext(tempfilename)
            return filepath, shotname, extension, tempfilename

        ENCODING = 'utf-8'  # 编码形式为utf-8

        # SCRIPT_NAME, IMAGE_NAME, JSON_NAME = argv  # 获得文件名参数

        img = self.process_img  # 所有图片的形成的列表信息
        # img_number = capture_file_info(img)[1]
        # imgs = sorted(img,key=lambda )

        out_file_path = self.out_file

        # imgtype = self.img_type

        out_file_type = self.out_file_type
        print("待处理的图片的数量:",len(img))
        if len(img) == 0:
            print("There was nothing under the specified path.")
            return 0
        for imgname in img:
            # midname = imgname[imgname.rindex("\\"):imgname.rindex("." + imgtype)]
            midname = capture_file_info(imgname)[1]   # midname:图片的名称，不带后缀名
            IMAGE_NAME = imgname
            # IMAGE_NAME = midname + imgtype
            JSON_NAME = midname + out_file_type
            # 读取二进制图片，获得原始字节码，注意 'rb'
            with open(IMAGE_NAME, 'rb') as jpg_file:
                byte_content = jpg_file.read()
            # 把原始字节码编码成 base64 字节码
            base64_bytes = b64encode(byte_content)
            # 将 base64 字节码解码成 utf-8 格式的字符串
            base64_string = base64_bytes.decode(ENCODING)
            # 用字典的形式保存数据
            """raw_data:用来存放加入特性的数据，img_raw_data:用来存放不加入特性的数据，只有图片的字符串数据"""
            # raw_data = {}
            # raw_data["name"] = IMAGE_NAME
            # raw_data["image_base64_string"] = base64_string
            img_raw_data = {}
            img_raw_data = base64_string
            # 将字典变成 json 格式，indent =2:表示缩进为 2 个空格
            # json_data = dumps(raw_data)
            json_img_data = dumps(img_raw_data)
            # 将 json 格式的数据保存到指定的文件中
            # with open(out_file_path+JSON_NAME, 'w') as json_file:
            #     json_file.write(json_img_data)
            return base64_string

def dict_json(imageData,shapes,imagePath,fillColor=None,lineColor=None):
    '''

    :param imageData: str
    :param shapes: list
    :param imagePath: str
    :param fillColor: list
    :param lineColor: list
    :return: dict
    '''
    return {"imageData":imageData,"shapes":shapes,"fillColor":fillColor,
            'imagePath':imagePath,'lineColor':lineColor}

def dict_shapes(points,label,fill_color=None,line_color=None):
    return {'points':points,'fill_color':fill_color,'label':label,'line_color':line_color}

# 注以下都是虚拟数据，仅为了说明问题
dirnpy='E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\front_img'#TODO：.npy格式二值图矩阵位置
dirpng='E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\'#TODO：.png格式原图位置
npyname = [name for name in os.listdir(dirnpy) if name.endswith('.npy')]
pathdict={}
for i in npyname:
    pathdict[os.path.join(dirnpy, i)]=os.path.join(dirpng, i[0:-4]+'.png')#TODO：用字典建立原图与二值图映射关系
# print(pathdict)
counter=0
for npy in pathdict:

    Binary_map=np.load(npy)
    contours_map = measure.find_contours(Binary_map,5)#TODO：找到二值连通域的边界坐标，返回一个List[np.array1,np.array2...]
    removelist = []#TODO:删除坐标数小于300的轮廓  也就是内轮廓 加入removelist，挨个删除
    for i in range(len(contours_map)):
        if contours_map[i].shape[0] < 300:
            removelist.append(i)
            print(removelist)
    if len(removelist) != 0:
        for j in range(len(removelist) - 1, -1, -1):
            print(removelist)
            del contours_map[removelist[j]]

    contours_map[0][:,[0,1]] = contours_map[0][:,[1,0]]#TODO：x坐标和y坐标互换 因为lebelme解析时x y坐标是相反的  本程序以图中有两个联通目标为例
    contours_map[1][:,[0,1]] = contours_map[1][:,[1,0]]

    trans = img_to_json(process_img_path=pathdict[npy])
    imageData=trans.en_decode()
    shapes=[]
    # 第一个对象
    points=contours_map[0].tolist() # 第一个对象坐标
    # fill_color=null
    label='mouse1'
    # line_color=null
    shapes.append(dict_shapes(points,label))

    # 第二个对象
    points=contours_map[1].tolist() # 第二个对象坐标
    label='mouse2'
    shapes.append(dict_shapes(points,label))

    fillColor=[255,0,0,128]

    imagePath=pathdict[npy]

    lineColor=[0,255,0,128]

    data=dict_json(imageData,shapes,imagePath,fillColor,lineColor)#TODO：写入字典

    # 写入json文件
    json_file = 'E:\\deeplabcut\\test2-jzh-2019-04-22\\labeled-data\\6\\front_img\\automatic_label\\'+npyname[counter][0:-4]+'.json'
    counter+=1
    json.dump(data,open(json_file,'w'))
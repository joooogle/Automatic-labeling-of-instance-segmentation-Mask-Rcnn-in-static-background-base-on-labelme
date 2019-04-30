# Automatic-labeling-of-instance-segmentation-Mask-Rcnn-in-static-background-base-on-labelme
静态背景下实例分割数据集自动标注工具，基于Labelme改进。可以自动生成labelme格式的json文件。(注意：本程序只适用于大量图片基于静态背景)原理是：背景减除后得到高质量的二值图，计算连通域外轮廓坐标，再将信息写入json文件。

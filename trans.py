import os
import numpy as np
import scipy.io as sio
import shutil
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import cv2


def make_voc_dir():
    # labels 目录若不存在，创建labels目录。若存在，则清空目录
    if not os.path.exists('../VOC2007/Annotations'):
        os.makedirs('../VOC2007/Annotations')
    if not os.path.exists('../VOC2007/ImageSets'):
        os.makedirs('../VOC2007/ImageSets')
        os.makedirs('../VOC2007/ImageSets/Main')
    if not os.path.exists('../VOC2007/JPEGImages'):
        os.makedirs('../VOC2007/JPEGImages')


if __name__ == '__main__':
# < class_label =1: pedestrians > 行人
                # < class_label =2: riders >      骑车的
                # < class_label =3: partially-visible persons > 遮挡的部分行人
                # < class_label =4: ignore regions > 一些假人，比如图画上的人
                # < class_label =5: crowd > 拥挤人群，直接大框覆盖了

    classes = {'1': 'person',
               '2': 'person',
               '3': 'person',
               '4':'person',
               '5':'person'
               }#这里如果自己只要人，可以把1-5全标记为people，也可以根据自己场景需要筛选
    VOCRoot = '../VOC2007'
    widerDir = '/home/prophetmu/archive2/WiderPerson'  # 数据集所在的路径
    wider_path = '/home/prophetmu/archive2/WiderPerson/test.txt'#这里第一次train，第二次test
    #这个函数第一次用注释掉，后面就要加注释了
    #make_voc_dir()
    with open(wider_path, 'r') as f:
        imgIds = [x for x in f.read().splitlines()]

    for imgId in imgIds:
        objCount = 0  # 一个标志位，用来判断该img是否包含我们需要的标注
        filename = imgId + '.jpg'
        img_path = '/home/prophetmu/archive2/WiderPerson/Images/' + filename
        print('Img :%s' % img_path)
        img = cv2.imread(img_path)
        width = img.shape[1]  # 获取图片尺寸
        height = img.shape[0]  # 获取图片尺寸 360

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'JPEGImages'
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = 'VOC2007/JPEGImages/%s' % filename
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = '%s' % width
        node_height = SubElement(node_size, 'height')
        node_height.text = '%s' % height
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        label_path = img_path.replace('Images', 'Annotations') + '.txt'
        # with open(label_path) as file:
        #     line = file.readline()
        #     count = int(line.split('\n')[0])  # 里面行人个数
        #     line = file.readline()
        #     while line:
        #         cls_id = line.split(' ')[0]
        #         xmin = int(line.split(' ')[1]) + 1
        #         ymin = int(line.split(' ')[2]) + 1
        #         xmax = int(line.split(' ')[3]) + 1
        #         ymax = int(line.split(' ')[4].split('\n')[0]) + 1
        #         line = file.readline()

        #         cls_name = classes[cls_id]

        #         obj_width = xmax - xmin
        #         obj_height = ymax - ymin

        #         difficult = 0
        #         if obj_height <= 6 or obj_width <= 6:
        #             difficult = 1

        #         node_object = SubElement(node_root, 'object')
        #         node_name = SubElement(node_object, 'name')
        #         node_name.text = cls_name
        #         node_difficult = SubElement(node_object, 'difficult')
        #         node_difficult.text = '%s' % difficult
        #         node_bndbox = SubElement(node_object, 'bndbox')
        #         node_xmin = SubElement(node_bndbox, 'xmin')
        #         node_xmin.text = '%s' % xmin
        #         node_ymin = SubElement(node_bndbox, 'ymin')
        #         node_ymin.text = '%s' % ymin
        #         node_xmax = SubElement(node_bndbox, 'xmax')
        #         node_xmax.text = '%s' % xmax
        #         node_ymax = SubElement(node_bndbox, 'ymax')
        #         node_ymax.text = '%s' % ymax
        #         node_name = SubElement(node_object, 'pose')
        #         node_name.text = 'Unspecified'
        #         node_name = SubElement(node_object, 'truncated')
        #         node_name.text = '0'

        image_path = VOCRoot + '/JPEGImages/' + filename
        xml = tostring(node_root, pretty_print=True)  # 'annotation'
        dom = parseString(xml)
        xml_name = filename.replace('.jpg', '.xml')
        xml_path = VOCRoot + '/Annotations/' + xml_name
        # with open(xml_path, 'wb') as f:
        #     f.write(xml)
        # widerDir = '../WiderPerson'  # 数据集所在的路径
        shutil.copy(img_path, '../VOC2007/JPEGImages/' + filename)


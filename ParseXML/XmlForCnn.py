import xml.etree.ElementTree as ET 
import os
from PIL import Image
import shutil
directory = 'C:/Users/gorin/OneDrive/Рабочий стол/UECFOOD100'
dir = os.listdir(directory)

def getmeta(dir):
    list=[]
    filemeta = open(dir)
    for line in filemeta:
        list.append(line);
    return list

classes=getmeta(directory+'/category.txt')
count_class=0
meta_count=1
for id_class in dir:
    print('обработан:',id_class)
    path_xml=directory+'/'+id_class+'/xml/';
    if (os.path.exists(path_xml)):
        shutil.rmtree(path_xml)
    else:
        os.mkdir(path_xml)
    meta=getmeta(directory+'/'+id_class+'/bb_info.txt')
    img_list=os.listdir(directory+'/'+id_class)
    for id_img in img_list: 
        if id_img=='bb_info.txt':
            break
        root = ET.Element("annotation")

        folder = ET.SubElement(root, "folder")
        folder.text=str(id_class)
        filename = ET.SubElement(root, "filename") 
        filename.text=str(id_img)

        size=ET.SubElement(root, "size")
        im = Image.open(directory+'/'+id_class+'/'+id_img)
        size_img = im.size
        width=ET.SubElement(size, "width")
        width.text=str(size_img[0])
        height=ET.SubElement(size, "height")
        height.text= str(size_img[1])
        segmented=ET.SubElement(root, "segmented")
        segmented.text=str(0)
        object=ET.SubElement(root, "object")

        name=ET.SubElement(object, "name")
        name.text=str(classes[count_class])
        bndbox=ET.SubElement(object, "bndbox")
        split_coord=meta[meta_count].split(" ")
        xmin=ET.SubElement(bndbox, "xmin")
        xmin.text=str(split_coord[1])
        ymin=ET.SubElement(bndbox, "ymin")
        ymin.text=str(split_coord[2])
        xmax=ET.SubElement(bndbox, "xmax")
        xmax.text=str(split_coord[3])
        ymax=ET.SubElement(bndbox, "ymax")
        ymax.text=str(split_coord[4][:len(split_coord[4])-1])
        
        tree = ET.ElementTree(root)
        name=str(id_img[:len(id_img)-4])+'.xml'
        tree.write(name)
        path=os.path.abspath(os.curdir)+'\\'+ name
        shutil.move(path, path_xml)
        meta_count=+1
    count_class+=1
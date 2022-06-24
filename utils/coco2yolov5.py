'''
把coco数据集合的所有标注转换到yolov5格式，不改变图片命名方式，
# 注意，原来有一些图片是黑白照片，检测出不是 RGB 图像，这样的图像不会被放到新的文件夹中
'''
from pycocotools.coco import COCO
import os, cv2, shutil
from lxml import etree, objectify
from tqdm import tqdm
from PIL import Image


def save_annotations(image_dir, anno_dir, filename, objs, filepath):
    #图片文件名，不含上级路径
    imgname = filename.split('/')[len(filename.split('/')) - 1]
    txtname = imgname.split('.')[0]

    dst_path = image_dir + "/" + imgname
    img_path = filepath
    try:
        img = cv2.imread(img_path)
    except Exception as e:
        print('image is not available: ', imgname)
        return 0
    else:
        shutil.copy(img_path, dst_path)  # 把原始图像复制到目标文件夹
        #创建label txt
        with open(os.path.join(anno_dir, '{}.txt'.format(txtname)), 'a') as f:
            for obj in objs:
                f.write(str(obj[0]) + ' ' + str(obj[1]) + ' ' + str(obj[2]) + 
                    ' ' + str(obj[3]) + ' ' + str(obj[4]) + '\n')
        return 1
    finally:
        pass
    
    # im = Image.open(img_path)
    # if im.mode != "RGB":
    #     # print(filename + " not a RGB image")
    #     im.close()
    #     return 0
    # im.close()
    


def showbycv(coco, dataType, img, classes, oldId2newId, origin_image_dir, image_dir, anno_dir, verbose=False):
    filename = img['file_name']
    filepath = os.path.join(origin_image_dir, filename)
    I = cv2.imread(filepath)
    image_w = img['width']
    image_h = img['height']
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            """Convert coco format data into yolo format data.
            Note: x,y in coco format are lowest left x and y. x,y in yolo format are center x,y.
            """
            x,y,w,h = ann['bbox']    
            cx = x+w/2.0
            cy = y+h/2.0
            obj = [oldId2newId[ann['category_id']], cx/image_w, cy/image_h, w/image_w, h/image_h]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (x, y), (x+w, y+h), (255, 0, 0))
                # xa = int(cx-w/2)
                # ya = int(cy-h/2)
                # print(xa, ya)
                # cv2.rectangle(I, (xa, ya), (xa+w, ya+h), (255, 0, 0))
                cv2.putText(I, name, (x, y), 3, 1, (0, 0, 255))


    result = save_annotations(image_dir, anno_dir, filename, objs, filepath)


    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)

    return result

def catid2name(coco):  # 将名字和id号建立一个字典
    classes = dict()
    oldId2newId = dict()
    i = 0
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        oldId2newId[cat['id']] = i
        # print(str(cat['id'])+":"+cat['name'])
        i+=1
    return classes, oldId2newId


def inaturalist2yolov5(root_path, image_path, annotation_file, dataType='train'):
    # 递归创建文件夹
    image_dir = os.path.join(root_path, 'images', dataType)  #不使用，直接沿用原来的图片地址
    anno_dir = os.path.join(root_path, 'labels', dataType)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    # annFile = 'instances_{}.json'.format(dataType)
    annpath = os.path.join(root_path, annotation_file)
    coco = COCO(annpath)
    classes, oldId2newId = catid2name(coco)
    imgIds = coco.getImgIds()
    # imgIds=imgIds[0:128] #测试用，抽取10张图片，看下存储效果

    grouptxt = os.path.join(root_path, '{}.txt'.format(dataType)) #保存数据分组txt
    file = open(grouptxt, 'w')

    for imgId in tqdm(imgIds):
        # imgId = imgId[6:12]
        img = coco.loadImgs(imgId)[0]
        result = showbycv(coco, dataType, img, classes, oldId2newId, root_path+image_path, image_dir, 
            anno_dir, verbose=False)
        # print(result)
        if result!=0:
            # file.write(result+'\n')
            # if result == -1:
            file.write(os.path.join('./images', dataType, 
                img['file_name'].split('/')[len(img['file_name'].split('/')) - 1]) + '\n')

    file.close()

if __name__ == "__main__":
    # inaturalist2yolov5('/home/Datasets/iNaturalist/2017', '', 'annotations/train_2017_Insecta_bboxes.json', 
        # 'train_insecta')
        inaturalist2yolov5('/home/Datasets/iNaturalist/2017', '', 'annotations/val_2017_Insecta_bboxes.json', 
        'val_insecta')
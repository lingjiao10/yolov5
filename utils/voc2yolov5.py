import os, cv2, shutil, collections
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from tqdm import tqdm
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List

IP102_CLASSES = ('Rice leaf roller', 'Rice leaf caterpillar', 'Paddy stem maggot', 'Asiatic rice borer', 'Yellow rice borer', 
    'Rice gall midge', 'Rice stemfly', 'Brown plant hopper', 'White backed plant hopper', 'Small brown plant hopper', 
    'Rice water weevil', 'Rice leafhopper', 'Grain spreader thrips', 'Rice shell pest', 'Grub', 'Mole cricket', 'Wireworm', 
    'White margined moth', 'Black cutworm', 'Large cutworm', 'Yellow cutworm', 'Red spider', 'Corn borer', 'Army worm', 
    'Aphids', 'Potosiabre vitarsis', 'Peach borer', 'English grain aphid', 'Green bug', 'Bird cherry-oataphid', 
    'Wheat blossom midge', 'Penthaleus major', 'Longlegged spider mite', 'Wheat phloeothrips', 'Wheat sawfly', 
    'Cerodonta denticornis', 'Beet fly', 'Flea beetle', 'Cabbage army worm', 'Beet army worm', 'Beet spot flies', 
    'Meadow moth', 'Beet weevil', 'Sericaorient alismots chulsky', 'Alfalfa weevil', 'Flax budworm', 'Alfalfa plant bug', 
    'Tarnished plant bug', 'Locustoidea', 'Lytta polita', 'Legume blister beetle', 'Blister beetle', 'Therioaphis maculata buckton', 
    'Odontothrips loti', 'Thrips', 'Alfalfa seed chalcid', 'Pieris canidia', 'Apolygus lucorum', 'Limacodidae', 'Viteus vitifoliae', 
    'Colomerus vitis', 'Brevipoalpus lewisi mcgregor', 'Oides decempunctata', 'Polyphagotars onemus latus', 
    'Pseudococcus comstocki kuwana', 'Parathrene regalis', 'Ampelophaga', 'Lycorma delicatula', 'Xylotrechus', 
    'Cicadella viridis', 'Miridae', 'Trialeurodes vaporariorum', 'Erythroneura apicalis', 'Papilio xuthus', 
    'Panonchus citri mcgregor', 'Phyllocoptes oleiverus ashmead', 'Icerya purchasi maskell', 'Unaspis yanonensis', 
    'Ceroplastes rubens', 'Chrysomphalus aonidum', 'Parlatoria zizyphus lucus', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus', 
    'Tetradacus c bactrocera minax', 'Dacus dorsalis(hendel)', 'Bactrocera tsuneonis', 'Prodenia litura', 'Adristyrannus', 
    'Phyllocnistis citrella stainton', 'Toxoptera citricidus', 'Toxoptera aurantii', 'Aphis citricola vander goot', 
    'Scirtothrips dorsalis hood', 'Dasineura sp', 'Lawana imitata melichar', 'Salurnis marginella guerr', 
    'Deporaus marginatus pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper', 'Rhytidodera bowrinii white', 
    'Sternochetus frigidus', 'Cicadellidae')


def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict

def showbycv(sample, dataType, label_names, origin_image_dir, image_dir, anno_dir, verbose=False):
    filename = sample['file_name']
    filepath = os.path.join(origin_image_dir, filename)
    I = cv2.imread(filepath)
    anns = sample['target']['annotation']
    image_w = int(anns['size']['width'])
    image_h = int(anns['size']['height'])
    # annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    

    objs = []
    for ann in anns['object']:
        name = label_names[int(ann['name'])]
        if 'bndbox' in ann:
            """Convert voc format data into yolo format data.
            Note: x,y in voc format are lowest left x and y, highest right x and y. 
            x,y in yolo format are center x,y,w,h.
            """
            box = ann['bndbox']
            x,y,w,h = (int(box['xmin']), int(box['ymin']), int(box['xmax']) - int(box['xmin']), 
                int(box['ymax']) - int(box['ymin']))  

            cx = x+w/2.0
            cy = y+h/2.0
            obj = [ann['name'], cx/image_w, cy/image_h, w/image_w, h/image_h]
            # print('obj: ', obj)
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (x, y), (x+w, y+h), (255, 0, 0))
                cv2.putText(I, name, (x, y), 3, 1, (0, 0, 255))

    # copy image and save annotation txt
    txtname = filename.split('.')[0]
    dst_path = image_dir + "/" + filename
    shutil.copy(filepath, dst_path)  # 把原始图像复制到目标文件夹
    #创建label txt
    with open(os.path.join(anno_dir, '{}.txt'.format(txtname)), 'a') as f:
        for obj in objs:
            f.write(str(obj[0]) + ' ' + str(obj[1]) + ' ' + str(obj[2]) + 
                ' ' + str(obj[3]) + ' ' + str(obj[4]) + '\n')
    #////////////////////


    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)

    return 1

def  voc2yolov5(root_path, image_path, dataType='train'):
    # 递归创建文件夹
    image_dir = os.path.join(root_path, 'images', dataType)
    anno_dir = os.path.join(root_path, 'labels', dataType)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    #读取list
    id_list_file = os.path.join(root_path, 'ImageSets/Main/{0}.txt'.format(dataType))

    imgIds = [id_.strip() for id_ in open(id_list_file)]
    # imgIds = [id_ for id_ in open(id_list_file)]
    # with open(id_list_file) as f:
    #     ids = f.readlines()
    # imgIds = [x.strip('\n') for x in ids]


    # imgIds = imgIds[0:10]
    # print(imgIds)
    label_names = IP102_CLASSES

    grouptxt = os.path.join(root_path, '{}.txt'.format(dataType)) #保存数据分组txt
    file = open(grouptxt, 'w')

    for imgId in tqdm(imgIds):
        sample = {
        'file_name': imgId + '.jpg',
        'target': parse_voc_xml(ET_parse(os.path.join(root_path, 'Annotations', imgId + '.xml')).getroot()),
        }
        result = showbycv(sample, dataType, label_names, os.path.join(root_path, image_path), image_dir, 
            anno_dir, verbose=True)
        # print(result)
        if result!=0:
            file.write(os.path.join('./images', dataType, imgId + '.jpg') + '\n')


if __name__ == "__main__":
        voc2yolov5('/home/Datasets/IP102_v1.1/Detection/VOC2007', 'JPEGImages', 'trainval')

# using object detection

import torch
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import os
import numpy as np


model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
categories=weights.meta["categories"]
model.eval()
transform = transforms.Compose([
    transforms.ToTensor()
])

def detect_merge(src,des,oclass,model):
    src_tensor=transform(src)
    objs=model([src_tensor])
    bbox=np.array([0,0,0,0])
    # for obj in objs:
    #     if obj['labels'][0]==oclass:
    #         bbox=obj['boxes'][0]
    #         break
    bbox=objs[0]['boxes'][0]
    cropped=src.crop(bbox.cpu().detach().numpy())
    r1=random.uniform(2.5,4.0)
    r1=cropped.width/(des.width/r1)
    cropped=cropped.resize((int(cropped.width/r1),int(cropped.height/r1)))
    rw=int(random.uniform(0,des.width-cropped.width))
    rh=int(random.uniform(0,des.height-cropped.height))                      
    des.paste(cropped,(rw,rh))
    
    return des

path1='./data/wrap_cats'
path2='./data/wrap_hats'
class1=categories.index('cat')
class2=categories.index('dog')
save_path='./data/wrap_cats_hats_od'
if not os.path.exists(save_path):
    os.makedirs(save_path)

imgs_path1=[x for x in Path(path1).iterdir() if x.is_file()]
imgs_path2=[x for x in Path(path2).iterdir() if x.is_file()]



for i in range(6):
    imgs1=random.sample(imgs_path1,k=1)
    imgs2=random.sample(imgs_path2,k=1)
    img1=Image.open(imgs1[0]).convert('RGB')
    img2=Image.open(imgs2[0]).convert('RGB')
    random.seed(1616+i*i*23)
    rd=random.random()
    if rd<0.5:
        des=img1
        src=img2
        oclass=class2
    else:
        des=img2
        src=img1
        oclass=class1
    res=detect_merge(src,des,oclass,model)
    res.save(os.path.join(save_path,f"{i+1}.png"))

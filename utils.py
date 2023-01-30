# use to generate pseudo merged specific images

from PIL import Image
from pathlib import Path
import random
import os

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

path1='./data/wrap_hats'
path2='./data/wrap_cats'
save_path='./data/wrap_cats_hats'
if not os.path.exists(save_path):
    os.makedirs(save_path)

imgs_path1=[x for x in Path(path1).iterdir() if x.is_file()]
imgs_path2=[x for x in Path(path2).iterdir() if x.is_file()]
nsize=(256,256)
for i in range(6):
    imgs1=random.sample(imgs_path1,k=2)
    imgs2=random.sample(imgs_path2,k=2)
    img1_1 = Image.open(imgs1[0])
    if not img1_1.mode == "RGB":
        img1_1 = img1_1.convert("RGB")
    img1_1=img1_1.resize(nsize)
    img1_2 = Image.open(imgs1[1])
    if not img1_2.mode == "RGB":
        img1_2 = img1_2.convert("RGB")
    img1_2=img1_2.resize(nsize)
    
    img2_1 = Image.open(imgs2[0])
    if not img2_1.mode == "RGB":
        img2_1 = img2_1.convert("RGB")
    img2_1=img2_1.resize(nsize)
    img2_2 = Image.open(imgs2[1])
    if not img2_2.mode == "RGB":
        img2_2 = img2_2.convert("RGB")
    img2_2=img2_2.resize(nsize)
    
    imgs=[img1_1,img1_2,img2_1,img2_2]
   
    
    tmp=random.sample(imgs,4)
    
    concat_h1=get_concat_h(tmp[0],tmp[1])
    concat_h2=get_concat_h(tmp[2],tmp[3])
    res=get_concat_v(concat_h1,concat_h2)

    res.save(os.path.join(save_path,f"{i+1}.png"))
        
        
        
        
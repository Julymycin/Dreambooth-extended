[1mdiff --git a/README.md b/README.md[m
[1mindex b05d6f5..d61e7d7 100644[m
[1m--- a/README.md[m
[1m+++ b/README.md[m
[36m@@ -1,6 +1 @@[m
 # Extended-Dreambooth[m
[31m-[m
[31m-## Requirements [m
[31m-```pytorch, diffuse, xformers, accelerate```[m
[31m-[m
[31m-[m
[1mdiff --git a/concepts_list.json b/concepts_list.json[m
[1mindex 0ee7150..6cbb38e 100644[m
[1m--- a/concepts_list.json[m
[1m+++ b/concepts_list.json[m
[36m@@ -1,22 +1,15 @@[m
 [[m
     {[m
[31m-        "instance_prompt":      "photo of sks cat",[m
[31m-        "class_prompt":         "a photo of cat",[m
[31m-        "instance_data_dir":    "data/wrap_cats",[m
[31m-        "class_data_dir":       "data/cats"[m
[32m+[m[32m        "instance_prompt":      "photo of sks dog",[m
[32m+[m[32m        "class_prompt":         "a photo of dog",[m
[32m+[m[32m        "instance_data_dir":    "data/wrap_dogs",[m
[32m+[m[32m        "class_data_dir":       "data/dogs"[m
     },[m
 [m
     {[m
[31m-        "instance_prompt":      "photo of efl hat",[m
[31m-        "class_prompt":         "a photo of hat",[m
[31m-        "instance_data_dir":    "data/wrap_hats",[m
[31m-        "class_data_dir":       "data/hats"[m
[31m-    },[m
[31m-[m
[31m-    {[m
[31m-        "instance_prompt":      "sks cats auq efl hats",[m
[31m-        "class_prompt":         "photo of cat and hat",[m
[31m-        "instance_data_dir":    "data/wrap_cats_hats",[m
[31m-        "class_data_dir":       "data/cats_hats"[m
[32m+[m[32m        "instance_prompt":      "photo of qgys rose",[m
[32m+[m[32m        "class_prompt":         "a photo of rose",[m
[32m+[m[32m        "instance_data_dir":    "data/wrap_roses",[m
[32m+[m[32m        "class_data_dir":       "data/roses"[m
     }[m
 ][m
\ No newline at end of file[m
[1mdiff --git a/train.sh b/train.sh[m
[1mindex abdff1c..b75ba68 100644[m
[1m--- a/train.sh[m
[1m+++ b/train.sh[m
[36m@@ -11,7 +11,7 @@[m [maccelerate launch train_dreambooth.py \[m
   --output_dir=$OUTPUT_DIR \[m
   --revision="fp16" \[m
   --with_prior_preservation --prior_loss_weight=1.0 \[m
[31m-  --seed=1616 \[m
[32m+[m[32m  --seed=1337 \[m
   --resolution=512 \[m
   --train_batch_size=1 \[m
   --train_text_encoder \[m
[36m@@ -23,7 +23,7 @@[m [maccelerate launch train_dreambooth.py \[m
   --lr_warmup_steps=0 \[m
   --num_class_images=50 \[m
   --sample_batch_size=4 \[m
[31m-  --max_train_steps=2400 \[m
[32m+[m[32m  --max_train_steps=1600 \[m
   --save_interval=400 \[m
[31m-  --save_sample_prompt="photo of sks cat wearing efl hat" \[m
[32m+[m[32m  --save_sample_prompt="photo of a sks dog with qgys rose" \[m
   --concepts_list="concepts_list.json"[m
\ No newline at end of file[m
[1mdiff --git a/train_dreambooth.py b/train_dreambooth.py[m
[1mindex 43d830b..9423581 100644[m
[1m--- a/train_dreambooth.py[m
[1m+++ b/train_dreambooth.py[m
[36m@@ -263,7 +263,7 @@[m [mdef parse_args(input_args=None):[m
 [m
     return args[m
 [m
[31m-# generate dataset from the definetion of concepts_list.json[m
[32m+[m
 class DreamBoothDataset(Dataset):[m
     """[m
     A dataset to prepare the instance and class images with the prompts for fine-tuning the model.[m
[1mdiff --git a/utils.py b/utils.py[m
[1mdeleted file mode 100644[m
[1mindex 9a47548..0000000[m
[1m--- a/utils.py[m
[1m+++ /dev/null[m
[36m@@ -1,63 +0,0 @@[m
[31m-# use to generate pseudo merged specific images[m
[31m-[m
[31m-from PIL import Image[m
[31m-from pathlib import Path[m
[31m-import random[m
[31m-import os[m
[31m-[m
[31m-def get_concat_h(im1, im2):[m
[31m-    dst = Image.new('RGB', (im1.width + im2.width, im1.height))[m
[31m-    dst.paste(im1, (0, 0))[m
[31m-    dst.paste(im2, (im1.width, 0))[m
[31m-    return dst[m
[31m-[m
[31m-def get_concat_v(im1, im2):[m
[31m-    dst = Image.new('RGB', (im1.width, im1.height + im2.height))[m
[31m-    dst.paste(im1, (0, 0))[m
[31m-    dst.paste(im2, (0, im1.height))[m
[31m-    return dst[m
[31m-[m
[31m-path1='./data/wrap_hats'[m
[31m-path2='./data/wrap_cats'[m
[31m-save_path='./data/wrap_cats_hats'[m
[31m-if not os.path.exists(save_path):[m
[31m-    os.makedirs(save_path)[m
[31m-[m
[31m-imgs_path1=[x for x in Path(path1).iterdir() if x.is_file()][m
[31m-imgs_path2=[x for x in Path(path2).iterdir() if x.is_file()][m
[31m-nsize=(256,256)[m
[31m-for i in range(6):[m
[31m-    imgs1=random.sample(imgs_path1,k=2)[m
[31m-    imgs2=random.sample(imgs_path2,k=2)[m
[31m-    img1_1 = Image.open(imgs1[0])[m
[31m-    if not img1_1.mode == "RGB":[m
[31m-        img1_1 = img1_1.convert("RGB")[m
[31m-    img1_1=img1_1.resize(nsize)[m
[31m-    img1_2 = Image.open(imgs1[1])[m
[31m-    if not img1_2.mode == "RGB":[m
[31m-        img1_2 = img1_2.convert("RGB")[m
[31m-    img1_2=img1_2.resize(nsize)[m
[31m-    [m
[31m-    img2_1 = Image.open(imgs2[0])[m
[31m-    if not img2_1.mode == "RGB":[m
[31m-        img2_1 = img2_1.convert("RGB")[m
[31m-    img2_1=img2_1.resize(nsize)[m
[31m-    img2_2 = Image.open(imgs2[1])[m
[31m-    if not img2_2.mode == "RGB":[m
[31m-        img2_2 = img2_2.convert("RGB")[m
[31m-    img2_2=img2_2.resize(nsize)[m
[31m-    [m
[31m-    imgs=[img1_1,img1_2,img2_1,img2_2][m
[31m-   [m
[31m-    [m
[31m-    tmp=random.sample(imgs,4)[m
[31m-    [m
[31m-    concat_h1=get_concat_h(tmp[0],tmp[1])[m
[31m-    concat_h2=get_concat_h(tmp[2],tmp[3])[m
[31m-    res=get_concat_v(concat_h1,concat_h2)[m
[31m-[m
[31m-    res.save(os.path.join(save_path,f"{i+1}.png"))[m
[31m-        [m
[31m-        [m
[31m-        [m
[31m-        [m
\ No newline at end of file[m

import cv2
import os
from os import listdir
from os.path import join
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
from torchvision import transforms
from vgg19 import vgg19
from PIL import Image
import clip

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".tif", ".bmp"])

normalization_mean = [0.485, 0.456, 0.406]
normalization_std = [0.229, 0.224, 0.225]

loader  = transforms.Compose([transforms.ToTensor(), 
                              transforms.Resize((224,224)),
                              transforms.Normalize(mean = normalization_mean, std = normalization_std)])

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to('cuda:0')


color_idx = [0, 1]
colors_per_class = [[255, 0, 0], [0, 0, 255]]

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image

def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate

    for image_path, label, x, y in zip(images, labels, tx, ty):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        label_t = 0
        if label == 'True':
            label_t = 1
        else:
            label_t = 0
        image = draw_rectangle_by_class(image, label_t)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    print('done======================')
    cv2.imwrite('/data/chemical/sample/figure/tsne_img_clip_thumb.jpg', tsne_plot)

def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def visualize_tsne(tsne, images, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as images
    visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)


#======================== CLIP show thumbnail ===============================================
#============================================================================================
parent_path = '/data/chemical'
folder = join('/data/chemical', '2021_kuvat')
out_folder = join('/data/chemical', 'sample/2021')
subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

#======================== CLIP show thumbnail ===============================================
label_dic = []
name_list = []
crop_list = []
model, preprocess = clip.load("ViT-B/32", device='cuda:0')
patch_size = 128

out_folder = join('/data/chemical', 'sample/2021')
img_filenames = sorted([x for x in listdir(out_folder) if is_img_file(x)])
start = 0
features = None
for i in img_filenames:
    img_path = join(out_folder, i)
    img = Image.open(img_path)
    width, height = img.size
    for i in [0, (width-patch_size)//2, width-patch_size]:
        for j in [0, (height-patch_size)//2, height-patch_size]:
            left = i
            top = j
            img = img.crop((left, top, left + patch_size, top + patch_size)) # (left, top, right, bottom)
            image = preprocess(img).unsqueeze(0).to('cuda:0')
            with torch.no_grad():
                data = model.encode_image(image)
            current_features = data.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features
            start = start + 1
            label_dic.append('False')
            name_list.append(img_path)

out_folder = join('/data/chemical', 'sample/2023')
img_filenames = sorted([x for x in listdir(out_folder) if is_img_file(x)])
for i in img_filenames:
    img_path = join(out_folder, i)
    img = Image.open(img_path)
    width, height = img.size
    for i in [0, (width-patch_size)//2, width-patch_size]:
        for j in [0, (height-patch_size)//2, height-patch_size]:
            left = i
            top = j
            img = img.crop((left, top, left + patch_size, top + patch_size)) # (left, top, right, bottom)
            image = preprocess(img).unsqueeze(0).to('cuda:0')
            with torch.no_grad():
                data = model.encode_image(image)
            current_features = data.cpu().numpy()
            if features is not None:
                features = np.concatenate((features, current_features))
            else:
                features = current_features
            start = start + 1
            label_dic.append('True')
            name_list.append(img_path)

print('done feature extraction ======================')
tsne = TSNE(n_components=2).fit_transform(features)
print('done t-SNE======================')

visualize_tsne(tsne, name_list, label_dic, plot_size=2000, max_image_size=200)
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
from transformers import AutoImageProcessor, ViTModel
from transformers import ViTImageProcessor

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


color_idx = [0, 1, 2, 3, 4]
colors_per_class = [[255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0], [0, 255, 255]]

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
        # label_t = 0
        # if label == 'True':
        #     label_t = 1
        # else:
        #     label_t = 0
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    print('done======================')
    cv2.imwrite('tsne_img_vit_thumb.jpg', tsne_plot)

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

# #======================== CLIP show thumbnail ===============================================
# label_dic = []
# name_list =[]
# model, preprocess = clip.load("ViT-B/32", device='cuda:0')

# out_folder = join('/data/chemical', 'sample/total')
# img_filenames = sorted([x for x in listdir(out_folder) if is_img_file(x)])
# start = 0
# features = None
# for i in img_filenames:
#     img_path = join(out_folder, i)
#     image = preprocess(Image.open(img_path)).unsqueeze(0).to('cuda:0')
#     with torch.no_grad():
#         data = model.encode_image(image)
#     current_features = data.cpu().numpy()
#     if features is not None:
#         features = np.concatenate((features, current_features))
#     else:
#         features = current_features
#     start = start + 1
#     label_dic.append('False')
#     name_list.append(img_path)

# out_folder = join('/data/chemical', 'sample/2023')
# img_filenames = sorted([x for x in listdir(out_folder) if is_img_file(x)])
# for i in img_filenames:
#     img_path = join(out_folder, i)
#     image = preprocess(Image.open(img_path)).unsqueeze(0).to('cuda:0')
#     with torch.no_grad():
#         data = model.encode_image(image)
#     current_features = data.cpu().numpy()
#     if features is not None:
#         features = np.concatenate((features, current_features))
#     else:
#         features = current_features
#     label_dic.append('True')
#     name_list.append(img_path)
#     start = start + 1


# #======================== HOG show thumbnail ===============================================
# hog = cv2.HOGDescriptor()
# features = None
# label_dic = []
# name_list =[]
# out_folder = join('/data/chemical', 'sample/total')
# img_filenames = sorted([x for x in listdir(out_folder) if is_img_file(x)])
# start = 0
# for i in img_filenames:
#     img_path = join(out_folder, i)
#     img = cv2.imread(img_path, 0)
#     img = cv2.resize(img, [160, 128])
#     h = hog.compute(img)
#     data = h.reshape((1, 49140))
#     if features is not None:
#         features = np.concatenate((features, data))
#     else:
#         features = data

#     label_dic.append('False')
#     name_list.append(img_path)



# #======================== VGG show thumbnail ===============================================
# vgg = vgg19(pre_trained=True, require_grad=False).to('cuda:0')
# features = None
# label_dic = []
# name_list =[]
# out_folder = join('/data/chemical', 'sample/total')
# img_filenames = sorted([x for x in listdir(out_folder) if is_img_file(x)])
# start = 0
# for i in img_filenames:
#     img_path = join(out_folder, i)
#     img = image_loader(img_path)
#     vgg_features = vgg(img)
#     data = getattr(vgg_features, 'pool5')
#     data = data.view(1, -1)
#     data = data.cpu().numpy()
#     if features is not None:
#         features = np.concatenate((features, data))
#     else:
#         features = data

#     label_dic.append('False')
#     name_list.append(img_path)


#======================== VIT show thumbnail ===============================================
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
model.to('cuda:0')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
features = None
out_folder = join('/data/chemical', 'data_v3/train')
label = []
size =[]
image_filenames = []

for i in ['1.0k', '2.0k', '5.0k', '10.0k']:
    data_path = join(out_folder, i)
    scale = float(i.split('.')[0])
    for x in [0, 1, 2, 3, 4]:
        image_path = join(data_path, str(x))
        for y in listdir(image_path):
            if is_img_file(y):
                name = join(image_path, y)
                image_filenames.append(name)
                label.append(x)
                size.append(scale)

start = 0
for i in image_filenames:
    img = Image.open(i).convert("L")
    rgb = Image.merge("RGB", (img, img, img))
    print(i)
    inputs = processor(images=rgb, return_tensors="pt").to('cuda:0')
    pixel_values = inputs.pixel_values
    with torch.no_grad():
        outputs = model(pixel_values)
    last_hidden_states = outputs.last_hidden_state
    data = last_hidden_states.view(1, -1)
    data = data.cpu().numpy()
    if features is not None:
        features = np.concatenate((features, data))
    else:
        features = data

print('done feature extraction ======================')
tsne = TSNE(n_components=2).fit_transform(features)
print('done t-SNE======================')

visualize_tsne(tsne, image_filenames, label, plot_size=2000, max_image_size=100)
import argparse
import matplotlib.pyplot as plt

from colorizers import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str,
                    default='/Users/Public/Restoration-and-recolourization-main/upload_output/final_output')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o', '--img_og_path', type=str,
                    default='/Users/Public/Restoration-and-recolourization-main/input_imgs',
                    help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

image_path_1 = "/Users/Public/Restoration-and-recolourization-main/input_imgs/color"
image_list_1 = os.listdir(image_path_1)

image_path_2 = "/Users/Public/Restoration-and-recolourization-main/input_imgs/bw"
image_list_2 = os.listdir(image_path_2)


for i in range(len(image_list_1)):
    img1 = load_img(os.path.join(opt.img_path, image_list_1[i][:-4] + '.png'))
    img_og = load_img(os.path.join(image_path_1, image_list_1[i]))
    img = np.asarray(Image.fromarray(img1).resize((img_og.shape[1], img_og.shape[0]), resample=3))

    plt.imsave('/Users/Public/Restoration-and-recolourization-main/restored_output/' + str(
        image_list_1[i][:-4]) + '_corrected.png', img)

for i in range(len(image_list_2)):
    img1 = load_img(os.path.join(opt.img_path, image_list_2[i][:-4] + '.png'))
    img_og = load_img(os.path.join(image_path_2, image_list_2[i]))
    img = np.asarray(Image.fromarray(img1).resize((img_og.shape[1], img_og.shape[0]), resample=3))
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    plt.imsave('/Users/Public/Restoration-and-recolourization-main/restored_output/' + str(
        image_list_2[i][:-4]) + '_corrected_1.png', out_img_eccv16)
    plt.imsave('/Users/Public/Restoration-and-recolourization-main/restored_output/' + str(
        image_list_2[i][:-4]) + '_corrected_2.png', out_img_siggraph17)

    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 2, 1)
    # plt.imshow(img_og)
    # plt.title('Original Input')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(img)
    # plt.title('Restored Image')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(out_img_eccv16)
    # plt.title('Colored Output-1')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 4)
    # plt.imshow(out_img_siggraph17)
    # plt.title('Colored Output-2')
    # plt.axis('off')
    # plt.show()


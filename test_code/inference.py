import sys
sys.path.append('./')

import matplotlib.pyplot as plt
import cartoonize
import os


model_path = './saved_models'
load_folder = './source-frames'
save_folder = './cartoonized_images'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

cartoonize.cartoonize(load_folder, save_folder, model_path)

source_image = plt.imread('./source-frames/image.jpg')
cartoonized_image = plt.imread('./cartoonized_images/image.jpg')

plt.subplot(1, 2, 1)
plt.imshow(source_image)
plt.title('Source image')
plt.subplot(1, 2, 2)
plt.imshow(cartoonized_image)
plt.title('Cartoonized image')
plt.show()

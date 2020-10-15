import sys

# sys.path.append('./White-box-Cartoonization/test_code')
import cv2
import cartoonize
import numpy as np
model_path = './saved_models'
load_folder = './source-frames'
save_folder = './cartoonized_images'

sess, final_out, input_photo = cartoonize.cartoonize_realtime(load_folder, save_folder, model_path)


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def read_img():
    image = cv2.imread(f'{load_folder}/image.jpg')
    image = resize_crop(image)
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    cv2.imwrite(f'{save_folder}/image.jpg', output)
    res_img = cv2.imread(f'{save_folder}/image.jpg')
    cv2.imshow('img2', res_img)


def cap_and_predict():
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        cv2.imshow('img', img)
        cv2.imwrite(f'{load_folder}/image.jpg', img)

        read_img()

        # Stop if 'q' key is pressed
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


cap_and_predict()

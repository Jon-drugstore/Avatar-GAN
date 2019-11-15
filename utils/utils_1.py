from __future__ import division
import pprint
import scipy.misc
import numpy as np
import copy
import cv2

try:
    _imread = scipy.misc.imread
except AttributeError:
    from imageio import imread as _imread

pp = pprint.PrettyPrinter()


def adjust_image(image, gamma=1.8, save_path=None):
    if isinstance(image, str):
        image = cv2.imread(image, flags=1)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(dtype=np.uint8)
    image = cv2.LUT(image, table)

    #Face detection and crop
    '''
    CASCADE_PATH = "data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(image, 1.1, 3, minSize=(100, 100))
    i = 0
    (x, y, w, h) = faces[0]
    '''
    #padding image to square and resize to 300x300
    desired_size = 300
    old_size = image.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    new_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    cv2.imwrite("test.jpg", new_img)
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('checkpoint/facedetect/deploy.prototxt.txt', 'checkpoint/facedetect/res10_300x300_ssd_iter_140000.caffemodel')
     
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    (h, w) = new_img.shape[:2]
    blob = cv2.dnn.blobFromImage(new_img, 1.0,
	    (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # get max confidence
    confidences = [detections[0, 0, i, 2] for i in range(0, detections.shape[2])]
    i = max(range(len(confidences)), key=confidences.__getitem__)

    confidence = detections[0, 0, i, 2]
    # compute the (x, y)-coordinates of the bounding box for the object
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    x = startX
    y = startY
    w = endX-startX
    h = endY-startY
    
    r = max(w, h) * 3/2
    centerx = x + w / 2
    centery = y + h / 2
    nx = max(int(centerx - r / 2), 0)
    ny = max(int(centery - r / 2 - r / 16), 0)
    nr = int(r)
    print(r, centerx, centery, nx, ny, nr)

    faceimg = new_img[ny:ny + nr, nx:nx + nr]
    lastimg = cv2.resize(faceimg, (256, 256))
    i += 1
    cv2.imwrite("image%d.jpg" % i, lastimg)


    if save_path is not None:
        cv2.imwrite(save_path, lastimg)
    return lastimg


def convert_grayscale(img, save_path=None):
    if isinstance(img, str):
        img = cv2.imread(img, flags=1)
    # transfer to gray scale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #gray_image = clahe.apply(gray_image)
    # convert uint8 format
    gray_image = gray_image.astype(dtype=np.uint8)
    if save_path is not None:
        cv2.imwrite(save_path, gray_image)
    return gray_image


class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
            return image


def load_test_data(image_path, fine_size=256, is_grayscale=True):
    img = imread(image_path, is_grayscale=is_grayscale)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img / 127.5 - 1
    return img


def load_train_data(image_path, load_size=256, fine_size=256, is_testing=False, is_grayscale=True):
    img_a = imread(image_path[0], is_grayscale=is_grayscale)
    img_b = imread(image_path[1], is_grayscale=is_grayscale)
    if not is_testing:
        img_a = scipy.misc.imresize(img_a, [load_size, load_size])
        img_b = scipy.misc.imresize(img_b, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_a = img_a[h1:h1 + fine_size, w1:w1 + fine_size]
        img_b = img_b[h1:h1 + fine_size, w1:w1 + fine_size]

        if np.random.random() > 0.5:
            img_a = np.fliplr(img_a)
            img_b = np.fliplr(img_b)
    else:
        img_a = scipy.misc.imresize(img_a, [fine_size, fine_size])
        img_b = scipy.misc.imresize(img_b, [fine_size, fine_size])

    img_a = img_a / 127.5 - 1.
    img_b = img_b / 127.5 - 1.
    if is_grayscale:
        img_a = np.reshape(img_a, newshape=(fine_size, fine_size, 1))
        img_b = np.reshape(img_b, newshape=(fine_size, fine_size, 1))
    img_ab = np.concatenate((img_a, img_b), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_ab


def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if is_grayscale:
        return _imread(path, flatten=True).astype(np.float)
    else:
        return _imread(path, mode='RGB').astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def boolean_string(bool_str):
    bool_str = bool_str.lower()

    if bool_str not in {"false", "true"}:
        raise ValueError("Not a valid boolean string!!!")

    return bool_str == "true"

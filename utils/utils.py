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

def ajust_lighting(image, gamma=1.8, save_path=None):
    if isinstance(image, str):
        image = cv2.imread(image, flags=1)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(dtype=np.uint8)
    image = cv2.LUT(image, table)
    if save_path is not None:
        cv2.imwrite(save_path, image)
    return image

def resize_image(image, save_path=None):
    if isinstance(image, str):
        image = cv2.imread(image, flags=1)
        
    # scale
    (h, w) = image.shape[:2]
    scale = 300.0/max(h,w)
    dim = (int(w*scale),int(h*scale))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    # padding
    (h, w) = image.shape[:2]
    dim_diff = h-w
    right = 0
    bottom = 0
    if dim_diff>0:
        right = dim_diff
    else:
        bottom = -dim_diff
    image = cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[255,255,255])
    (h, w) = image.shape[:2]
    dim = (w,h)

    #Face detection and crop
    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe('checkpoint/facedetect/deploy.prototxt.txt', 'checkpoint/facedetect/res10_300x300_ssd_iter_140000.caffemodel')
     
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    #(h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
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
  
    face_w = box[2]-box[0]
    face_h = box[3]-box[1]
    
    # scale
    scale = 116.0/face_w
    dim = (int(scale*w),int(scale*h))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    right = 0
    bottom = 0
    if dim[0]<256:
        right = 256-dim[0]
    if dim[1]<256:
        bottom = 256-dim[1]
    image = cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[255,255,255])
    (h, w) = image.shape[:2]
    dim = (w,h) 
    
    # center face
    left = 70                       #(256-116)/2.0, Face x is from 70 to 186
    top = (256-scale*face_h)/2.0
    x = scale*box[0]
    y = scale*box[1]
    M = np.float32([[1,0,left-x],[0,1,top-y]])
    image = cv2.warpAffine(image,M,dim, borderValue=(255,255,255))
    lastimg = image[0:256,0:256]
    
    if save_path is not None:
        cv2.imwrite(save_path, lastimg)
    return lastimg, int(top)
    
def adjust_image(image, gamma=1.0, val_adj=210, sat_adj=70, save_path=None):
    # color adjustments
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(dtype=np.uint8)
    lastimg = cv2.LUT(image, table)
    
    imghsv = cv2.cvtColor(lastimg, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s[s<sat_adj] = sat_adj
    v[v>val_adj] = val_adj
    imghsv = cv2.merge([h,s,v])
    lastimg = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    
    if save_path is not None:
        cv2.imwrite(save_path, lastimg)
    return lastimg


def convert_grayscale(img, top, med_value=[135,150], save_path=None, brightness=False):
    if isinstance(img, str):
        img = cv2.imread(img, flags=1)
    # convert to gray scale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # adjust contrast
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(12,12))
    gray_image = clahe.apply(gray_image)
    # convert uint8 format
    gray_image = gray_image.astype(dtype=np.uint8)
        
    if brightness:
        # adjust brightness
        orig_med = int(median([gray_image[i,j] for i in range(70,186) for j in range(top, 256-top)]))
        if orig_med < med_value[0]:
            bright_diff = med_value[0]-orig_med
            gray_image[gray_image>255-bright_diff] = 255-bright_diff
        elif orig_med > med_value[1]:
            bright_diff = med_value[1]-orig_med
            gray_image[gray_image<-bright_diff] = -bright_diff 
        else:
            bright_diff = 0     
        gray_image = gray_image + bright_diff    
        gray_image[gray_image>240+bright_diff] = 255
    else:
        bright_diff = 0 
        # convert uint8 format
        gray_image = gray_image.astype(dtype=np.uint8)
    
    if save_path is not None:
        cv2.imwrite(save_path, gray_image)
    return gray_image, bright_diff

def median(lst):
    quotient, remainder = divmod(len(lst), 2)
    if remainder:
        return sorted(lst)[quotient]
    return sum(sorted(lst)[quotient - 1:quotient + 1]) / 2.0
    
    
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

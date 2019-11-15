import cv2
import numpy as np
import tensorflow as tf
from scipy.misc import imread


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 256
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pre-trained deeplab model."""
        self.graph = tf.Graph()

        graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
            image: A PIL.Image object, raw input image.
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        #height, width, channel = image.shape
        #resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        #target_size = (int(resize_ratio * height), int(resize_ratio * width))
        # target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        #resized_image = cv2.resize(image, dsize=target_size)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            # feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
            feed_dict={self.INPUT_TENSOR_NAME: [image]})
        seg_map = batch_seg_map[0]

        return image, seg_map


def draw_segment(base_img, mat_img):
    # width, height = base_img.size
    height, width, channel = base_img.shape
    # dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
    dummy_img = np.zeros([height, width, channel], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = mat_img[y, x]
            b, g, r = base_img[y, x]
            # (r, g, b) = base_img.getpixel((x, y))
            if color == 0:
                dummy_img[y, x] = [255, 255, 255]
                # dummy_img[y, x, 3] = 0
            else:
                dummy_img[y, x] = [b, g, r]
                # dummy_img[y, x] = [r, g, b, 255]
    # img = Image.fromarray(dummy_img)
    img = dummy_img
    return img


def run_bg_removal(img, save_path=None, model_type="mobile_net_model"):
    # model_type = "mobile_net_model" or "xception_model"
    if isinstance(img, str):
        img = imread(img)
    model = DeepLabModel(model_type)
    resized_img, seg_map = model.run(img)
    new_img = draw_segment(resized_img, seg_map)
    # new_img_ = Image.new("RGB", img.size, (255, 255, 255))
    # new_img_.paste(new_img, new_img)
    if save_path is not None:
        cv2.imwrite(save_path, new_img)
        # new_img_.save(save_path)
    # return new_img_
    return new_img

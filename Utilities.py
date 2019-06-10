import tempfile
from wand.image import Image
from wand.color import Color
import numpy as np
import math
import os
import cv2
import gc
from sklearn.cluster import DBSCAN


def scale_img(img_list, ref_shape=(3840, 2160)):
    # selected document
    res_imgs = []
    ref_px = ref_shape[0] * ref_shape[1]
    for img in img_list:
        # Current image resolution
        px = img.shape[0] * img.shape[1]
        # Calculate value to scale img dimensions
        val = math.sqrt(ref_px / px)
        # Resize image
        img_res = cv2.resize(img, (int(img.shape[1] * val), int(img.shape[0] * val)), interpolation=cv2.INTER_CUBIC)
        res_imgs.append(img_res)

    return res_imgs, val

def generate_images(tmp_file,pages):
    """Convert PDFS to images.
    Return: A dictionnary with 'pages' as only key, pages is a list of
    dictionnaries. For every page we keep: the raw image of the page,
    the size of the page, the number of the page.
    """
    img_list = list()
    lst_pages = range(pages-1)
    try:
        for p in lst_pages:
            # Build file name of the image per page.
            # Open, convert, and save page.
            with Image(filename="{}[{}]".format(tmp_file.name, p),
                       resolution=300) as img:

                img.background_color = Color("white")
                img.alpha_channel = 'remove'
                with img.convert('PNG') as converted:   
                    # Don't save page as converted image
                    img_bytes = bytearray(converted.make_blob())
                    img_buffer = np.asarray(img_bytes, dtype=np.uint8)
                    im = cv2.imdecode(img_buffer,flags=cv2.IMREAD_COLOR)

            img_list.append(im)
    except Exception:
        os.remove(tmp_file.name)
        raise
    
    return img_list


def connected_components(img_list, thres=5e3,
                         img_idxs=None, to_grayscale=True):
    """
        Remove elements with more than n pixes connected (thres)
    """
    if img_idxs is None:
        img_idxs = [i for i in range(len(img_list))]

    for idx, im in enumerate(img_list):
        if idx in img_idxs:
            image = im
            if to_grayscale:
                image = im[:, :]
            _, image_bin = cv2.threshold(image, 127, 255,cv2.THRESH_BINARY_INV)
            image_bin = image_bin.astype('uint8')

            comp, out, stats, _ = cv2.connectedComponentsWithStats(image_bin,
                                                                   8)
            sizes = stats[:, -1]
            max_labels = []
            # first component is whole doc
            for i in range(1, comp):
                if sizes[i] > thres:
                    max_labels.append(i)

            new_img = np.zeros(out.shape, dtype=np.uint8)
            for lab in max_labels:
                new_img[out == lab] = 255

            img_list[idx] = new_img + image

    return img_list

def preprocess_images(img_list):
    """
       Apply smoothing and adaptative thresholding
    """
    processed_imgs = []

    for image in img_list:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 7, 100, 100)
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 20)
        processed_imgs.append(binary)

    cleaned = connected_components(processed_imgs, thres=3e3)

    return cleaned
	
	
def to_grayscale(img_list):
    """
    Given an image list, convert all images to grayscale and 
    return a new list
    """
    gray_list = []

    img_idxs = [i for i in range(len(img_list))]

    for idx, im in enumerate(img_list):
        if idx in img_idxs:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_list.append(gray)
    
    return gray_list
	
	

def _is_valid_chunk(w, h, p_size):
    """
    Check whether a chunk is valid or not based on position and size.

    Args:
        :w:  Width of the chunk.
        :h:  Height of the chunk.
        :p_size: Dimensions of the image.

    Returns:
        True if the chunk is valid, False if the chunk is not valid.

    """
    valid = True

    # Area too small, smaller than 75 x 75 pixels
    if w * h <= 25 * 25:
        valid = False

    # Area too large, similar to the page's area
    if w * h >= p_size[0] * p_size[1] * 0.75:
        valid = False

    if w < 30 or h < 35:
        valid = False

    # Extreme aspect ratio, either too tall or too wide
    if min(w, h) / max(w, h) <= 0.03:
        valid = False

    return valid

def check_dims(coord, shape_img, thres, option):
    """
        Check if a padding with pixel of image can be applied
    """
    w, h = shape_img

    if option in ['top', 'left']:
        return coord - thres >= 0
    elif option == 'bottom':
        return coord + thres < h
    elif option == 'right':
        return coord + thres < w
    
def _is_valid_chunk_thres(w, h, p_size, thres=0.02):
    """
    Check whether a chunk is valid or not based on the chunk size.

    Args:
        :w:  Width of the chunk.
        :h:  Height of the chunk.
        :p_size: Dimensions of the image.
        :thres: Threshold to compare the chunks size.

    Returns:
        True if the chunk is valid, False if the chunk is not valid.

    """
    if h*w < thres or min(w, h) / max(w, h) <= 0.04:
        return False
  
    return True

def cluster_extraction(img_list,page_type, img_idxs=None,
                       thres_x=7, thres_y=7,
                       index_start=0, all_chunks_valid=False, 
                       min_chunk_size=800):
    """
        Extract chunks by DBSCAN algorithm.
        There is a different extractions base on type of page (page_type)
    """
    
    processed_img_list = list(img_list)

    # Select type of extraction
    options = {
        'text': {'eps': 25, 'min_samples': 125, 'fx': 0.5, 'fy': 1.0},
        'hr': {'eps': 25, 'min_samples': 125, 'fx': 0.5, 'fy': 2.0}
    }
    args = options[page_type]

    #Convert to grayscale
    try:
        gray_list = to_grayscale(processed_img_list) 
    except:
        gray_list = processed_img_list

    if img_idxs is None:
        img_idxs = [i for i in range(len(gray_list))]

    img_chunk_list = list()

    for im_idx, pro_im in enumerate(gray_list):
        if im_idx in img_idxs:
            # Resize image
            img = cv2.resize(pro_im, None, fx=args['fx'], fy=args['fy'],interpolation=cv2.INTER_AREA)

            # Apply threshold
            _, img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)


            # Transform image to coordinates
            coord = np.where(img != 255)
            X = np.vstack((coord[1], coord[0])).T
            # Free memory
            img = None
            gc.collect()

            # Apply DBSCAN
            db = DBSCAN(eps=args['eps'], min_samples=args['min_samples'],algorithm = 'ball_tree').fit(X)

            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Get values by cluster
            cluster_values = dict()
            for cluster_id in np.unique(labels):
                if cluster_id != -1:
                    values = np.where(labels == cluster_id)[0]
                    cluster_values[cluster_id] = values

            # Create chunks
            lst_chunks_out = list()
            real_shape = pro_im.shape

            for key, values in cluster_values.items():
                cluster_ix = X[values]
                x_positions = np.hsplit(cluster_ix, [0,1])[1]
                y_positions = np.hsplit(cluster_ix, [0,1])[2]
                x1 = int(np.amin(x_positions)/args['fx'])
                if check_dims(x1, real_shape, thres_x, 'left'):
                    x1 -= thres_x
                x2 = int(np.amax(x_positions)/args['fx'])
                if check_dims(x2, real_shape, thres_x, 'right'):
                    x2 += thres_x
                y1 = int(np.amin(y_positions)/args['fy'])
                if check_dims(y1, real_shape, thres_y, 'top'):
                    y1 -= thres_y
                y2 = int(np.amax(y_positions)/args['fy'])
                if check_dims(y2, real_shape, thres_y, 'bottom'):
                    y2 += thres_y

                w = x2 - x1
                h = y2 - y1

                if _is_valid_chunk(w, h, pro_im.shape) or \
                        (all_chunks_valid and self._is_valid_chunk_thres(w, h, pro_im.shape,thres=min_chunk_size)):
                    chk = {'ID': int(key)+index_start,
                           'Prev_Type': 'H',
                           'Type': 'H',
                           'combined': None,
                           'page': im_idx,
                           'position': (x1, y1),
                           'rectangle': {'bottom': y2, 'left': x1,
                                         'right': x2, 'top': y1},
                           'size': (w, h),
                           'text': None
                          }

                    lst_chunks_out.append(chk)

            img_chunk_list.append(lst_chunks_out)
    return img_chunk_list
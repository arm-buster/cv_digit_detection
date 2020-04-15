import h5py
from glob import glob
import os
import cv2
import numpy as np
import random
import pdb
import torch
#  CITATION:
#  I got the following three functions for reading the .mat files from
#  this (https://stackoverflow.com/questions/41176258/h5py-access-data-in-datasets-in-svhn)
#  SO thread (get_box_data, print_attrs, get_name)



def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])



# All the following code is mine and newly written for this project #


def get_file_paths(directory):
    """
    returns all file paths to pngs in directory in sorted order
    """

    data = glob(os.path.join(directory, "*.png"))
    img_indices = []
    
    # get the number in each file name
    for s in data:
        img_indices.append(int("".join([i for i in s if i.isdigit()])))

    data = list(zip(data, img_indices))
    data.sort(key = lambda x: x[1])
    data = [i[0] for i in data]
    return data


def load_bboxes(bbox_file_name, indices = None):
    """
    loads all bounding boxes in indices
    loads all bounding boxes if indices is None
    returns list[dict] of bounding boxes
    """

    # load bounding boxes
    mat_data = h5py.File(bbox_file_name)
    size = mat_data['/digitStruct/name'].size
    boxes = []

    if indices is None:
        indices = range(size)

    for ind in indices:
        boxes.append(get_box_data(ind, mat_data))
    return boxes


def read_in_images(img_paths, greyscale = False):
    """
    reads in the images and converts to greyscale optionally
    """
    
    imgs = [cv2.imread(path) for path in img_paths]
    if greyscale:
        imgs = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in pos]
    return imgs



def get_full_bbox(boxes, scale, img_shapes):
    
    """
    boxes: list of dicts containing bounding boxes
    
    gets bounding box that contains all digit bounding boxes
    returns boxes with this bounding box added - box[i]['full_bbox'] = (left, top, right, bottom)
    """
    
    for i in range(len(boxes)):
        bbox = boxes[i]
        left = []
        right = []
        bottom = []
        top = []
        y,x = img_shapes[i]
        y_scale = y / scale
        x_scale = x / scale

        for j in range(len(bbox['height'])):
            left.append(int(bbox['left'][j]))
            top.append(int(bbox['top'][j]))
            right.append(int(bbox['left'][j] + bbox['width'][j]))
            bottom.append(int(bbox['top'][j] + bbox['height'][j]))
        
        l = round(min(left) / x_scale)
        t = round(min(top) / y_scale)
        r = round(max(right) / x_scale)
        b = round(max(bottom) / y_scale)
        
        boxes[i]['full_bbox'] = (l,t,r,b)
    
    return boxes





def translate_left(box, eps = 2, w_factor = 1):
    """
    shift box to left to try to find next house number
    """
    (x1, y1), (x2, y2) = box
    w = int((x2-x1) * w_factor)
    x1 -= (w+eps)
    x2 -= (w+eps)
    return ((x1, y1), (x2, y2))


def translate_right(box, eps = 2, w_factor = 1):
    """
    shift box to right to try to find next house number
    """
    (x1, y1), (x2, y2) = box
    w = int((x2-x1) * w_factor)
    x1 += (w+eps)
    x2 += (w+eps)
    return ((x1, y1), (x2, y2))




def left_right_search(bb, img, model, trans):
    """
    look to the left and right of each identified bbox
    _left_right_search actually does the work
    """

    bb.sort(key = lambda x: x[0][0])
    leftmost_box = bb[0]
    rightmost_box = bb[-1]
    left = []
    right = []
    for i in range(12):
        left.append(_left_right_search(leftmost_box, img, model, direction = "left", trans = trans,shift= i))
        right.append(_left_right_search(rightmost_box, img, model, direction = "right", trans = trans,shift= i))

    return get_highest_confidence(left)[0], get_highest_confidence(right)[0]


def get_highest_confidence(lst):
    """
    lst: list of from list[(str, confidence score)]
    returns list element with highest confidene score
    """
    lst = [i for i in lst if i[0] != ""]
    if len(lst) == 0:
        return ("", 0)
    conf = [i[1] / len(i[0]) for i in lst]
    ind =  conf.index(max(conf))
    return lst[ind]
    
def get_digits(boxes, images):

    """
    separate out each digit in images
    return 2 list of equal size, one with digits, one with numeric labels
    """

    digits = []
    labels = []
    for i in range(len(boxes)):
        bbox = boxes[i]

        for j in range(len(bbox['height'])):
            l = max(int(bbox['left'][j]), 0)
            t = max(int(bbox['top'][j]), 0)
            r = min(int(bbox['left'][j] + bbox['width'][j]), images[i].shape[1])
            b = min(int(bbox['top'][j] + bbox['height'][j]), images[i].shape[0])
            digits.append(images[i][t:b, l:r, :])
            if not all([i != 0 for i in digits[-1].shape]):
                pdb.set_trace()
            lab = bbox['label'][j]
            labels.append(lab if lab != 10 else 0)
    return digits, labels
    

def _left_right_search(box, img, model, direction, trans, shift = 2, w_size = 1, confidence = 0, depth = 0,  found_digits = ""):
    """
    each function call takes a single box and direction, left or right to search in
    makes 10-20 recursive calls to search cutouts in the neighbood directly to the left
    or right depending on the value of direction

    returns either the longest, most frequent, or highest confidence string of digits that results
    from these recursive calls, and all of their recursive calls

    box: current bbox
    img: full image
    direction: left or right
    trans: image transforms to be applied for each model prediction
    shift: how many pixels in between this bbox and next bbox to skip
    w_size: factor to multiply width of next box by
    confidence: summed confidence of all boxes found so far
    """
    assert direction in {"left", "right"}, "direction must be left or right"

    translate_funcs = {'left': translate_left, 'right':translate_right}
    
    (x1, y1), (x2, y2) = translate_funcs[direction](box, shift, w_size)
    cutout = img[y1:y2, x1:x2,:]
    if not all([i>0 for i in cutout.shape]):
        return found_digits, confidence

    indices, predictions, conf = model([cutout])
    if len(indices) > 0:
        result = get_predictions([((x1, y1), (x2, y2))], indices, predictions)
        if direction == "left": # add to left of found digits
            found_digits = result + found_digits 
        elif direction == "right": # add to right of found digits
            found_digits = found_digits + result
        confidence += conf[0]

        results = []
        for w_size in np.arange(0.5, 1.5, 0.3):
            for sh in range(0, 8, 2):
                results.append(_left_right_search(((x1, y1),
                              (x2, y2)), img,
                              model, direction,
                              trans, sh,
                              w_size, confidence, depth+1, found_digits))
        return get_highest_confidence(results)
        
    else:
        return found_digits, confidence
        




def get_cutouts(vstep, hstep, wsize, images, greyscale = False):
    """
    vstep: how many vertical cutouts, for each horizontal cutout
    to return for each image
    hstep: how many horizontal cutouts
    wsize: tuple, (x, y) proportion of x size and y size that window should take up
    images: list of images

    return list of cutouts
    """
    cutouts = []
    coords = []
    for im in images:
        xsize, ysize = int(wsize[0] * im.shape[1]), int(wsize[1] * im.shape[0])
        ystep, xstep = (im.shape[0] - ysize) // vstep, (im.shape[1] - xsize) // hstep

        for y in range(0, im.shape[0], ystep)[:vstep+1]:
            for x in range(0, im.shape[1], xstep)[:hstep+1]:
                cutouts.append(im[y:y+ysize, x:x+xsize, :])
                coords.append((int(x+0.5*xsize), int(y+0.5*ysize)))

    if greyscale:
        return [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in cutouts], coords
    else:
        return cutouts, coords



def write_mser_image(bboxes, mser_ids, img, imgname):
    colors = [(0,0,255), (0,255,0), (255, 0, 0), (255, 0, 255)]
    for ix, (a,b) in enumerate(bboxes):
        cv2.rectangle(img, a, b, colors[mser_ids[ix]], 1)
    cv2.imwrite(os.path.join("graded_images", imgname), img)



def create_mser():
    """
    initializes opencv MSER object and returns it
    """
    mser = cv2.MSER_create(_delta = 4,
    _max_variation = 0.4,
    _min_diversity = 5)
    mser.setMinArea(68)
    mser.setMaxArea(177)
    return mser



def apply_mser(img, mser, imgname):
    """
    applies mser object passed in to img, returns ROIs

    img: a numpy array (image) to run mser on
    mser: cv2 mser object, already initialized
    imgname: name of image to write to file
    """
    
    temp = img.copy()
    images = []
    size_factors = [1]
    temp = cv2.resize(temp, (1024, 1024))
    images.append(temp)
    rectangle_ids = []

    current = temp.copy()
    for i in range(3):
        current = cv2.pyrDown(current).copy()
        images.append(current)
        #print(images[-1].shape)
        size_factors.append(2 ** (i+1))
    
    rec = []
    for ix, (im, s) in enumerate(zip(images, size_factors)):
        rectangles = _apply_mser(im, mser, s)
        rec.extend(rectangles)
        rectangle_ids.extend([ix] * len(rectangles))
    #    for a,b in rectangles:
    #        cv2.rectangle(img, a, b, (0,255,0), 1)
    #cv2.imwrite(os.path.join("mser_images", imgname), img)

    return rec, rectangle_ids



def _apply_mser(img, mser, size_factor):
    """
    runs opencv mser algorithm on input images, returns bounding boxes with
    aspect ratio between upper and lower bound
    """
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions = mser.detectRegions(grey)
    rec = [cv2.boundingRect(region.reshape(-1,1,2)) for region in regions[0]]
    rec = [(x,y,w,h) for (x,y,w,h) in rec if 1.17 < h/w < 3]
    rec = [((x * size_factor,y * size_factor), ((x+w)*size_factor, (y+h)*size_factor)) for (x,y,w,h) in rec]
    rec = list(set(rec))
    return rec



def get_mser_cutouts(bboxes, img):
    """
    bboxes - list of bbox tuples, list[(topx, topy), (bottomx, bottomy)]
    retuns the image cutouts inside bbox, resized to 256, 256
    """
    cutouts = []
    for (x1, y1), (x2, y2) in bboxes:
        cutout = img[y1:y2, x1:x2, :]
        cutout = cv2.resize(cutout, (256,256))
        cutouts.append(cutout.copy())
    return cutouts

def get_model_prediction_and_confidence(output):
    """
    output of model, shape n_patches x n_outputs
    """
    pred = torch.argmax(output,axis=1)
    smax_sorted, _ = output.sort(axis=1, descending = True)
    confidence = smax_sorted[:,0] - smax_sorted[:,1]
    return pred, confidence.detach()



def post_mser_nms(proposals, t, mser_ids, model_scores = None):
    """
    proposals: list[((x1,y1), (x2,y2))] of bounding boxes

    CITATION: I got some help with this function from
    I wrote the code, but used the algorithm explained at this link
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    

    keep = []
    xleft, xright, ytop, ybottom = [], [], [], []
    for (xl, yt), (xr, yb) in proposals:
        xleft.append(xl)
        xright.append(xr)
        ytop.append(yt)
        ybottom.append(yb)

    xleft = np.array(xleft)
    xright = np.array(xright)
    ytop = np.array(ytop)
    ybottom = np.array(ybottom)


    if model_scores is not None:
        idx = np.argsort(model_scores)
    else:
        idx = np.argsort(ybottom)

    area = (xright - xleft + 1) * (ybottom - ytop + 1)
    while len(idx) > 0:

        current_idx = len(idx)-1
        current_box = idx[current_idx]
        keep.append(current_box)

        xl_max = np.maximum(xleft[current_box], xleft[idx[:-1]])
        yt_max = np.maximum(ytop[current_box], ytop[idx[:-1]])
        xr_max = np.minimum(xright[current_box], xright[idx[:-1]])
        yb_max = np.minimum(ybottom[current_box], ybottom[idx[:-1]])

        x_overlap = xr_max - xl_max + 1
        y_overlap = yb_max - yt_max + 1
        
        if isinstance(x_overlap, np.ndarray):
            x_overlap[x_overlap < 0] = 0
            y_overlap[y_overlap < 0] = 0
        else:
            x_overlap = max(x_overlap, 0)
            y_overlap = max(y_overlap, 0)

        overlap_ratio = x_overlap * y_overlap / (area[idx[:-1]])
        idx = np.delete(idx, np.concatenate(([len(idx) - 1], np.where(overlap_ratio > t)[0])))
    
    proposals = np.array(proposals)
    proposals = proposals[np.array(keep)]
    proposals = [(tuple(x[0]), tuple(x[1])) for x in proposals]
    if model_scores is not None:
        return proposals, model_scores[np.array(keep)]
    else:
        return proposals, list(np.array(mser_ids)[np.array(keep)])

def get_predictions(bboxes, indices, pred):
    """
    returns predicted values from left to right or from top
    to bottom if standard deviation of x coordinates of bounding
    boxes is small
    """
    bboxes = np.array(bboxes)
    bboxes = bboxes[indices]
    x_left = [x[0][0] for x in bboxes]
    y_left = [x[0][1] for x in bboxes]
    #print("stdev: %f" % np.std(x_left))
    if 0 < np.std(x_left) < 2:
        indices = indices[np.argsort(y_left)]
    else:
        indices = indices[np.argsort(x_left)]
    result = list(pred[indices])
    result = [str(int(i)) for i in result]
    result = "".join(result)
    return result

def add_noise(img, k):
    noise = np.zeros(img.shape)
    cv2.randu(noise, 0, 256)
    img+=np.array(noise * k, dtype = np.uint8)
    return img


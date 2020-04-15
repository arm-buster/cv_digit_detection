from glob import glob
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from stage1.utils import *
from stage2.stage2_utils import ModelEnsemble
import pdb
from time import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--images_directory", type=str, default = "input_images")

mser = create_mser()
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

model = torch.load("saved_models/regular.pth", map_location=torch.device('cpu')) # takes normal images
model_v = torch.load("saved_models/v_flipped.pth", map_location=torch.device('cpu')) # takes vertically flipped images
model_h = torch.load("saved_models/h_flipped.pth", map_location=torch.device('cpu')) # takes horizontall flipped images
model_hv = torch.load("saved_models/hv_flipped.pth", map_location=torch.device('cpu')) # takes vertically and horizontally flipped images


def predict_ensemble(img, models, mser, trans, imgname):
        
    img = cv2.resize(img, (1024, 1024))
    bboxes, mser_ids = apply_mser(img, mser, imgname)

    # non maximum supression
    bboxes, mser_ids = post_mser_nms(bboxes, 0.15, mser_ids)
    write_mser_image(bboxes, mser_ids, img.copy(), imgname)
    cutouts_normal = get_mser_cutouts(bboxes, img)

    # save each transformation as lambda function
    flips = (lambda x: x, lambda x: cv2.flip(x, 0), lambda x: cv2.flip(x, 1), lambda x: cv2.flip(cv2.flip(x, 1), 0))
    
    
    ensemble = ModelEnsemble(models, trans, flips)
    indices, predictions, _ = ensemble(cutouts_normal)

    selected_bboxes = [bboxes[i] for i in indices]
    if len(indices) > 0:
        result = get_predictions(bboxes, indices, predictions)
        additional_left, additional_right = left_right_search(selected_bboxes, img, ensemble, trans)
        result = additional_left + result + additional_right
    else:
        result = ""
    
    return result



if __name__ == "__main__":
    args = parser.parse_args()

    assert os.path.exists(args.images_directory), "folder %s does not exist!" % args.images_directory

    if not os.path.exists("graded_images"):
        os.makedirs("graded_images")

    test_houses = glob(os.path.join(args.images_directory, "*.png"))
    test_houses = sorted(test_houses)
    for path in test_houses:
        img = cv2.imread(path)
        fname = path.split("/")[-1]
        result = predict_ensemble(img, (model, model_v, model_h, model_hv), mser, trans, fname)
        print("%s: %s" % (fname, result))


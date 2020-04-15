import cv2
import os
from stage1.utils import *
import json
import random

prop_negative_train = 3
prop_negative_test = 4


if __name__ == "__main__":

    greyscale = False
    scale = 64
    # load train and test sets
    train = get_file_paths("data/train")
    train_boxes = load_bboxes('data/train/digitStruct.mat')
    pos_train = read_in_images(train, greyscale = greyscale)


    test = get_file_paths("data/test")
    test_boxes = load_bboxes('data/test/digitStruct.mat')
    pos_test = read_in_images(test, greyscale = greyscale)
    
    # keep only 4000 examples in validation set, move the rest to trainset
    pos_train += pos_test[4000:]
    pos_test = pos_test[:4000]
    train_boxes += test_boxes[4000:]
    test_boxes = test_boxes[:4000]

    pos_test, test_digits = get_digits(test_boxes, pos_test)
    pos_train, train_digits = get_digits(train_boxes, pos_train)

    # how to resize bounding boxes
    pos_train = [cv2.resize(image, (64, 64)) for image in pos_train]
    pos_test = [cv2.resize(image, (64, 64)) for image in pos_test]

    # generate negative image patches (patches number in them)
    if os.path.exists("data/negative_examples_train"):
        neg_train_full_images = [cv2.imread(k) for k in glob("data/negative_examples_train/*.png")]
        neg_test_full_images = [cv2.imread(k) for k in glob("data/negative_examples_test/*.png")]

        neg_train, coords = get_cutouts(56, 56, (0.04, 0.04), neg_train_full_images)
        neg_train = [cv2.resize(image, (64, 64)) for image in neg_train]
        neg_test, coords = get_cutouts(38, 38, (0.04, 0.04), neg_test_full_images)
        neg_test = [cv2.resize(image, (64, 64)) for image in neg_test]
        
        n_negative_test = int(len(pos_test) * prop_negative_test)
        neg_test_indices = random.sample(range(len(neg_test)), n_negative_test)
        neg_test = [neg_test[i] for i in neg_test_indices]

        if greyscale:
            neg_train = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in neg_train]
            neg_test = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in neg_test]
    else:
        neg_train, neg_test = [], []

    
    # combine positive labels, negative labels
    train_digits += [10] * len(neg_train)
    test_digits += [10] * len(neg_test)

    # combine positive examples, negative examples
    train = pos_train + neg_train
    test = pos_test + neg_test

    # shuffle train and test sets
    reorder_ind_train = list(range(len(train)))
    reorder_ind_test = list(range(len(test)))
    random.shuffle(reorder_ind_train)
    random.shuffle(reorder_ind_test)
    train = [train[i] for i in reorder_ind_train]
    train_digits = [train_digits[i] for i in reorder_ind_train]
    test = [test[i] for i in reorder_ind_test]
    test_digits = [test_digits[i] for i in reorder_ind_test]

    if not os.path.exists("data/preprocessed"): # training script expects this file structure
        os.mkdir("data/preprocessed")
        os.mkdir("data/preprocessed/testdigits")
        os.mkdir("data/preprocessed/traindigits")


    with open("data/preprocessed/testdigits/labels.json", 'w') as fp:
        json.dump(test_digits, fp)
    with open("data/preprocessed/traindigits/labels.json", 'w') as fp:
        json.dump(train_digits, fp)
    
    train = np.stack(train)
    test = np.stack(test)
    np.save("data/preprocessed/testdigits/test_reduced_negative.npy", test, fix_imports = False)
    np.save("data/preprocessed/traindigits/train_reduced_negative.npy", train, fix_imports = False)
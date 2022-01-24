# import the necessary packages
import os
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2
import skimage
import string
from math import sqrt
from statistics import mode


def process_lp_text(text):
    if not text[0].isalnum():
        return False
    if text[2].isnumeric():
        return False
    try:
        a = int(text.replace(text[2], ""))
    except:
        return False
    if len(text) < 7 or len(text) > 9:
        return False
    return text


def most_common(lst):
    return mode(lst)


def distance_calc(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_4_points(contours, w, h):
    list_point = np.squeeze(contours)
    # contours = list_point.tolist()
    list_1, list_2, list_3, list_4 = [], [], [], []

    for point in list_point:
        x, y = point
        list_1.append(distance_calc(0, 0, x, y))
        list_2.append(distance_calc(w, 0, x, y))
        list_3.append(distance_calc(w, h, x, y))
        list_4.append(distance_calc(0, h, x, y))
    points = []
    points.append(list_point[list_1.index(min(list_1))])
    points.append(list_point[list_2.index(min(list_2))])
    points.append(list_point[list_3.index(min(list_3))])
    points.append(list_point[list_4.index(min(list_4))])

    return points


def process_lp(plate):
    # plate = cv2.imread(r"D:\IC-Lab\yolor_tracking\ANPR\LP_from_video\72.jpg")
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    thresh_value = skimage.filters.threshold_otsu(gray)
    thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, np.ones((2, 2), dtype=np.uint8))
    thresh = cv2.dilate(thresh, np.ones((2, 2), dtype=np.uint8))
    # V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
    # T = threshold_local(V, 29, offset=10, method="gaussian")
    # thresh = (V > T).astype("uint8") * 255
    # thresh = cv2.bitwise_not(thresh)
    plate = imutils.resize(plate, width=400)
    thresh = imutils.resize(thresh, width=400)
    # cv2.imshow("Thresh", thresh)
    lp_text = ""
    if is_LP_square(thresh):
        LP = bboxes_square_LP(thresh)
    else:
        LP = bboxes_rec_LP(thresh)
    lp_text = ""
    for x, y, w, h in LP:
        crop = plate[y:y + h, x:x + w]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.threshold(crop, 50, 255, cv2.THRESH_BINARY_INV)[1]
        crop = add_extend(crop)
        crop = cv2.resize(crop, dsize=None, fx=3, fy=3)
        cv2.rectangle(plate, (x - 1, y - 1), (x + w + 1, y + h + 1), (0, 0, 255), 2)

    return thresh, plate, lp_text


def is_LP_square(image):
    H, W = image.shape[:2]
    if W / H > 2.5:
        return False
    return True


def CCA(image):
    H, W = image.shape[:2]
    output = cv2.connectedComponentsWithStats(
        image, 2, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    bboxes = []
    for i in range(2, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w * h < 50:
            continue
        if H / h > 10:
            continue
        if (y <= 10 or y + h > H - 10) and (x < 10 or x + w > W - 10):
            continue
        bboxes.append([x, y, w, h])
    return bboxes


def bboxes_square_LP(image):
    bboxes = CCA(image)
    y_mean = sum([i[1] for i in bboxes]) / len(bboxes)
    line1 = []
    line2 = []
    for bbox in bboxes:
        x, y, w, h = bbox
        if y < y_mean:
            line1.append([x, y, w, h])
        else:
            line2.append([x, y, w, h])
    line1 = sorted(line1)
    line2 = sorted(line2)
    bboxes_LP = line1 + line2
    return bboxes_LP


def bboxes_rec_LP(image):
    bboxes = CCA(image)
    return sorted(bboxes)


def process_LP(image):
    if is_LP_square(image):
        LP = bboxes_square_LP(image)
    else:
        LP = bboxes_rec_LP(image)
    return LP


def add_extend(image, type="black", size=5):
    H, W = image.shape[:2]
    if type == "black":
        blank_image = np.zeros((H + size * 2, W + size * 2), np.uint8)
    elif type == "white":
        blank_image = np.full((H + size * 2, W + size * 2), 255, np.uint8)
    blank_image[size:H + size, size:W + size] = image
    return blank_image


def process_bboxes(bboxes):
    try:
        bboxes_new = []
        bboxes = sorted(bboxes, key=lambda box: box[2])
        w_mean = bboxes[2][2]
        for x, y, w, h in bboxes:
            if w / w_mean > 1.7:
                bboxes_new.append([x, y, w // 2, h])
                bboxes_new.append([x + w // 2, y, w // 2, h])
            else:
                bboxes_new.append([x, y, w, h])
        return sorted(bboxes_new, key=lambda box: box[0])
    except:
        return sorted(bboxes, key=lambda box: box[0])


if __name__ == "__main__":
    path = r"D:\IC-Lab\yolor_tracking\ANPR\LP_from_video"
    for imageName in os.listdir(path):
        image = cv2.imread(f"{path}/{imageName}")
        plate, lp_text = process_lp(image)
        print(lp_text)
        cv2.imshow("Image", plate)
        cv2.waitKey()

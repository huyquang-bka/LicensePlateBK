import time

import cv2
import numpy as np
import string

from lib_detection import load_model, detect_lp, im2single
import easyocr
import imutils
from Plate import *

allow_list = string.ascii_uppercase + string.digits
reader = easyocr.Reader(lang_list=["en"])
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 400
Dmin = 200

digit_w = 20  # Kich thuoc ki tu
digit_h = 40  # Kich thuoc ki tu

data_path = r"D:\Lab IC\yolov4-tf-LP\LP_from_video"
video_path = r"D:\LP_BK\2021-12-23\goc_thap_nhat.mp4"
video_path_2 = r"D:\LP_BK\2021-12-23\goc_gan_thap.mp4"
video_path_3 = r"D:\LP_BK\2021-12-23\ch01_00000000095000000.mp4"
cap = cv2.VideoCapture(video_path)
# count = 8500
fourcc = 'mp4v'  # output video codec
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_writer = cv2.VideoWriter(r"D:\Lab IC\demo\bien_so_cam_thap_nhat.mp4", cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
count = 1000
cap.set(cv2.CAP_PROP_POS_FRAMES, count)
# Đọc file ảnh đầu vào
# for imageName in os.listdir(data_path):
check = 1
LP = []
lp_text = ""
left_text = 1

while True:
    count += 1
    # print("Loading: ", count)
    ret, Ivehicle = cap.read()
    image_copy = Ivehicle.copy()
    x1_box, y1_box, x2_box, y2_box = 50, 200, 1385, 568
    # x1_box, y1_box, x2_box, y2_box = 200, 200, 1385, 568
    Ivehicle = Ivehicle[y1_box: y2_box, x1_box:x2_box]
    time_check = 0

    if check == 1:
        try:
            t = time.time()
            # Ivehicle = cv2.imread(f"{data_path}/{imageName}")

            # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
            ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
            side = int(ratio * Dmin)
            bound_dim = min(side, Dmax)

            _, LpImg, lp_type, cor = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.8);
            print(lp_type)
            # Cau hinh tham so cho model SVM

            if not (len(LpImg)):
                continue
                # Chuyen doi anh bien so
            time_check = t
            roi = cv2.convertScaleAbs(LpImg[0], alpha=255.0)
            roi = imutils.resize(roi, width=400)
            print(roi.shape)
            # cv2.imwrite(f"ROI/{count}.jpg", roi)
            # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # gray = cv2.bilateralFilter(gray, 15, 30, 30)
            # # Ap dung threshold de phan tach so va nen
            #
            # # Segment kí tự
            # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            # thre_mor = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel3)
            #
            # cv2.imshow("thre_mor", thre_mor)
            cv2.imshow("Roi", roi)
            #
            # squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # # gray = cv2.bilateralFilter(gray, 15, 50, 50)
            # light = cv2.morphologyEx(gray, cv2.MORPH_DILATE, squareKern)
            # # light = cv2.threshold(light, 0, 255,
            # #                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # thresh = cv2.adaptiveThreshold(light, 255,
            #                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
            # canny = cv2.Canny(gray, 100, 200)
            # cv2.imshow("Light", light)
            # cv2.imshow("Canny", canny)
            # cv2.imshow("Adaptive Threshold", thresh)
            #
            # binary = cv2.threshold(thre_mor, 0, 255,
            #                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            #
            # cv2.imshow("Anh bien so sau threshold", binary)
            if lp_type == 2:
                print("Square")
                H, W = roi.shape[:2]
                roi1 = roi[:H // 2, :]
                roi2 = roi[H // 2:, :]
                text1 = reader.readtext(roi1, detail=0, allowlist=allow_list)[0].strip()
                text2 = reader.readtext(roi2, detail=0, allowlist=allow_list)[0].strip()
                print(text1 + text2)
                text = process_lp_text(text1 + text2)
                if text:
                    LP.append(text)
            else:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 15, 30, 30)
                # Ap dung threshold de phan tach so va nen
                cv2.imshow("Gray", gray)
                # Segment kí tự
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                thre_mor = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel3)
                light = cv2.threshold(thre_mor, 0, 255,
                                      cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                cv2.imshow("Light", light)
                bboxes = bboxes_rec_LP(light)
                text = ""
                for x, y, w, h in bboxes:
                    crop = light[y:y + h, x:x + w]
                    # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    # crop = cv2.threshold(crop, 50, 255, cv2.THRESH_BINARY_INV)[1]
                    crop = add_extend(crop)
                    crop = cv2.resize(crop, dsize=None, fx=3, fy=3)
                    text1 = reader.readtext(crop, detail=0, allowlist=allow_list)[0].strip()
                    text += text1
                    # cv2.imshow("Crop", crop)
                    # cv2.waitKey()
                text = process_lp_text(text)
                if text:
                    print(text)
                    LP.append(text)
            print(time.time() - t)
        except:
            pass
        cv2.rectangle(image_copy, (20, 70), (220, 110), (0, 0, 0), -1)
        if left_text == 1:
            cv2.putText(image_copy, lp_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(image_copy, lp_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", image_copy)
    else:
        cv2.rectangle(image_copy, (20, 70), (220, 110), (0, 0, 0), -1)
        if left_text == 1:
            cv2.putText(image_copy, lp_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(image_copy, lp_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", image_copy)
    # print(LP)
    if time.time() - time_check > 2 and LP:
        time_check = time.time()
        text = most_common(LP)
        if text:
            lp_text = text
            left_text = -left_text
        with open("testLP.txt", "w+") as f:
            f.write(f"{count}: {text}\n")
        LP = []
    vid_writer.write(image_copy)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == 32:
        cv2.waitKey()
    elif key == ord("c"):
        check = -check
    # elif key == 2:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, count + 250)
    # elif key == 3:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, count - 250)

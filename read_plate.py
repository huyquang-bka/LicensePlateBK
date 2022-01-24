import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
import os


# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


# Dinh nghia cac ky tu tren bien so
char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'


# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString


wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
img_path = r"D:\Lab IC\LP\data17-7\data17-7-2021\1_1.jpg"

digit_w = 30  # Kich thuoc ki tu
digit_h = 60  # Kich thuoc ki tu

####Load SVM
# model_svm = cv2.ml.SVM_load('svm.xml')

data_path = r"D:\Lab IC\LP\data17-7\data17-7-2021"
# Đọc file ảnh đầu vào
count = 0
video_path = r"D:\LP_BK\2021-12-23\ch01_00000000104000000.mp4"
cap = cv2.VideoCapture(video_path)
# for imageName in os.listdir(data_path):
while True:
    count += 1
    print("Loading: ", count)
    # Ivehicle = cv2.imread(f"{data_path}/{count}_{count}.jpg")
    ret, Ivehicle = cap.read()

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _, LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)

    # Cau hinh tham so cho model SVM

    if not (len(LpImg)):
        continue
        # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

    roi = LpImg[0]
    cv2.imshow("Roi", roi)
    cv2.waitKey(1)
    # # Chuyen anh bien so ve gray
    # gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)
    #
    # # Ap dung threshold de phan tach so va nen
    # binary = cv2.threshold(gray, 0, 255,
    #                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #
    # cv2.imshow("Anh bien so sau threshold", binary)
    # cv2.waitKey()
    #
    # # Segment kí tự
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    #
    # cv2.imshow("thre_mor", thre_mor)
    # cv2.waitKey()
    # cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # plate_info = ""
    #
    # for c in sort_contours(cont):
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     raito = h / w
    #     height, width = gray.shape
    #     if height / h > 4:
    #         continue
    #     if raito < 1.2:
    #         continue
    #     if width / float(w) < 5:
    #         continue
    #     area = h * w
    #     if area < 100:
    #         continue
    #     if h / roi.shape[0] >= 0.6:  # Chon cac contour cao tu 60% bien so tro len
    #
    #         # Ve khung chu nhat quanh so
    #
    #         # Tach so va predict
    #         curr_num = thre_mor[y - 2:y + h + 2, x - 2:x + w + 2]
    #         curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
    #         _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
    #         cv2.imshow("Digits", curr_num)
    #         cv2.waitKey()
    #         curr_num = np.array(curr_num, dtype=np.float32)
    #         curr_num = curr_num.reshape(-1, digit_w * digit_h)
    #
    #         # Dua vao model SVM
    #         result = model_svm.predict(curr_num)[1]
    #         result = int(result[0, 0])
    #
    #         if result <= 9:  # Neu la so thi hien thi luon
    #             result = str(result)
    #         else:  # Neu la chu thi chuyen bang ASCII
    #             result = chr(result)
    #
    #         plate_info += result
    #
    # cv2.imshow("Cac contour tim duoc", roi)
    # cv2.waitKey()
    #
    # # Viet bien so len anh
    # cv2.putText(Ivehicle, fine_tune(plate_info), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255),
    #             lineType=cv2.LINE_AA)
    #
    # # Hien thi anh
    # print("Bien so=", plate_info)
    # cv2.imshow("Hinh anh output", Ivehicle)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single
import os
import pytesseract
import re

# Ham sap xep contour tu trai sang phai
def sort_contours(cnts):

    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

# Dinh nghia cac ky tu tren bien so
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    index = 0
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
            if lp[i].isdigit():
                index = i
    if index > 2 and newString[0] == "1":
        newString = newString[1:]
    newString = list(newString)
    if newString[2] == "6":
        newString[2] = "G"
    elif newString[2] == "0":
        newString[2] = "D"
    elif newString[2] == "4":
        newString[2] = "A"
    elif newString[2] == "7":
        newString[2] = "Y"
    result = "".join(newString)
    if len(result) > 8 and result[-1] == 1:
        result = result[:-1]
    return result


wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

# Đường dẫn ảnh, các bạn đổi tên file tại đây để thử nhé
img_path = r"D:\Lab IC\LP\data17-7\data17-7-2021\1_1.jpg"

digit_w = 30 # Kich thuoc ki tu
digit_h = 60 # Kich thuoc ki tu

####Load SVM

data_path = r"D:\Lab IC\yolov4-tf\test_image"

f = open("result.csv", "w+")
f.write("Name,Plate Number\n")
# Đọc file ảnh đầu vào
count = 0
for imageName in sorted(os.listdir(data_path)):
    count += 1
    print(f"Loading {count}")
    Ivehicle = cv2.imread(f"{data_path}/{imageName}")

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    # try:

    _ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


    # Cau hinh tham so cho model SVM

    if not (len(LpImg)):
        continue
        # Chuyen doi anh bien so
    roi = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    roi = cv2.resize(roi, fy=2, fx=2, dsize=None)
    roi = cv2.medianBlur(roi, 3)
    cv2.imshow("LP", roi)
    cv2.waitKey()
    # cv2.imwrite(f"LiciencePlate/{imageName}", roi)
    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor( roi, cv2.COLOR_BGR2GRAY)


    # Ap dung threshold de phan tach so va nen
    binary = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imshow("Anh bien so sau threshold", binary)
    cv2.waitKey()

    # plate_info = pytesseract.image_to_string(binary, config='--psm 13 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUV0123456789',
    #                                     lang="eng")
    # # Segment kí tự
    # plate_info = re.sub("[\W_]+", "", plate_info)
    # print("Raw plate: ", plate_info)
    # plate_info = fine_tune(plate_info)
    # # Viet bien so len anh
    # H, W = 30, Ivehicle.shape[1] // 2
    #
    # cv2.rectangle(Ivehicle, (0, 0), (W, H), (0, 0, 0), cv2.FILLED)
    # cv2.putText(Ivehicle,plate_info,(5, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), lineType=cv2.LINE_AA)
    # cv2.imshow("Image", Ivehicle)
    # cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite(f"Result/{imageName}", Ivehicle)
    # f.write(f"{imageName},{plate_info}")
    # except:
    #     # f.write(f"{imageName},Fail")
    #     pass
    # cv2.imshow("Image", Ivehicle)
    # cv2.waitKey()
    # Hien thi anh
    # print("Bien so=", plate_info)

f.close()



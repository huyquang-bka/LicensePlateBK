from Module.Plate import *
import time
import string
from random import shuffle

# import tensorflow as tf

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_path = r"D:\Lab IC\LP\data17-7\data17-7-2021"
data_video_path = r"LP.mp4"



# cap = cv2.VideoCapture(data_video_path)
imageList = os.listdir(data_path)
# shuffle(imageList)
for imageName in sorted(imageList):
    s = time.time()
    # ret, image = cap.read()
    image = cv2.imread(f"{data_path}/{imageName}")
    # image = cv2.resize(image, dsize=None, fx=5, fy=5)
    H, W, _ = image.shape
    status, LP_image = plate_detection(image)
    if status:
        image_processed, text = process_image_chracter(LP_image)
        cv2.rectangle(image, (0, 0), (120, 80), (0, 0, 0), -1)
        cv2.putText(image, str(text), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        text = fine_tune(text)
        cv2.putText(image, str(text), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("LP", image_processed)
    cv2.imshow("Video", image)
    key = cv2.waitKey()
    cv2.destroyAllWindows()
    # if key == ord("q"):
    #     break
    # elif key == 32:
    #     cv2.waitKey()

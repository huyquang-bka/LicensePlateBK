import os
import cv2

for imageName in os.listdir("Result"):
    img = cv2.imread(f"Result/{imageName}")
    cv2.imshow("Image", img)
    cv2.waitKey()
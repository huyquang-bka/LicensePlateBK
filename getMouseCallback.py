import numpy as np
import cv2

# mouse callback function
point = []
count = 1

def draw_circle(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        cv2.circle(img, (x, y), 3, (0, 0, 255), 5)
    # elif event == cv2.EVENT_RBUTTONDOWN:
    #     x1, y1, x2, y2 = point[0], point[1]
    #     distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #     point = []
    #     f = open('CSV/distance_2_wheels.csv', 'a+')
    #     f.write(f"{count} {distance}")
    #     f.close()



# Create a black image, a window and bind the function to window
img = cv2.imread('for_cut.jpg')
# img = cv2.resize(img,dsize=None,fy=0.8, fx=0.8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)
while 1:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
#
# 2.0 889.0
# 1663.0 648.0
# 1694.0 718.0
#1149 870
# 6.0 1039.0

# (-0.1751956652619242, 889.3503913305238)
# (-0.2197867298578199, 1040.318720379147)
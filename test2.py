import cv2

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

while True:
    ret, Ivehicle = cap.read()
    image_copy = Ivehicle.copy()
    x1_box, y1_box, x2_box, y2_box = 50, 200, 1385, 568
    # x1_box, y1_box, x2_box, y2_box = 200, 200, 1385, 568
    Ivehicle = Ivehicle[y1_box: y2_box, x1_box:x2_box]
    vid_writer.write(image_copy)
    cv2.imshow("image", image_copy)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == 32:
        cv2.waitKey()
    elif key == ord("c"):
        check = -check

import cv2
import numpy as np

def clahe(img, clip_limit=2.0, grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def card_detector(src):
    hsv = cv2.cvtColor(src.copy(), cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 120])
    upper_blue = np.array([180, 38, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(src, src, mask=mask)
    b, g, r = cv2.split(result)
    g = clahe(g, 5, (3, 3))
    img_blur = cv2.blur(g, (9, 9))
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, 2)

    contours, hierarchy = cv2.findContours(img_th,
                                            cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    max_brightness = 0
    canvas = src.copy()
    for cnt in contours:
        rect = cv2.boundingRect(cnt)
        x, y, w, h = rect
        if w*h > 4000 and w*h<40000:
            mask = np.zeros(src.shape, np.uint8)
            mask[y:y+h, x:x+w] = src[y:y+h, x:x+w]
            brightness = np.sum(mask)
            if brightness > max_brightness:
                brightest_rectangle = rect
                max_brightness = brightness

    x, y, w, h = brightest_rectangle
    print("The width of card is", w)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), 5)
    while True:
        cv2.imshow("Card Detected", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            cv2.destroyWindow("Card Detected")
            break
    return canvas,w

def pupil_detect(img):
    gray_resized = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    classifier =cv2. CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    boxes = classifier.detectMultiScale(gray_resized)
    pupilh=[]
    wh=0
    hh=0
    for box in boxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        cv2.rectangle(img, (x, y), (x2, y2), (0,255,0), 5)
        pupilh.append([int((x+x2)/2), int((y+y2)/2)])
        wh=width
        hh=height
    print("The pupil coordinates according to haarcascade are",pupilh)
    img = cv2.circle(img, (int(pupilh[0][0]),int(pupilh[0][1])), 5, color=(0, 0, 255), thickness=-1)
    img = cv2.circle(img, (int(pupilh[1][0]),int(pupilh[1][1])), 5, color=(0, 0, 255), thickness=-1)
    # cv2.imwrite("haar_pupil.jpg",img)
    img=cv2.line(img, (int(pupilh[0][0]),int(pupilh[0][1])), (int(pupilh[1][0]),int(pupilh[1][1])), (0, 255, 0), thickness=5)
    # while True:
    #     cv2.imshow("Pupil Detected", img)

    #     k = cv2.waitKey(1)
    #     if k%256 == 27:
    #         # ESC pressed
    #         print("Escape hit, closing...")
    #         cv2.destroyWindow("Pupil Detected")
    #         break
    return img,pupilh

import math
PARALLEL_PLANE_SCALING_FACTOR=1

def scaleit(l,r,w):
    l=np.array(l) #choosing haarcascade pupil coordinates
    r=np.array(r)
    pupil_dist_in_pixel=float(math.sqrt((l[0]-r[0])*(l[0]-r[0])+(l[1]-r[1])*(l[1]-r[1])))
    print("The pupil distance in pixel is", pupil_dist_in_pixel)
    print("The pupil distance in cm is", float((pupil_dist_in_pixel*9)/w)*PARALLEL_PLANE_SCALING_FACTOR)
    return float((pupil_dist_in_pixel*9)/w)*PARALLEL_PLANE_SCALING_FACTOR

# im = cv2.imread("C://Users//tanus//OneDrive//Desktop//CREATIVITY//Cynaptics and ML//Jain_software//SLA task 1//In//img6.jpeg")

cam = cv2.VideoCapture(0)

cv2.namedWindow("PD_calculator")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("PD_calculator", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        rect,w=card_detector(frame)
        fin,dim=pupil_detect(frame)
        val=scaleit(dim[0],dim[1],w)
        fin = cv2.putText(fin, str(math.floor(val))+"cm", tuple(((np.array(dim[0])+np.array(dim[1]))/2).astype('int')), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        while True:
            cv2.imshow("Pupil Detected", fin)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                cv2.destroyWindow("Pupil Detected")
                break
        cv2.imwrite(img_name, fin)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
import cv2
import numpy as np

async def find_max_radi(circles):
    index = -1
    max_index = 0
    radi = -1
    for i in circles[0, :]:
        index += 1
        if i[2] > radi:
            radi = i[2]
            max_index = index
    return max_index

async def primitive(img, x, y, r):
    r = r - 20
    try:
        cropped = img[y - r:y + r, x - r:x + r]
    except Exception as e:
        print("resize error")
        return 1
    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        objCor = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        if objCor == 6:
            string = "L"
            return 0
        elif objCor == 12:
            string = "X"
            return 2
        elif objCor == 8:
            string = "T"
            return 1
        else:
            string = None
            return None
    else:
        return None

async def search_circle(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 1)
    # pimg = await preprocess_circle(img)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT_ALT, 1.5, 20,
                               param1=220, param2=0.75, minRadius=30, maxRadius=150)
    if circles is None:
        return [None, None, None]

    circles = np.uint16(np.around(circles))
    index = await find_max_radi(circles)
    x, y, r = circles[0][index][0], circles[0][index][1], circles[0][index][2]

    return [x, y, r]

import math
import cv2
import numpy as np


async def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_canny = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=3)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    cv2.imwrite("arrow.png", img_erode)
    return img_erode


async def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])


async def get_tail(cnt, tip):
    max_distance = 0
    tail = None
    for [[x1, y1]] in cnt:
        distance = get_length(tip, (x1, y1))
        if distance > max_distance:
            max_distance = distance
            tail = (x1, y1)
    return tail


async def angle_between_points(a, b):
    arrow_slope = (a[0] - b[0]) / (a[1] - b[1])
    arrow_angle = math.degrees(math.atan(arrow_slope))
    return arrow_angle


async def rotation_between_points(tip, tail):
    # Store the tip and tail points:
    tipX = tip[0]
    tipY = tip[1]
    tailX = tail[0]
    tailY = tail[1]

    tipX = tipX - 240
    tailX = tailX - 240
    tipY = 320 - tipY
    tailY = 320 - tailY
    # Compute the sides of the triangle:
    dif_x = tipX - tailX
    dif_y = tipY - tailY
    # Compute the angle alpha
    try:
        alpha = math.degrees(math.atan(dif_y / dif_x))
    except ZeroDivisionError:
        alpha=0
    temp = alpha
    field = ""
    if dif_x > 0 and dif_y > 0:  # 1st field
        field = "first"
        alpha = 90 - alpha
    elif dif_x < 0 < dif_y:  # 2nd field
        field = "second"
        alpha = -90 - alpha
    elif dif_x < 0 and dif_y < 0:  # 3rd field
        field = "third"
        alpha = -(90 + alpha)
    elif dif_x > 0 > dif_y:  # 4th field
        field = "fourth"
        alpha = 90 + alpha
    else:
        print("UNEXPECTED JOURNEY")
    # print(temp, f"----{field}-----", alpha)
    return alpha


async def get_length(p1, p2):
    line_length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return line_length


async def mid_point(tip, tail):
    centerX = (tip[0] + tail[0]) // 2
    centerY = (tip[1] + tail[1]) // 2
    return centerX, centerY


async def angleFinder(img):
    processed = cv2.GaussianBlur(img, (5, 5), 1)
    contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in c:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        hull = cv2.convexHull(approx, returnPoints=False)
        sides = len(hull)

        if 6 > sides > 3 and sides + 2 == len(approx):
            M = cv2.moments(cnt)
            # Find x-axis centroid using image moments
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                return [None, None, None, None, None]
            arrow_tip = await find_tip(approx[:, 0, :], hull.squeeze())
            if arrow_tip:
                # arrow_tail = get_tail(cnt, arrow_tip)
                arrow_tail = (cx, cy)
                angle = await rotation_between_points(arrow_tip, arrow_tail)
                centerX, centerY = await mid_point(arrow_tip, arrow_tail)
                return [angle, centerX, centerY, arrow_tip, arrow_tail]
    return [None, None, None, None, None]

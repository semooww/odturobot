import cv2


async def detect_line(image):
    h, w = image.shape[:2]
    half_image = image[0:int(h / 2), :]
    blur = cv2.GaussianBlur(half_image, (5, 5), 1)
    # Find all contours in frame
    contours, hierarchy = cv2.findContours(blur.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find largest contour area and image moments
        c = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True)
        M = cv2.moments(c)

        # Find x-axis centroid using image moments
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M["m01"] / M["m00"])
        except ZeroDivisionError:
            return None, None, None
        return cx, cy, approx
    else:
        return None, None, None

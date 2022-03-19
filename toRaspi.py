import sys
from threading import Thread
import cv2
import numpy as np
from follow_line import detect_line
from circle_finder import search_circle, primitive
from arrow_finder import angleFinder
import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)


async def helperDrawings(img, centerX, centerY, approx=None, circle=None, letter=None, arrow=None):
    global controller
    global yawThreshold
    global vehicle_state

    if img is None:
        return
    # helper drawings
    imgCopy = img.copy()
    cX, cY = None, None
    if centerX is not None and centerY is not None:
        cX = centerX - cw
        cY = ch - centerY

    # Cartesian plane
    cv2.line(imgCopy, (0, ch), (w, ch), green, 2)
    cv2.line(imgCopy, (cw, 0), (cw, h), green, 2)
    # threshold lines for yaw
    cv2.line(imgCopy, (cw - yawThreshold, 0), (cw - yawThreshold, ch), green, 1)
    cv2.line(imgCopy, (cw + yawThreshold, 0), (cw + yawThreshold, ch), green, 1)
    cv2.putText(imgCopy, f"Mode:{vehicle_state}", (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
    cv2.rectangle(imgCopy, (cw - yawThreshold, ch - yawThreshold), (cw + yawThreshold, ch + yawThreshold), green, 2)
    if control_centerX is None or control_centerY is None:
        return imgCopy
    try:
        if vehicle_state == "Line":
            cv2.circle(imgCopy, (centerX, centerY), 3, red, 3)
            if approx is not None:
                cv2.drawContours(imgCopy, [approx], 0, cyan, 5)
            # from center of the largest area to the closest threshold line
            if cX is not None:
                if cX >= yawThreshold:
                    # print(cX, ">", yawThreshold)
                    cv2.line(imgCopy, (int(cw + yawThreshold), centerY), (centerX, centerY), red, 1)
                elif cX <= -yawThreshold:
                    # print(cX, "<", -yawThreshold)
                    cv2.line(imgCopy, (int(cw - yawThreshold), centerY), (centerX, centerY), red, 1)
            if circle is not None:
                cv2.circle(imgCopy, (circle[0], circle[1]), 3, red, 3)
                cv2.circle(imgCopy, (circle[0], circle[1]), circle[2], red, 5)
        elif vehicle_state == "Circle":
            cv2.circle(imgCopy, (circle[0], circle[1]), 3, red, 3)
            cv2.circle(imgCopy, (circle[0], circle[1]), circle[2], red, 5)
        elif vehicle_state == "Letter":
            cv2.circle(imgCopy, (circle[0], circle[1]), 3, red, 3)
            cv2.circle(imgCopy, (circle[0], circle[1]), circle[2], red, 5)
            if letter is not None:
                cv2.putText(imgCopy, f"Letter:{letter}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
        elif vehicle_state == "Arrow":
            if arrow is not None:
                cv2.circle(imgCopy, arrow[0], 3, green, 3)
                cv2.circle(imgCopy, arrow[1], 3, blue, 3)
                cv2.line(imgCopy, arrow[1], arrow[0], red, 2)
                cv2.putText(imgCopy, f"Angle: {round(arrow[2], 3)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
                cv2.circle(imgCopy, (centerX, centerY), 3, red, 3)
            if approx is not None:
                cv2.drawContours(imgCopy, [approx], 0, cyan, 5)
                cv2.circle(imgCopy, (centerX, centerY), 3, red, 3)
        elif vehicle_state == "Symbol":
            if circle is not None:
                cv2.circle(imgCopy, (circle[0], circle[1]), 3, red, 3)
                cv2.circle(imgCopy, (circle[0], circle[1]), circle[2], red, 5)
            elif arrow is not None:
                cv2.circle(imgCopy, arrow[0], 3, green, 3)
                cv2.circle(imgCopy, arrow[1], 3, blue, 3)
                cv2.line(imgCopy, arrow[1], arrow[0], red, 2)
                cv2.circle(imgCopy, (centerX, centerY), 3, red, 3)
                cv2.putText(imgCopy, f"Angle: {round(arrow[2], 3)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, red, 2)
    except Exception as e:
        print(f"Error occured as {e}")

    return imgCopy


async def black_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    ret, mask = cv2.threshold(img, black_threshold, 255, 0)
    return mask


async def camera():
    global controller
    global control_centerX
    global control_centerY
    global angle
    global vehicle_state
    global time_stamp
    global time_circle
    global time_arrow
    global arrow_counter
    circle_counter = 0
    letter_counter = 0
    approvals = [0, 0, 0]  # L T X
    cap = cv2.VideoCapture(0)

    while 1:
        if not initialize:
            continue
        if controller:
            print("Mission Done, Camera is closed.")
            cv2.destroyAllWindows()
            break
        ret, img = cap.read()
        if not ret:
            print("Image is none")
            sys.exit(0)
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
        imgCopy = img.copy()
        black_img = await black_mask(img)

        if vehicle_state == "Line":
            control_centerX, control_centerY, approx = await detect_line(black_img)
            circle_control = [None, None, None]
            if time_circle < time_stamp:
                circle_control = await search_circle(black_img)
            imgCopy = await helperDrawings(img, control_centerX, control_centerY, approx, circle=circle_control)
            if circle_control[0] is not None:
                circle_counter += 1
                if circle_counter > 10:
                    vehicle_state = "Circle"
                    print(f"Vehicle state is changed to {vehicle_state}")
                    circle_counter = 0
            else:
                circle_counter -= 2
                if circle_counter < 0:
                    circle_counter = 0

        elif vehicle_state == "Circle":
            [control_centerX, control_centerY, radius] = await search_circle(black_img)
            imgCopy = await helperDrawings(img, control_centerX, control_centerY,
                                           circle=[control_centerX, control_centerY, radius])
            if control_centerX == None:
                cv2.imshow("Image", imgCopy)
                cv2.waitKey(1)
                continue

        elif vehicle_state == "Letter":
            if letter_counter >= 10:
                state_index = np.argmax(approvals)
                if state_index == 0:
                    vehicle_state = "Line"
                    time_circle = time_stamp + 10
                elif state_index == 1:
                    vehicle_state = "Land"
                elif state_index == 2:
                    vehicle_state = "Arrow"
                print(f"Vehicle state is changed to {vehicle_state}")
                letter_counter = 0
                approvals = [0, 0, 0]
            else:
                [control_centerX, control_centerY, radius] = await search_circle(black_img)
                imgCopy = await helperDrawings(img, control_centerX, control_centerY,
                                               circle=[control_centerX, control_centerY, radius])
                if control_centerX == None:
                    cv2.imshow("Image", imgCopy)
                    continue
                index = await primitive(black_img, control_centerX, control_centerY, radius)
                if index is not None:
                    approvals[index] += 1
                    letter_counter += 1
                    imgCopy = await helperDrawings(img, control_centerX, control_centerY,
                                                   circle=[control_centerX, control_centerY, radius], letter=mapping[index])
                else:
                    imgCopy = await helperDrawings(img, control_centerX, control_centerY,
                                                   circle=[control_centerX, control_centerY, radius])

        elif vehicle_state == "Arrow":
            if arrow_counter < 6:
                control_centerX, control_centerY, approx = await detect_line(black_img)
            [angle, tempX, tempY, tip, tail] = await angleFinder(black_img)  # angle, X, Y, tip, tail
            if tempX is not None or tempY is not None:
                if arrow_counter == 6:
                    approx = None
                arrow_counter += 1
                control_centerX = tempX
                control_centerY = tempY
                imgCopy = await helperDrawings(img, control_centerX, control_centerY, approx=approx, arrow=[tip, tail, angle])
            else:
                imgCopy = await helperDrawings(img, control_centerX, control_centerY)

        elif vehicle_state == "Symbol":
            radius = None
            angle = None
            if time_circle < time_stamp:
                [circleX, circleY, radius] = await search_circle(black_img)
            if time_arrow < time_stamp:
                [angle, arrowX, arrowY, tip, tail] = await angleFinder(black_img)
            if angle is not None:
                cX = arrowX - cw
                control_centerX, control_centerY = arrowX, arrowY
                if symbol_slice >= cX >= -symbol_slice:
                    arrow_counter += 1
                    if arrow_counter > 15:
                        arrow_counter = 0
                        circle_counter = 0
                        vehicle_state = "Arrow"
                        print(f"Vehicle state is changed to {vehicle_state}")
            if radius is not None:
                cX = circleX - cw
                control_centerX, control_centerY = circleX, circleY
                if symbol_slice >= cX >= -symbol_slice:
                    circle_counter += 1
                    if circle_counter > 8:
                        circle_counter = 0
                        arrow_counter = 0
                        vehicle_state = "Circle"
                        print(f"Vehicle state is changed to {vehicle_state}")

            else:
                imgCopy = await helperDrawings(img, control_centerX, control_centerY)
        video_writer.write(cv2.resize(imgCopy, (1280, 720)))
        cv2.imshow("Image", imgCopy)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('x'):
            cv2.destroyAllWindows()
            controller = True
            break


async def control_drone():
    global control_centerX
    global control_centerY
    global angle
    global vehicle_state
    global controller
    global time_stamp
    global time_circle
    global time_arrow
    global arrow_counter
    global my_relative_altitude
    div = 350
    global vehicle
    await vehicle.connect(system_address="serial:///dev/ttyACM0:57600")
    print("Vehicle connected.")
    async for state in vehicle.core.connection_state():
        if state.is_connected:
            print(f"Drone discovered!")
            break
    print("-- Arming")
    await vehicle.action.arm()

    print("Taking off")
    await vehicle.action.takeoff()
    await asyncio.sleep(5)
    print("-- Setting initial BODY setpoint")
    await vehicle.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard")
    try:
        await vehicle.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: \
                     {error._result.result}")
        print("Landing -- Disarming")
        await vehicle.action.land()
        await vehicle.action.disarm()
        return

    print("Initialize completed.")
    global initialize
    initialize = True
    time_circle = time_stamp + delay_fist_circle
    while True:
        if control_centerX is None or control_centerY is None:
            if vehicle_state != "Symbol":
                await asyncio.sleep(0.25)
                continue
        else:
            cX = control_centerX - cw
            cY = ch - control_centerY
        if vehicle_state == "Line":
            div = 350
        elif vehicle_state == "Circle":
            div = 375
            if 30 >= cX >= -30 and 30 >= cY >= -30:
                vehicle_state = "Letter"
                print(f"Vehicle state is changed to {vehicle_state}")
                continue
        elif vehicle_state == "Letter":
            await asyncio.sleep(0.25)
            continue
        elif vehicle_state == "Arrow":
            div = 375
            if angle is not None:
                if -5 <= angle <= 5 and -30 <= cX <= 30 and -30 <= cY <= 30:
                    vehicle_state = "Symbol"
                    arrow_counter = 0
                    time_circle = time_stamp + delay_circle
                    time_arrow = time_stamp + delay_arrow
                    print(f"Vehicle state is changed to {vehicle_state}")
                    await asyncio.sleep(0)
                    continue
                else:
                    if 30 >= cX >= -30 and 30 >= cY >= -30:
                        await vehicle.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, angle))
                        await asyncio.sleep(0.25)
        elif vehicle_state == "Symbol":
            await vehicle.offboard.set_velocity_body(VelocityBodyYawspeed(0.5, 0, 0, 0))
            await asyncio.sleep(0.25)
            continue
        elif vehicle_state == "Land":
            try:
                await vehicle.offboard.stop()
            except OffboardError as error:
                print(f"Stopping offboard mode failed with error code: {error._result.result}")
            await vehicle.action.land()
            controller = True
            await vehicle.action.disarm()
            break
        asyncio.ensure_future(get_position())
        if my_relative_altitude > 1.7:
            await vehicle.offboard.set_velocity_body(VelocityBodyYawspeed(cY / div, cX / div, 0.1, cX / 7.5))
            await asyncio.sleep(0.25)
        elif my_relative_altitude < 1.5:
            await vehicle.offboard.set_velocity_body(VelocityBodyYawspeed(cY / div, cX / div, -0.1, cX / 7.5))
            await asyncio.sleep(0.25)
        else:
            await vehicle.offboard.set_velocity_body(VelocityBodyYawspeed(cY / div, cX / div, 0, cX / 7.5))
            await asyncio.sleep(0.25)


async def get_position():
    global my_relative_altitude
    async for position in vehicle.telemetry.position():
        my_relative_altitude = position.relative_altitude_m


async def timer():
    global controller
    global time_stamp
    global initialize
    while 1:
        if not initialize:
            continue
        if controller:
            break
        time_stamp += 1
        await asyncio.sleep(1)


def thread_timer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(timer())
    loop.close()
    print("********Timer func closed.**********")


def thread_control_drone():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(control_drone())
    loop.close()
    print("********Control Drone func closed.**********")


def thread_camera():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(camera())
    loop.close()
    print("********Camera func closed.**********")


# RGB codes for some colors
white = (255, 255, 255)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
cyan = (0, 255, 255)
black = (0, 0, 0)
purple = (255, 0, 255)

black_threshold = int(sys.argv[1])
symbol_slice = int(sys.argv[2])
delay_circle = int(sys.argv[3])
delay_arrow = int(sys.argv[4])
delay_fist_circle = int(sys.argv[5])
yawThreshold = 30  # Threshold value for yaw
controller = False  # check variable for the job
initialize = False  # check the timer to start
time_stamp = 0  # time counter in second form
time_circle = time_stamp + 15
time_arrow = 0
arrow_counter = 0
my_relative_altitude = 0
vehicle_state = "Line"
w, h = 640, 480
cw, ch = 320, 240
mapping = {0: "L", 1: "T", 2: "X"}

# Variables for control functions
control_centerX = None
control_centerY = None
angle = None

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output.avi', fourcc, 20, (1280, 720))

print(f"Yaw threshold value: {yawThreshold}")

vehicle = System()
print("System initialized.")

t1 = Thread(target=thread_timer)
t2 = Thread(target=thread_camera)
t3 = Thread(target=thread_control_drone)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

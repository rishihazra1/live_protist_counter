import cv2 as cv
video_cap = cv.VideoCapture("video.mp4")
_, frame = video_cap.read()
prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
frameCount = 0
totalCount = 0
liveCount = 0

while frameCount < 100:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = video_cap.read()
    frameCount = frameCount + 1
    current = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # check for total objects
    _, threshold = cv.threshold(current, 166, 255, cv.THRESH_BINARY)
    totalContours, _ = cv.findContours(threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    i = 0
    for contour in totalContours:
        # here we are ignoring first counter because findContours function detects whole image as shape
        if i == 0:
            i = 1
            continue
        # cv.drawContours(current, [contour], 0, (0, 255, 0), 1) # thin line

    # check for moving objects
    flow = cv.calcOpticalFlowFarneback(prvs, current, None, 0.5, 1, 5, 5, 7, 1.5, 0) # tune these
    mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mag = (cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX) > 25) * 255
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20))
    closing = cv.morphologyEx(mag.astype('uint8'), cv.MORPH_CLOSE, kernel)
    liveContours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(current, liveContours, -1, (0, 255, 0), 2) # thick line

    cv.imshow("Contours", current)
    print("Total objects in frame ", frameCount, " : ", len(totalContours) - 1, " Live objects: ", len(liveContours))
    totalCount = totalCount + len(totalContours) - 1
    liveCount = liveCount + len(liveContours)
    prvs = current.copy()
    if cv.waitKey(2) == ord('q'):
        break

print("Total Objects: ", totalCount/frameCount)
print("Live Objects:", liveCount/frameCount)
video_cap.release()
# wait for any key presses
cv.waitKey(0)
cv.destroyAllWindows()


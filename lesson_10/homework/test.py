import cv2

# capture video from built-in camera
cap = cv2.VideoCapture(0)

# capture video from the file
#cap = cv2.VideoCapture("./cockroach_from_corfu.mp4")

# create CSRT tracker
tracker = cv2.TrackerCSRT_create()

# create KCF tracker
#tracker = cv2.TrackerKCF_create()
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)

tracker.init(img, bbox)


def drawRect(img, bbox, fps):
    x, y, width, height = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
    cv2.putText(img, "FPS: " + str(int(fps)), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)


while True:
    timer = cv2.getTickCount()
    success, img = cap.read()

    # exit if Escape is hit
    if cv2.waitKey(10) == 27:
        break

    success, bbox = tracker.update(img)
    print(success, bbox)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if success:
        drawRect(img, bbox, fps)
    else:
        cv2.putText(img, "Lost", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
        cv2.putText(img, "FPS: " + str(int(fps)), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    cv2.imshow("Tracking", img)

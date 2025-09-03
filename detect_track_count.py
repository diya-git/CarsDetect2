import cv2
from ultralytics import YOLO
from tracker import Tracker

# Load YOLO model
model = YOLO('yolov8s.pt')

# COCO classes (same as your list; keep as-is if you need other classes)
class_list = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]

tracker = Tracker(max_distance=50)
cap = cv2.VideoCapture('input_video.mp4')

# Counting containers: use sets to prevent duplicate counting
counter_down = set()
counter_up = set()
# intermediate markers to detect crossing order
seen_red = {}   # id -> True when object seen at red line first
seen_blue = {}  # id -> True when object seen at blue line first

# lines and offsets
red_line_y = 198
blue_line_y = 268
offset = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    # Run YOLO prediction with a confidence filter (adjust conf as needed)
    results = model(frame, conf=0.25, iou=0.45)[0]

    cars_list = []
    if hasattr(results, "boxes") and len(results.boxes) > 0:
        det = results.boxes.data.cpu().numpy()  # nx6: x1,y1,x2,y2,conf,cls
        for box in det:
            x1, y1, x2, y2, conf, cls = box
            cls = int(cls)
            # only keep 'car' (cls index 2). If you also want buses/trucks, include them.
            if cls == 2:
                cars_list.append([int(x1), int(y1), int(x2), int(y2)])

    # update tracker
    bbox_id = tracker.update(cars_list)

    # draw lines
    cv2.line(frame, (172, red_line_y), (774, red_line_y), (0, 0, 255), 3)
    cv2.putText(frame, 'red line', (172, red_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.line(frame, (8, blue_line_y), (927, blue_line_y), (255, 0, 0), 3)
    cv2.putText(frame, 'blue line', (8, blue_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx = (x3 + x4) // 2    # correct midpoint
        cy = (y3 + y4) // 2

        # draw box and id for debugging
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(obj_id), (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

        # detect crossing order + count once using sets
        # going down: red -> blue
        if red_line_y - offset < cy < red_line_y + offset:
            seen_red[obj_id] = True

        if obj_id in seen_red and blue_line_y - offset < cy < blue_line_y + offset:
            if obj_id not in counter_down:
                counter_down.add(obj_id)
                # once counted, remove seen flags so it won't be counted again
                seen_red.pop(obj_id, None)
                seen_blue.pop(obj_id, None)

        # going up: blue -> red
        if blue_line_y - offset < cy < blue_line_y + offset:
            seen_blue[obj_id] = True

        if obj_id in seen_blue and red_line_y - offset < cy < red_line_y + offset:
            if obj_id not in counter_up:
                counter_up.add(obj_id)
                seen_blue.pop(obj_id, None)
                seen_red.pop(obj_id, None)

    # show counts
    cv2.putText(frame, f'going down - {len(counter_down)}', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'going up   - {len(counter_up)}', (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

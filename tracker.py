import math

class Tracker:
    def __init__(self, max_distance=50):
        # Store the center positions of the objects {id: (cx, cy)}
        self.center_points = {}
        # ID counter for new objects
        self.id_count = 0
        # distance threshold to consider same object across frames
        self.max_distance = max_distance

    def update(self, objects_rect):
        """
        objects_rect: list of [x1, y1, x2, y2] (ints)
        returns: list of [x1, y1, x2, y2, id]
        """
        objects_bbs_ids = []

        for rect in objects_rect:
            # be safe: convert to ints
            x1, y1, x2, y2 = map(int, rect)

            # correct center calculation (midpoint)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            same_object_detected = False
            for object_id, pt in list(self.center_points.items()):
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < self.max_distance:
                    # update existing object's center
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, object_id])
                    same_object_detected = True
                    break

            # new object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        # cleanup: keep only centers for objects seen this frame
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            new_center_points[object_id] = self.center_points[object_id]

        self.center_points = new_center_points.copy()
        return objects_bbs_ids

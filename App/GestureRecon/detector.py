from collections import defaultdict
import math
from typing import Any


class GestureAnalyzer:
    def __init__(self, fps=30):
        self.history = defaultdict(
            lambda: {
                "left_hidden_frames": 0,
                "right_hidden_frames": 0,
                "surrender_frames": 0,
                "aiming_frames": 0,
                "fist_frames": 0,
                "threat_frames": 0,
            }
        )

        self.fps = fps
        self.thresh_hidden = max(3, int(self.fps * 0.35))
        self.thresh_surrender = max(3, int(self.fps * 0.3))
        self.thresh_aiming = max(4, int(self.fps * 0.4))
        self.thresh_fist = max(2, int(self.fps * 0.2))
        self.thresh_threat = max(3, int(self.fps * 0.22))

    def _get_keypoint(self, keypoints, idx):
        kp = keypoints[idx]
        return kp[0], kp[1], kp[2]

    def _distance(self, point_a, point_b):
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def _inside_box(self, point, box, margin_x=0.0, margin_y=0.0):
        if box is None:
            return False

        x1, y1, x2, y2 = box
        return (
            (x1 - margin_x) <= point[0] <= (x2 + margin_x)
            and (y1 - margin_y) <= point[1] <= (y2 + margin_y)
        )

    def _build_torso_box(self, shoulders, hips, shoulder_width):
        visible_points = [point for point in shoulders + hips if point[2] > 0.35]
        if len(visible_points) < 2:
            return None

        xs = [point[0] for point in visible_points]
        ys = [point[1] for point in visible_points]
        margin_x = max(12.0, shoulder_width * 0.22)
        margin_y = max(10.0, shoulder_width * 0.12)
        return [
            min(xs) - margin_x,
            min(ys) - margin_y,
            max(xs) + margin_x,
            max(ys) + margin_y,
        ]

    def _arm_extension_features(self, shoulder, elbow, wrist, shoulder_width, torso_box):
        if min(shoulder[2], elbow[2], wrist[2]) <= 0.35:
            return 0.0, 0.0, False

        upper_arm = self._distance(shoulder, elbow)
        lower_arm = self._distance(elbow, wrist)
        shoulder_to_wrist = self._distance(shoulder, wrist)
        full_arm = upper_arm + lower_arm

        if full_arm <= 1e-6:
            return 0.0, 0.0, False

        straightness = shoulder_to_wrist / full_arm
        reach_ratio = shoulder_to_wrist / max(shoulder_width, 1.0)
        wrist_in_torso = self._inside_box(
            wrist,
            torso_box,
            margin_x=shoulder_width * 0.08,
            margin_y=shoulder_width * 0.08,
        )
        return straightness, reach_ratio, wrist_in_torso

    def _confirm_gesture(self, track_id, key, active, threshold, decay=2, reset=False):
        if reset and not active:
            self.history[track_id][key] = 0
        else:
            self._update_counter(track_id, key, active, cooldown=decay)

        return self.history[track_id][key] >= threshold

    def analyze(self, track_id, keypoints, box=None, hand_context=None):
        """
        Analisa pose corporal combinada com deteccao de maos.
        """
        alerts = []
        hand_context = hand_context or {}

        ls_x, ls_y, ls_c = self._get_keypoint(keypoints, 5)
        rs_x, rs_y, rs_c = self._get_keypoint(keypoints, 6)

        lw_x, lw_y, lw_c = self._get_keypoint(keypoints, 9)
        rw_x, rw_y, rw_c = self._get_keypoint(keypoints, 10)

        le_x, le_y, le_c = self._get_keypoint(keypoints, 7)
        re_x, re_y, re_c = self._get_keypoint(keypoints, 8)

        lh_x, lh_y, lh_c = self._get_keypoint(keypoints, 11)
        rh_x, rh_y, rh_c = self._get_keypoint(keypoints, 12)

        conf_thresh = 0.35

        shoulder_dist = 40.0
        if ls_c > conf_thresh and rs_c > conf_thresh:
            shoulder_dist = self._distance((ls_x, ls_y), (rs_x, rs_y)) + 0.1

        torso_height = 60.0
        visible_hips = [(lh_x, lh_y, lh_c), (rh_x, rh_y, rh_c)]
        visible_hip_points = [point for point in visible_hips if point[2] > conf_thresh]
        if visible_hip_points and (ls_c > conf_thresh or rs_c > conf_thresh):
            visible_shoulder_points = [
                point
                for point in [(ls_x, ls_y, ls_c), (rs_x, rs_y, rs_c)]
                if point[2] > conf_thresh
            ]
            if visible_shoulder_points:
                avg_shoulder_y = sum(point[1] for point in visible_shoulder_points) / len(
                    visible_shoulder_points
                )
                avg_hip_y = sum(point[1] for point in visible_hip_points) / len(
                    visible_hip_points
                )
                torso_height = max(20.0, avg_hip_y - avg_shoulder_y)

        if shoulder_dist < 20.0:
            shoulder_width = torso_height * 0.45
        else:
            shoulder_width = max(shoulder_dist, 40.0)

        torso_box = self._build_torso_box(
            [(ls_x, ls_y, ls_c), (rs_x, rs_y, rs_c)],
            [(lh_x, lh_y, lh_c), (rh_x, rh_y, rh_c)],
            shoulder_width,
        )

        left_visible = bool(hand_context.get("left_visible"))
        right_visible = bool(hand_context.get("right_visible"))
        left_closed = bool(hand_context.get("left_closed"))
        right_closed = bool(hand_context.get("right_closed"))
        left_in_torso = bool(hand_context.get("left_in_torso"))
        right_in_torso = bool(hand_context.get("right_in_torso"))

        is_aiming = False
        left_straightness, left_reach, left_wrist_in_torso = self._arm_extension_features(
            (ls_x, ls_y, ls_c),
            (le_x, le_y, le_c),
            (lw_x, lw_y, lw_c),
            shoulder_width,
            torso_box,
        )
        right_straightness, right_reach, right_wrist_in_torso = self._arm_extension_features(
            (rs_x, rs_y, rs_c),
            (re_x, re_y, re_c),
            (rw_x, rw_y, rw_c),
            shoulder_width,
            torso_box,
        )

        if left_straightness > 0.82 and left_reach > 1.05 and not left_wrist_in_torso:
            is_aiming = True
        if right_straightness > 0.82 and right_reach > 1.05 and not right_wrist_in_torso:
            is_aiming = True

        aiming_confirmed = self._confirm_gesture(
            track_id,
            "aiming_frames",
            is_aiming,
            self.thresh_aiming,
            decay=5,
            reset=left_visible or right_visible,
        )
        if aiming_confirmed:
            alerts.append("Braco Estendido")

        is_surrendering = False
        left_hands_up = False
        right_hands_up = False
        if not is_aiming:
            margin_y = shoulder_width * 0.4

            left_hands_up = (
                ls_c > conf_thresh and lw_c > conf_thresh and lw_y < (ls_y - margin_y)
            )
            right_hands_up = (
                rs_c > conf_thresh and rw_c > conf_thresh and rw_y < (rs_y - margin_y)
            )

            left_behind_head = False
            right_behind_head = False

            if ls_c > conf_thresh and lw_c > conf_thresh and le_c > conf_thresh:
                if le_x < (ls_x - margin_y) and lw_y < (ls_y + margin_y) and lw_x > le_x:
                    left_behind_head = True

            if rs_c > conf_thresh and rw_c > conf_thresh and re_c > conf_thresh:
                if re_x > (rs_x + margin_y) and rw_y < (rs_y + margin_y) and rw_x < re_x:
                    right_behind_head = True

            if left_hands_up or right_hands_up or left_behind_head or right_behind_head:
                is_surrendering = True

        surrender_confirmed = self._confirm_gesture(
            track_id,
            "surrender_frames",
            is_surrendering,
            self.thresh_surrender,
            decay=5,
            reset=(not left_hands_up and not right_hands_up),
        )
        if surrender_confirmed:
            alerts.append("Rendicao")

        fist_detected = left_closed or right_closed
        fist_confirmed = self._confirm_gesture(
            track_id,
            "fist_frames",
            fist_detected,
            self.thresh_fist,
            decay=6,
            reset=left_visible or right_visible,
        )
        if fist_confirmed:
            alerts.append("Mao Fechada")

        threat_detected = fist_detected and is_aiming
        threat_confirmed = self._confirm_gesture(
            track_id,
            "threat_frames",
            threat_detected,
            self.thresh_threat,
            decay=6,
            reset=(not fist_detected or not is_aiming),
        )
        if threat_confirmed:
            alerts.append("Mao Fechada + Braco Estendido")

        hand_state = {
            "left": {
                "visible": left_visible,
                "closed": left_closed,
                "in_torso": left_in_torso,
            },
            "right": {
                "visible": right_visible,
                "closed": right_closed,
                "in_torso": right_in_torso,
            },
        }
        hidden_debug: dict[str, dict[str, Any]] = {
            "left": {
                "hand_visible": left_visible,
                "hand_in_torso": left_in_torso,
                "elbow_inside": False,
                "wrist_inside": False,
                "wrist_missing": False,
                "cross_body": False,
                "arm_towards_torso": False,
                "hidden": False,
            },
            "right": {
                "hand_visible": right_visible,
                "hand_in_torso": right_in_torso,
                "elbow_inside": False,
                "wrist_inside": False,
                "wrist_missing": False,
                "cross_body": False,
                "arm_towards_torso": False,
                "hidden": False,
            },
        }

        def hand_hidden(side):
            if side == "left":
                shoulder = (ls_x, ls_y, ls_c)
                elbow = (le_x, le_y, le_c)
                wrist = (lw_x, lw_y, lw_c)
                hand_visible = left_visible
            else:
                shoulder = (rs_x, rs_y, rs_c)
                elbow = (re_x, re_y, re_c)
                wrist = (rw_x, rw_y, rw_c)
                hand_visible = right_visible
            side_debug = hidden_debug[side]

            if shoulder[2] <= conf_thresh:
                return False

            if hand_visible:
                return False

            elbow_inside = elbow[2] > conf_thresh and self._inside_box(
                elbow,
                torso_box,
                margin_x=shoulder_width * 0.18,
                margin_y=shoulder_width * 0.10,
            )
            wrist_inside = wrist[2] > conf_thresh and self._inside_box(
                wrist,
                torso_box,
                margin_x=shoulder_width * 0.22,
                margin_y=shoulder_width * 0.16,
            )

            cross_body = False
            if elbow[2] > conf_thresh:
                if side == "left":
                    cross_body = elbow[0] > (shoulder[0] + shoulder_width * 0.08)
                else:
                    cross_body = elbow[0] < (shoulder[0] - shoulder_width * 0.08)

            wrist_missing = wrist[2] <= conf_thresh and elbow[2] > conf_thresh
            arm_towards_torso = wrist_inside or elbow_inside or (wrist_missing and cross_body)

            side_debug["elbow_inside"] = elbow_inside
            side_debug["wrist_inside"] = wrist_inside
            side_debug["wrist_missing"] = wrist_missing
            side_debug["cross_body"] = cross_body
            side_debug["arm_towards_torso"] = arm_towards_torso
            side_debug["hidden"] = arm_towards_torso

            return arm_towards_torso

        left_hidden = False
        right_hidden = False
        if torso_box is not None:
            left_hidden = hand_hidden("left")
            right_hidden = hand_hidden("right")

        self._update_counter(track_id, "left_hidden_frames", left_hidden, cooldown=6)
        self._update_counter(track_id, "right_hidden_frames", right_hidden, cooldown=6)

        if (
            self.history[track_id]["left_hidden_frames"] >= self.thresh_hidden
            or self.history[track_id]["right_hidden_frames"] >= self.thresh_hidden
        ):
            alerts.append("Mao Oculta")

        return {
            "alerts": alerts,
            "hand_context": hand_state,
            "hidden_debug": hidden_debug,
        }

    def _update_counter(self, track_id, key, active, cooldown=1):
        if active:
            self.history[track_id][key] += 1
        else:
            self.history[track_id][key] = max(0, self.history[track_id][key] - cooldown)

    def clean_old_tracks(self, current_tracks):
        missing_tracks = set(self.history.keys()) - set(current_tracks)
        for track_id in missing_tracks:
            del self.history[track_id]

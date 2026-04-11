from collections import defaultdict


class GestureAnalyzer:
    def __init__(self, fps=30):
        self.history = defaultdict(
            lambda: {
                "hidden_frames": 0,
                "surrender_frames": 0,
                "aiming_frames": 0,
                "fist_frames": 0,
                "threat_frames": 0,
            }
        )

        self.fps = fps
        self.thresh_hidden = int(self.fps * 1.2)
        self.thresh_surrender = int(self.fps * 0.5)
        self.thresh_aiming = int(self.fps * 0.8)
        self.thresh_fist = int(self.fps * 0.4)
        self.thresh_threat = int(self.fps * 0.35)

    def _get_keypoint(self, keypoints, idx):
        kp = keypoints[idx]
        return kp[0], kp[1], kp[2]

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

        conf_thresh = 0.5

        shoulder_dist = 40.0
        if ls_c > conf_thresh and rs_c > conf_thresh:
            shoulder_dist = abs(ls_x - rs_x) + 0.1

        if shoulder_dist < 20.0 and ls_c > conf_thresh and lh_c > conf_thresh:
            shoulder_width = abs(ls_y - lh_y) * 0.4
        else:
            shoulder_width = max(shoulder_dist, 40.0)

        left_visible = bool(hand_context.get("left_visible"))
        right_visible = bool(hand_context.get("right_visible"))
        left_closed = bool(hand_context.get("left_closed"))
        right_closed = bool(hand_context.get("right_closed"))
        left_in_torso = bool(hand_context.get("left_in_torso"))
        right_in_torso = bool(hand_context.get("right_in_torso"))

        is_aiming = False
        if ls_c > conf_thresh and lw_c > conf_thresh and le_c > conf_thresh:
            arm_length = abs(ls_x - le_x) + abs(le_x - lw_x)
            if arm_length > 10:
                if abs(lw_y - ls_y) < (arm_length * 0.5) and abs(lw_x - ls_x) > (
                    arm_length * 0.7
                ):
                    is_aiming = True

        if rs_c > conf_thresh and rw_c > conf_thresh and re_c > conf_thresh:
            arm_length = abs(rs_x - re_x) + abs(re_x - rw_x)
            if arm_length > 10:
                if abs(rw_y - rs_y) < (arm_length * 0.5) and abs(rw_x - rs_x) > (
                    arm_length * 0.7
                ):
                    is_aiming = True

        self._update_counter(track_id, "aiming_frames", is_aiming, cooldown=2)
        aiming_confirmed = self.history[track_id]["aiming_frames"] > self.thresh_aiming
        if aiming_confirmed:
            alerts.append("Braco Estendido")

        is_surrendering = False
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

        self._update_counter(track_id, "surrender_frames", is_surrendering, cooldown=2)
        if self.history[track_id]["surrender_frames"] > self.thresh_surrender:
            alerts.append("Rendicao")

        fist_detected = left_closed or right_closed
        self._update_counter(track_id, "fist_frames", fist_detected, cooldown=1)
        fist_confirmed = self.history[track_id]["fist_frames"] > self.thresh_fist
        if fist_confirmed:
            alerts.append("Mao Fechada")

        threat_detected = fist_detected and is_aiming
        self._update_counter(track_id, "threat_frames", threat_detected, cooldown=2)
        if self.history[track_id]["threat_frames"] > self.thresh_threat:
            alerts.append("Mao Fechada + Braco Estendido")

        is_hidden = False
        left_side_visible = ls_c > conf_thresh and lh_c > conf_thresh
        right_side_visible = rs_c > conf_thresh and rh_c > conf_thresh

        if left_side_visible or right_side_visible:
            min_x = min(
                [
                    x
                    for x, c in [(ls_x, ls_c), (rs_x, rs_c), (lh_x, lh_c), (rh_x, rh_c)]
                    if c > conf_thresh
                ]
            )
            max_x = max(
                [
                    x
                    for x, c in [(ls_x, ls_c), (rs_x, rs_c), (lh_x, lh_c), (rh_x, rh_c)]
                    if c > conf_thresh
                ]
            )
            min_y = min(
                [y for y, c in [(ls_y, ls_c), (rs_y, rs_c)] if c > conf_thresh]
            )
            max_y = max(
                [y for y, c in [(lh_y, lh_c), (rh_y, rh_c)] if c > conf_thresh]
            )

            if max_x - min_x < 10:
                min_x -= shoulder_width / 2
                max_x += shoulder_width / 2

            margin = (max_x - min_x) * 0.2
            waist_y = max_y

            left_hidden = False
            right_hidden = False

            if left_side_visible:
                if left_visible:
                    left_hidden = False
                elif lw_c > conf_thresh:
                    if (min_x - margin) < lw_x < (max_x + margin) and min_y < lw_y < waist_y:
                        left_hidden = True
                elif le_c > conf_thresh:
                    if (min_x - margin) < le_x < (max_x + margin) and le_y < waist_y:
                        left_hidden = True

            if right_side_visible:
                if right_visible:
                    right_hidden = False
                elif rw_c > conf_thresh:
                    if (min_x - margin) < rw_x < (max_x + margin) and min_y < rw_y < waist_y:
                        right_hidden = True
                elif re_c > conf_thresh:
                    if (min_x - margin) < re_x < (max_x + margin) and re_y < waist_y:
                        right_hidden = True

            if left_hidden and right_hidden and lw_c > conf_thresh and rw_c > conf_thresh:
                dist_between_hands = abs(lw_x - rw_x) + abs(lw_y - rw_y)
                if dist_between_hands < (shoulder_width * 1.0) and lw_y > (
                    ls_y + shoulder_width
                ):
                    left_hidden = False
                    right_hidden = False

            # Se a mao esta visivel bem na frente do torso, nao tratamos como oculta.
            if left_in_torso:
                left_hidden = False
            if right_in_torso:
                right_hidden = False

            if left_hidden or right_hidden:
                is_hidden = True

        self._update_counter(track_id, "hidden_frames", is_hidden, cooldown=1)
        if self.history[track_id]["hidden_frames"] > self.thresh_hidden:
            alerts.append("Mao Oculta")

        return alerts

    def _update_counter(self, track_id, key, active, cooldown=1):
        if active:
            self.history[track_id][key] += 1
        else:
            self.history[track_id][key] = max(0, self.history[track_id][key] - cooldown)

    def clean_old_tracks(self, current_tracks):
        missing_tracks = set(self.history.keys()) - set(current_tracks)
        for track_id in missing_tracks:
            del self.history[track_id]

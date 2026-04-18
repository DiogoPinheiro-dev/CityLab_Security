from collections import defaultdict


class GestureAnalyzer:
    def __init__(self, fps=30, track_ttl_frames=18):
        # Historico de frames continuos em que a pessoa esta fazendo o gesto.
        self.history = defaultdict(
            lambda: {
                "hidden_frames": 0,
                "surrender_frames": 0,
                "aiming_frames": 0,
                "missing_frames": 0,
            }
        )

        # Limiares de tempo (frames) para confirmar o gesto.
        self.fps = fps
        self.track_ttl_frames = track_ttl_frames
        self.thresh_hidden = max(4, int(self.fps * 1.5))  # oculto por algum tempo
        self.thresh_surrender = max(3, int(self.fps * 0.55))  # maos para o alto
        self.thresh_aiming = max(4, int(self.fps * 0.8))  # braco estendido

    def _get_keypoint(self, keypoints, idx):
        # Retorna (x, y, conf) do keypoint.
        kp = keypoints[idx]
        return kp[0], kp[1], kp[2]

    def analyze(self, track_id, keypoints, box=None):
        """
        Analisa os limiares de pose e atualiza o historico.
        Retorna uma lista de alertas ativos para a pessoa.
        """
        alerts = []
        self.history[track_id]["missing_frames"] = 0

        # Indices do COCO:
        # 5: L Shoulder, 6: R Shoulder
        # 7: L Elbow,    8: R Elbow
        # 9: L Wrist,    10: R Wrist
        # 11: L Hip,     12: R Hip
        # 15: L Ankle,   16: R Ankle

        # Extraindo coordenadas e confiancas.
        ls_x, ls_y, ls_c = self._get_keypoint(keypoints, 5)
        rs_x, rs_y, rs_c = self._get_keypoint(keypoints, 6)

        lw_x, lw_y, lw_c = self._get_keypoint(keypoints, 9)
        rw_x, rw_y, rw_c = self._get_keypoint(keypoints, 10)

        le_x, le_y, le_c = self._get_keypoint(keypoints, 7)
        re_x, re_y, re_c = self._get_keypoint(keypoints, 8)

        lh_x, lh_y, lh_c = self._get_keypoint(keypoints, 11)
        rh_x, rh_y, rh_c = self._get_keypoint(keypoints, 12)

        conf_thresh = 0.5  # Confianca minima para considerar keypoint valido.

        # Largura do ombro como base de escala para usar como medida relativa.
        shoulder_dist = 40.0
        if ls_c > conf_thresh and rs_c > conf_thresh:
            shoulder_dist = abs(ls_x - rs_x) + 0.1

        # De lado, ombros se alinham no eixo X. Se a largura for muito pequena,
        # estimamos com base ombro->quadril.
        if shoulder_dist < 20.0 and ls_c > conf_thresh and lh_c > conf_thresh:
            shoulder_width = abs(ls_y - lh_y) * 0.4
        else:
            shoulder_width = max(shoulder_dist, 40.0)

        # --- 1. Braco estendido (agressao) ---
        is_aiming = False
        if ls_c > conf_thresh and lw_c > conf_thresh and le_c > conf_thresh:
            arm_length = abs(ls_x - le_x) + abs(le_x - lw_x)
            if arm_length > 10:
                if abs(lw_y - ls_y) < (arm_length * 0.5) and abs(lw_x - ls_x) > (arm_length * 0.7):
                    is_aiming = True

        if rs_c > conf_thresh and rw_c > conf_thresh and re_c > conf_thresh:
            arm_length = abs(rs_x - re_x) + abs(re_x - rw_x)
            if arm_length > 10:
                if abs(rw_y - rs_y) < (arm_length * 0.5) and abs(rw_x - rs_x) > (arm_length * 0.7):
                    is_aiming = True

        if is_aiming:
            self.history[track_id]["aiming_frames"] += 1
        else:
            self.history[track_id]["aiming_frames"] = max(0, self.history[track_id]["aiming_frames"] - 2)

        if self.history[track_id]["aiming_frames"] > self.thresh_aiming:
            alerts.append("Braco Estendido (Agressao)")

        # --- 2. Rendicao (maos para o alto / na nuca) ---
        is_surrendering = False
        if not is_aiming:
            margin_y = shoulder_width * 0.4

            left_hands_up = ls_c > conf_thresh and lw_c > conf_thresh and lw_y < (ls_y - margin_y)
            right_hands_up = rs_c > conf_thresh and rw_c > conf_thresh and rw_y < (rs_y - margin_y)

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

        if is_surrendering:
            self.history[track_id]["surrender_frames"] += 1
        else:
            self.history[track_id]["surrender_frames"] = max(0, self.history[track_id]["surrender_frames"] - 2)

        if self.history[track_id]["surrender_frames"] > self.thresh_surrender:
            alerts.append("Rendicao")

        # --- 3. Mao oculta na jaqueta/cintura ---
        is_hidden = False

        left_side_visible = ls_c > conf_thresh and lh_c > conf_thresh
        right_side_visible = rs_c > conf_thresh and rh_c > conf_thresh

        if left_side_visible or right_side_visible:
            min_x = min([x for x, c in [(ls_x, ls_c), (rs_x, rs_c), (lh_x, lh_c), (rh_x, rh_c)] if c > conf_thresh])
            max_x = max([x for x, c in [(ls_x, ls_c), (rs_x, rs_c), (lh_x, lh_c), (rh_x, rh_c)] if c > conf_thresh])
            min_y = min([y for y, c in [(ls_y, ls_c), (rs_y, rs_c)] if c > conf_thresh])
            max_y = max([y for y, c in [(lh_y, lh_c), (rh_y, rh_c)] if c > conf_thresh])

            if max_x - min_x < 10:
                min_x -= shoulder_width / 2
                max_x += shoulder_width / 2

            margin = (max_x - min_x) * 0.2
            waist_y = max_y

            left_hidden = False
            right_hidden = False

            if left_side_visible:
                if lw_c > conf_thresh:
                    if (min_x - margin) < lw_x < (max_x + margin) and min_y < lw_y < waist_y:
                        left_hidden = True
                elif le_c > conf_thresh:
                    if (min_x - margin) < le_x < (max_x + margin) and le_y < waist_y:
                        left_hidden = True

            if right_side_visible:
                if rw_c > conf_thresh:
                    if (min_x - margin) < rw_x < (max_x + margin) and min_y < rw_y < waist_y:
                        right_hidden = True
                elif re_c > conf_thresh:
                    if (min_x - margin) < re_x < (max_x + margin) and re_y < waist_y:
                        right_hidden = True

            if left_hidden and right_hidden and lw_c > conf_thresh and rw_c > conf_thresh:
                dist_between_hands = abs(lw_x - rw_x) + abs(lw_y - rw_y)
                if dist_between_hands < (shoulder_width * 1.0) and lw_y > (ls_y + shoulder_width):
                    left_hidden = False
                    right_hidden = False

            if left_hidden or right_hidden:
                is_hidden = True

        if is_hidden:
            self.history[track_id]["hidden_frames"] += 1
        else:
            self.history[track_id]["hidden_frames"] = max(0, self.history[track_id]["hidden_frames"] - 1)

        if self.history[track_id]["hidden_frames"] > self.thresh_hidden:
            alerts.append("Mao Oculta")

        return alerts

    def clean_old_tracks(self, current_tracks):
        """Remove historico apenas apos algumas perdas consecutivas de tracking."""
        missing_tracks = set(self.history.keys()) - set(current_tracks)
        for track_id in missing_tracks:
            self.history[track_id]["missing_frames"] += 1
            if self.history[track_id]["missing_frames"] > self.track_ttl_frames:
                del self.history[track_id]

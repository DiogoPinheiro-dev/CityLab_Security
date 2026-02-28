import numpy as np
from collections import defaultdict

class GestureAnalyzer:
    def __init__(self, fps=30):
        # Histórico de frames contínuos em que a pessoa está fazendo o gesto
        self.history = defaultdict(lambda: {
            "hidden_frames": 0,
            "surrender_frames": 0,
            "aiming_frames": 0
        })
        
        # Limiares de tempo (frames) para confirmar o gesto
        # Considerando 30 FPS.
        self.fps = fps
        self.thresh_hidden = int(self.fps * 2.5)  # 2.5 segundos oculto
        self.thresh_surrender = int(self.fps * 1.0) # 1.0 segundo com mão pro alto
        self.thresh_aiming = int(self.fps * 1.5)    # 1.5 segundo apontando

    def _get_keypoint(self, keypoints, idx):
        # Retorna (x, y, conf) do keypoint
        # YOLOv8 pose default tem 17 keypoints.
        kp = keypoints[idx]
        return kp[0], kp[1], kp[2]

    def analyze(self, track_id, keypoints, box=None):
        """
        Analisa os limiares de pose e atualiza o histórico.
        Retorna uma lista de alertas ativos para a pessoa.
        """
        alerts = []
        
        # Índices do COCO:
        # 5: L Shoulder, 6: R Shoulder
        # 7: L Elbow,    8: R Elbow
        # 9: L Wrist,    10: R Wrist
        # 11: L Hip,     12: R Hip
        # 15: L Ankle,   16: R Ankle

        # Extraindo coordenadas e confianças
        ls_x, ls_y, ls_c = self._get_keypoint(keypoints, 5)
        rs_x, rs_y, rs_c = self._get_keypoint(keypoints, 6)
        
        lw_x, lw_y, lw_c = self._get_keypoint(keypoints, 9)
        rw_x, rw_y, rw_c = self._get_keypoint(keypoints, 10)
        
        le_x, le_y, le_c = self._get_keypoint(keypoints, 7)
        re_x, re_y, re_c = self._get_keypoint(keypoints, 8)

        lh_x, lh_y, lh_c = self._get_keypoint(keypoints, 11)
        rh_x, rh_y, rh_c = self._get_keypoint(keypoints, 12)
        
        la_x, la_y, la_c = self._get_keypoint(keypoints, 15)
        ra_x, ra_y, ra_c = self._get_keypoint(keypoints, 16)

        conf_thresh = 0.5 # Confiança mínima para considerar o keypoint válido

        # Largura do ombro como base de escala para usar como medida relativa
        # (Se estiver de lado, a largura será muito pequena, então pegamos a distância do ombro ao quadril como alternativa se necessário)
        shoulder_dist = 40.0
        if ls_c > conf_thresh and rs_c > conf_thresh:
            shoulder_dist = abs(ls_x - rs_x) + 0.1
            
        # De lado, ombros se alinham no eixo X. Se a largura for menor que 20 (muito fina), 
        # a pessoa está de perfil. Calculamos uma margem de escala com base na distância ombro-quadril
        if shoulder_dist < 20.0 and ls_c > conf_thresh and lh_c > conf_thresh:
            shoulder_width = abs(ls_y - lh_y) * 0.4 # Estima a largura média do ombro baseada no "pescoço ao quadril"
        else:
            shoulder_width = max(shoulder_dist, 40.0) # Usa o ombro, ou valor default de segurança

        # --- 1. Apontando Arma (Braços Estendidos) ---
        # Pulsos na mesma linha dos ombros (Y similar) e distantes do ombro
        is_aiming = False
        if ls_c > conf_thresh and lw_c > conf_thresh and le_c > conf_thresh:
            arm_length = abs(ls_x - le_x) + abs(le_x - lw_x)
            if arm_length > 10: # Evitar divisão por zero e braço muito curto na câmera
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

        # --- 2. Rendição (Mãos para o alto ou Mãos na nuca) ---
        # Funciona mesmo se apenas 1 braço estiver visível (margem da câmera)
        is_surrendering = False

        if not is_aiming:
            margin_y = shoulder_width * 0.4
            
            # Condição A: Mãos para o alto (Pulsos acima da linha dos ombros)
            left_hands_up = (ls_c > conf_thresh and lw_c > conf_thresh and lw_y < (ls_y - margin_y))
            right_hands_up = (rs_c > conf_thresh and rw_c > conf_thresh and rw_y < (rs_y - margin_y))
            
            # Condição B: Mãos na nuca/atrás da cabeça (por braço)
            left_behind_head = False
            right_behind_head = False
            
            if ls_c > conf_thresh and lw_c > conf_thresh and le_c > conf_thresh:
                if le_x < (ls_x - margin_y) and lw_y < (ls_y + margin_y) and lw_x > le_x:
                    left_behind_head = True
            
            if rs_c > conf_thresh and rw_c > conf_thresh and re_c > conf_thresh:
                if re_x > (rs_x + margin_y) and rw_y < (rs_y + margin_y) and rw_x < re_x:
                    right_behind_head = True

            # Se detectar QUALQUER um dos braços levantados ou na nuca, já aciona
            if left_hands_up or right_hands_up or left_behind_head or right_behind_head:
                is_surrendering = True
        
        if is_surrendering:
            self.history[track_id]["surrender_frames"] += 1
        else:
            self.history[track_id]["surrender_frames"] = max(0, self.history[track_id]["surrender_frames"] - 2)

        if self.history[track_id]["surrender_frames"] > self.thresh_surrender:
            alerts.append("Rendicao")

        # --- 3. Mão Oculta na Jaqueta/Cintura ---
        # Funciona para cada braço separadamente
        is_hidden = False
        
        # Só conseguimos calcular a "área do torso" se tivermos pelo menos 1 ombro e 1 quadril do mesmo lado
        left_side_visible = (ls_c > conf_thresh and lh_c > conf_thresh)
        right_side_visible = (rs_c > conf_thresh and rh_c > conf_thresh)
        
        if left_side_visible or right_side_visible:
            # Pega as margens baseadas no que estiver visível (se faltar um lado, assume a posição do outro)
            min_x = min([x for x, c in [(ls_x, ls_c), (rs_x, rs_c), (lh_x, lh_c), (rh_x, rh_c)] if c > conf_thresh])
            max_x = max([x for x, c in [(ls_x, ls_c), (rs_x, rs_c), (lh_x, lh_c), (rh_x, rh_c)] if c > conf_thresh])
            min_y = min([y for y, c in [(ls_y, ls_c), (rs_y, rs_c)] if c > conf_thresh])
            max_y = max([y for y, c in [(lh_y, lh_c), (rh_y, rh_c)] if c > conf_thresh])
            
            # Se só um lado for visível, min e max serão iguais. Criamos uma "caixa" imaginária com a largura do ombro
            if max_x - min_x < 10:
                min_x -= shoulder_width / 2
                max_x += shoulder_width / 2
            
            margin = (max_x - min_x) * 0.2

            left_hidden = False
            right_hidden = False

            # Para evitar falsos positivos com braços relaxados ou braços cruzados normais:
            # 1. Os pulsos precisam estar ACIMA da linha do quadril (se o braço está solto pra baixo cruzando a perna, não é perigoso).
            # 2. Se o pulso não for visto (baixa convicção), o cotovelo tem que estar ativamente dobrado pra cima/frente.
            
            # Altura do Bounding Box (quadril ou ombro ao invés de max_y pra ter certeza que a mão está alta o suficiente)
            waist_y = max_y # linha do quadril

            # Avalia braço esquerdo
            if left_side_visible:
                if lw_c > conf_thresh:
                    # Se o pulso está visível, ele tem que estar na região do peito/barriga (não solto pra baixo)
                    # e voltado para dentro da "caixa" do corpo.
                    if (min_x - margin) < lw_x < (max_x + margin) and min_y < lw_y < waist_y:
                        left_hidden = True
                elif le_c > conf_thresh:
                    # Se o pulso sumiu, o cotovelo precisa estar dobrado na direção do peito (Y menor que o quadril)
                    if (min_x - margin) < le_x < (max_x + margin) and le_y < waist_y:
                        left_hidden = True

            # Avalia braço direito
            if right_side_visible:
                if rw_c > conf_thresh:
                    if (min_x - margin) < rw_x < (max_x + margin) and min_y < rw_y < waist_y:
                        right_hidden = True
                elif re_c > conf_thresh:
                    if (min_x - margin) < re_x < (max_x + margin) and re_y < waist_y:
                        right_hidden = True

            # Exceção para Braços Cruzados "Relaxados":
            # Se as duas mãos estão visíveis, juntas (x e y próximos) e no centro inferior do corpo, é braço cruzado de descanso.
            if left_hidden and right_hidden and lw_c > conf_thresh and rw_c > conf_thresh:
                dist_between_hands = abs(lw_x - rw_x) + abs(lw_y - rw_y)
                # Se as mãos estiverem muito pertos uma da outra e perto do umbigo/quadril, ignora.
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

        # (Aiming logic was moved above to prevent surrender false positives)

        return alerts

    def clean_old_tracks(self, current_tracks):
        """Remove histórico de IDs que não estão mais na tela"""
        missing_tracks = set(self.history.keys()) - set(current_tracks)
        for track_id in missing_tracks:
            del self.history[track_id]

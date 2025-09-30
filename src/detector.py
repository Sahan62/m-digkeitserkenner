import mediapipe as mp
import cv2
import numpy as np

class Detector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Augenpunkte für EAR
        self.left_eye_idxs = [362, 385, 387, 263, 373, 380]
        self.right_eye_idxs = [33, 160, 158, 133, 153, 144]

    def compute_EAR(self, eye_pts):
        A = np.linalg.norm(np.array(eye_pts[1]) - np.array(eye_pts[5]))
        B = np.linalg.norm(np.array(eye_pts[2]) - np.array(eye_pts[4]))
        C = np.linalg.norm(np.array(eye_pts[0]) - np.array(eye_pts[3]))
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        EAR = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Landmarks einzeichnen
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Augenpunkte sammeln
                left_eye_pts = [(int(face_landmarks.landmark[idx].x * w),
                                 int(face_landmarks.landmark[idx].y * h)) for idx in self.left_eye_idxs]

                right_eye_pts = [(int(face_landmarks.landmark[idx].x * w),
                                  int(face_landmarks.landmark[idx].y * h)) for idx in self.right_eye_idxs]

                # Punkte auf Frame zeichnen
                for pt in left_eye_pts:
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)
                for pt in right_eye_pts:
                    cv2.circle(frame, pt, 2, (0, 0, 255), -1)

                # EAR berechnen
                left_EAR = self.compute_EAR(left_eye_pts)
                right_EAR = self.compute_EAR(right_eye_pts)
                EAR = (left_EAR + right_EAR) / 2.0

                cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, EAR

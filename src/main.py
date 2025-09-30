import cv2
from detector import Detector
from logger import Logger

# Logger initialisieren
logger = Logger("events.log")

# Detector initialisieren
detector = Detector()

# Kamera starten
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Gesichter & EAR erkennen
    frame, EAR = detector.process_frame(frame)

    # Wenn EAR unter Threshold, Logging
    if EAR is not None and EAR < 0.2:
        logger.log_event("Müdigkeit erkannt!")

    # Frame anzeigen
    cv2.imshow("Müdigkeitserkennung", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

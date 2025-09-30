# MÜDIGKEITSERKENNER – FACIAL LANDMARK DETECTION & EAR

## ÜBERSICHT
Dies ist ein **Python-Projekt zur Müdigkeitserkennung**, das **Live-Kamerabilder** analysiert und **Facial Landmarks** verwendet.  
Die Augen werden erkannt und anhand des **Eye Aspect Ratio (EAR)** kann Müdigkeit oder Augenschließen erkannt werden.

---

## FEATURES
- **Echtzeit-Gesichtserkennung** mit Mediapipe Face Mesh  
- **Augen, Augenbrauen, Lippen** werden hervorgehoben  
- **EAR-Berechnung** zur Müdigkeitserkennung (Augen schließen)  
- Optional: **Videoaufnahme** der Live-Kamera  

---

## VORAUSSETZUNGEN
- Python 3.11  
- OpenCV  
- Mediapipe  
- Matplotlib (optional, für Visualisierungen)

## ANMERKUNGEN
- Drücke 'q', um die Live-Kamera zu schließen
- Videoaufzeichnung wird optional im Projektordner gespeichert
- EAR-Algorithmen und Landmark-Punkte sind in detector.py definiert


```bash
pip install opencv-python mediapipe matplotlib

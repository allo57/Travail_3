from ultralytics import YOLO

# Charger un modèle pré-entraîné
model = YOLO("yolov8n.pt")

# Image à analyser
resultats = model("Ydger.jpg", show=True)

# Afficher les résultats dans la console
for r in resultats:
    print(r.boxes)

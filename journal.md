# Journal de bord — Projet YOLOv8
## Équipe : William-Jacob
## Semaine 2 — Exploration technique

### 📅 Date :
2025-11-18

---

## 🔍 1. Lecture de la documentation YOLOv8

Cette semaine, j’ai consulté la documentation officielle de YOLOv8 (Ultralytics).  
Référence : https://docs.ultralytics.com/

### Informations importantes retenues :
- YOLOv8 est un modèle de détection d’objets basé sur PyTorch.
- Plusieurs variantes existent : `n`, `s`, `m`, `l`, `x`.
- YOLOv8 supporte plusieurs tâches : détection, classification, segmentation, pose estimation.
- L’architecture est divisée en : backbone, neck et head.
- Le modèle peut être utilisé facilement via la librairie `ultralytics`.

---

## 🖼️ 2. Tests de YOLO sur images

### Script utilisé :
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
resultats = model("Ydger.jpg", show=True)

for r in resultats:
    print(r.boxes)
```

### Résultats obtenus :
- YOLO a détecté **1 personne** dans l’image.
- Score de confiance : **0.9022**
- Classe détectée : `person` (classe 0)
- Bounding box générée.

Exemple de sortie :
```
cls: tensor([0.])
conf: tensor([0.9022])
xyxy: tensor([[41.4, 41.2, 1246.9, 1248.8]])
```

---

## 🧠 3. Compréhension des concepts techniques

### Bounding boxes :
YOLO retourne les coordonnées sous différents formats :
- `xyxy` : (x_min, y_min, x_max, y_max)
- `xywh` : (x_center, y_center, width, height)
- versions normalisées (`xyxyn`, `xywhn`)

### Score de confiance :
- Indique la certitude du modèle.
- Valeurs élevées (0.80–1.00) = très fiable.
- Dans notre test → **0.9022**, YOLO est très confiant.

---

## 🤖 4. Utilisation de l’IA cette semaine

### L’IA a été utilisée pour :
- comprendre les formats `xyxy`, `xywh`, `conf`, `cls`
- résoudre l’erreur : `FileNotFoundError: Ydger.jpg does not exist`
- comprendre pourquoi le script doit être exécuté dans le bon dossier
- obtenir des explications claires sur l’architecture YOLOv8
- organiser correctement notre projet

### Exemple de prompt utilisé :
> « Explique-moi les valeurs retournées dans Boxes par YOLOv8. »

---

## ⚙️ 5. Problèmes rencontrés

- L’image `Ydger.jpg` n’était pas trouvée par YOLO → problème de mauvais répertoire d’exécution.
- Corrigé en exécutant Python dans :  
  `C:\Users\willi\OneDrive\Bureau\Yolo\Travail_3`

---

## 🎯 6. Objectifs pour la semaine suivante

- Détection sur **vidéo**.
- Détection en **webcam (temps réel)**.
- Création d’un script Python organisé.
- Préparer la structure du projet pour le dépôt GitHub.

---

## ✔️ Statut : Semaine 2 terminée avec succès.

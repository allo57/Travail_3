from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import cv2
import time


def generer_rapport(resultats, source):
    """
    Génère un fichier report_YYYYMMDD_HHMMSS.txt dans le dossier 'reports'
    avec le résumé des détections.
    """
    base_dir = Path(__file__).resolve().parent
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"report_{timestamp}.txt"

    lignes = []
    lignes.append("===== RAPPORT YOLOv8 =====\n")
    lignes.append(f"Source analysée : {source}\n\n")

    total_objets = 0
    compteur_classes = {}

    for r in resultats:
        boxes = r.boxes
        names = r.names  # dict id -> nom de classe

        if boxes is None or len(boxes) == 0:
            continue

        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            nom_classe = names[int(cls_id)]
            total_objets += 1
            compteur_classes[nom_classe] = compteur_classes.get(nom_classe, 0) + 1
            lignes.append(f"- {nom_classe} (confiance: {conf:.3f})\n")

    lignes.append("\nRésumé par classe :\n")
    for nom_classe, count in compteur_classes.items():
        lignes.append(f"- {nom_classe} : {count}\n")

    lignes.append(f"\nTotal d’objets détectés : {total_objets}\n")
    lignes.append("===========================\n")

    report_path.write_text("".join(lignes), encoding="utf-8")
    print(f"Rapport généré : {report_path}")


def webcam_avec_fps(model):
    """
    Ouvre la webcam, affiche les détections YOLOv8
    + affiche les FPS en temps réel.
    Appuie sur 'q' pour fermer.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERREUR : Impossible d’ouvrir la webcam.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERREUR : Impossible de lire l’image de la webcam.")
            break

        # Lancer YOLO sur l’image courante
        resultats = model(frame)

        # Annoter l’image avec les boîtes
        annotated_frame = resultats[0].plot()

        # Calcul des FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        # Afficher les FPS sur l’image
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Webcam YOLOv8", annotated_frame)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Session webcam terminée.")


def main():
    # Charger le modèle YOLOv8 (nano, rapide)
    model = YOLO("yolov8n.pt")

    print("===== MENU YOLOv8 — Détection d’objets =====")
    print("1 - Détection sur une image")
    print("2 - Détection sur une vidéo")
    print("3 - Détection en temps réel (webcam avec FPS)")
    choix = input("Choisis une option (1, 2 ou 3) : ")

    if choix == "1":
        saisie = input("Chemin de l'image (ex: Ydger.jpg) : ")
        base_dir = Path(__file__).resolve().parent
        chemin = (base_dir / saisie).resolve()

        print("Chemin utilisé :", chemin)
        if not chemin.exists():
            print("ERREUR : Le fichier n’existe pas.")
        else:
            resultats = model(str(chemin), show=True, save=True)
            print("Détection terminée sur l'image.")
            # Générer un rapport texte
            generer_rapport(resultats, chemin)

    elif choix == "2":
        saisie = input("Chemin de la vidéo (ex: video.mp4) : ")
        base_dir = Path(__file__).resolve().parent
        chemin = (base_dir / saisie).resolve()
        print("Chemin utilisé :", chemin)

        if not chemin.exists():
            print("ERREUR : Le fichier vidéo n’existe pas.")
        else:
            resultats = model(str(chemin), show=True, save=True)
            print("Détection terminée sur la vidéo.")
            # Générer un rapport texte
            generer_rapport(resultats, chemin)

    elif choix == "3":
        print("Ouverture de la webcam... (appuie sur 'q' pour fermer la fenêtre)")
        webcam_avec_fps(model)

    else:
        print("Choix invalide. Relance le programme et choisis 1, 2 ou 3.")


if __name__ == "__main__":
    main()

from ultralytics import YOLO
from pathlib import Path

def main():
    # Charger le modèle YOLOv8 (nano, rapide)
    model = YOLO("yolov8n.pt")

    print("===== MENU YOLOv8 — Détection d’objets =====")
    print("1 - Détection sur une image")
    print("2 - Détection sur une vidéo")
    print("3 - Détection en temps réel (webcam)")
    choix = input("Choisis une option (1, 2 ou 3) : ")

    if choix == "1":
        saisie = input("Chemin de l'image (ex: Ydger.jpg) : ")

        base_dir = Path(__file__).resolve().parent
        sortie = base_dir / "detect_output"

        chemin = (base_dir / saisie).resolve()

        print("Chemin utilisé :", chemin)

        if not chemin.exists():
            print("ERREUR : L'image n'existe pas.")
        else:
            resultats = model(
                str(chemin),
                show=True,
                save=True,
                project=str(sortie),
                name="exp"
            )
            print("Détection terminée.")

    elif choix == "2":
        saisie = input("Chemin de la vidéo (ex: video.mp4) : ")

        base_dir = Path(__file__).resolve().parent
        sortie = base_dir / "detect_output"

        chemin = (base_dir / saisie).resolve()

        print("Chemin utilisé :", chemin)

        if not chemin.exists():
            print("ERREUR : La vidéo n'existe pas.")
        else:
            resultats = model(
                str(chemin),
                show=True,
                save=True,
                project=str(sortie),
                name="exp"
            )
            print("Détection terminée sur la vidéo.")


    elif choix == "3":
        print("Ouverture de la webcam... (appuie sur 'q' pour fermer la fenêtre)")

        # 0 = première webcam du système
        resultats = model(0, show=True)
        print("Session webcam terminée.")

    else:
        print("Choix invalide. Relance le programme et choisis 1, 2 ou 3.")


if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from pathlib import Path
import cv2
import time
from datetime import datetime
from collections import Counter


class YoloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 – Interface graphique")
        self.root.geometry("900x650")

        # Couleur de fond
        self.root.configure(bg="#20232a")

        # Charger le modèle YOLO une seule fois
        self.model = YOLO("yolov8n.pt")

        # Dossiers de sortie
        base_dir = Path(__file__).resolve().parent
        self.output_dir = base_dir / "detect_output"
        self.reports_dir = base_dir / "reports"
        self.output_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)

        # Titre
        title_label = tk.Label(
            root,
            text="Projet YOLOv8 – Détection d’objets",
            font=("Segoe UI", 18, "bold"),
            bg="#20232a",
            fg="white"
        )
        title_label.pack(pady=10)

        # Sous-titre
        subtitle_label = tk.Label(
            root,
            text="Image · Vidéo · Webcam (FPS + rapport)",
            font=("Segoe UI", 11),
            bg="#20232a",
            fg="#bbbbbb"
        )
        subtitle_label.pack(pady=(0, 15))

        # Frame boutons
        self.btn_frame = tk.Frame(root, bg="#20232a")
        self.btn_frame.pack(pady=10)

        Button(
            self.btn_frame,
            text="🖼 Détection Image",
            width=20,
            font=("Segoe UI", 11),
            command=self.detect_image
        ).grid(row=0, column=0, padx=10)

        Button(
            self.btn_frame,
            text="🎬 Détection Vidéo",
            width=20,
            font=("Segoe UI", 11),
            command=self.detect_video
        ).grid(row=0, column=1, padx=10)

        Button(
            self.btn_frame,
            text="📷 Webcam (FPS)",
            width=20,
            font=("Segoe UI", 11),
            command=self.detect_webcam
        ).grid(row=0, column=2, padx=10)

        # Zone d'affichage (image détectée)
        self.display_label = Label(root, bg="#20232a")
        self.display_label.pack(pady=20)

        # Label de statut
        self.status_label = tk.Label(
            root,
            text="Prêt.",
            font=("Segoe UI", 10),
            bg="#20232a",
            fg="#bbbbbb"
        )
        self.status_label.pack(pady=5)

    # ─────────────────────────────────────────────
    # Utils : génération de rapport texte
    # ─────────────────────────────────────────────
    def generate_report_from_results(self, results, source_label: str):
        """
        Génère un rapport texte (résumé des classes + confidences)
        à partir d'une liste de Results Ultralytics.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"report_{timestamp}.txt"

        lines = []
        lines.append("===== RAPPORT YOLOv8 =====\n")
        lines.append(f"Source analysée : {source_label}\n\n")

        total_objects = 0
        class_counts = Counter()

        for r in results:
            boxes = r.boxes
            names = r.names  # dict id -> nom de classe

            if boxes is None or len(boxes) == 0:
                continue

            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                class_name = names[int(cls_id)]
                total_objects += 1
                class_counts[class_name] += 1
                lines.append(f"- {class_name} (confiance: {conf:.3f})\n")

        lines.append("\nRésumé par classe :\n")
        if class_counts:
            for class_name, count in class_counts.items():
                lines.append(f"- {class_name} : {count}\n")
        else:
            lines.append("Aucun objet détecté.\n")

        lines.append(f"\nTotal d’objets détectés : {total_objects}\n")
        lines.append("===========================\n")

        report_path.write_text("".join(lines), encoding="utf-8")
        print(f"Rapport généré : {report_path}")
        return report_path

    def generate_report_from_counter(self, counter: Counter, source_label: str):
        """
        Génère un rapport à partir d'un Counter (utilisé pour la webcam).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"report_{timestamp}.txt"

        lines = []
        lines.append("===== RAPPORT YOLOv8 (Webcam) =====\n")
        lines.append(f"Source : {source_label}\n\n")

        total_objects = sum(counter.values())
        if counter:
            lines.append("Résumé par classe (sur toute la session) :\n")
            for class_name, count in counter.items():
                lines.append(f"- {class_name} : {count}\n")
        else:
            lines.append("Aucune détection enregistrée.\n")

        lines.append(f"\nTotal d’objets détectés : {total_objects}\n")
        lines.append("===========================\n")

        report_path.write_text("".join(lines), encoding="utf-8")
        print(f"Rapport généré : {report_path}")
        return report_path

    # ─────────────────────────────────────────────
    # 1) Détection sur image
    # ─────────────────────────────────────────────
    def detect_image(self):
        fichier = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not fichier:
            return

        self.status_label.config(text=f"Analyse de l’image : {fichier}")
        self.root.update()

        try:
            results = self.model(
                fichier,
                save=True,
                project=str(self.output_dir),
                name="image"
            )

            # YOLO crée un fichier dans detect_output/image/
            output_folder = self.output_dir / "image"
            images = sorted(output_folder.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
            if images:
                image_sortie = images[-1]
                self.show_image(image_sortie)

            report_path = self.generate_report_from_results(results, fichier)
            messagebox.showinfo(
                "YOLOv8",
                f"Détection terminée sur l’image.\nRapport généré :\n{report_path}"
            )
            self.status_label.config(text="Analyse de l’image terminée.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l’analyse de l’image :\n{e}")
            self.status_label.config(text="Erreur lors de l’analyse de l’image.")

    # ─────────────────────────────────────────────
    # 2) Détection vidéo
    # ─────────────────────────────────────────────
    def detect_video(self):
        fichier = filedialog.askopenfilename(
            title="Choisir une vidéo",
            filetypes=[("Vidéos", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not fichier:
            return

        self.status_label.config(text=f"Analyse de la vidéo : {fichier}")
        self.root.update()

        try:
            results = self.model(
                fichier,
                save=True,
                project=str(self.output_dir),
                name="video"
            )

            report_path = self.generate_report_from_results(results, fichier)
            messagebox.showinfo(
                "Vidéo",
                "Détection terminée !\n"
                "La vidéo détectée est dans detect_output/video/\n\n"
                f"Rapport généré :\n{report_path}"
            )
            self.status_label.config(text="Analyse de la vidéo terminée.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l’analyse de la vidéo :\n{e}")
            self.status_label.config(text="Erreur lors de l’analyse de la vidéo.")

    # ─────────────────────────────────────────────
    # 3) Détection via webcam (avec FPS + rapport)
    # ─────────────────────────────────────────────
    def detect_webcam(self):
        self.status_label.config(
            text="Webcam en cours… (appuie sur 'q' dans la fenêtre vidéo pour quitter)"
        )
        self.root.update()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'ouvrir la webcam.")
            self.status_label.config(text="Erreur : webcam non disponible.")
            return

        prev_time = time.time()
        class_counts = Counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            res = results[0]
            boxes = res.boxes
            names = res.names

            # Comptage des classes détectées sur ce frame
            if boxes is not None and len(boxes) > 0:
                for cls_id in boxes.cls.tolist():
                    class_name = names[int(cls_id)]
                    class_counts[class_name] += 1

            # Annoter l’image avec les boîtes
            annotated = res.plot()

            # Calcul des FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # Afficher les FPS sur l’image
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Webcam YOLOv8 - Appuie sur q pour quitter", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Générer le rapport de la session webcam
        report_path = self.generate_report_from_counter(class_counts, "Webcam (session)")
        messagebox.showinfo(
            "Webcam",
            f"Session webcam terminée.\nRapport généré :\n{report_path}"
        )
        self.status_label.config(text="Session webcam terminée.")

    # ─────────────────────────────────────────────
    # Afficher une image dans Tkinter
    # ─────────────────────────────────────────────
    def show_image(self, path: Path):
        img = Image.open(path)
        # Resize en gardant une taille raisonnable
        img = img.resize((750, 450))
        img_tk = ImageTk.PhotoImage(img)

        self.display_label.configure(image=img_tk)
        self.display_label.image = img_tk  # empêcher le garbage collector


# ─────────────────────────────────────────────
# Lancer l’application
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = YoloApp(root)
    root.mainloop()

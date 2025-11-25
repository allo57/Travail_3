import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
from pathlib import Path
import cv2


class YoloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 – Interface graphique")
        self.root.geometry("800x600")

        # Charger modèle YOLO
        self.model = YOLO("yolov8n.pt")

        # Frame boutons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=20)

        Button(self.btn_frame, text="Détection Image", width=20,
               command=self.detect_image).grid(row=0, column=0, padx=10)

        Button(self.btn_frame, text="Détection Vidéo", width=20,
               command=self.detect_video).grid(row=0, column=1, padx=10)

        Button(self.btn_frame, text="Webcam", width=20,
               command=self.detect_webcam).grid(row=0, column=2, padx=10)

        # Zone d'affichage (image détectée)
        self.display_label = Label(root)
        self.display_label.pack()

        # Dossier sortie YOLO
        self.output_dir = Path(__file__).resolve().parent / "detect_output"

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

        result = self.model(
            fichier,
            save=True,
            project=str(self.output_dir),
            name="image"
        )

        # YOLO crée un fichier dans detect_output/image/
        image_sortie = list((self.output_dir / "image").glob("*.jpg"))[-1]

        # Affichage Tkinter
        self.show_image(image_sortie)

    # ─────────────────────────────────────────────
    # 2) Détection vidéo
    # ─────────────────────────────────────────────
    def detect_video(self):
        fichier = filedialog.askopenfilename(
            title="Choisir une vidéo",
            filetypes=[("Vidéos", "*.mp4 *.avi *.mov")]
        )
        if not fichier:
            return

        self.model(
            fichier,
            save=True,
            project=str(self.output_dir),
            name="video"
        )

        tk.messagebox.showinfo("Vidéo", "Détection terminée !\nLa vidéo détectée est dans detect_output/video/")

    # ─────────────────────────────────────────────
    # 3) Détection via webcam
    # ─────────────────────────────────────────────
    def detect_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            tk.messagebox.showerror("Erreur", "Impossible d'ouvrir la webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            annotated = results[0].plot()

            cv2.imshow("Webcam YOLOv8 - Appuie sur q pour quitter", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ─────────────────────────────────────────────
    # Méthode pour afficher une image dans Tkinter
    # ─────────────────────────────────────────────
    def show_image(self, path):
        img = Image.open(path)
        img = img.resize((700, 500))
        img_tk = ImageTk.PhotoImage(img)

        self.display_label.configure(image=img_tk)
        self.display_label.image = img_tk  # prévention garbage collector


# ─────────────────────────────────────────────
# Lancer l’application
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = YoloApp(root)
    root.mainloop()

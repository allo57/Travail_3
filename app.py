import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import time
from pathlib import Path
from collections import Counter

# Configuration de la page
st.set_page_config(
    page_title="YOLOv8 - Exploration IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés pour un look "Premium"
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border-color: #ff3333;
    }
    h1 {
        color: #fafafa;
    }
    h2, h3 {
        color: #e0e0e0;
    }
    .report-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4c4c4c;
    }
    </style>
    """, unsafe_allow_html=True)

# Chargement du modèle (mis en cache pour la performance)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Titre et Introduction
st.title("🤖 Projet 3 : Exploration IA avec YOLOv8")
st.markdown("### Détection d'objets en temps réel")
st.markdown("---")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choisir le mode :", ["🖼️ Image", "🎬 Vidéo", "📷 Webcam"])

st.sidebar.markdown("---")
st.sidebar.info(
    "Ce projet explore l'utilisation de l'IA pour la vision par ordinateur. "
    "Il utilise le modèle **YOLOv8** pour détecter des objets dans des images, vidéos et flux webcam."
)

# ─────────────────────────────────────────────
# MODE IMAGE
# ─────────────────────────────────────────────
if mode == "🖼️ Image":
    st.header("Détection sur Image")
    
    uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image originale", use_container_width=True)
        
        if st.button("Lancer la détection"):
            with st.spinner("Analyse en cours..."):
                results = model(image)
                res = results[0]
                annotated_img = res.plot()
                st.image(annotated_img, caption="Résultat de la détection", use_container_width=True)
                
                # Affichage des résultats sous l'image
                st.markdown("### 📋 Résultats de la détection")
                boxes = res.boxes
                if boxes:
                    class_counts = Counter()
                    names = res.names
                    
                    # Collecter les objets détectés
                    detected_items = []
                    for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                        class_name = names[int(cls_id)]
                        class_counts[class_name] += 1
                        detected_items.append(f"{class_name} ({conf:.2f})")
                    
                    # Afficher le résumé
                    st.success(f"Objets détectés : {len(boxes)}")
                    
                    # Afficher le détail
                    st.markdown("**Détails :**")
                    st.write(", ".join(detected_items))
                    
                    # Afficher le comptage par classe
                    st.markdown("**Résumé par classe :**")
                    col_metrics = st.columns(len(class_counts))
                    for idx, (name, count) in enumerate(class_counts.items()):
                        with col_metrics[idx % len(col_metrics)]:
                            st.metric(label=name, value=count)
                else:
                    st.warning("Aucun objet détecté.")

# ─────────────────────────────────────────────
# MODE VIDÉO
# ─────────────────────────────────────────────
elif mode == "🎬 Vidéo":
    st.header("Détection sur Vidéo")
    
    uploaded_video = st.file_uploader("Choisissez une vidéo...", type=['mp4', 'mov', 'avi', 'mkv'])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("Analyser la vidéo"):
            st.warning("L'analyse vidéo peut prendre du temps...")
            
            st_frame = st.empty()
            st_results = st.empty() # Placeholder pour les résultats sous la vidéo
            
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame)
                res = results[0]
                annotated_frame = res.plot()
                
                # Convertir BGR vers RGB
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, caption="Traitement en cours...", use_container_width=True)
                
                # Afficher les résultats du frame courant sous la vidéo
                boxes = res.boxes
                if boxes:
                    names = res.names
                    detected_in_frame = [f"{names[int(c)]}" for c in boxes.cls.tolist()]
                    counts = Counter(detected_in_frame)
                    summary = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                    st_results.info(f"**Détecté dans ce cadre :** {summary}")
                else:
                    st_results.info("Rien détecté dans ce cadre.")
            
            cap.release()
            st.success("Analyse terminée !")
            
            # Nettoyage du fichier temporaire
            try:
                import os
                os.unlink(video_path)
            except Exception as e:
                print(f"Erreur lors de la suppression du fichier temporaire : {e}")

# ─────────────────────────────────────────────
# MODE WEBCAM
# ─────────────────────────────────────────────
elif mode == "📷 Webcam":
    st.header("Détection Webcam en Temps Réel")
    
    run = st.checkbox('Démarrer la Webcam')
    FRAME_WINDOW = st.image([])
    RESULTS_WINDOW = st.empty() # Placeholder pour les résultats
    
    if run:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Impossible d'accéder à la webcam.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erreur de lecture du flux webcam.")
                    break
                
                results = model(frame)
                res = results[0]
                annotated_frame = res.plot()
                
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)
                
                # Afficher les résultats sous la webcam
                boxes = res.boxes
                if boxes:
                    names = res.names
                    detected_in_frame = [f"{names[int(c)]}" for c in boxes.cls.tolist()]
                    counts = Counter(detected_in_frame)
                    summary = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                    RESULTS_WINDOW.info(f"**Détecté :** {summary}")
                else:
                    RESULTS_WINDOW.info("Rien détecté.")
            
            cap.release()
    else:
        st.info("Cochez la case ci-dessus pour activer la webcam.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Projet réalisé dans le cadre du cours 'Explorer une technologie avec l'IA'.<br>"
    "Propulsé par YOLOv8 et Streamlit."
    "</div>",
    unsafe_allow_html=True
)

if __name__ == "__main__":
    import sys
    import subprocess
    import os
    
    # Évite la récursion infinie : si on a déjà lancé le sous-processus, on ne fait rien
    if not os.environ.get("STREAMLIT_FROM_SUBPROCESS"):
        # On prépare l'environnement avec le flag
        env = os.environ.copy()
        env["STREAMLIT_FROM_SUBPROCESS"] = "true"
        
        # On lance streamlit dans un processus séparé
        # sys.executable assure qu'on utilise le même python
        subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]], env=env)
        sys.exit(0)

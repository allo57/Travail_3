# Journal de bord — Projet YOLOv8
## Équipe : William - Jacob
## Semaine 3 — Mise en pratique

### 📅 Date :
2025-11-25

---

## 🚀 1. Début de la mise en pratique

Cette semaine, nous avons transformé le projet YOLO en un **prototype fonctionnel complet**, incluant une interface graphique, la détection sur image, vidéo et webcam, ainsi que la génération automatique de rapports.

L’objectif était d’aller au-delà des tests simples faits en Semaine 2 et d’obtenir une application capable d’être présentée à l’enseignant.

---

## 🖥️ 2. Création d’une interface graphique (Tkinter)

Nous avons développé une interface graphique permettant d’utiliser YOLO facilement sans passer par le terminal.

### Fonctionnalités ajoutées :
- 🖼 **Détection sur image**  
  Chargement d’une image, détection YOLO, affichage dans l'interface et génération de rapport.

- 🎬 **Détection sur vidéo**  
  Lecture d’une vidéo, analyse frame par frame, sauvegarde de la vidéo annotée.

- 📷 **Détection en temps réel via webcam**  
  YOLO analyse les images en temps réel avec affichage des FPS.

### Améliorations visuelles :
- Thème sombre (fond gris foncé)
- Gros boutons centrés
- Titre et sous-titre
- Zone d’affichage des images traitées
- Label d’état dynamique (« Prêt », « Analyse en cours », etc.)

---

## 📑 3. Génération automatique de rapports

YOLO génère maintenant automatiquement un rapport texte :

```
reports/report_YYYYMMDD_HHMMSS.txt
```

Chaque rapport inclut :
- nombre d’objets détectés
- classes trouvées (ex. person, car, dog…)
- confiances associées
- résumé complet de l'analyse

Pour la **webcam**, le rapport contient même :
- toutes les classes détectées au total pendant la session

Cela renforce la valeur professionnelle du prototype.

---

## 🎞️ 4. Calcul et affichage des FPS (webcam)

Nous avons ajouté un compteur FPS pour mesurer la performance du modèle.  
Les FPS sont calculés en temps réel :

```
FPS: 27.3 William pc
FPS: 16.3 Jacob pc
```

Cela démontre :
- la vitesse de YOLOv8
- la capacité du modèle à fonctionner en temps réel
- la stabilité du prototype

---

## 🐞 5. Bugs corrigés avec l’aide de l’IA

Nous avons eu un bug important :  
> YOLO ne trouvait pas les fichiers même s’ils étaient dans le dossier.

Grâce à l’aide de ChatGPT, nous avons identifié que :
- Python exécutait parfois le script depuis **un autre dossier**
- Cela causait un `FileNotFoundError`
- On devait utiliser `Path(__file__).resolve().parent` pour construire correctement le chemin des fichiers

### Résultat :
✔ Le programme trouve maintenant les fichiers à chaque fois  
✔ L’IA nous a aidés à **comprendre, localiser et corriger** ce bug  
✔ Le projet est plus stable et professionnel

---

## ⚙️ 6. Difficultés rencontrées

- Gestion des chemins de fichiers
- Fermeture propre de la webcam
- Intégration YOLO + Tkinter
- Mise à jour de l’interface pendant un traitement
- Gestion des dossiers YOLO (`detect_output` et `reports`)
- Redimensionnement des images affichées

---

## ✔ 7. Ce qui est terminé pour la semaine 3

- Prototype complet fonctionnel  
- Interface graphique  
- Détection image  
- Détection vidéo  
- Détection webcam + FPS  
- Comptage automatique des objets  
- Rapports générés automatiquement  
- Correction de bugs grâce à l’IA  
- Interface améliorée visuellement  

---

## 🎯 8. Objectifs de la semaine 4

- Améliorée l interface car Ydger.jpg est partout
- Documenter le projet pour le rapport final  
- Préparer la démonstration orale  
- Ajouter des captures d’écran dans le GitHub  

---

## ✔ Statut : Semaine 3 terminée avec succès !


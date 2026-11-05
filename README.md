<div align="center">

# Village → post-catastrophe

**Simulez une catastrophe sur une vraie image aérienne et récupérez les labels de dégâts au pixel près.**

Séisme · Ouragan · Tornade · Inondation · Feu de forêt

![exemple](output/previews/muenster_62_tornado_0_preview.jpg)

*Village de Münster (Allemagne) frappé par une tornade.*

</div>

<details>
<summary><b>Lire cette image</b></summary>

<br>

**Ligne du haut : les images**

| Panneau | Ce qu'il montre |
|---|---|
| **BEFORE** | La vraie photo aérienne d'OpenEarthMap : un lotissement en haut, un bois au centre, des champs en bas. |
| **AFTER** | La même scène après la tornade, générée par le code. Le long d'une bande diagonale, le bois et le champ sont roussis par le vent ; en haut à droite, les maisons sont effondrées ou ont le toit abîmé (bâches bleues). Hors de la bande, rien ne change. |
| **HAZARD FIELD** | La carte d'aléa, c'est-à-dire la force de la tornade en chaque point (noir = rien, jaune clair = maximum). On y voit le couloir de la tornade, et c'est lui qui décide de tout le reste. |

**Ligne du bas : les labels**

| Panneau | Ce qu'il montre |
|---|---|
| **MASK BEFORE (RescueNet)** | Le masque d'origine : vert clair = bâtiment intact, vert foncé = arbre, gris = route, bleu = eau, noir = sol. |
| **MASK AFTER (RescueNet)** | Le masque après la catastrophe : les maisons touchées changent de couleur selon leur niveau de dégâts. Il correspond exactement à l'image AFTER, car le code l'écrit en même temps qu'il dessine. |
| **DAMAGE** | La vue xBD, réduite aux bâtiments. En gris, les pixels hors bâtiment modifiés par la tornade (végétation abîmée, débris). |

**Couleurs des dégâts** : 🟩 intact · 🟨 mineur · 🟧 majeur · 🟥 détruit.

Le titre du dernier panneau donne le bilan : `no-da:43 minor:9 major:7 destr:19`, soit 43 maisons
intactes, 9 avec des dégâts mineurs, 7 avec des dégâts majeurs et 19 détruites.

Les maisons rouges et orange se trouvent toutes dans le couloir jaune de la carte d'aléa : les dégâts
suivent un phénomène cohérent au lieu d'être tirés au hasard maison par maison.

</details>

---

## À quoi ça sert

Les images aériennes prises **après** une catastrophe, et annotées, sont rares. Ce projet part d'un
**village intact** dont on connaît déjà chaque bâtiment, arbre, route ou cours d'eau, et fabrique la
**même scène après la catastrophe**. Comme c'est le code qui dessine les dégâts, il écrit en même
temps les **labels exacts**, prêts pour entraîner un modèle de détection ou de segmentation des
dégâts.

Tout se pilote depuis un **site web local** : on choisit un village et une catastrophe, on clique là
où elle frappe, et le résultat s'affiche en quelques secondes.

---

## Lancer le site

Il faut **Python 3.10 ou plus récent** ([python.org](https://www.python.org/downloads/)).

```bash
git clone https://github.com/Moha-df/post-disaster-villages.git
cd post-disaster-villages
python -m venv .venv
```

Activer l'environnement, puis installer et lancer :

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Ouvrir **http://127.0.0.1:7860** dans le navigateur.

> **`pip` introuvable sous Windows ?** C'est souvent que Python n'est pas installé : la commande
> `python` ouvre alors seulement le Microsoft Store. Installez Python (par exemple
> `winget install Python.Python.3.12`), rouvrez le terminal, et utilisez `python -m pip` à la place
> de `pip`.

---

## Le site

### 1. Simulation

<div align="center"><img src="docs/site_simulation.jpg" alt="Onglet Simulation" width="900"></div>

- **Village** : 8 villages fournis (Allemagne, Autriche, États-Unis, Philippines, Seychelles, Zanzibar), ou
  importer **votre propre image** avec son masque OpenEarthMap.
- **Catastrophe** : séisme, ouragan, tornade, inondation ou feu de forêt.
- **Sévérité** de 0,05 à 1, et une **graine** pour retrouver exactement le même résultat.
- **Cliquer sur l'image** pour placer l'épicentre du séisme, le passage de la tornade ou le point de
  départ du feu.
- **Direction du vent** (ouragan, tornade, feu) ou **surface inondée** (inondation).
- **Rapide (512 px)** pour explorer, **Complet (1000 px)** pour la qualité finale.
- Option **aspect post-événement** : l'image « après » prend l'allure d'une autre prise de vue
  (couleurs, bruit, ombres de nuages).

### 2. Résultats

<div align="center"><img src="docs/site_resultats.jpg" alt="Résultats" width="900"></div>

- **Curseur avant / après** pour comparer les deux images.
- Onglets **Dégâts** (niveau de chaque maison), **Masque RescueNet**, **Carte d'aléa** et
  **Changements**.
- **Télécharger le résultat** en `.zip` : images, masques, labels xBD et métadonnées.

### 3. Albumentations

Montre l'effet du rendu « post-événement » (rendu brut vs. avec Albumentations) et génère des
**crops alignés** avant / après / masques, comme ceux utilisés pour l'entraînement.

### 4. Export en lot

<div align="center"><img src="docs/site_export.jpg" alt="Export en lot" width="900"></div>

Génère d'un coup **tous les villages × toutes les catastrophes**, en plusieurs versions chacune
(point d'impact, vent et sévérité tirés au hasard dans une plage), avec jusqu'à 16 crops par scène.
Le tout sort dans **un seul `.zip`** : images, masques, labels xBD, crops et `manifest.csv`.

---

## Comment ça marche

```
image + masque OpenEarthMap
        │
        ▼
 ① séparation des maisons      ──  watershed sur l'image
        │
        ▼
 ② carte d'aléa  [0, 1]        ──  un modèle physique simplifié par catastrophe
        │
        ▼
 ③ niveau de dégâts par maison ──  intact · mineur · majeur · détruit
        │
        ▼
 ④ rendu procédural            ──  toits, décombres, arbres, eau, brûlures, fumée
        │                           (le masque est écrit en même temps)
        ▼
 ⑤ raccord capteur + Albumentations
        │
        ▼
 image « après » + masques RescueNet / xBD + crops
```

### ① Partir d'un village annoté

L'entrée est une photo aérienne et son **masque de segmentation** issu
d'**OpenEarthMap** [1], annoté à la main sur 44 pays (8 = bâtiment, 4 = route, 5 = arbre, 6 = eau…).
Il est converti dans les classes de **RescueNet** [2].

Les bâtiments collés les uns aux autres sont séparés en maisons individuelles par une
**ligne de partage des eaux (watershed)** [8] calculée sur l'image, à partir des empreintes érodées.
Les pixels orphelins vont à la maison la plus proche grâce à une **transformée de distance** [9].

### ② Carte d'aléa : où la catastrophe frappe

Les dégâts ne sont pas tirés maison par maison au hasard : ils suivent un champ continu, si bien que
des maisons voisines ont des dégâts proches, comme dans la réalité. Le relief et les variations
locales viennent d'un **bruit fractal (fBm)** multi-octaves [10, 11].

| Catastrophe | Modèle d'aléa | Inspiré de |
|---|---|---|
| **Séisme** | Atténuation exponentielle avec la distance à l'épicentre + amplification par les sols mous (bruit) | Lois d'atténuation et cartes d'intensité de type ShakeMap [13] |
| **Ouragan** | Rafales : bruit fractal étiré dans la direction du vent | — |
| **Tornade** | Couloir étroit et légèrement sinueux, dégâts maximaux au centre | Relevés de traces de tornade, échelle de Fujita améliorée [14] |
| **Inondation** | Pseudo-relief : bas près des rivières et des routes ; le niveau d'eau est fixé pour couvrir la surface voulue | Hauteur au-dessus du réseau de drainage (HAND) [15] |
| **Feu de forêt** | Front de feu qui avance avec le vent, plus vite en aval (forme elliptique), et brûle là où il y a du combustible | Propagation de Rothermel [16] et forme elliptique d'Anderson [17] ; végétation repérée par l'indice *Excess Green* [12] |

### ③ Niveau de dégâts de chaque maison

Chaque maison reçoit un score : **aléa moyen sur son emprise + vulnérabilité + bruit**. Les petites
maisons sont un peu plus fragiles que les grandes. Ce score est comparé à trois seuils propres à
chaque catastrophe pour donner l'un des 4 niveaux de l'échelle **xBD** [3] (intact, mineur, majeur,
détruit). C'est un **modèle ordinal à seuils** [18], dans l'esprit des courbes de fragilité de
**HAZUS** [19]. Pour l'inondation, c'est la profondeur d'eau juste autour de la maison qui compte.

### ④ Rendu des dégâts, 100 % procédural

Aucune texture n'est copiée d'un autre jeu de données : les décombres sont faits avec le propre toit
de la maison.

- **Toits abîmés** : tuiles arrachées dans le sens du vent, voligeage apparent, bâches bleues.
- **Effondrement** partiel ou total : le toit est cassé en **fragments de Voronoï** [20], chacun
  pivoté et assombri, avec de la poussière couleur du sol local ; des pans de murs restent debout, et
  l'ombre portée du bâtiment disparaît.
- **Traînées de débris** emportées par le vent.
- **Arbres couchés** dans le sens du vent : le sol caché sous la couronne est reconstruit par
  **inpainting de Telea** [21].
- **Végétation** jaunie par le vent ou couverte de boue après l'inondation.
- **Eau de crue** boueuse, transparente là où elle est peu profonde, avec traînées de sédiments,
  reflets, ligne de débris flottants au bord et sol mouillé autour.
- **Feu** : cendres, zones carbonisées, lisière roussie, maisons brûlées et panaches de fumée.
- **Poussière** après un séisme.

Pour que les zones redessinées ne paraissent pas « trop propres », le flou et le bruit du vrai capteur
sont mesurés (estimation robuste du bruit par la **médiane des écarts absolus**, MAD [22]) puis
appliqués aux zones modifiées.

### ⑤ Albumentations

**Albumentations** [4] intervient à deux endroits :

1. **Aspect post-événement** : l'image après doit sembler prise par un autre vol, un autre jour.
   Seules des transformations de pixels sont utilisées pour que les masques restent valides :
   courbe de tons, teinte, bruit de capteur, compression JPEG, baisse de résolution, et selon la
   catastrophe des **ombres de nuages** ou une lumière inégale (bruit « plasma », algorithme
   *diamond-square* [23]).
2. **Crops d'entraînement** : la **même** transformation géométrique (symétries D4, petite rotation,
   zoom, recadrage centré sur les zones endommagées) est appliquée à l'image avant, à l'image après
   et à tous les masques, qui restent alignés au pixel ; les couleurs sont tirées
   **indépendamment** pour avant et après.

---

## Ce que l'on récupère

| Dossier | Contenu |
|---|---|
| `before/`, `after/` | Images avant et après |
| `masks_before/`, `masks/` | Masques **RescueNet** [2] : 0 fond, 1 débris, 2 eau, 3 à 6 bâtiment intact → détruit, 8 route, 9 arbre… |
| `xbd/`, `labels_xbd/` | Dégâts au format **xBD / xView2** [3] : image et polygones JSON |
| `change/`, `hazard/` | Pixels modifiés, carte d'aléa |
| `augmented/` | Crops Albumentations alignés |
| `previews/` | Aperçus en 6 panneaux |

---

## En ligne de commande (optionnel)

Le site suffit pour tout faire, mais on peut aussi générer le jeu de données sans interface :

```bash
python generate.py                     # 8 villages x 5 catastrophes, dans output/
python generate.py --scenarios flood wildfire --n-per-scenario 3 --severity 0.6
```

Télécharger d'autres villages depuis OpenEarthMap :

```bash
python download_villages.py --n 20 --out data/villages_big
python generate.py --images data/villages_big/images --masks data/villages_big/masks --out output_big
```

| Fichier | Rôle |
|---|---|
| `app.py` | Le site (Gradio [5]) |
| `generate.py` | Génération du jeu de données en ligne de commande |
| `download_villages.py` | Choix et téléchargement des villages OpenEarthMap |
| `postdisaster/hazard.py` | Cartes d'aléa et niveaux de dégâts |
| `postdisaster/effects.py` | Rendu des dégâts |
| `postdisaster/augment.py` | Albumentations |
| `postdisaster/pipeline.py` | Enchaînement complet d'un scénario |
| `postdisaster/classes.py` | Classes OpenEarthMap, RescueNet et xBD |

---

## Limites et licences

- Les images générées sont **synthétiques**. Un modèle entraîné dessus doit être évalué sur de
  **vraies** images post-catastrophe (xBD, RescueNet…).
- Les modèles d'aléa sont des **simplifications** inspirées des références ci-dessous : ils visent un
  rendu plausible, pas une simulation physique.
- Les données OpenEarthMap sont réservées à la **recherche non commerciale**
  (CC BY-NC-SA 4.0).

---

## Références

**Données et formats de labels**

1. J. Xia, N. Yokoya, B. Adriano, C. Broni-Bediako. *OpenEarthMap: A Benchmark Dataset for Global
   High-Resolution Land Cover Mapping*. WACV 2023. — [open-earth-map.org](https://open-earth-map.org)
2. M. Rahnemoonfar, T. Chowdhury, R. Murphy. *RescueNet: A High Resolution UAV Semantic Segmentation
   Dataset for Natural Disaster Damage Assessment*. Scientific Data, 2023.
3. R. Gupta et al. *Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery*.
   CVPR Workshops, 2019. — [xview2.org](https://xview2.org)

**Bibliothèques**

4. A. Buslaev et al. *Albumentations: Fast and Flexible Image Augmentations*. Information 11(2), 2020.
   — [albumentations.ai](https://albumentations.ai)
5. A. Abid et al. *Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild*.
   arXiv:1906.02569, 2019. — [gradio.app](https://www.gradio.app)
6. G. Bradski. *The OpenCV Library*. Dr. Dobb's Journal of Software Tools, 2000.
7. C. R. Harris et al. *Array programming with NumPy*. Nature 585, 2020.

**Traitement d'image**

8. F. Meyer. *Color Image Segmentation*. 4th International Conference on Image Processing and its
   Applications, 1992. (watershed implémenté par OpenCV)
9. G. Borgefors. *Distance Transformations in Digital Images*. Computer Vision, Graphics, and Image
   Processing 34(3), 1986.
10. K. Perlin. *An Image Synthesizer*. SIGGRAPH 1985.
11. D. S. Ebert, F. K. Musgrave, D. Peachey, K. Perlin, S. Worley. *Texturing and Modeling: A
    Procedural Approach*. 3rd ed., Morgan Kaufmann, 2003.
12. D. M. Woebbecke, G. E. Meyer, K. Von Bargen, D. A. Mortensen. *Color Indices for Weed
    Identification Under Various Soil, Residue, and Lighting Conditions*. Transactions of the ASAE
    38(1), 1995.

**Modèles d'aléa et de dégâts**

13. D. J. Wald et al. *ShakeMap Manual: Technical Manual, User's Guide, and Software Guide*. USGS
    Techniques and Methods 12-A1, 2005.
14. Wind Science and Engineering Center. *A Recommendation for an Enhanced Fujita Scale (EF-Scale)*.
    Texas Tech University, 2006.
15. A. D. Nobre et al. *Height Above the Nearest Drainage – a hydrologically relevant new terrain
    model*. Journal of Hydrology 404, 2011.
16. R. C. Rothermel. *A Mathematical Model for Predicting Fire Spread in Wildland Fuels*. USDA Forest
    Service, Research Paper INT-115, 1972.
17. H. E. Anderson. *Predicting Wind-Driven Wild Land Fire Size and Shape*. USDA Forest Service,
    Research Paper INT-305, 1983.
18. P. McCullagh. *Regression Models for Ordinal Data*. Journal of the Royal Statistical Society B
    42(2), 1980.
19. FEMA. *Hazus Earthquake Model Technical Manual*. Federal Emergency Management Agency.

**Rendu**

20. S. Worley. *A Cellular Texture Basis Function*. SIGGRAPH 1996.
21. A. Telea. *An Image Inpainting Technique Based on the Fast Marching Method*. Journal of Graphics
    Tools 9(1), 2004.
22. D. L. Donoho, I. M. Johnstone. *Ideal Spatial Adaptation by Wavelet Shrinkage*. Biometrika 81(3),
    1994.
23. A. Fournier, D. Fussell, L. Carpenter. *Computer Rendering of Stochastic Models*. Communications
    of the ACM 25(6), 1982.

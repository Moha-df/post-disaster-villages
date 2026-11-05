"""
site_doc.py : contenu de l'onglet « Documentation » du site (une page par partie).

Chaque page est un couple (titre, html). Les valeurs citées (seuils, tailles, probabilités) sont
celles du code : si vous modifiez un paramètre, mettez la page à jour.
"""

OVERVIEW = """<div class='explain doc'>
<p>Le projet transforme une <b>scène intacte</b> (photo + masque de segmentation annoté à la main) en la
<b>même scène après une catastrophe</b>, et écrit en même temps les <b>labels exacts</b> des dégâts.
Ces pages expliquent chaque étape, avec les formules et les valeurs utilisées dans le code.</p>

<h3>Les étapes</h3>
<pre>① données     photo + masque  ->  classes RescueNet, maisons séparées       (Données d'entrée)
② aléa        où et avec quelle force la catastrophe frappe : carte [0, 1]   (Aléa et dégâts)
③ dégâts      un niveau par maison : intact, mineur, majeur, détruit         (Aléa et dégâts)
④ rendu       dessin des dégâts + écriture du masque                         (une page par catastrophe)
                 rendu AÉRIEN (2D)   si la résolution est ≥ 12 cm/px
                 rendu DRONE (2,5D)  si la résolution est < 12 cm/px         (Moteur drone)
⑤ finition    bruit du capteur, autre prise de vue, crops alignés            (Capteur et sorties)</pre>

<h3>Deux rendus, une seule simulation</h3>
<table>
<tr><th></th><th>Rendu aérien</th><th>Rendu drone</th></tr>
<tr><td>Images</td><td>OpenEarthMap, 0,25 à 0,5 m/px</td><td>Semantic Drone Dataset, environ 2 à 3 cm/px</td></tr>
<tr><td>Une maison fait</td><td>20 à 60 px de côté</td><td>300 à 800 px de côté</td></tr>
<tr><td>Principe</td><td>textures dessinées à plat</td><td>chaque débris a une hauteur, il est éclairé et projette une ombre</td></tr>
<tr><td>Fichier</td><td><code>postdisaster/effects.py</code></td><td><code>postdisaster/drone.py</code></td></tr>
</table>
<p>Les étapes ①, ②, ③ et ⑤ sont <b>les mêmes</b> pour les deux rendus : seule la manière de dessiner change.
Le choix est automatique (option <b>Rendu</b> sur <i>Auto</i>) mais peut être forcé.</p>

<h3>Pourquoi deux rendus ?</h3>
<p>Vu de haut à 0,3 m/px, un toit effondré est une tache de quelques dizaines de pixels : des fragments de
couleur bien choisis suffisent à tromper l'œil. À 2 cm/px, on voit chaque tuile : des fragments plats ressemblent
à une mosaïque collée sur le toit. L'œil attend du <b>relief</b> (des morceaux posés les uns sur les autres) et
de la <b>lumière</b> (faces éclairées, ombres portées). Le rendu drone construit donc une carte de hauteur et
l'éclaire avec le soleil de la photo.</p>

<h3>Les tailles sont en mètres</h3>
<p>Toutes les tailles du rendu (une tuile, une planche, un tas de gravats) sont écrites en <b>mètres</b> puis
converties en pixels avec la résolution au sol (<i>gsd</i>, en m/px) : <code>pixels = mètres / gsd</code>.
Une tuile fait donc toujours 25 cm, que l'image soit à 2 cm/px ou à 0,5 m/px.</p>

<h3>Reproductibilité</h3>
<p>Tous les tirages aléatoires passent par un générateur <code>numpy.random.Generator</code> initialisé avec la
graine, le nom de la scène et la catastrophe : même graine = même image, au pixel près.</p>
</div>"""

DATA = """<div class='explain doc'>
<h3>OpenEarthMap (images aériennes)</h3>
<p>Tuiles de 1024 px, 0,25 à 0,5 m/px, dans 44 pays, avec un masque annoté à la main en 9 classes :
0 inconnu, 1 sol nu, 2 prairie, 3 aménagé, 4 route, 5 arbre, 6 eau, 7 agriculture, 8 bâtiment.
<code>download_villages.py</code> ne télécharge que la colonne « masque » de chaque tuile (quelques dizaines de Ko)
par requêtes HTTP partielles, calcule un score « village » (maisons dispersées, beaucoup de champs et d'arbres,
quelques routes) et ne télécharge que les images retenues. Les régions de vraies catastrophes sont écartées, car
certaines de leurs tuiles montrent déjà des dégâts. <code>data/zones/</code> ajoute des tuiles urbaines (Accra) et
périurbaines (Münster).</p>

<h3>Semantic Drone Dataset (images drone)</h3>
<p>400 photos prises à la verticale par un drone, à 5–30 m du sol, au-dessus de maisons de Graz (Autriche),
en 6000 × 4000 px, avec 24 classes. L'archive Kaggle fait 4 Go : <code>download_drone.py</code> lit seulement
l'index du zip puis les fichiers choisis (classe <code>RangeFile</code> : un fichier « virtuel » qui lit l'archive
distante par requêtes HTTP <i>Range</i>).</p>
<p><b>Conversion des classes</b> vers OpenEarthMap (<code>SDD_TO_OEM</code>) :</p>
<table>
<tr><th>Semantic Drone Dataset</th><th>devient</th></tr>
<tr><td>toit, mur, fenêtre, porte</td><td>bâtiment</td></tr>
<tr><td>zone pavée, voiture</td><td>route</td></tr>
<tr><td>arbre, arbre sans feuilles, végétation</td><td>arbre</td></tr>
<tr><td>eau, piscine</td><td>eau</td></tr>
<tr><td>gazon</td><td>prairie</td></tr>
<tr><td>terre, gravier, roche</td><td>sol nu</td></tr>
<tr><td>clôture, personne, chien, vélo, marqueur, obstacle, non annoté</td><td>aménagé (sol)</td></tr>
</table>
<p><b>Résolution au sol.</b> Le jeu de données ne la donne pas. Elle est estimée avec les voitures : une voiture
fait environ 4,5 m et environ 770 px en pleine résolution, soit 0,55 cm/px. Les images sont réduites à 1536 px de
large (environ 2,2 cm/px) ; le site les affiche à 1000 px (3,3 cm/px) ou 512 px (6,5 cm/px). La valeur est
enregistrée dans <code>data/drone/scenes.json</code>.</p>

<h3>Classes RescueNet</h3>
<p>Le masque est converti dans les classes RescueNet (<code>classes.oem_to_rescuenet</code>) : 0 fond, 1 débris,
2 eau, 3 à 6 bâtiment intact / mineur / majeur / détruit, 7 véhicule, 8 route, 9 arbre, 10 piscine, 11 sable.
Prairie, agriculture et aménagé deviennent « fond », sol nu devient « sable ».</p>

<h3>Séparer les maisons (<code>pipeline.split_buildings</code>)</h3>
<p>Dans le masque, des maisons mitoyennes forment une seule tache « bâtiment ». Pour les séparer :</p>
<ol>
<li>les empreintes sont <b>érodées</b> (rayon 2 px) : les ponts fins entre maisons disparaissent, il reste un
« noyau » par maison, qui sert de graine ;</li>
<li>une <b>ligne de partage des eaux</b> (watershed d'OpenCV) part de ces graines sur l'image floutée : les
frontières se placent sur les arêtes des toits ;</li>
<li>les pixels perdus (lignes de partage, parties fines) reviennent à la maison la plus proche par
<b>transformée de distance</b> ;</li>
<li>les petites parties sans graine deviennent leur propre maison ; tout ce qui fait moins de 6 m² est ignoré.</li>
</ol>
<p>Chaque maison garde son numéro, sa boîte englobante, son aire et son centre.</p>
</div>"""

HAZARD = """<div class='explain doc'>
<h3>La carte d'aléa</h3>
<p>Les dégâts ne sont pas tirés au hasard maison par maison. Une carte continue <code>field</code> (valeurs de 0 à 1)
dit en chaque pixel avec quelle force la catastrophe frappe ; elle vient d'un modèle physique simplifié propre à
chaque catastrophe (voir les pages suivantes). Des maisons voisines reçoivent donc des dégâts proches, comme
dans les vraies images post-catastrophe. Cette carte est affichée dans l'onglet <b>Carte d'aléa</b>.</p>

<h3>Bruit fractal (fBm)</h3>
<p>La plupart des modèles utilisent <code>hazard.fbm(h, w, rng, scale, octaves)</code> : une somme de bruits
aléatoires lissés, chaque « octave » ayant 2 fois plus de détails et 2 fois moins d'amplitude que la précédente.
<code>scale</code> est le nombre de cellules de la première octave sur le grand côté de l'image. Le résultat
ressemble à du relief naturel. <code>directional_blur</code> étire un champ dans une direction (rafales, fumée,
traînées).</p>

<h3>Du champ au niveau de dégât (<code>hazard.damage_levels</code>)</h3>
<p>Pour chaque maison, on calcule un score latent :</p>
<pre>z = aléa moyen sur l'emprise
  + 0,08 × clip( ln( aire médiane / aire de la maison ), -1, 1 )     vulnérabilité
  + N(0 ; 0,08)                                                        bruit</pre>
<p>Les petites maisons (plus légères) sont donc un peu plus fragiles. Pour l'inondation, on prend la
<b>profondeur maximale juste autour</b> de la maison (anneau de 3 px) au lieu de la moyenne : c'est l'eau qui
l'entoure qui compte. Le score est comparé à trois seuils : le niveau est le nombre de seuils dépassés.</p>
<table>
<tr><th>Catastrophe</th><th>mineur si z &gt;</th><th>majeur si z &gt;</th><th>détruit si z &gt;</th></tr>
<tr><td>Séisme</td><td>0,22</td><td>0,42</td><td>0,60</td></tr>
<tr><td>Ouragan</td><td>0,30</td><td>0,55</td><td>0,76</td></tr>
<tr><td>Tornade</td><td>0,18</td><td>0,45</td><td>0,70</td></tr>
<tr><td>Inondation</td><td>0,08</td><td>0,35</td><td>0,80</td></tr>
<tr><td>Feu</td><td>0,35</td><td>0,52</td><td>0,66</td></tr>
</table>
<p>C'est un <b>modèle ordinal à seuils</b> (McCullagh, 1980), dans l'esprit des courbes de fragilité HAZUS :
plus l'aléa est fort, plus la probabilité de dépasser chaque seuil augmente. Les 4 niveaux sont ceux de l'échelle
xBD : <i>no-damage, minor-damage, major-damage, destroyed</i>.</p>

<h3>La sévérité</h3>
<p>Le curseur <b>Sévérité</b> (0,05 à 1) agit sur l'aléa : il multiplie la carte (séisme, ouragan), élargit le
couloir (tornade), augmente la surface inondée ou la portée du feu. Avec la même graine, seule la sévérité
change : on peut comparer des intensités sur la même scène.</p>

<h3>Ordre de dessin</h3>
<p>Les maisons sont dessinées de la moins touchée à la plus touchée, pour que les débris des maisons détruites
passent par-dessus le reste. Les couronnes d'arbres qui surplombaient une maison abîmée sont remises par-dessus
(sauf en cas de feu).</p>
</div>"""

EARTHQUAKE = """<div class='explain doc'>
<h3>Aléa (<code>hazard.earthquake</code>)</h3>
<p>Inspiré des lois d'atténuation et des cartes d'intensité de type ShakeMap (USGS) : la secousse décroît avec la
distance à l'épicentre, et les sols mous l'amplifient localement.</p>
<pre>épicentre : cliqué sur l'image, sinon tiré à une distance de 0,2 à 1,2 × D du centre
            (D = grand côté de l'image ; il peut être hors de l'image)
atténuation = exp( -distance / (D × U(0,6 ; 1,2)) )
sol         = fBm (3 cellules, 3 octaves)                     sols mous
champ       = normalise(0,7 × atténuation + 0,3 × sol) × (0,45 + 0,65 × sévérité)</pre>

<h3>Rendu aérien</h3>
<ul>
<li><b>Mineur</b> (<code>roof_damage</code>) : 2 à 6 zones sur le toit, de rayon 5 à 14 % de la taille de la
maison (entre 0,4 et 4 m) : tuiles manquantes (toit assombri à 42 %, 45 % des cas), voligeage apparent (30 %),
bâche bleue (25 %).</li>
<li><b>Majeur</b> : l'emprise est coupée par une ligne aléatoire rendue irrégulière par du bruit ; un côté
s'effondre en <b>fragments de Voronoï</b> de 0,7 m (<code>shatter</code>) : chaque cellule copie un morceau
du vrai toit, tourné, assombri au hasard, et certaines cellules deviennent de la poussière couleur du sol ;
l'autre côté reçoit des dégâts de toit.</li>
<li><b>Détruit</b> : l'ombre portée de la maison est effacée (<code>remove_shadow</code> : les pixels sombres
collés au bâtiment sont éclaircis) ; l'emprise et un débordement irrégulier sur le sol deviennent des fragments de
0,6 m (45 % de poussière) ; des <b>restes de murs</b> clairs subsistent sur une partie du contour.</li>
<li><b>Débris</b> (<code>debris_trail</code>) : traînée de 2 + 4 × niveau mètres, faite du matériau du toit.</li>
<li><b>Poussière</b> (<code>haze</code>) : voile irrégulier sur toute l'image, opacité 0,12 + 0,12 × sévérité.</li>
</ul>

<h3>Rendu drone</h3>
<p>Le principe du moteur (hauteur, soleil, ombres) est expliqué dans la page <b>Moteur drone</b>.
Pour le séisme :</p>
<ul>
<li><b>Mineur</b> : 2 ou 3 zones de <b>tuiles manquantes</b> de 0,3 à 0,9 m (<code>missing_tiles</code>).
Elles sont découpées <b>rangée par rangée</b> (rangées de 33 cm, tuiles de 25 cm) en suivant les joints : le bord
du trou forme des marches, comme sur un vrai toit. Le sens des rangées est mesuré sur la photo par un
<b>tenseur de structure</b> (<code>roof_axis</code>). Dans le trou : l'écran sous toiture noir avec les
<b>liteaux</b> en bois tous les 33 cm, ou (10 %) des panneaux de voligeage de 1,22 × 2,44 m avec leurs joints ;
12 % des tuiles restent accrochées. Le trou est 5 cm plus bas que le toit : le bord côté soleil y projette une
fine ombre. Puis 3 à 6 <b>fissures</b> (<code>cracks</code>) : marches aléatoires de 8 à 20 segments de 25 cm.
Les tuiles tombées gisent dans un anneau de 1,2 m autour de la maison, avec leur ombre.</li>
<li><b>Majeur</b> : une partie du toit, d'un côté d'une ligne irrégulière, <b>s'effondre</b>. Sur le reste :
2 à 4 zones de tuiles manquantes et des fissures. Dans la partie effondrée : un tas de gravats bas (0,8 m), 35 %
de <b>trous sombres</b> vers l'intérieur de la maison, des <b>pans du toit</b> qui pendent dans le trou (40 %
des morceaux, glissés de 0,3 m au plus) et des <b>chevrons</b> encore en place, orientés dans le sens de la pente.</li>
<li><b>Détruit</b> : l'ombre de la maison disparaît ; l'emprise, plus un débordement de 1 à 2,5 m sur le sol,
devient un <b>tas de gravats</b> de 1,6 m de haut au plus (<code>rubble_layer</code>), puis de <b>grands pans du
vrai toit</b> (<code>roof_sections</code>, 45 % des morceaux) sont posés dessus, puis des petits débris
par-dessus. On reconnaît encore la maison et son toit, comme sur les photos de séismes réels.</li>
<li><b>Débris au sol</b> (<code>scatter_debris</code>) : un séisme ne projette pas loin : les morceaux tombent à
moins de 0,8 + 0,6 × niveau mètres des murs, avec une probabilité qui décroît avec la distance ; très peu de
planches (2 %).</li>
<li><b>Poussière</b> (<code>dust</code>) : nuage <b>local</b> autour des effondrements (flou de 2,5 m modulé par
du bruit, opacité 0,25 + 0,2 × sévérité, au plus 45 %), au lieu d'un voile sur toute l'image.</li>
</ul>
<p><b>Matériaux des gravats</b> : 46 % morceaux du vrai toit, 24 % béton, 16 % enduit, 10 % brique, 4 % bois.</p>
</div>"""

WIND = """<div class='explain doc'>
<h3>Aléa de l'ouragan (<code>hazard.hurricane</code>)</h3>
<p>Le vent souffle dans une direction (choisie ou aléatoire) ; ses <b>rafales</b> sont des bandes allongées dans
le sens du vent.</p>
<pre>rafales = fBm (6 cellules, 4 octaves) étiré sur 0,25 × D dans le sens du vent
champ   = normalise(0,35 + rafales) × (0,4 + 0,65 × sévérité)</pre>
<p>Tout le quartier est touché, avec des variations : beaucoup de dégâts mineurs, peu de maisons détruites.</p>

<h3>Aléa de la tornade (<code>hazard.tornado</code>)</h3>
<p>Inspiré des relevés de traces de tornade (échelle de Fujita améliorée) : un <b>couloir étroit</b> traverse
l'image, avec des dégâts maximaux au centre et presque rien à côté.</p>
<pre>axe          : passe par le point cliqué (sinon entre 30 et 70 % de l'image), direction choisie ou aléatoire
ondulation   = sin(position le long de l'axe) × 0,04 × D             trajectoire légèrement sinueuse
demi-largeur = D × U(0,07 ; 0,13) × (0,6 + 0,6 × sévérité)
cœur         = clip(1,25 - |distance à l'axe + ondulation| / demi-largeur, 0, 1)
champ        = cœur × (0,85 + 0,3 × fBm)</pre>
<p>Les débris partent le long de la trajectoire (rotation de la tornade).</p>

<h3>Rendu aérien</h3>
<ul>
<li>Dégâts de toit arrachés <b>dans le sens du vent</b> (ellipses orientées), bâches, voligeage.</li>
<li>Majeur / détruit : comme le séisme, avec une traînée de débris plus longue (+6 m) poussée par le vent.</li>
<li><b>Arbres couchés</b> (<code>topple_trees</code>) : un arbre isolé (2 à 12 m) tombe avec une probabilité
proportionnelle à l'aléa ; sa couronne glisse de 0,7 à 1,2 fois sa taille sous le vent et s'allonge de 35 % (vue
de côté) ; le sol caché dessous est reconstruit par <b>inpainting de Telea</b> ; le tronc et l'ombre sont dessinés.</li>
<li><b>Végétation</b> jaunie / grise là où le vent est fort (<code>vegetation_stress</code>).</li>
</ul>

<h3>Rendu drone</h3>
<ul>
<li><b>Mineur</b> : 2 à 5 grandes plaques de tuiles arrachées (0,5 à 1,6 m), voligeage plus souvent à nu (35 %),
et dans 45 % des cas une <b>bâche bleue</b> (<code>tarp</code>) de 2 à 4 m sur 1,5 à 3 m, alignée sur le toit :
ses <b>plis</b> sont un relief de 12 cm (bruit étiré perpendiculairement), éclairé par le soleil ; des planches ou
sacs de sable la tiennent sur les bords.</li>
<li><b>Majeur / détruit</b> : comme le séisme, mais les gravats sont plus bas (0,9 m) et contiennent plus de
<b>bois</b> de charpente (matériaux : 45 % toit, 25 % bois, 18 % enduit, 12 % béton, 15 % de planches dans les
blocs) ; les grands pans de toit sont moins nombreux (25 %) car le vent les emporte.</li>
<li><b>Débris projetés sous le vent</b> sur 4 + 3 × niveau mètres : la probabilité est plus forte dans la direction
du vent ; 12 % de planches, orientées dans le sens du vent.</li>
<li><b>Feuilles et branches</b> (<code>wind_litter</code>) : à cette échelle, déplacer un arbre entier donnerait
une découpe collée. Les arbres restent en place ; des feuilles (amas de 4 à 15 cm, verts ou bruns) et des
branches (0,3 à 1,2 m) partent des couronnes, là où l'aléa est fort, et sont poussées jusqu'à 3 m sous le vent,
avec leur ombre.</li>
<li>Végétation jaunie comme en aérien.</li>
</ul>
</div>"""

FLOOD = """<div class='explain doc'>
<h3>Aléa (<code>hazard.flood</code>)</h3>
<p>Inspiré du modèle HAND (hauteur au-dessus du réseau de drainage) : l'eau monte d'abord dans les endroits bas,
près des rivières. On fabrique un <b>pseudo-relief</b> :</p>
<pre>s'il y a de l'eau : relief = 0,65 × clip(distance à l'eau / (0,5 × D)) + 0,35 × fBm
sinon             : une vallée traverse l'image (direction et position aléatoires)
relief            = relief - 0,02 sur les routes + 0,03 sur les bâtiments   (routes basses, maisons surélevées)
surface visée     = 0,10 + 0,30 × sévérité (± 0,05), ou le curseur « Surface inondée »
niveau d'eau      = quantile du relief qui couvre exactement cette surface de sol
profondeur        = (niveau - relief), normalisée de 0 à 1 ; les rivières existantes ≥ 0,6</pre>
<p>Le niveau de dégât d'une maison dépend de la profondeur maximale <b>autour</b> d'elle.</p>

<h3>Rendu aérien</h3>
<ul>
<li><b>Eau boueuse</b> gris-brun, plus sombre en profondeur, opacité 0,9 + 0,4 × profondeur (le sol reste
visible là où il y a peu d'eau), <b>traînées de sédiments</b> étirées sur 25 m dans le sens du courant, reflets
du ciel.</li>
<li>Couronnes d'arbres visibles au-dessus de l'eau peu profonde ; les rivières gardent une partie de leur aspect.</li>
<li><b>Laisse de crue</b> : bande de débris flottants et d'écume de 1,2 m au bord ; sol <b>mouillé</b> (plus
sombre) sur 4 m autour ; végétation couverte de boue.</li>
</ul>

<h3>Rendu drone</h3>
<ul>
<li><b>Eau à l'échelle de la rue</b> (<code>flood_depth</code>) : à 2 cm/px, la carte d'aléa est trop « bosselée »
(elle donnerait des flaques rondes). On garde seulement sa <b>pente générale</b> (très lissée, flou de D/8) et on
ajoute le <b>type de sol</b> : routes 12 cm plus bas, arbres 4 cm plus haut, petit relief de ± 4 cm, bâtiments au
sec. Le niveau est fixé pour couvrir la même surface : l'eau remplit la rue et les jardins avec des bords nets.</li>
<li><b>Transparence selon la profondeur</b> (<code>flood_water</code>) : la profondeur visible augmente avec la
distance au bord (sur 2 m). Au bord, l'opacité est de 30 % : on voit le sol dessous, légèrement <b>flou</b>
(réfraction) et teinté ; au large elle monte à 95 % et l'eau s'assombrit de 25 %.</li>
<li><b>Vaguelettes</b> : petit relief de 3 cm (longueur d'onde environ 35 cm, étiré perpendiculairement au
courant), éclairé par le soleil comme les débris : la surface n'est plus un aplat. Les pentes les mieux
orientées donnent des <b>éclats de lumière</b> ; un reflet du ciel diffus est ajouté. Des <b>volutes de
sédiments</b> s'étirent sur 1,5 m dans le sens du courant.</li>
<li><b>Bord</b> : écume clairsemée dans les 35 premiers cm, sol mouillé sur 0,8 m (jusqu'à 35 % plus sombre).</li>
<li><b>Objets flottants</b> : planches (35 %) et petits morceaux entre 0,2 et 1,5 m du bord, orientés par le
courant, avec leur ombre sur l'eau.</li>
<li><b>Maisons</b> : elles ne s'effondrent pas en tas (ce serait incohérent sous l'eau) ; leurs dégâts se lisent
par l'eau qui les entoure, et les plus touchées perdent des tuiles (1 zone si majeur, 3 si détruit).</li>
</ul>
</div>"""

FIRE = """<div class='explain doc'>
<h3>Aléa (<code>hazard.wildfire</code>)</h3>
<p>Inspiré du modèle de propagation de Rothermel et de la forme elliptique d'Anderson : le feu avance avec le
vent et brûle là où il y a du <b>combustible</b>.</p>
<pre>combustible = 0,4 × arbres + 0,6 × clip(4 × ExG) + 0,25 × sol nu/herbe, lissé (D/80)
              ExG = 2V - R - B (indice Excess Green : végétation repérée sur la couleur)
portée      = 0,2 + 0,55 × sévérité
avec un point de départ : distance elliptique = max(devant, 0) + 3 × max(derrière, 0) + 1,5 × côté
                          le feu va 3 fois moins loin contre le vent et 1,5 fois moins sur les côtés
sans point            : un front entre par un côté de l'image et avance avec le vent
champ       = front × (0,45 + 0,8 × combustible) × (0,6 + 0,6 × bruit étiré dans le sens du vent)</pre>

<h3>Rendu aérien</h3>
<ul>
<li><b>Sol brûlé</b> (<code>burn_scar</code>) : cœur carbonisé là où le champ dépasse 0,35, qui garde la
<b>structure</b> de ce qui était là (squelettes d'arbres, motifs des champs) mais très sombre ; plaques de cendre
grise ; lisière roussie (brune) autour ; les arbres brûlés perdent leur label « arbre ».</li>
<li><b>Maisons</b> : mineur = bord du toit noirci du côté du feu (<code>scorch</code>) ; majeur / détruit =
effondrement recouvert de charbon et de cendre.</li>
<li><b>Fumée</b> (<code>smoke</code>) : panaches qui partent du front actif et dérivent sous le vent.</li>
</ul>

<h3>Rendu drone</h3>
<ul>
<li><b>Mineur</b> : <b>suie</b> sur les 1,5 premiers mètres du bord du toit (irrégulière), 1 ou 2 zones de tuiles
éclatées.</li>
<li><b>Majeur / détruit</b> : gravats <b>carbonisés</b> : 45 % charbon, 25 % cendre blanche, 15 % béton, 15 % tôle
rouillée ; poussière sombre ; pas de grands pans de toit (ils ont brûlé) et pas de débris projetés.</li>
<li><b>Sol brûlé</b> (<code>burnt_texture</code>) : à cette échelle, tout noircir uniformément donnait un
résultat faux (dalles et gravier qui « brûlent » comme de l'herbe). On distingue :
<ul>
<li>la <b>végétation</b> (ExG &gt; 10 ou classe arbre) devient du charbon avec un grain fin de 1,5 cm et un grain
de 8 cm, et des <b>plaques de cendre</b> grise diffuses ;</li>
<li>le <b>reste</b> (dalles, gravier, objets) est seulement couvert de <b>suie</b> : 45 à 65 % de sa luminosité,
sa texture reste lisible.</li>
</ul></li>
<li>Fumée comme en aérien.</li>
</ul>
</div>"""

ENGINE = """<div class='explain doc'>
<p>Cette page décrit <code>postdisaster/drone.py</code>, utilisé quand la résolution est plus fine que 12 cm/px.</p>

<h3>1. Le calque : couleur, couverture, hauteur</h3>
<p>Tout ce que le rendu drone dessine (gravats, tuiles tombées, bâche, feuilles, objets flottants) est un
<b>calque</b> de trois images de la taille de la zone :</p>
<ul>
<li><code>col</code> : la couleur propre du matériau (<i>albédo</i>), sans lumière ;</li>
<li><code>cov</code> : la couverture (1 là où il y a quelque chose) ;</li>
<li><code>z</code> : la <b>hauteur en mètres</b> au-dessus du sol.</li>
</ul>
<p><code>compose()</code> éclaire ce calque et le pose sur la photo ; il projette aussi l'ombre du calque sur le
sol autour (seulement là où le sol est éclairé, pour ne pas assombrir deux fois une ombre existante).</p>

<h3>2. Trouver le soleil sur la photo (<code>estimate_sun</code>)</h3>
<p>Les ombres doivent tomber du même côté que celles de la photo. On décale les empreintes des bâtiments de
1,2 m dans <b>24 directions</b> et on mesure la luminosité moyenne du sol ainsi découvert. Du côté des ombres,
ce sol est sombre : la direction la plus sombre (après un léger lissage) est celle des ombres, le soleil est à
l'opposé. Le rapport luminosité à l'ombre / au soleil (borné entre 0,3 et 0,75) donne la <b>lumière ambiante</b>.
S'il n'y a pas d'ombre nette (écart de moins de 12 niveaux), on prend un éclairage par défaut. L'élévation du
soleil est fixée à 42°.</p>

<h3>3. Éclairage de Lambert (<code>Light.lambert</code>)</h3>
<p>La normale d'une surface de hauteur z(x, y) se calcule avec ses pentes, en mètres :</p>
<pre>n = ( -dz/dx , -dz/dy , 1 ) / norme          L = vecteur vers le soleil
éclairage = (n · L) / sin(élévation)         = 1 pour un sol plat (il garde sa luminosité)
                                               > 1 pour une face tournée vers le soleil (borné à 1,3)
                                               < 1 pour une face à contre-jour</pre>
<p>C'est l'ombrage de relief (<i>hillshading</i>, Horn 1981) appliqué aux débris.</p>

<h3>4. Ombres portées (<code>Light.shadow</code>)</h3>
<p>Pour chaque pixel, on « regarde » vers le soleil pas à pas (jusqu'à hauteur max / tan(élévation), en 60
pas au plus). À la distance d, le rayon vers le soleil est à la hauteur z + d × tan(élévation) : si la carte de
hauteur est plus haute à cet endroit, quelque chose cache le soleil, le pixel est à l'ombre. Le calcul est
vectorisé : à chaque pas, toute la carte est décalée d'un coup. L'ombre est adoucie de 0,8 px.</p>

<h3>5. Occlusion ambiante</h3>
<p>Les creux reçoivent moins de lumière du ciel. On compare la hauteur à sa moyenne locale (flou de 25 cm) :
plus le pixel est en dessous, plus il est assombri (jusqu'à 45 %). Les interstices entre les gravats deviennent
sombres, ce qui donne beaucoup de profondeur.</p>
<pre>lumière finale = ambiante + (1 - ambiante) × Lambert × (1 - ombre)
                 × (1 - 0,45 × occlusion)</pre>

<h3>6. Matériaux (<code>Palette</code>)</h3>
<p>Les gravats sont faits des matériaux de la maison : des morceaux du <b>vrai toit</b> (copiés à un endroit au
hasard du toit, tournés), du béton, de l'enduit, de la brique et du bois. Chaque matériau a un <b>grain</b> à sa
taille réelle (3 cm pour le béton) ; le bois a un veinage dans le sens de la planche. Les proportions dépendent de
la catastrophe (voir chaque page).</p>

<h3>7. Poser des morceaux (<code>draw_pieces</code>)</h3>
<p>Un morceau est un <b>polygone</b> de 4 à 7 sommets (ou une <b>planche</b> de 0,6 à 2 m sur 6 à 14 cm). Il est
posé <b>sur</b> ce qui est déjà là : sa base est au 70<sup>e</sup> centile de la hauteur dessous, son épaisseur
est de 4 à 12 cm (+10 cm pour le béton et la brique), et c'est une <b>plaque inclinée</b> (pente aléatoire) : sa
face est donc éclairée différemment de ses voisines. Son bord cassé est un peu plus sombre.</p>

<h3>8. Un tas de gravats (<code>rubble_layer</code>)</h3>
<ol>
<li><b>Monticule</b> : la hauteur croît avec la distance au bord de la zone (jusqu'à 3 m de distance), jusqu'à la
hauteur du tas (1,6 m pour un séisme), modulée par du bruit (± 35 %).</li>
<li><b>Fond</b> : gravier et poussière, mélange de béton, couleur du sol et couleur du toit, avec deux grains
(2,5 et 8 cm).</li>
<li><b>Morceaux</b>, du plus gros au plus petit : dalles de 0,5 à 1,5 m (0,5 par m², surtout du toit), blocs de
15 à 45 cm (1,2 par m²), petits débris de 4 à 12 cm (3 par m²).</li>
<li><b>Poussière</b> par-dessus, irrégulière, couleur du sol local mélangée à du gris neutre ; bord du tas adouci
sur 4 cm.</li>
</ol>

<h3>9. Grands pans de toit (<code>roof_sections</code>)</h3>
<p>Le vrai toit est découpé en cellules de 4 à 8 m², dont les frontières sont rendues <b>déchiquetées</b> par une
déformation de bruit (± 25 cm), avec un espace de 6 cm entre les morceaux. Une partie des cellules est conservée ;
chacune <b>glisse vers le centre</b> de la maison (0,1 à 0,5 m), <b>tourne</b> un peu (± 6°) et se pose
<b>inclinée</b> (pente 0,15 à 0,55) sur le tas. Grâce à l'éclairage, chaque pan a une luminosité différente :
c'est ce qui donne l'impression d'un toit effondré plutôt que d'une mosaïque.</p>

<h3>10. Tuiles manquantes (<code>missing_tiles</code>, <code>roof_axis</code>)</h3>
<p>Le sens des rangées de tuiles est mesuré avec un <b>tenseur de structure</b> (Bigün et Granlund) : on somme les
produits des gradients de l'image autour du point ; la direction dominante du gradient est perpendiculaire aux
rangées. Le trou est construit rangée par rangée (33 cm), chaque rangée coupée sur des joints de tuiles (25 cm),
plus courte vers les bords : on obtient des bords en marches. Dedans : écran noir et liteaux, ou voligeage.</p>

<h3>Coût</h3>
<p>Environ 0,3 s en mode Rapide (512 px) et 1 à 4 s en mode Complet (1000 px) par image sur un processeur
ordinaire, sans carte graphique.</p>
</div>"""

OUTPUTS = """<div class='explain doc'>
<h3>Raccord capteur (<code>effects.match_sensor</code>)</h3>
<p>Les zones redessinées sont « trop propres » par rapport à la photo. On mesure le bruit du vrai capteur sur les
zones non modifiées (estimation robuste : 1,4826 × médiane des écarts absolus du détail fin, MAD) et on applique
ce bruit, avec un léger flou, aux zones modifiées.</p>

<h3>Aspect post-événement (<code>augment.apply_post_event</code>)</h3>
<p>Une vraie image « après » vient d'un autre vol, un autre jour. Albumentations applique des transformations de
<b>pixels uniquement</b> (les masques restent valides) : courbe de tons, teinte / saturation, bruit de capteur,
baisse de résolution, compression JPEG, et selon la catastrophe : ombres de nuages (ouragan, inondation,
tornade), lumière inégale (séisme, feu). Chaque transformation a sa propre graine : elles ne se déclenchent pas
toutes ensemble. L'onglet <b>Albumentations</b> compare le rendu brut et le rendu transformé.</p>

<h3>Crops d'entraînement (<code>augment.paired_crops</code>)</h3>
<p>La <b>même</b> transformation géométrique (symétries du carré, rotation de ± 20°, zoom de 0,85 à 1,2, recadrage
centré sur les zones endommagées) est appliquée à l'image avant, à l'image après et à tous les masques, qui restent
alignés au pixel. Les couleurs sont tirées <b>indépendamment</b> pour avant et après.</p>

<h3>Fichiers produits</h3>
<table>
<tr><th>Dossier</th><th>Contenu</th></tr>
<tr><td><code>before/</code>, <code>after/</code></td><td>images avant et après (JPEG)</td></tr>
<tr><td><code>masks_before/</code>, <code>masks/</code></td><td>masques RescueNet (PNG, une valeur de classe par pixel)</td></tr>
<tr><td><code>xbd/</code></td><td>dégâts par pixel, échelle xBD : 0 fond, 1 intact, 2 mineur, 3 majeur, 4 détruit</td></tr>
<tr><td><code>labels_xbd/</code></td><td>JSON au format xView2 : un polygone WKT par bâtiment, sous-type de dégât, aléa, aire en m², métadonnées (sévérité, graine, rendu, vent, épicentre…)</td></tr>
<tr><td><code>change/</code></td><td>pixels modifiés (label changé ou pixel nettement redessiné)</td></tr>
<tr><td><code>hazard/</code></td><td>carte d'aléa (0 à 255)</td></tr>
<tr><td><code>augmented/</code></td><td>crops alignés</td></tr>
<tr><td><code>manifest.csv</code>, <code>summary.json</code></td><td>une ligne par image ; totaux par catastrophe</td></tr>
</table>
</div>"""

CODE = """<div class='explain doc'>
<h3>Architecture</h3>
<pre>app.py                  site (Gradio) : onglets, paramètres, export zip
site_doc.py             contenu de cet onglet Documentation
generate.py             génération en ligne de commande, previews, labels xBD JSON
download_villages.py    sélection + téléchargement partiel des tuiles OpenEarthMap
download_drone.py       téléchargement partiel + conversion du Semantic Drone Dataset
postdisaster/
  classes.py            classes OpenEarthMap -> RescueNet -> xBD, palettes
  hazard.py             cartes d'aléa (5 catastrophes) + niveaux de dégâts
  pipeline.py           enchaînement d'un scénario, choix du rendu
  effects.py            rendu AÉRIEN (2D)
  drone.py              rendu DRONE (2,5D)
  augment.py            Albumentations</pre>

<h3>Appel d'un scénario</h3>
<pre>from postdisaster import classes as C, pipeline as P
import numpy as np, cv2

img  = cv2.imread("data/villages/images/muenster_62.png")
mask = C.oem_to_rescuenet(cv2.imread("data/villages/masks/muenster_62.png", 0))
blds = P.split_buildings(img, mask, min_area=int(6 / 0.3 ** 2))
rng  = np.random.default_rng(7)
after, mask_after, info = P.generate(img, mask, "earthquake", 0.75, 0.3, rng, buildings=blds,
                                     opts={"point": (500, 300), "render": "auto"})
xbd = C.rescuenet_to_xbd(mask_after)</pre>
<p><code>info</code> contient <code>buildings</code> (avec <code>damage_level</code> et <code>hazard</code> pour
chaque maison), <code>b_lab</code> (image des maisons), <code>hazard</code> (la carte) et <code>extra</code>
(rendu utilisé, épicentre, vent, surface inondée ou brûlée…).</p>

<h3>Options (<code>opts</code>)</h3>
<table>
<tr><th>Clé</th><th>Effet</th></tr>
<tr><td><code>point</code></td><td>(x, y) en pixels : épicentre, centre de la tornade ou départ du feu</td></tr>
<tr><td><code>wind_deg</code></td><td>direction du vent en degrés (0 = vers l'est, 90 = vers le bas)</td></tr>
<tr><td><code>coverage</code></td><td>fraction du sol inondé (0,02 à 0,9)</td></tr>
<tr><td><code>render</code></td><td><code>auto</code>, <code>aerial</code> ou <code>drone</code></td></tr>
</table>

<h3>Ajouter une catastrophe</h3>
<ol>
<li>Dans <code>hazard.py</code> : une fonction qui renvoie <code>{"field": ...}</code> (et <code>wind</code> si
le vent compte), ajoutée à <code>compute</code>, avec ses seuils dans <code>THRESHOLDS</code>.</li>
<li>Son nom dans <code>pipeline.SCENARIOS</code> et dans <code>LABELS</code> du site.</li>
<li>Son rendu dans <code>damage_building</code> de <code>effects.py</code> et de <code>drone.py</code>, et ses
effets de scène dans <code>pipeline.generate</code>.</li>
<li>Ses transformations météo dans <code>augment.post_event</code>.</li>
</ol>

<h3>Ajouter un jeu de données</h3>
<ol>
<li>Convertir ses masques en classes OpenEarthMap (exemple : <code>sdd_to_oem</code> dans
<code>download_drone.py</code>).</li>
<li>Ranger <code>images/</code> et <code>masks/</code> avec les mêmes noms de fichiers.</li>
<li>Donner la résolution au sol : <code>--gsd</code> en ligne de commande, ou un <code>scenes.json</code> comme
celui de <code>data/drone/</code> pour le site.</li>
</ol>

<h3>Ligne de commande</h3>
<pre>python generate.py --scenarios earthquake flood --n-per-scenario 3 --severity 0.6
python download_drone.py --ids 128 239 016 287
python generate.py --images data/drone/images --masks data/drone/masks --gsd 0.0215 --render drone</pre>
</div>"""

PAGES = [
    ("Vue d'ensemble", OVERVIEW),
    ("Données d'entrée", DATA),
    ("Aléa et dégâts", HAZARD),
    ("Séisme", EARTHQUAKE),
    ("Ouragan et tornade", WIND),
    ("Inondation", FLOOD),
    ("Feu de forêt", FIRE),
    ("Moteur drone", ENGINE),
    ("Capteur et sorties", OUTPUTS),
    ("Code", CODE),
]

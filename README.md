# Optimizer Race – Autograd et Visualisation avec Manim

## 1. Objectif du projet

Ce projet a pour objectif de :

- Implémenter un moteur d’auto-différentiation (autograd) simple  
- Implémenter plusieurs algorithmes d’optimisation  
- Visualiser leur comportement sur une fonction de perte 2D  
- Comparer leur convergence  

L’objectif pédagogique est de comprendre comment différents optimiseurs descendent une surface de perte complexe.

## 2. Choix du dataset

Il n’y a pas de dataset classique (images, textes, etc.).  
À la place, on utilise une fonction mathématique artificielle :

z = f(u, v)

Cette fonction est définie dans le code par :

- `funky_func_numpy(u, v)` pour l’affichage  
- `funky_func_tensor(u, v)` pour le calcul des gradients  

Elle combine :
- des exponentielles,
- des sinus et cosinus,
- plusieurs pics et vallées.

Cela crée une surface avec plusieurs minima locaux, ce qui est idéal pour tester les optimiseurs.

Dans ce projet, la fonction joue le rôle du **dataset et de la fonction de perte**.

## 3. Moteur d’auto-différentiation

La classe `Tensor` est une implémentation minimale d’un système d’autograd, inspiré de PyTorch.

Chaque objet `Tensor` contient :
- `data` : la valeur numérique
- `grad` : le gradient
- `_prev` : les tenseurs parents dans le graphe de calcul
- `_backward` : la fonction de rétro-propagation locale

Les opérations (+, *, **, etc.) créent un graphe de calcul.
Quand on appelle loss.backward(), le programme parcourt le graphe en sens inverse et applique la règle de la chaîne pour calculer les dérivées partielles.

Cela permet de calculer automatiquement les dérivées partielles de f par rapport à u et v.

## 4. Fonctions mathématiques différentiables

Les fonctions suivantes sont définies :

- sin_d
- cos_d
- exp_d

Ce sont des versions différentiables de sin, cos et exp.
Elles sont nécessaires pour que la fonction de perte soit entièrement dérivable.

Exemple :
La dérivée de sin(x) est cos(x).

## 5. Optimiseurs implémentés

Quatre optimiseurs sont implémentés :

Nom | Description
----|------------
SGD | Descente de gradient classique
Momentum | SGD avec inertie
RMSProp | Pas adaptatif basé sur le carré du gradient
Adam | Combinaison de Momentum et RMSProp

Chaque optimiseur met à jour les paramètres u et v pour minimiser la fonction f(u, v).

## 6. Visualisation avec Manim

Manim est utilisé pour afficher :

- Les axes (u, v, z)
- La surface z = f(u, v)
- Une sphère représentant la position de l’optimiseur
- La trajectoire suivie par chaque algorithme

Chaque optimiseur démarre au même point initial (u0, v0) et sa trajectoire est affichée sur la surface.

Cela permet de comparer visuellement la vitesse de convergence, la stabilité et les oscillations.

## 7. Organisation du code

Le projet est structuré en trois parties principales :

Autograd
- Tensor
- sin_d, cos_d, exp_d

Optimiseurs
- SGD
- Momentum
- RMSProp
- Adam

Visualisation
- OptimizerRace(ThreeDScene)
- Génération de la surface
- Animation des trajectoires

## Résultats

Les résultats de l’expérience sont visibles dans les vidéos fournies avec le projet.  
Ces vidéos montrent la surface de perte ainsi que les trajectoires des différents optimiseurs (SGD, Momentum, RMSProp et Adam) à partir du même point initial.

https://drive.google.com/open?id=1-nzipeiaCw9xVdnwspxs_womyH8ILkxX&usp=drive_copy

https://drive.google.com/open?id=1ZRvMQeuC3ML8ygzywbMeZiOTwbuUYpUW&usp=drive_copy

Chaque vidéo affiche le déplacement des optimiseurs sur la surface z = f(u, v).

## Analyse des résultats

### Stochastic Gradient Descent (SGD)

SGD converge très lentement et n’atteint pas le minimum local dans la durée de l’animation. Sa trajectoire suit d’abord les directions où le gradient est le plus fort, ce qui le fait progresser principalement selon une dimension avant de se diriger progressivement vers le minimum. Ce comportement est typique sur des surfaces mal conditionnées, où SGD oscille et avance lentement vers l’optimum.

### Momentum (0.9)

L’optimiseur avec momentum converge beaucoup plus rapidement et atteint le minimum local. L’accumulation d’une vitesse permet d’accélérer le mouvement dans les directions cohérentes. Cependant, cette inertie provoque des oscillations autour du minimum, dues à des dépassements dans les zones de forte courbure.

### RMSProp

RMSProp présente la trajectoire la plus stable et la plus directe vers le minimum. Grâce à son taux d’apprentissage adaptatif, il réduit les pas dans les directions raides et les augmente dans les directions plus plates. Cela lui permet de converger rapidement avec très peu d’oscillations.

### Adam

Adam converge également vers le minimum et suit une trajectoire globalement efficace. En combinant momentum et adaptation du pas, il bénéficie d’une bonne vitesse de convergence. Cependant, la composante de momentum introduit encore de légères oscillations près du minimum, ce qui le rend légèrement moins stable que RMSProp dans ce cas.

---

install uv :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run project : 

```bash
uv run manim optimizer_race.py
```
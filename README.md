# ldpc

Ce projet implémente le décodage d'une code LDPC avec
l'algorithme de la propagation de croyance en python
uniquement.

Les programmes de simulations sur canal AWGN sont
[bpsk.py](./src/bpsk.py) pour la modulation BPSK,
[4ask.py](./src/4ask.py) pour la modulation 4-ASK (ou 4-PAM)
et [8ask.py](./src/8ask.py) pour la 8-ASK.

Su canal alpha-stable, les programmes de simulation
équivalents sont [bpsk-astable.py](./src/bpsk-astable.py),
[4ask-astable.py](./src/4ask-astable.py) et
[8ask-astable.py](./src/8ask-astable.py) respectivement.

## Format des codes LDPC

Le format des fichiers décrivant les codes LDPC est une
suite de séquences se terminant par un point-virgule ';'
sauf la dernière qui se termine par un point '.'. Chaque
séquence représente une parité en listant les variables qui
y participent. Les variables sont indexées par un nombre
entre 0 inclu et $n$ exclu avec $n$ la longueur du code.

Par exemple, le code de Hamming [7, 4, 3] dont la matrice
est $$
    H = \begin{pmatrix}
        1&0&1&0&1&0&1\\
        0&1&1&0&0&1&1\\
        0&0&0&1&1&1&1
        \end{pmatrix} $$
et représenté par le fichier
```
0 2 4 6;
1 2 5 6;
3 4 5 6.
```

## Simulation

La simulation utilise le truc de l'envoi du code tout à zéro
en BPSK ou encore l'utilisation d'un offset aléatoire en
ASK. Il faut atteindre un nombre minimal d'erreurs binaire
et aussi d'erreurs sur les mots pour stopper la simulation
en cours pour chaque $E_b/N_0$.

## Alpha-stable

La simulation dans bpsk-astable.py utilise la librairie
astable.py pour simuler un canal alpha-stable et utilisant
le LLR pour décoder.

# ldpc

Ce projet implémente le décodage d'une code LDPC avec
l'algorithme de la propagation de croyance en python
uniquement. Un programme d'exemple est donné dans le fichier
`bpsk.py`.

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

La simulation utilise le truc de l'envoi du code tout
à zéro. Il faut atteindre un nombre minimal d'erreurs
binaire et aussi d'erreurs sur les mots pour stopper la
simulation en cours pour chaque $E_b/N_0$.

## Alpha-stable

La simulation dans bpsk-astable.py utilise la librairie
astable.py pour simuler un canal alpha-stable et utilisant
le LLR pour décoder.

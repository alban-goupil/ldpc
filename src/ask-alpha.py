### Simulation d'une modulation 4ASK sur un canal
### alpha-stable pour plusieurs gamma avec un décodage par
### propagation de croyance. L'alphabet de la 4ASK
### est -3, -1, +1, +3 et le mapping est 11, 10, 00, 01.

### Une séquence de symboles s_1,..., s_m code un mot
### binaire c_1,..., c_n avec n = 2m. Les premiers bits
### c_1,...c_m utilisent les bits 0 de la 4ASK et les
### derniers bits c_{m+1},...,c_n utilisent les bits 1. Par
### exemple la séquence ASK [+3 +1 +1 -3 -3] code le mot
### binaire [0 0 0 1 1 | 1 0 0 1 1] où '|' sert à indiquer
### la séparation entre les bits 0 et les bits 1.

### Le truc du mot de code tout à zéro n'est plus utilisable
### ici. Mais pour éviter un codage, on utilise un offset
### aléatoire mais connu du récepteur, et le mot de code
### émis est de la forme offset + [0 0 0 0... 0].

import numpy as np
import ldpc
import time
from astable import SaS


## * Paramètres du système

alpha = 1.6                      # Exposant du bruit SaS
gammas = np.linspace(.1, .5, 4) # Échelle du bruit
minberrors = 5000               # min erreurs bit à observer
minwerrors = 100                # min erreurs mot à observer
bpitmax = 100                   # max itérations

codefile = '../data/MacKay96-963.ldpc' # Fichier LDPC
ppevery = 1000       # Affichage temporaire tout les 10 mots


## * Construction du code

code = ldpc.LDPC(codefile)


## * Variables de travail

sw = np.empty(code.length//2)   # Mot émis en ASK
rw = np.zeros_like(sw)          # Mot reçu
off = np.empty(code.length, dtype=np.bool) # Offset en binaire
illr = np.zeros(code.length)               # LLR de l'entrée
ollr = np.zeros(code.length)               # Mot décodé

rng = np.random.default_rng()   # Générateur aléatoire


## * LLRs

def L0(y, N, gamma):
  num = np.log(N.pdf(y - 1, gamma) + N.pdf(y - 3, gamma))
  den = np.log(N.pdf(y + 1, gamma) + N.pdf(y + 3, gamma))
  return num - den


def L1(y, N, gamma):
  num = np.log(N.pdf(y - 1, gamma) + N.pdf(y + 1, gamma))
  den = np.log(N.pdf(y - 3, gamma) + N.pdf(y + 3, gamma))
  return num - den



## * Affichage des paramètres

print(f'# LDPC file:\t{codefile}')
print(f'# rate:\t\t{code.rate}')
print(f'# nvars:\t{code.length}')
print(f'# nchks:\t{code.nchecks}')
print(f'# nedgs:\t{code.nedges}')
print(f'# maxiter:\t{bpitmax}')
print(f'# alpha:\t{alpha}')
print(f'# minbiterror:\t{minberrors}')
print(f'# minworderror:\t{minwerrors}')
print('# Gamma #it/cw  #codeword            '
      'BER       #bit error '
      'FER       #word error')


## * Simulation
N = SaS(alpha)                  # v.a. du bruit

tic = time.time()
for gamma in gammas:
  ncw = 0                       # Nombre de mots transmis
  nbe = 0                       # Nombre d'erreurs bit
  nwe = 0                       # Nombre d'erreurs mot
  its = 0                       # Nombre d'itérations
  
  while nbe < minberrors or nwe < minwerrors:
    ncw += 1

    sw = -3 + 2 * rng.integers(4, size=sw.shape)
    rw = sw + N.samples(gamma=gamma, size=sw.shape)
    off[:sw.size] = sw < 0         # bit 0
    off[sw.size:] = np.abs(sw) > 2 # bit 1

    # Calcul du LLR avant décodage
    illr[:sw.size] = L0(rw, N, gamma)
    illr[sw.size:] = L1(rw, N, gamma)
    illr[off] = -illr[off]
    
    # Décodage
    it = code.bp(illr, ollr, bpitmax)
    its += it

    # Comptage des erreurs
    err = np.sum(ollr <= 0.0)
    nbe += err
    if err > 0: nwe += 1

    # Affichage
    if 0 == ncw % ppevery:
      print(f'\r'
            f'{gamma:6.5f}'
            f'{its/ncw: 7.2f}  {ncw:<20d} '
            f'{nbe/ncw/code.length:<9.2e} {nbe:<10d} '
            f'{nwe/ncw:<9.2e} {nwe:<10d}', end='\r')

    
  # Affichage
  print(f'{gamma:6.5f}'
        f'{its/ncw: 7.2f}  {ncw:<20d} '
        f'{nbe/ncw/code.length:<9.2e} {nbe:<10d} '
        f'{nwe/ncw:<9.2e} {nwe:<10d}')

tac = time.time()
print(f"Time: {tac - tic}")

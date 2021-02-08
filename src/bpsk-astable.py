### Simulation d'une BPSK sur un canal gaussien pour
### plusieurs EbN0 avec un décodage par propagation de
### croyance.

import numpy as np
import ldpc
import time
from astable import SaS


## * Paramètres du système

alpha = 1.6                      # Exposant du bruit SaS
gammas = np.linspace(.01, .5, 4) # Échelle du bruit
minberrors = 5000               # min erreurs bit à observer
minwerrors = 100                # min erreurs mot à observer
bpitmax = 100                   # max itérations

codefile = '../data/MacKay96-963.ldpc' # Fichier LDPC
ppevery = 1000       # Affichage temporaire tout les 10 mots


## * Construction du code

code = ldpc.LDPC(codefile)


## * Variables de travail

cw = np.ones(code.length)       # Mot de code en BPSK
rw = np.zeros_like(cw)          # Mot/signal reçu
illr = np.zeros_like(cw)        # LLR de l'entrée
ollr = np.zeros_like(cw)        # Mot décodé

rng = np.random.default_rng()   # Générateur aléatoire


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
N = SaS(1.6)                  # v.a. du bruit

tic = time.time()
for gamma in gammas:

  ncw = 0                       # Nombre de mots transmis
  nbe = 0                       # Nombre d'erreurs bit
  nwe = 0                       # Nombre d'erreurs mot
  its = 0                       # Nombre d'itérations
  
  while nbe < minberrors or nwe < minwerrors:
    # Émission du mot de code tout à zero (+1 en BPSK)
    ncw += 1
    np.copyto(rw, cw)
    rw = cw + gamma * N.samples(size=cw.shape)

    # Calcul du LLR avant décodage
    illr = N.logpdf((rw - 1.0) / gamma) - N.logpdf((rw + 1.0) / gamma)
    
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

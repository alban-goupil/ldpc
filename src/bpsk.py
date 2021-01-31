### Simulation d'une BPSK sur un canal gaussien pour
### plusieurs EbN0 avec un décodage par propagation de
### croyance.

import numpy as np
import ldpc


## * Paramètres du système

ebnos = np.linspace(1, 5, 8)    # Le rapport signal à bruit
minberrors = 100         # # erreurs bit minimum à observer
minwerrors = 10         # # erreurs mot minimum à observer
bpitmax = 100             # # maximum d'itération du BP

codefile = '../data/MacKay96-963.ldpc'


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
print(f'# minbiterror:\t{minberrors}')
print(f'# minworderror:\t{minwerrors}')
print('# EbNo  #it/cw  #codeword            '
      'BER       #bit error '
      'FER       #word error')


## * Simulation

for ebno in ebnos:
  sigma2 = 10 ** (-ebno / 10.0) / 2 / code.rate
  sigma = np.sqrt(sigma2)       # Ecart-type du bruit
  
  ncw = 0                       # Nombre de mots transmis
  nbe = 0                       # Nombre d'erreurs bit
  nwe = 0                       # Nombre d'erreurs mot
  its = 0                       # Nombre d'itérations
  
  while nbe < minberrors or nwe < minwerrors:
    # Émission du mot de code tout à zero (+1 en BPSK)
    ncw += 1
    np.copyto(rw, cw)
    rw = cw + rng.normal(scale=sigma, size=cw.shape)

    # Calcul du LLR avant décodage
    illr = 2.0 * rw / sigma2
    
    # Décodage
    it, _ = code.bp(illr, bpitmax, out=ollr)
    its += it

    # Comptage des erreurs
    err = np.sum(ollr <= 0.0)
    nbe += err
    if err > 0: nwe += 1

  # Affichage
  print(f'{ebno: 6.2f} '
        f'{its/ncw: 7.2f}  {ncw:<20d} '
        f'{nbe/ncw/code.length:<9.2e} {nbe:<10d} '
        f'{nwe/ncw:<9.2e} {nwe:<10d}')

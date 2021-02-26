### Simulation d'une modulation 8ASK sur un canal gaussien
### pour plusieurs EbN0 avec un décodage par propagation de
### croyance. L'alphabet de la 8ASK est -7, -5, -3, -1, +1,
### +3, +5, +7 et le mapping est 110, 111, 101, 100, 000,
### 001, 010, 011.

### Une séquence de symboles s_1,..., s_m code un mot
### binaire c_1,..., c_n avec n = 3m. Les premiers bits
### c_1,...c_m utilisent les bits 0 de la 8ASK, les suivants
### c_{m+1},...,c_{2m} les bits 1 et les derniers
### c_{2m+1},...,c_n utilisent les bits 2. Par exemple la
### séquence ASK [+3 +5 +1 -3 -7] code le mot binaire [0
### 0 0 1 1 | 1 1 0 0 1 | 1 0 0 1 0] où '|' sert à indiquer
### la séparation entre les bits 0 et les bits 1.

### Le truc du mot de code tout à zéro n'est plus utilisable
### ici. Mais pour éviter un codage, on utilise un offset
### aléatoire mais connu du récepteur, et le mot de code
### émis est de la forme offset + [0 0 0 0... 0].

import numpy as np
from scipy.special import logsumexp
import ldpc
import time


## * Paramètres du système

ebnos = np.linspace(3, 8, 6)    # Le rapport signal à bruit
minberrors = 1000               # min erreurs bit à observer
minwerrors = 100                # min erreurs mot à observer
bpitmax = 100                   # max itérations

codefile = '../data/MacKay96-963.ldpc' # Fichier LDPC
ppevery = 1000       # Affichage temporaire tout les 10 mots


## * Construction du code

code = ldpc.LDPC(codefile)


## * Variables de travail

sw = np.empty(code.length//3)   # Mot émis en ASK
rw = np.zeros_like(sw)          # Mot reçu
off = np.empty(code.length, dtype=np.bool) # Offset en binaire
illr = np.zeros(code.length)               # LLR de l'entrée
ollr = np.zeros(code.length)               # Mot décodé

rng = np.random.default_rng()   # Générateur aléatoire


## * LLRs

def L0(y, sigma):
  num = logsumexp(-np.vstack(((rw - 1)/sigma, (rw - 3)/sigma,
                              (rw - 5)/sigma, (rw - 7)/sigma))**2, axis=0)
  den = logsumexp(-np.vstack(((rw + 1)/sigma, (rw + 3)/sigma,
                              (rw + 5)/sigma, (rw + 7)/sigma))**2, axis=0)
  return num - den

def L1(y, sigma):
  num = logsumexp(-np.vstack(((rw - 1)/sigma, (rw - 3)/sigma,
                              (rw + 1)/sigma, (rw + 3)/sigma))**2, axis=0)
  den = logsumexp(-np.vstack(((rw - 5)/sigma, (rw - 7)/sigma,
                              (rw + 5)/sigma, (rw + 7)/sigma))**2, axis=0)
  return num - den

def L2(y, sigma):
  num = logsumexp(-np.vstack(((rw - 1)/sigma, (rw - 5)/sigma,
                              (rw + 1)/sigma, (rw + 7)/sigma))**2, axis=0)
  den = logsumexp(-np.vstack(((rw - 3)/sigma, (rw - 7)/sigma,
                              (rw + 3)/sigma, (rw + 5)/sigma))**2, axis=0)
  return num - den



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

tic = time.time()
for ebno in ebnos:
  # Eb/N0 = Es / (2*M*R*sigma2) avec Es l'énergie moyenne
  # par symbole, M le nombre de bits par symbole, R le
  # rendement du code et sigma2 la variance du bruit sur une
  # voie (I ou Q).
  sigma2 = 10 ** (-ebno / 10.0) * 7 / 4 / code.rate
  sigma = np.sqrt(sigma2)       # Ecart-type du bruit
  
  ncw = 0                       # Nombre de mots transmis
  nbe = 0                       # Nombre d'erreurs bit
  nwe = 0                       # Nombre d'erreurs mot
  its = 0                       # Nombre d'itérations
  
  while nbe < minberrors or nwe < minwerrors:
    # Émission d'un mot de code avec offset en ASK
    ncw += 1

    sw = -7 + 2 * rng.integers(8, size=sw.shape)
    rw = sw + rng.normal(scale=sigma, size=sw.shape)
    off[:sw.size] = sw < 0                  # bit 0
    off[sw.size:2*sw.size] = np.abs(sw) > 4 # bit 1
    off[2*sw.size:] = (sw > 6) \
                    | ((2 < sw) & (sw < 4)) \
                    | ((-6 < sw) & (sw < -2)) # bit 2

    # Calcul du LLR avant décodage
    illr[:sw.size] = L0(rw, sigma)          # bit 0
    illr[sw.size:2*sw.size] = L1(rw, sigma) # bit 1
    illr[2*sw.size:] = L2(rw, sigma)        # bit 2
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
            f'{ebno: 6.2f} '
            f'{its/ncw: 7.2f}  {ncw:<20d} '
            f'{nbe/ncw/code.length:<9.2e} {nbe:<10d} '
            f'{nwe/ncw:<9.2e} {nwe:<10d}', end='\r')

    
  # Affichage
  print(f'{ebno: 6.2f} '
        f'{its/ncw: 7.2f}  {ncw:<20d} '
        f'{nbe/ncw/code.length:<9.2e} {nbe:<10d} '
        f'{nwe/ncw:<9.2e} {nwe:<10d}')

tac = time.time()
print(f"Time: {tac - tic}")

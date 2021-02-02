## Bibliothèque pour le décodage des codes LDPC.
import re
import numpy as np
from numba import jit

p = re.compile('\d+|;|\.')


def _tokens(f):
  """Lit le fichier `f` caractère par caractère pour en
  extraire les nombres entiers positifs, les ';' et qui
  s'arrête au premier '.' pour satisfaire le format des
  fichiers décrivant les LDPC.""" 
  num = None
  c = f.read(1)
  while c:
    if c.isdigit():
      num = int(c) + (10 * num if num else 0)
    elif num is not None:
        yield num
        num = None
    if c == ';': yield c
    if c == '.': break
    c = f.read(1)



class LDPC:
  def __init__(self, codefile):
    """Lit le fichier nommé `codefile` pour construire le code
    LDPC qui y est décrit."""
    checks = [[]]
    with open(codefile) as f:
      for tok in _tokens(f):
        if tok == ';':
          checks.append([])
        else:
          checks[-1].append(tok)
          
    self._vedges = np.concatenate(checks)
    self._cedges = np.cumsum([0] + list(map(len, checks)))
          
    self.nedges = self._vedges.size
    self.nchecks = self._cedges.size-1
    self.length = 1 + self._vedges.max()
    self.rate = (self.length - self.nchecks) / self.length

    self._v2c = np.zeros(self.nedges) # Messages var -> chk
    self._c2v = np.zeros(self.nedges) # Messages chk -> var

    
  def check(self, illr):
    """Vérifie si le tableau `illr` contient le LLR d'un mot de
    code."""
    return numba_check(self._vedges, self._cedges, illr)

  
  def bp(self, illr, ollr, maxiter):
    """Infère le mot de code selon l'algorithme du BP et
    retourne dans `ollr` les LLR de fin de décodage et le
    nombre d'itérations nécessaire.
    """
    return numba_bp(self._vedges, self._cedges, self._v2c, self._c2v,
                    illr, ollr, maxiter)




@jit(nopython=True, fastmath=True)
def numba_check(vedges, cedges, illr):
  iscodeword = True
  for c in range(cedges.shape[0]-1):
    for e in range(cedges[c], cedges[c+1]):
      if illr[vedges[e]] < 0.0:
        iscodeword = not iscodeword
    if not iscodeword: return False
  return iscodeword

  

@jit(nopython=True, fastmath=True)
def numba_bp(vedges,
             cedges,
             v2c,
             c2v,
             illr,
             ollr,
             maxiter):  
  # Initialisation
  for v in range(illr.size): ollr[v] = illr[v]
  for e in range(v2c.size): v2c[e] = np.tanh(ollr[vedges[e]]/2.0)

  # # Itération
  for it in range(maxiter):
    # Arrêt prématuré
    if numba_check(vedges, cedges, ollr):
      return 1+it

    # Check pass
    for c in range(cedges.size-1):
      for e in range(cedges[c], cedges[c+1]):
        m = 1.0
        for ep in range(cedges[c], cedges[c+1]):
          if e != ep:
            m *= v2c[ep]
        if m >= 1.0: c2v[e] = 1e300
        elif m <= -1.0: c2v[e] = -1e300
        else: c2v[e] = 2.0 * np.arctanh(m)

    # Data pass
    for v in range(illr.size): ollr[v] = illr[v]
    for e in range(vedges.shape[0]):
      ollr[vedges[e]] += c2v[e]
    for e in range(vedges.shape[0]):
      v2c[e] = np.tanh((ollr[vedges[e]] - c2v[e])/2.0)
      
  return maxiter

## Bibliothèque pour le décodage des codes LDPC.
import re
import numpy as np

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
    self._checks = checks
          
    self._vedges = np.concatenate(checks)
    self._cedges = np.cumsum([0] + list(map(len, checks)))
          
    self.nedges = self._vedges.size
    self.length = 1 + self._vedges.max()
    self.nchecks = len(checks)
    self.rate = (self.length - self.nchecks) / self.length

    self._v2c = np.zeros(self.nedges) # Messages var -> chk
    self._c2v = np.zeros(self.nedges) # Messages chk -> var

    
  def check(self, illr):
    """Vérifie si le tableau `illr` contient le LLR d'un mot de
    code."""
    iscodeword = True
    for c in range(self.nchecks):
      for e in range(self._cedges[c], self._cedges[c+1]):
        if illr[self._vedges[e]] < 0.0:
          iscodeword = not iscodeword
      if not iscodeword: return False
    return iscodeword


  def bp(self, illr, maxiter, out=None):
    """Infère le mot de code selon l'algorithme du BP et
    retourne les LLR de fin de décodage."""
    if out is None:
      out = np.zeros_like(illr)

    # Initialisation  
    np.copyto(out, illr)
    for e, v in enumerate(self._vedges): self._v2c[e] = np.tanh(out[v]/2.0)

    # Itération
    for it in range(maxiter):
      # Arrêt prématuré
      if self.check(out):
        break

      # Check pass
      for c in range(self.nchecks):
        for e in range(self._cedges[c], self._cedges[c+1]):
          m = 1.0
          for ep in range(self._cedges[c], self._cedges[c+1]):
            if e != ep:
              m *= self._v2c[ep]
          self._c2v[e] = 2.0 * np.arctanh(m)
              
      # Data pass
      np.copyto(out, illr)
      for e, v in enumerate(self._vedges): out[v] += self._c2v[e]
      np.negative(self._c2v, self._v2c)
      for e, v in enumerate(self._vedges): self._v2c[e] += out[v]
      np.divide(self._v2c, 2.0, out=self._v2c)
      np.tanh(self._v2c, out=self._v2c)
      
    return 1+it, out


ldpc = LDPC(codefile)

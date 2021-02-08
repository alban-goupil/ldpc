import numpy as np
from scipy.integrate import quad
from scipy.stats import levy_stable





def _pdf(xs, alpha):
  """Calcule la pdf en `x` d'une loi alpha-stable symétrique
  de paramètre `alpha` et de facteur d'échelle 1.
  """
  def phi(t):
    ta = t ** alpha
    return np.exp(-ta) if ta < 700.0 else 0.0
  
  # if np.abs(alpha - 2.0) < 1e-5: # Gaussian case (gamma = sigma/sqrt(2))
  #   return np.exp(- 0.25 * np.power(xs, 2.0)) / np.sqrt(4.0 * np.pi)

  ps = np.empty_like(xs)
  for i, x in enumerate(xs):
    ps[i] = quad(phi, 0.0, np.inf, weight="cos",
                 wvar=np.abs(x), limlst=1000)[0] / np.pi
  return ps



def _samples(rng, alpha, size=None, eps=1e-5):
  """Génère des échantillons d'une loi alpha-stable symétrique
  de paramètre `alpha` et d'un générateur `rng`. On suppose
  1 < alpha < 2. Pour les lois de Cauchy et de Gauss, il
  faut utiliser les méthodes classiques.
  """
  phi = (rng.uniform(size=size) - 0.5) * np.pi
  w = -np.log(rng.uniform(size=size))
  return np.sin(alpha * phi) / np.power(np.cos(phi), 1.0/alpha) \
    * np.power((np.cos((1.0-alpha)*phi)/w), 1.0/alpha-1)



class SaS:
  def __init__(self, alpha, rng=None):
    assert 1.0 < alpha < 2.0
    self.alpha = alpha
    self.rng = rng or np.random.default_rng()

    # Construction de la lut
    self.xlut = np.linspace(0, 0.9999999 * np.pi/2, 2000)
    # self.ylut = _pdf(np.tan(self.xlut), alpha)
    # self.yplut = np.log(self.ylut)
    self.ylut = levy_stable.pdf(np.tan(self.xlut), alpha, 0.0)
    self.yplut = levy_stable.logpdf(np.tan(self.xlut), alpha, 0.0)
    
    
  def samples(self, gamma=1.0, size=None):
    return gamma * _samples(self.rng, self.alpha, size)

  
  def pdf(self, x, gamma=1.0):
    t = np.arctan(np.abs(x) / gamma) / self.xlut[1]
    i = np.array(t, dtype=np.int)
    f = t - i
    i[i >= self.xlut.size - 1] = self.xlut.size - 3
    p = self.ylut[i.astype(int)] * (1-f) + self.ylut[i.astype(int)+1] * f
    return p / gamma

  
  def logpdf(self, x, gamma=1.0):
    t = np.arctan(np.abs(x) / gamma) / self.xlut[1]
    i = np.array(t, dtype=np.int)
    f = t - i
    i[i >= self.xlut.size - 1] = self.xlut.size - 2
    p = self.yplut[i.astype(int)] * (1-f) + self.yplut[i.astype(int)+1] * f - np.log(gamma)
    p[p < -100.0] = -100.0
    return p


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  x = np.linspace(-10, 10, 1000)

  Xs = [SaS(a) for a in [1.2, 1.4, 1.6, 1.8]]

  # Tracés des pdf
  plt.figure()
  for X in Xs: plt.plot(x, X.pdf(x), label=f'alpha = {X.alpha}')
  plt.grid()
  plt.xlabel('$x$')
  plt.ylabel('pdf')
  plt.legend()

  plt.figure()
  for X in Xs: plt.plot(x, _pdf(x, X.alpha) - X.pdf(x), label=f'alpha = {X.alpha}')
  plt.grid()
  plt.xlabel('$x$')
  plt.ylabel('pdf - interpolation')
  plt.legend()

  plt.figure()
  for X in Xs: plt.plot(x, np.log(_pdf(x, X.alpha)) - X.logpdf(x), label=f'alpha = {X.alpha}')
  plt.grid()
  plt.xlabel('$x$')
  plt.ylabel('logpdf - interpolation')
  plt.legend()

  plt.figure()
  for X in Xs: plt.plot(x, np.log(_pdf(x, X.alpha)) - np.log(X.pdf(x)), label=f'alpha = {X.alpha}')
  plt.grid()
  plt.xlabel('$x$')
  plt.ylabel('logpdf - log(interpolation)')
  plt.legend()

  
  # Tracés des échantillons
  fig, axes = plt.subplots(nrows=len(Xs))
  for X, ax in zip(Xs, axes):
    ax.plot(X.samples(size=1000))
    plt.title(f'alpha: {X.alpha}')

  # Fonction caractéristique
  ts = np.linspace(0, 2*np.pi, 100)
  plt.figure()
  for X in Xs:
    A = X.samples(size=100000)
    # fc empirique
    emp = np.exp(-1j*(ts[None,:] * A[:,None])).mean(0)
    # fc théorique
    the = np.exp(-np.power(ts, X.alpha))
    plt.plot(ts, np.abs(emp - the), label=f'alpha: {X.alpha}')
  plt.legend()
  
  plt.show(block=False)

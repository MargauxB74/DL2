import utils
import numpy as np
import matplotlib.pyplot as plt

"""
3.1 Construction d’un RBM et test sur Binary AlphaDigits
  RESTRICTED BOLTZMAN MACHINE 
  Dans principal RBM alpha nous implementons des fonction pour l'initialisation, l'entrainement et 
  la génération d'images d'un RBM
"""

class RBM:
  """ Methode pour l'initalisation du RBM (constructeur de la classe)
  """
  def __init__(self, p, q):
    """
  p: dimension input (v)
  q: dimension output (h)
    """
    self.p = p
    self.q = q
    self.W = np.random.normal(loc = 0, scale=0.1, size=(self.p, self.q))*0.1 # centered with variance 10^-2
    self.a = np.zeros(shape=(1, self.p))
    self.b = np.zeros(shape=(1, self.q))

def init_RBM(p, q):
  """
  p: dimension input (v)
  q: dimension output (h)
    """
  return RBM(p,q)


def entree_sortie_RBM(self, V):
  """
  self : objet du type RBM
  V : matrice de taille m * p
  """

  H = V @ self.W + self.b
  #retourne une distribution de bernoulli pour les probabilités échantillonnées
  return utils.sigmoid(H)


def sortie_entree_RBM(self, H):
  """
  self : objet du type RBM
  H : matrice de taille m * p
  """

  V = H @ self.W.T + self.a
  #retourne une distribution de bernoulli pour les probabilités échantillonnées
  return utils.sigmoid(V)



def train_RBM(self, X, nb_iter=50, batch_size=200, eps=0.01):
    """
    -> entraie un rbm et renvoie le rbm entrainé et la loss associée.
    self : objet du type RBM.
    X : data en input (les chiffres pour alpha digits).
    nb_iter : nombre de répétition de l'entrainement.
    batch_size : Le batch size définit le nombre d'échantillons qui seront propagés dans le réseau.
    eps : valeur de mises à jour pour contrôler la vitesse de convergence.
    """
    #init de la taille des données
    n = X.shape[0]
    loss = []
    for i in range(nb_iter):
      #crée une copie pour ne pas detruire la donnée
        X_shuffled = X.copy()
        #shuffle 
        np.random.shuffle(X_shuffled) 
        for batch_index in range(0,n,batch_size):
            #init du batch et du batch_size
            X_batch = X_shuffled[batch_index:min(batch_index + batch_size, n)]
            current_batch_size = X_batch.shape[0]
            V0 = X_batch
            #forward
            P_H0 = entree_sortie_RBM(self, V0)
            probs = np.random.rand(current_batch_size,self.q)
            H0 = (probs < P_H0) * 1.
            P_V1 = sortie_entree_RBM(self, H0)
            probs = np.random.rand(current_batch_size,self.p)
            V1 = (probs < P_V1) * 1.
            P_H1 = entree_sortie_RBM(self, V1)
            #maj des gradients
            grad_a = np.sum(X_batch - V1,axis=0)
            grad_b = np.sum(P_H0 - P_H1,axis=0)
            grad_W = V0.T @ P_H0 - V1.T @ P_H1
            #maj des paramètres    
            self.a += eps * (grad_a/current_batch_size)
            self.b += eps * (grad_b/current_batch_size)
            self.W += eps * (grad_W/current_batch_size)
        #calcul et affichage de la loss
        current_loss = np.mean((V1 - V0)**2)
        loss.append(current_loss)
        print(f'Epoch: {i} ------ Reconstruction error: {current_loss}')
    return self, loss



def generer_images_RBM(rbm, nb_images, iter_gibbs, visualize = True):
    """
    -> genere et affiche des images à partir d'un rbm entrainé
    rbm : objet de type rbm entrainé.
    nb_images : nombre d'images.
    iter_gibbs : le nombre d'itérations pour générer une image (plus ce nombre est grand, plus la qualité de l'image reconstruite est bonne).
    visualize : savoir si a la sortie on veut les images ou non.
    """
    p, q = rbm.a.shape[1], rbm.b.shape[1]
    imgs = []
    for i in range(0, nb_images):
      #init d'une image aléatoire
      v = 1* np.random.rand(1,p)<0.5
      #echantillonnage de gibbs
      for j in range(0, iter_gibbs):
        p_h = entree_sortie_RBM(rbm, v)
        h = 1* np.random.rand(1,q)<p_h
        p_v = sortie_entree_RBM(rbm, h)
        v = 1* np.random.rand(1,p)<p_v
      #fin generation
      #dimensionnage de l'image
      imgs.append(1 * v.reshape(20, 16))
      #affichage
      if visualize:
          plt.figure()
          plt.imshow(imgs[-1], cmap='gray') # AlphaDigits
          plt.title("Generated image after {0} iterations".format(iter_gibbs))
          plt.show()

    return np.array(imgs)

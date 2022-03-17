import utils
import principal_RBM_alpha as RBM
import numpy as np
import matplotlib.pyplot as plt
"""
3.2 Construction d’un DBN et test sur Binary AlphaDigits
  DEEP BELIEF NATIVE 
  Dans principal DBN alpha nous implementons des fonctions pour l'initialisation, le pré-entrainement et 
  la génération d'images d'un DBN
"""

class DBN:
  """ Methode pour l'initalisation du DBN (constructeur de la classe)
  """
  def __init__(self, layers, hidden_units):
        """
        layers: > 1
        hidden_units: nb layers + 1
        """
        assert (layers >= 1)
        self.layers = [None] * layers
        self.hidden_units = hidden_units
        self.num_layers = layers
        for layer, layer_units in enumerate(hidden_units):
            self.layers[layer] = RBM.init_RBM(layer_units[0], layer_units[1])


def init_DNN(layers, hidden_units):
  """
  layers: > 1
  hidden_units: nb layers + 1
  """
  return DBN(layers, hidden_units)


def pretrain_DNN(dbn, epochs, eps, batch_size, X):
  """
  -> pré-entraie un dnn et renvoie le dnn pré-entrainé et la loss associée.
  dbn : objet de type dbn
  epochs : nombre de répétition du pré-entrainement
  eps : valeur de mises à jour pour contrôler la vitesse de convergence.
  batch_size : Le batch size définit le nombre d'échantillons qui seront propagés dans le réseau.
  X : X en input (alpha digits).
        """  
  loss = []
  #crée une copie pour ne pas detruire la donnée
  x = X.copy()
  for i in range(dbn.num_layers):
      dbn.layers[i], err_eqm = RBM.train_RBM(dbn.layers[i],x, epochs, batch_size, eps)
      loss.append(err_eqm)
      x = RBM.entree_sortie_RBM(dbn.layers[i], x)
  return dbn, loss


def generer_image_DBN(dbn, nb_images, iter_gibbs, visualize = True):
  """
  -> genere et affiche des images à partir d'un dnn entrainé
  dbn : objet de type dbn pré-entrainé 
  nb_images : nombre d'images.le nombre d'itérations pour générer une image (plus ce nombre est grand, plus la qualité de l'image reconstruite est bonne).he number of sampling iterations to generate an image, the bigger this number is the better quality of reconstructed image we get
  visualize : savoir si a la sortie on veut les images ou non.
        """ 
  p, q = dbn.layers[0].a.shape[1], dbn.layers[-1].b.shape[1]
  imgs = []
  for i in range(0, nb_images):
    #init d'une image aléatoire
    v = 1* np.random.rand(1,dbn.layers[-1].W.shape[0])<0.5
    #echantillonnage de gibbs
    for j in range(0, iter_gibbs):
      p_h = RBM.entree_sortie_RBM(dbn.layers[-1], v)
      h = 1* np.random.rand(p_h.shape[0],p_h.shape[1])<p_h
      p_v = RBM.sortie_entree_RBM(dbn.layers[-1], h)
      v = 1* np.random.rand(p_v.shape[0],p_v.shape[1])<p_v
    for l in range(dbn.num_layers-2, -1, -1):
      proba = RBM.sortie_entree_RBM(dbn.layers[l], v)
      v = 1* np.random.rand(proba.shape[0], proba.shape[1])<proba
    #dimensionnage de l'image
    imgs.append(1 * v.reshape(20, 16))
    #affichage
    if visualize:
        plt.figure()
        plt.imshow(imgs[-1], cmap='gray') # AlphaDigits
        plt.title("Generated image after {0} iterations".format(iter_gibbs))
        plt.show()
  return np.array(imgs)


import utils
import principal_RBM_alpha as RBM
import principal_DBN_alpha as DBN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
"""
3.3 Construction d’un DNN et test sur MNIST
  DEEP NEURAL NETWORK 
  Dans principal DNN mnist nous implementons des fonctions pour calculer le softmax, l'entrée sortie reseau la retropropagation
   et une fonction de test du DNN.
"""

def calcul_softmax(rbm, X):
    """ fonction pour calculer le softmax d'un rbm en input 
    rbm : objet de type rbm
    X :  data de type mnist ici
    """
    A = np.dot(X, rbm.W) + rbm.b
    e = np.exp(A - np.max(A, axis=1).reshape((-1, 1)))
    return  e / e.sum(axis=1).reshape((-1,1))

def entree_sortie_reseau(dnn, X):
    """
    dnn : objet de type dnn
    X : data de type mnist ici
    """
    sorties_couches = []
    sorties_couches.append(RBM.entree_sortie_RBM(dnn.layers[0], X))
    for i in range(1, dnn.num_layers-1):
        sorties_couches.append(RBM.entree_sortie_RBM(dnn.layers[i], sorties_couches[-1]))
    sorties_couches.append(calcul_softmax(dnn.layers[-1], sorties_couches[-1]))
    return sorties_couches

def copy_dnn(dnn):
    new_dnn = DBN.init_DNN(dnn.num_layers, [dnn.layers[i].W.shape for i in range(dnn.num_layers)])
    for i in range(dnn.num_layers):
        new_dnn.layers[i].a = dnn.layers[i].a.copy()
        new_dnn.layers[i].b = dnn.layers[i].b.copy()
        new_dnn.layers[i].W = dnn.layers[i].W.copy()
    return new_dnn


def retropropagation(dnn, X_train, Y_train, nb_iter, eps, batch_size, pre_trained, visualize=True, imagepath=None):
    """
    -> retourne un fine_tuned dnn
    dnn : objet de type dnn
    X_train : échantillon pour l'entrainement.
    Y_train : échantillon pour l'entrainement.
    nb_iter : nombre de répétition de l'entrainement.
    lr : valeur de mises à jour pour contrôler la vitesse de convergence.
    batch_size : Le batch size définit le nombre d'échantillons qui seront propagés dans le réseau.
    pre_trained : réseau pré entrainé.
    visualize : savoir si a la sortie on veut les images ou non.
    image_path : chemin pour accéder aux images.
    """
    if type(X_train) == 'pandas.core.frame.DataFrame':
        X_train = X_train.values
    entrop_crois = []
    for ploud in range(nb_iter):
        indices = np.arange(0, X_train.shape[0], 1)
        #shuffle
        np.random.shuffle(indices)
        #batch iterations
        for j in range(0, X_train.shape[0], batch_size):
            new_dnn = copy_dnn(dnn)
            batch_ind = indices[j:min(j + batch_size, X_train.shape[0])]
            X = X_train[batch_ind, :]
            sorties_couches = entree_sortie_reseau(dnn, X)
            #début dernière couche
            matrice_c = sorties_couches[-1] - Y_train[batch_ind]
            der_w = np.dot(sorties_couches[dnn.num_layers - 2].transpose(), matrice_c) / X.shape[0]
            der_b = matrice_c.sum(0) / X.shape[0]
            new_dnn.layers[-1].W -= eps * der_w  # /batch
            new_dnn.layers[-1].b -= eps * der_b
            #fin dernière couche
            for couche in range(dnn.num_layers - 2, -1, -1):

                if couche == 0:
                    inpute = X
                else:
                    inpute = sorties_couches[couche - 1]

                h_mult = sorties_couches[couche] * (1 - sorties_couches[couche])
                matrice_c = np.dot(matrice_c, dnn.layers[couche + 1].W.transpose()) * h_mult
                der_w = np.dot(inpute.transpose(), matrice_c) / X.shape[0]
                der_b = matrice_c.sum(0) / X.shape[0]
                new_dnn.layers[couche].W -= eps * der_w
                new_dnn.layers[couche].b -= eps * der_b

            dnn = copy_dnn(new_dnn)
        #forward 
        sorties_couches = entree_sortie_reseau(dnn, X_train)
        classif = -np.log10(sorties_couches[-1])[Y_train == 1]
        #calcul de la loss
        erreur = classif.sum()
        entrop_crois.append(erreur)

    if visualize:
        f = plt.figure(figsize=(10, 7))
        plt.plot(range(nb_iter), entrop_crois)
        plt.legend(['Entropie croisée'])
        plt.title("Évolution de l'entropie croisée au cours des iterations")
        plt.xlabel("nombre d'itérations")
        plt.ylabel('entropie croisée')
        plt.show()
        #if imagepath is not None:
            #f.savefig(imagepath+'retropropagation_{}.png'.format(pre_trained))
        return dnn

    else:
        return dnn, entrop_crois[-1]


def accuracy_score(y_test, y_pred):
    result = (y_test != y_pred)
    return "Accuracy : {}%".format(round((1 - (result.sum()//2)/y_test.shape[0])*100*100)/100)


def matrice_de_confusion(y_test, y_pred, erreur, pre_trained, imagepath=None):
    y_t = []
    y_p = []
    for i in range(y_test.shape[0]):
        for j in range(y_test.shape[1]):
            if y_test[i,j] == 1:
                y_t.append(j)
            if y_pred[i,j] == 1:
                y_p.append(j)
    df_cm = pd.DataFrame(confusion_matrix(y_t, y_p), index = range(10), columns = range(10))
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.title("MATRICE DE CONFUSION {} \n {}\n Entropie croisée : {}".format(pre_trained, accuracy_score(y_test, y_pred), round(erreur*100)/100))
    #if imagepath is not None:
        #plt.savefig(imagepath + 'confusion_mat_{}.png'.format(pre_trained))



def test_DNN(dnn, X_test, y_test, pre_trained, visualize = True, imagepath=None):
    """
    dnn : objet de type dnn
    X_test : échantillon pour le test.
    Y_train : échantillon pour le test.
    pre_trained : réseau pré entrainé.
    visualize : savoir si a la sortie on veut les images ou non.
    image_path : chemin pour accéder aux images.
    """
    #le dernier element de la liste est la proba de la classe
    y_pred = entree_sortie_reseau(dnn, X_test)[-1]
    classif = -np.log10(y_pred)[y_test==1]
    #calcul de la loss
    erreur = classif.sum() 
    print("entropie croisée :", erreur)
    for i in range(y_pred.shape[0]):
        for j in range(y_pred.shape[1]):
            if y_pred[i,j] == max(y_pred[i,:]):
                y_pred[i,j] = 1
            else:
                y_pred[i,j] = 0
    print(accuracy_score(y_test, y_pred))
    if visualize:
      matrice_de_confusion(y_test, y_pred, erreur, pre_trained, imagepath)

    return float(accuracy_score(y_test, y_pred).split(':')[-1].split('%')[0])/100, erreur
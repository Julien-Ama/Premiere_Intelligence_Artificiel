import numpy as np

x_entrer = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4,1.5]), dtype=float)
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]),dtype=float) # donnée de sortie 1 = rouge / 0 = bleu

x_entrer = x_entrer/np.amax(x_entrer, axis=0)

# on a des valeur entre 0 et 1

X = np.split(x_entrer,[8])[0]
xPrediction = np.split(x_entrer,[8])[1]

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2 # synapse d'entrée
        self.outputSize = 1 # la valeur de sortie (on en veut que une)
        self.hiddenSize = 3 # les synapse cachées

        # poid des synapses
        # entre synapses d'entré et synapses cachés
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # Matrice 2x3
        #entre synapses cachés et synapses du neurone final
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # Matrice 3x1

    def forward(self,X):

        self.z = np.dot(X,self.W1) # valeur d'entrer z = 0.54545455 * 0.14557919 +1 * 0.31888304 = 39828987...
        self.z2 = self.sigmoid(self.z) # valeur caché
        self.z3 = np.dot(self.z2,self.W2) # valeur final
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self,s):
        return 1/(1+np.exp(-s))



    #comment modifier les poids des synapse ( methode : Algorithme du gradient )
    # _ Calculer la marge d'erreur (input - output = erreur)
    # _ Appliquer la derivé de la fonction d'activation( la sigmoid a cet erreur) = l'erreur_delta
    # _ Multiplication matricielle entre W2 et erreur_delat = l'erreur_2_cachée
    # _ Appliquer la dérivé de la sigmoid a l'erreur_2_cachée = l'erreur_2_delta_cachée
    # _ Ajuster W1 et W2

    def sigmoidPrime(self,s):
        return s * (1-s)

    # . X = valeur d'entrée . y = valeur qu'on devrai avoir . o = valeur de sortie
    def backward(self,X,y,o):

        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        # mettre à jour no poids
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self,X,y):
        o = self.forward(X)
        self.backward(X,y,o)

    def predict(self):
        print("Donnée predite apres entrainement:")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))

        if(self.forward(xPrediction) < 0.5):
            print("la fleur est bleu ! \n")
        else:
            print("la fleur est rouge ! \n")


NN = Neural_Network()

for i in range(3000):
    print("#" + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie Actuelle: \n" + str(y))
    print("Sortie predite: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.predict()

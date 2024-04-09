"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import keras
import os
from keras.optimizers import Adam

from helpers.ImageCollection import ImageCollection
import helpers.classifiers as classifiers

VERBOSE = False
data_processing = False #Leave that to False. If not, please delete .txt file before.
analyse_data = True
deocrelate_data = analyse_data
test_set = analyse_data

neural_network = False#analyse_data
ppv = analyse_data
bayesien = False#analyse_data


#######################################
def problematique_APP2():
    img = ImageCollection(load_all = True)
    if VERBOSE:
        print(f'The shape of the input is: {img.images.shape}')
        print(f'The shape of the label is: {img.labels.shape}')
        print(img.labels[0])

        N = 10
        im_list = img.get_samples(N)

        img.images_display(im_list)
        img.view_histogrammes(im_list)

        plt.show()

    #If the user is doing a test to accelerate the process 
    images_test = img.images[:6]
    label_test = img.labels[:6]
                
    img.generateRepresentation(img.images, img.labels, data_processing, analyse_data, deocrelate_data, test_set)


    if neural_network:
            # Exemple de RN
            n_neurons = 6
            n_layers = 2

            beset_model = keras.callbacks.ModelCheckpoint(filepath='3classes_prob'+os.sep+'.keras', monitor='val_loss', verbose=1)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.95, patience=5, min_delta=0.001)
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=25, mode='min')

            nn1 = classifiers.NNClassify_APP2(train_data=img.training_data, train_label=img.training_target, test_data=img.test_data, test_label=img.test_target,
                                          extent=img.extent, n_layers=n_layers, n_neurons=n_neurons, innerActivation='tanh',
                                          outputActivation='softmax', optimizer=Adam(learning_rate=0.025), loss='categorical_crossentropy',
                                          metrics=['accuracy'],
                                          callback_list=[beset_model, reduce_lr, early_stopping], 
                                          experiment_title='NN Simple',
                                          n_epochs = 1000, savename=None,
                                          ndonnees_random=5000, gen_output=True, view=True)

    if ppv: 
        # Exemples de ppv avec ou sans k-moy
        # 1-PPV avec comme représentants de classes l'ensemble des points déjà classés
        n_neighbors = 20
        ppv1 = classifiers.PPVClassify_APP2(train_data=img.training_data, train_label=img.training_target, test_data=img.test_data, test_label=img.test_target,
                                            n_neighbors=n_neighbors, extent=img.extent, 
                                            experiment_title=f'{n_neighbors}-PPV avec données orig comme représentants',
                                            gen_output=True, view=True)
        # 1-mean sur chacune des classes
        # suivi d'un 1-PPV avec ces nouveaux représentants de classes
        # n_neighbors = 5
        # n_representants = 12
        # ppv1km1 = classifiers.PPVClassify_APP2(train_data=img.training_data, train_label=img.training_target, test_data=img.test_data, test_label=img.test_target, 
        #                                     n_neighbors=n_neighbors, extent=img.extent, experiment_title=f'{n_neighbors}-PPV sur le {n_representants}-moy',
        #                                     useKmean=True, n_representants=n_representants,
        #                                     gen_output=True, view=False)


    if bayesien:
        # Exemple de classification bayésienne
        apriori = [324/881, 295/881, 262/881]
        cost = [[0, 1, 3], 
                [1, 0, 1], 
                [2, 1, 0]]
        # Bayes gaussien les apriori et coûts ne sont pas considérés pour l'instant
        bg1 = classifiers.BayesClassify_APP2(train_data=img.training_data, train_label=img.training_target, test_data=img.test_data, test_label=img.test_target,
                                             apriori=apriori, costs=cost,
                                             experiment_title='probabilités gaussiennes',
                                             gen_output=True, view=True, extent=img.extent) 


######################################
if __name__ == '__main__':
    problematique_APP2()

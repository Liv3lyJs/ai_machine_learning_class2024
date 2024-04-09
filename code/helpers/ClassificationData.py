"""
Classe "ClassificationData" : vise à contenir la représentation à utiliser pour classifier par la suite
Constructeur:
    par défaut, charge les 3 classes du laboratoire
    peut aussi utiliser un array ou une liste de listes
Membres :
    pour simplifier l'utilisation dans différents modes d'entraînement des techniques et dans les prédictions subséquentes,
    les données sont représentées sous 2 formes
        i. Liste de listes ou array_like
            -> Format nécessaire quand on veut constuire l'objet à partir de données existantes.
                TODO problématique: construire un object classification data à partir des quantités retenues pour la représentation
            -> dataLists : array_like de dimension K * L * M, K le nombre de classes distinctes, L les données pour chaque classe
            Note: il n'existe pas d'étiquette explicite pour ce format, on utilise automatiquement la position dans l'array selon la première dimension
        ii. 1 seul long vecteur. Généré automatiquement à partir du format précédent
            -> data1array: array N x M où N est le nombre total de données disponibles (N = K * L)
                et M la dimension de l'espace de représentation
            -> labels1array: étiquettes de classes. Entiers de 0 à L-1 générés automatiquement à partir de la dimension L dans dataLists
    extent: la plage utile des données
Méthodes:
    getStats: calcule des statistiques descriptives sur chaque classe
    getBorders: visualise les nuages de points de chaque classe avec ou sans les frontières calculées
"""

import numpy as np
import os
import helpers.analysis as an
import helpers.classifiers as classifiers


folder_name = 'prob_000'


class ClassificationData:

    def __init__(self, existingData=None, problematique=False):
        self.dataLists = []
        if existingData is not None:
            for data in existingData:
                if np.asarray(data).any():
                    self.dataLists.append(data)
        else:
            if not problematique:
                # Import data from text files in subdir
                self.dataLists.append(np.loadtxt('data'+os.sep+'data_3classes'+os.sep+'C1.txt'))
                self.dataLists.append(np.loadtxt('data'+os.sep+'data_3classes'+os.sep+'C2.txt'))
                self.dataLists.append(np.loadtxt('data'+os.sep+'data_3classes'+os.sep+'C3.txt'))
            else:
                #Load data for the problematique. 
                self.dataLists.append(np.loadtxt('final_data'+os.sep+f'{folder_name}'+os.sep+'coast.txt'))
                self.dataLists.append(np.loadtxt('final_data'+os.sep+f'{folder_name}'+os.sep+'forest.txt'))
                self.dataLists.append(np.loadtxt('final_data'+os.sep+f'{folder_name}'+os.sep+'street.txt'))

        self.classification = len(self.dataLists) #Number of classes to analyse
        # reorganisation en 1 seul vecteur pour certains entraînements et les predicts
        self.data1array = np.vstack(self.dataLists)
        _, self._x = self.data1array.shape
        self.variable_number = self._x #Number of variable per images
        self.ndata = len(self.data1array)

        # assignation des classes d'origine 0 à 2 pour C1 à C3 respectivement
        self.labels1array = np.zeros([self.ndata, 1])
        self.labelsLists = []
        self.labelsLists.append(self.labels1array[range(len(self.dataLists[0]))])
        for i in range(1,self.classification):
            self.labels1array[range(i * len(self.dataLists[i]), (i + 1) * len(self.dataLists[i]))] = i
            self.labelsLists.append(self.labels1array[range(i * len(self.dataLists[i]), (i + 1) * len(self.dataLists[i]))])

        for i in range(self.classification ):
            print(len(self.labelsLists[i]))

        # Min et max des données
        self.extent = an.Extent(ptList=self.data1array)

        self.m = []
        self.cov = []
        self.valpr = []
        self.vectpr = []
        self.coeffs = []

        if existingData is None:
             
            #Normalise data between 0 and 1. 
            self.data1array, _ =an.scaleDataPerColumn(self.data1array) #  scaleDataPerColumn  scaleData
            self.dataLists_norm = []
            for i in range(self.classification):
                temp, _ = an.scaleDataPerColumn(self.dataLists[i])
                self.dataLists_norm.append(temp)

            self.getStats(gen_print=True, save_cov_txt=True)

        else:
            self.getStats(gen_print=True)
            self.getBorders()

    def getStats(self, gen_print=False, save_cov_txt=False):
        if not self.m:
            for i in range(self.classification):
                _m, _cov, _valpr, _vectpr = an.calcModeleGaussien(self.dataLists_norm[i])
                self.m.append(_m)
                self.cov.append(_cov)
                self.valpr.append(_valpr)
                self.vectpr.append(_vectpr)
            #Calculate for all the values in the dataset. 
            self.m_3classes, self.cov_3classes, self.valpr_3classes, self.vectpr_3classes = an.calcModeleGaussien(self.data1array)
        if gen_print:
            #Save the average and covariance matrix information into a .txt file to facilitate it analyse. 
            for i in range(self.classification):
                an.printModeleGaussien(
                    self.m[i], self.cov[i], self.valpr[i], self.vectpr[i], f'\nClasse {i + 1}' if gen_print else '')
            an.printModeleGaussien(self.m_3classes, self.cov_3classes, self.valpr_3classes, self.vectpr_3classes, message='All classes at the same time.')
                
        if save_cov_txt:
            file_name = f'{folder_name}.txt'
            file2_name = f'{file_name}_all'
            folder = 'covariance_matrix'
            output_file = f'/home/jean-sebastien/Documents/s7/APP2/Problematique/s7-app2/Analyse_image2/{folder}'
            if not os.path.exists(output_file):
                os.makedirs(output_file)
            file1 = os.path.join(output_file, file_name)
            file2 = os.path.join(output_file, file2_name)

            with open(file1, 'w') as file, open(file2, 'w') as file2: 
                for i in range(self.classification):
                    #Write the mean value
                    file.write(f'Classe {i}: \n')
                    file.write(f'Mean: \n')
                    mean_info = ' '.join(f'{valeur:.6f}' for valeur in self.m[i]) + '\n'
                    file.write(mean_info)
                    file.write(f'Covariance: \n')
                    for ligne in self.cov[i]: 
                        file.write(' '.join(f'{value:.6f}' for value in ligne) + '\n')
                #Write the mean value
                file2.write(f'Mean: \n')
                mean_info = ' '.join(f'{valeur:.6f}' for valeur in self.m_3classes) + '\n'
                file2.write(mean_info)
                file2.write(f'Covariance: \n')
                for ligne in self.cov_3classes: 
                    file2.write(' '.join(f'{value:.6f}' for value in ligne) + '\n')     


        return self.m, self.cov, self.valpr, self.vectpr

    def getBorders(self, view=False):
        if not self.coeffs:
            self.coeffs = classifiers.get_gaussian_borders(self.dataLists)
        if view:
            an.view_classes(self.dataLists, self.extent, self.coeffs)
        return self.coeffs
    
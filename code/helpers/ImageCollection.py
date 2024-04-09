"""
Classe "ImageCollection" pour charger et visualiser les images de la problématique
Membres :
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, changer le flag load_all du constructeur à True)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes pour la problématique :
    generateRGBHistograms : calcul l'histogramme RGB de chaque image, à compléter
    generateRepresentation : vide, à compléter pour la problématique
Méthodes génériques :
    generateHistogram : histogramme une image à 3 canaux de couleurs arbitraires
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
from enum import IntEnum, auto
from PIL import Image

from skimage import color as skic
from skimage import io as skiio
from scipy.ndimage import label
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from scipy.signal import convolve2d
from skimage import filters

import helpers.analysis as an
from helpers.ClassificationData import ClassificationData


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """
    class imageLabels(IntEnum):
        coast = auto()
        forest = auto()
        street = auto()

    def __init__(self, load_all=False):
        # liste de toutes les images
        self.image_folder = r"data" + os.sep + "baseDeDonneesImages"
        self._path = glob.glob(self.image_folder + os.sep + r"*.jpg")
        image_list = os.listdir(self.image_folder)
        # Filtrer pour juste garder les images
        self.image_list = [i for i in image_list if '.jpg' in i]

        self.all_images_loaded = False
        self.images = []

        # Crée un array qui contient toutes les images
        # Dimensions [980, 256 ,256, 3]
        #            [Nombre image, hauteur, largeur, channels]
        if load_all:
            self.images = np.array([np.array(skiio.imread(image)) for image in self._path])
            self.all_images_loaded = True

        self.labels = []
        for i in image_list:
            if 'coast' in i:
                self.labels.append(ImageCollection.imageLabels.coast)
            elif 'forest' in i:
                self.labels.append(ImageCollection.imageLabels.forest)
            elif 'street' in i:
                self.labels.append(ImageCollection.imageLabels.street)
            else:
                raise ValueError(i)
            
        self.labels = np.array(self.labels)

    def standardization(self, data) -> float:
        """
        Standardize data between 0 and 1. 
        """
        return data / 255.

    def get_standardization(self, data_set) -> np.array:
        """
        Do standardisation on the data set. 
        """
        std_data_set = []
        for sample in data_set: 
            std_data_set.append(self.standardization(sample))

        return np.array(std_data_set)

    def get_samples(self, N):
        """
        Get a sample from the test set. 
        """
        return np.sort(random.sample(range(np.size(self.image_list, 0)), N))

    def generateHistogram(self, image, n_bins=256):
        """
        Generate an histogram 
        """
        # Construction des histogrammes
        # 1 histogram per color channel
        n_channels = 3
        pixel_values = np.zeros((n_channels, n_bins))
        for i in range(n_bins):
            for j in range(n_channels):
                pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)
        return pixel_values

    def get_generateHistograms(self, images_norm):
        """
        Calcule les histogrammes RGB de toutes les images
        """
        rgb_histograms = []
        for sample in images_norm: 
            rgb_histograms.append(self.generateHistogram(sample))

        return np.array(rgb_histograms)
    
    def convert_rgb2lab(self, data_set):
        """
        Convert rgb to lab 
        """
        return skic.rgb2lab(data_set) 
    
    def convert_rgb2hsv(self, data_set):
        """
        Convert rgb to hsv
        """
        return skic.rgb2hsv(data_set)
    
    def convolve2d(self, image, kernel):
        """
        Apply 2D convolution without padding
        """
        kernel = np.flipud(np.fliplr(kernel))
        output = np.zeros_like(image)

        #Convolution 
        image_padded = np.pad(image, ((kernel.shape[0]//2, kernel.shape[1]//2),
                                    (kernel.shape[0]//2, kernel.shape[1]//2)), 
                            mode='constant', constant_values=0).astype(np.float32)
        #Apply the convolution on the gray image. 
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                output[y, x] = (kernel * image_padded[y: y+kernel.shape[0], x: x+kernel.shape[1]]).sum()
        
        return output
    
    def edge_detection(self, images):
        """
        Do edge detection for one image in the data set. 
        """
        #Calculate the gradients with the sobel filter. 
        gradient_x = filters.sobel_h(images) #convolve2d(images, sobel_x)
        gradient_y = filters.sobel_v(images)

        #Combine into one resulting image
        image_result = np.sqrt(gradient_x**2 + gradient_y**2)
        image_result = (image_result / image_result.max()) * 255

        #Display information for debugging reasons if wanted. 
        verbose = 0
        if verbose: 
            Image.fromarray(image_result.astype('uint8'), 'L').show()

        return image_result, gradient_x, gradient_y
    
    def rgb_to_grayscale(self, rgb_image):
        """
        Convert a rgb image into a grayscale image. 
        """
        R = rgb_image[:, :, 0]
        G = rgb_image[:, :, 1]
        B = rgb_image[:, :, 2]

        #Manually convert a GGB image into grayscale. 
        grayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B

        #Visualise the image if wanted. 
        verbose = 0
        if verbose: 
            Image.fromarray(rgb_image.astype('uint8')).show()
            Image.fromarray(grayscale.astype('uint8'), 'L').show()

        return grayscale
    
    def get_edge_detection(self, images):
        """
        Apply edge detection for all the images of the dataset. 
        Start by converting the images in grayscale and then apply the edge detection 
        algorithm on every images. 
        """
        num_images = len(images)
        #Initialize numpy array to allow the program to work quicker. 
        images_result = np.zeros((num_images, 256, 256))
        gradients_x = np.zeros((num_images, 256, 256))
        gradients_y = np.zeros((num_images, 256, 256))

        for i, img in enumerate(images):
            print(f'The program is analysing image number: {i}')
            #convert the image into grayscale image. 
            image_gray = self.rgb_to_grayscale(img)
            #Apply the edgedetection 
            image_result, gradient_x, gradient_y = self.edge_detection(image_gray)

            #Keep result in memory 
            images_result[i] = image_result
            gradients_x[i] = gradient_x
            gradients_y[i] = gradient_y
        
        return images_result, gradients_x, gradients_y
    
    def count_contours(self, image, threshold):
        """
        count the connected region in one image
        """
        #For visualisation purpose. 
        verbose = 0
        if verbose:
            Image.fromarray(image.astype('uint8'), 'L').show()
        #Put the values under the treshold at 0.
        new_image = (image > threshold).astype(int)
        #Visualise the transformation. 
        if verbose:
            Image.fromarray(new_image.astype('uint8'), 'L').show()

        labeled_array, num_features = label(new_image)

        return labeled_array, num_features
    
    def contour_lengths(self, labeled_array, num_features):
        """
        Count the contour length for one image
        """
        lengths = np.zeros(num_features)

        for i in range(1, num_features+1):
            lengths[i-1] = np.sum(labeled_array == i)

        #Extract sub values for the precise features
        mean_lengths = np.mean(lengths)
        total_length = np.sum(lengths)
        std_length = np.std(lengths)

        return mean_lengths, total_length, std_length
    
    def contour_orientations(self, Gx, Gy, labeled_array, num_features):
        """
        Find the contour orientation in one image
        """
        orientations = np.zeros(num_features)
        #Itterate over every features fond in the image. 
        for i in range(1, num_features+1):
            contour_pixels = labeled_array == i
            gx = Gx[contour_pixels]
            gy = Gy[contour_pixels]
            angle = np.arctan2(gy, gx)
            #Convert back to deg. 
            orientations[i-1] = np.rad2deg(np.mean(angle))

        #Extract sub values for the precise features
        mean_orientation = np.mean(orientations)
        std_orientation = np.std(orientations)
        
        return mean_orientation, std_orientation
    
    def mean_value(self, image):
        """
        Calculate the mean value of the image by channel
        """
        verbose = 1
        if verbose:
            Image.fromarray(image.astype('uint8')).show()
        mean_red = np.mean(image[:, :, 0])
        mean_green = np.mean(image[:, :, 1])
        mean_blue = np.mean(image[:, :, 2])

        return mean_red, mean_green, mean_blue

    def median_value(self, image):
        """
        Calculate the median value of the image by channel
        """
        meidan_red = np.median(image[:, :, 0])
        meidan_green = np.median(image[:, :, 1])
        meidan_blue = np.median(image[:, :, 2])

        return meidan_red, meidan_green, meidan_blue
    
    def variance_value(self, image):
        """
        Calculate the variance of the images by channels
        """
        var_red = np.var(image[:, :, 0])
        var_green = np.var(image[:, :, 1])
        var_blue = np.var(image[:, :, 2])

        return var_red, var_green, var_blue
    
    def pourcentile_value(self, image):
        """
        Calculate the percentile value from the image channels
        """
        percentile25_red = np.percentile(image[:, :, 0], 25)
        percentile75_red = np.percentile(image[:, :, 0], 75)

        percentile25_green = np.percentile(image[:, :, 1], 25)
        percentile75_green = np.percentile(image[:, :, 1], 75)

        percentile25_blue = np.percentile(image[:, :, 2], 25)
        percentile75_blue = np.percentile(image[:, :, 2], 75)

        return percentile25_red, percentile75_red, percentile25_green, percentile75_green, percentile25_blue, percentile75_blue
    
    def texture_extraction(self, image):
        """
        Extract the texture of the image
        """
        #Convert the image with values between 0 and 255
        gray_image = skic.rgb2gray(image)
        gray_image = (gray_image*255).astype('uint8')

        #Extract the GLCM value 
        distances = [1]
        #Set the angles values. 
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        #List of texture features to extract
        features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        texture_features = {feature: graycoprops(glcm, feature)[0, 0] for feature in features}

        return features, texture_features

    def get_feature_extraction(self, input_data, label_test):
        """
        get the feature extraction. 
        """
        #Flag used to know which feature to extract. It is done when you are searching the optimal parameters. 
        edge_detection = True
        color_detection = False
        texture_detection = True
        categories = ['coast', 'forest', 'street']
        image_features = {}

        #Create distionaries to temporary contain the data. 
        for categorie in categories:
            if edge_detection:
                #Edge detection 
                image_features[f'images_num_features_{categorie}'] = []
                image_features[f'images_mean_lengths_{categorie}'] = []
                image_features[f'images_total_length_{categorie}'] = []
                image_features[f'images_total_std_length_{categorie}'] = []
                image_features[f'images_mean_orientation_{categorie}'] = []
                image_features[f'images_std_orientation_{categorie}'] = []
            if color_detection: 
                #Color in the image
                image_features[f'images_mean_red_{categorie}'] = []
                image_features[f'images_mean_green_{categorie}'] = []
                image_features[f'images_mean_blue_{categorie}'] = []
                image_features[f'images_meidan_red_{categorie}'] = []
                image_features[f'images_meidan_green_{categorie}'] = []
                image_features[f'images_meidan_blue_{categorie}'] = []
                image_features[f'images_std_red_{categorie}'] = []
                image_features[f'images_std_green_{categorie}'] = []
                image_features[f'images_std_blue_{categorie}'] = []               
                image_features[f'images_percentile25_red_{categorie}'] = []
                image_features[f'images_percentile75_red_{categorie}'] = []
                image_features[f'images_percentile25_green_{categorie}'] = []
                image_features[f'images_percentile75_green_{categorie}'] = []
                image_features[f'images_percentile25_blue_{categorie}'] = []
                image_features[f'images_percentile75_blue_{categorie}'] = []
            if texture_detection:
                #Texture in the image
                image_features[f'images_contrast_{categorie}'] = []
                image_features[f'images_dissimilarity_{categorie}'] = []
                image_features[f'images_homogeneity_{categorie}'] = []
                image_features[f'images_energy_{categorie}'] = []
                image_features[f'images_correlation_{categorie}'] = []
                image_features[f'images_ASM_{categorie}'] = []

        #Pretraitement for edge detection, which is finding the edges 
        if edge_detection: 
            images_edge_detection, Gx, Gy = self.get_edge_detection(input_data)
            #Flag to print information on screen, can be used when debugging or to show information. 
            verbose = 0
            if verbose:
                print(f'The shape of the input image is: {images_edge_detection.shape}')
                print(f'The shape of the gradient x is: {Gx.shape} and y is: {Gx.shape}')

        for i, img in enumerate(input_data):
            if edge_detection: 
                #Edge detection 
                threshold = 50 #Number that can be change when converting a gray image to a edgedetection image. 
                labeled_array, num_features = self.count_contours(images_edge_detection[i], threshold)
                mean_lengths, total_length, std_length = self.contour_lengths(labeled_array, num_features)
                mean_orientation, std_orientation = self.contour_orientations(Gx[i], Gy[i], labeled_array, num_features)
            if color_detection: 
                #Color in the image
                change_color_lab = False #Flag used when working with a different color format. 
                change_color_hsv= False

                if change_color_lab: 
                    img_color = self.convert_rgb2lab(img)
                    color_name = 'Lab'
                elif change_color_hsv:
                    img_color = self.convert_rgb2hsv(img)
                    color_name = 'HSV'
                else:
                    img_color = img
                    color_name = 'RGB'

                #Extract color features
                mean_red, mean_green, mean_blue = self.mean_value(img_color)
                meidan_red, meidan_green, meidan_blue = self.median_value(img_color)
                std_red, std_green, std_blue = self.variance_value(img_color)
                percentile25_red, percentile75_red, percentile25_green, percentile75_green, percentile25_blue, percentile75_blue = self.pourcentile_value(img_color)
                print(f'The program is analysing image number: {i}')

            if texture_detection:
                #Texture in the image
                _, texture_features = self.texture_extraction(img)
                
                #Extract texture features
                contrast = texture_features['contrast']
                dissimilarity = texture_features['dissimilarity']
                homogeneity = texture_features['homogeneity']
                energy = texture_features['energy']
                correlation = texture_features['correlation']
                ASM = texture_features['ASM']
                print(f'The program is analysing image number: {i}')

            if label_test[i] == 1: 
                location = 'coast'
            if label_test[i] == 2: 
                location = 'forest'
            if label_test[i] == 3: 
                location = 'street'

            #Append the feature to the dictionaries for the image analysed. 
            if edge_detection:
                #Edge detection 
                image_features[f'images_num_features_{location}'].append(num_features)
                image_features[f'images_mean_lengths_{location}'].append(mean_lengths)
                image_features[f'images_total_length_{location}'].append(total_length)
                image_features[f'images_total_std_length_{location}'].append(std_length)
                image_features[f'images_mean_orientation_{location}'].append(mean_orientation)
                image_features[f'images_std_orientation_{location}'].append(std_orientation)
            if color_detection:
                #Color in the image
                image_features[f'images_mean_red_{location}'].append(mean_red)
                image_features[f'images_mean_green_{location}'].append(mean_green)
                image_features[f'images_mean_blue_{location}'].append(mean_blue)
                image_features[f'images_meidan_red_{location}'].append(meidan_red)
                image_features[f'images_meidan_green_{location}'].append(meidan_green)
                image_features[f'images_meidan_blue_{location}'].append(meidan_blue)
                image_features[f'images_std_red_{location}'].append(std_red)
                image_features[f'images_std_green_{location}'].append(std_green)
                image_features[f'images_std_blue_{location}'].append(std_blue)              
                image_features[f'images_percentile25_red_{location}'].append(percentile25_red)
                image_features[f'images_percentile75_red_{location}'].append(percentile75_red)
                image_features[f'images_percentile25_green_{location}'].append(percentile25_green)
                image_features[f'images_percentile75_green_{location}'].append(percentile75_green)
                image_features[f'images_percentile25_blue_{location}'].append(percentile25_blue)
                image_features[f'images_percentile75_blue_{location}'].append(percentile75_blue)
            if texture_detection:
                #Texture in the image
                image_features[f'images_contrast_{location}'].append(contrast)
                image_features[f'images_dissimilarity_{location}'].append(dissimilarity)
                image_features[f'images_homogeneity_{location}'].append(homogeneity)
                image_features[f'images_energy_{location}'].append(energy)
                image_features[f'images_correlation_{location}'].append(correlation)
                image_features[f'images_ASM_{location}'].append(ASM)

        #Write the information in a .txt file, at first save everything but later save only feature that help classification. 
        for location in categories: 
            if edge_detection: 
                #Edge detection 
                first = image_features[f'images_num_features_{location}']
                second = image_features[f'images_mean_lengths_{location}']
                third = image_features[f'images_total_length_{location}']
                fourth = image_features[f'images_total_std_length_{location}']
                fifth = image_features[f'images_mean_orientation_{location}']
                six = image_features[f'images_std_orientation_{location}']

                t1 = image_features[f'images_contrast_{location}']
                t3 = image_features[f'images_homogeneity_{location}']


                test_name = 'prob_000'
                destination_folder = f'final_data/{test_name}'
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                file_path = os.path.join(destination_folder, f'{location}.txt')

                with open(file_path, 'w') as file:
                    for i in range(len(first)):
                        file.write(f'{first[i]:.5f} {t1[i]} {t3[i]}\n')
                        #file.write(f'{first[i]:.5f} {second[i]:.5f} {third[i]:.5f} {fourth[i]:.5f} {fifth[i]:.5f} {six[i]:.5f}\n')

            if color_detection: 
                #Color in the image
                c1 = image_features[f'images_mean_red_{location}']
                c2 = image_features[f'images_mean_green_{location}']
                c3 = image_features[f'images_mean_blue_{location}']
                c4 = image_features[f'images_meidan_red_{location}']
                c5 = image_features[f'images_meidan_green_{location}']
                c6 = image_features[f'images_meidan_blue_{location}']
                c7 = image_features[f'images_std_red_{location}']
                c8 = image_features[f'images_std_green_{location}']
                c9 = image_features[f'images_std_blue_{location}']
                c10 = image_features[f'images_percentile25_red_{location}']
                c11 = image_features[f'images_percentile75_red_{location}']
                c12 = image_features[f'images_percentile25_green_{location}']
                c13 = image_features[f'images_percentile75_green_{location}']
                c14 = image_features[f'images_percentile25_blue_{location}']
                c15 = image_features[f'images_percentile75_blue_{location}']

                test_name = f'prob_004_color_detection_{color_name}'
                destination_folder = f'/home/jean-sebastien/Documents/s7/APP2/s7-app2/data2/{test_name}'
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                file_path = os.path.join(destination_folder, f'{location}.txt')

                with open(file_path, 'w') as file:
                    for i in range(len(c4)):
                        #file.write(f'{c4[i]:.5f} {c13[i]:.5f}\n')
                        file.write(f'{c7[i]:.5f}\n')
                        #file.write(f'{c1[i]:.5f} {c2[i]:.5f} {c3[i]:.5f} {c4[i]:.5f} {c5[i]:.5f} {c6[i]:.5f} {c7[i]:.5f} {c8[i]:.5f} {c9[i]:.5f} {c10[i]:.5f} {c11[i]:.5f} {c12[i]:.5f} {c13[i]:.5f} {c14[i]:.5f} {c15[i]:.5f}\n')

            if texture_detection:
                #Texture in the image
                t1 = image_features[f'images_contrast_{location}']
                t2 = image_features[f'images_dissimilarity_{location}']
                t3 = image_features[f'images_homogeneity_{location}']
                t4 = image_features[f'images_energy_{location}']
                t5 = image_features[f'images_correlation_{location}']
                t6 = image_features[f'images_ASM_{location}']

                test_name = 'prob_000_texture_detection'
                destination_folder = f'/final_data/{test_name}'
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                file_path = os.path.join(destination_folder, f'{location}.txt')

                with open(file_path, 'w') as file:
                    for i in range(len(t3)):
                        pass
                        #file.write(f'{t1[i]} {t3[i]}\n')
                        #file.write(f'{t1[i]} {t2[i]} {t3[i]} {t4[i]} {t5[i]} {t6[i]}\n')

    def generateRepresentation(self, input_data=None, label_test=None, data_processing=False, analyse_data=False, deocrelate_data=False, test_set=False):
        if data_processing:
            #Extract features from images
            self.get_feature_extraction(input_data, label_test)
            print('Data processing has benn executed successusfully. Go analyse the data now...')
        
        if analyse_data: 
            #Produce a ClassificationData object usable by the classifiers
            self.data3classes = ClassificationData(problematique=True)
            print('Do it for other variables...')
            print('If every variables done, do PCA.')

            #Display 3D graph of classes before PCA
            an.view3D(self.data3classes.dataLists_norm, self.data3classes.labelsLists, 'Data before PCA')

        if deocrelate_data:   
            pca3 = PCA(n_components=3)
            pca3.fit(self.data3classes.data1array)
            data3D = pca3.transform(self.data3classes.data1array)

            #Calculate Silhouette score
            silhouette_avg = silhouette_score(data3D, self.data3classes.labels1array.flatten().astype(int))
            print(f"Le score de silhouette moyen est : {silhouette_avg}")
            #To allow data visualisation with machine learning algorithms
            varianceexplained = pca3.explained_variance_ratio_
            print(f"Variance expliquée : {varianceexplained}")
            self.extent = an.Extent(ptList=data3D) 
            
            variable_number = 3
            tailles = [360, 328, 292]
            donnees_projetees_par_classe = []
            start_idx = 0

            for taille in tailles:
                segment = data3D[start_idx:start_idx + taille]
                donnees_projetees_par_classe.append(segment)
                start_idx += taille

            #Display the transformed input data after PCA has been applied. 
            an.view3D(donnees_projetees_par_classe, self.data3classes.labelsLists, 'After PCA')
            an.calcModeleGaussien(data3D, '\nPCA 3d')

            plt.show()

        if test_set:
            #Create a traain and test set. 90% train and valid and 10% test set
            self.training_data = []
            self.test_data = []
            self.training_target = []
            self.test_target = []

            for i in range(len(donnees_projetees_par_classe)):
                temp_training_data, temp_test_data, temp_training_target, temp_test_target = train_test_split(donnees_projetees_par_classe[i], self.data3classes.labelsLists[i], test_size=0.1, random_state=82)
                self.training_data.append(temp_training_data)
                self.test_data.append(temp_test_data)
                self.training_target.append(temp_training_target)
                self.test_target.append(temp_test_target)

            self.training_data = np.concatenate(self.training_data)
            self.test_data = np.concatenate(self.test_data)
            self.training_target = np.concatenate(self.training_target)
            self.test_target = np.concatenate(self.test_target)

    def images_display(self, indexes):
        """
        fonction pour afficher les images correspondant aux indices
        indexes: indices de la liste d'image (int ou list of int)
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig2 = plt.figure()
        ax2 = fig2.subplots(len(indexes), 1)
        for i in range(len(indexes)):
            if self.all_images_loaded:
                im = self.images[indexes[i]]
            else:
                im = skiio.imread(self.image_folder + os.sep + self.image_list[indexes[i]])
            ax2[i].imshow(im)

    def view_histogrammes(self, indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 3)

        espacement_vertical = 0.5
        espacement_horizontal = 0.5
        fig.subplots_adjust(hspace=espacement_vertical, wspace=espacement_horizontal)

        for image_counter in range(len(indexes)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[indexes[image_counter]]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[indexes[image_counter]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  
            imageHSV = skic.rgb2hsv(imageRGB)  

            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins) # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)
            histtvaluesLab = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[0, start:end], s=3, c='red')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[1, start:end], s=3, c='green')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[2, start:end], s=3, c='blue')
            ax[image_counter, 0].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 0].set_title(f'histogramme RGB de {image_name}')

            # 2e histogramme
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[0, start:end], s=3, c='red')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[1, start:end], s=3, c='green')
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[2, start:end], s=3, c='blue')
            ax[image_counter, 1].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 1].set_title(f'histogramme RGB de {image_name}')

            # 3e histogramme
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[0, start:end], s=3, c='red')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[1, start:end], s=3, c='green')
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[2, start:end], s=3, c='blue')
            ax[image_counter, 2].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 2].set_title(f'histogramme RGB de {image_name}')

            print(f'The image type is: {image_name}')

        print('Les histogrammes sont prets a etre visualises')

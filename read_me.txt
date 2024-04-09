- Le script main se retrouve dans le fichier problematique.py. 

- Installer requirement.txt

- Dans le script problématique.py il y a des flags qui sont utilises pour gérer ce que le code exécute dans la boucle principale. 
	- Le flag data_processing est utilisé pour réaliser l'extraction des features sur les images. Ce fichier va aller exécuter le module 
	  get_feature_extraction qui va permettre d'extraire les caractéristiques souhaité par l'utilisateur. Ces caractéristiques peuvent être le 
	  edge détection, les caractéristiques des couleurs ou les textures des images. 
	- Il y a d'autres flags dans ce scipts situés dans le module get_feature extraction pour l'extraction de ces caractéristiques qui doivent être mise
	  a True afin d'extraire les caractéristiques. Aussi le il y a d'autres flags si on utilise l'extraction des couleurs pour savoir si on souhaite avoir
	  une image en format RGB, Lab et HSV. 
	- Il est important de mettre des paths valide pour les fichiers de sorties de feature extraction qui sont: 
		destination_folder = f'/home/jean-sebastien/Documents/s7/APP2/s7-app2-Js/final_data/{test_name}'
		destination_folder = f'/home/jean-sebastien/Documents/s7/APP2/s7-app2/data2/{test_name}'
		destination_folder = f'/home/jean-sebastien/Documents/s7/APP2/s7-app2/final_data/{test_name}'
	  Ces paths sont localisés à la fin du même module. 

- Une fois les caractéristiques extraite, mettre le flag du fichier main à True. Il s'agit des Flags de analyse_data et mettre les flags à false pour data_processing. 
- Ensuite ajouter le path des fichiers nouvellement extrait dans classificationData.py Il est important d'indiquer les bons noms de paths des données à analyser. 
		folder_name = 'prob_000'
 		self.dataLists.append(np.loadtxt('final_data'+os.sep+f'{folder_name}'+os.sep+'coast.txt'))
                self.dataLists.append(np.loadtxt('final_data'+os.sep+f'{folder_name}'+os.sep+'forest.txt'))
                self.dataLists.append(np.loadtxt('final_data'+os.sep+f'{folder_name}'+os.sep+'street.txt'))

- Une fois ces modifications effectués, aller dans le fichier main et lancer le code. Le code va aller effectuer le chargement des caractéristiques extraites pour les trois
  classes comme le laboratoire le faisait, il va normaliser les données et calculer les paramètres de moyenne, de la matrice de covariance et les valeurs et vecteurs propres. 
  Ce script va aussi calculer ces paramètres pour chacune des classes et aussi pour l'ensemble des données, les résultats des matrices de covariances vont être stockés au path suivant: 

	output_file = f'/home/jean-sebastien/Documents/s7/APP2/Problematique/s7-app2/Analyse_image2/{folder}'
	Dans le script classification data, merci de bien renommer le path si on souhaite analyser les matrice de covariances. Vers ligne 114. 

- Le reste du code est totalement automatisé et va par la suite effectuer les étapes suivantes dans le script ImageCollection.py
	1- Analyser les données initiales sans l'application de la PCA
	2- Appliquer la PCA sur les données d'origines
	3- Analyser la PCA
	4- Séparation des données en entraînement et en validation 

- Ensuite le code effecture les algorithmes d'apprentissage machine. 
	1- Le réseau de neurones
	2- Le ppv et le k-moyen
	3- Le réseau Bayésien 


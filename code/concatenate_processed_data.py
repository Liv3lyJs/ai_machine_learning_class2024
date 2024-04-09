"""
Author: Jean-Sebastien Giroux
Date: 03/08/2024
Description: Concatene two text file into a new one to allow the user to
             combine feature extraction techniques when working with the
             machine learning algorithms.
             For exemple, work with edgedetection, colordetection and texteure
             detection all at the same time
"""
import os

test_number = 'End06'

filepath1 = f'/home/jean-sebastien/Documents/s7/APP2/Problematique/s7-app2/data2/prob_000_edge_detection'
filepath2 = f'/home/jean-sebastien/Documents/s7/APP2/Problematique/s7-app2/data2/prob_004_color_detection_Lab'
#filepath3 = f'/home/jean-sebastien/Documents/s7/APP2/Problematique/s7-app2/data2/prob_006_texture_detection'
# filepath4 = f'/home/jean-sebastien/Documents/s7/APP2/Problematique/s7-app2/data/prob_Final01_color_detection_Lab'
# filepath5 = f'/home/jean-sebastien/Documents/s7/APP2/Problematique/s7-app2/data/prob_Final01_color_detection_HSV'

filepaths = [filepath1, filepath2] #filepath2 filepath5
output_file = f'/home/jean-sebastien/Documents/s7/APP2/Problematique/s7-app2/data2/prob_{test_number}_mix_variables'
file_name = ['coast.txt', 'forest.txt', 'street.txt']


if not os.path.exists(output_file):
    os.makedirs(output_file)

for name in file_name:
    print(f'concatenationg file: {name}')
    file1 = os.path.join(filepath1, name)
    file2 = os.path.join(filepath2, name)

    output = os.path.join(output_file, name)
    with open(file1, 'r') as f1, \
         open(file2, 'r') as f2, \
         open(output, 'w') as out:


        for line1, line2, line3 in zip(f1, f2): #f3, f4 line3, f2, f4, line2, line4, 
            element1 = line1.strip().split()
            element2 = line2.strip().split()

            merged_line = element1 +  element2 #element2 + + element5

            merged_line_str = ' '.join(merged_line) + '\n'

            out.write(merged_line_str)

print('The contenation of the files have work as expected.')
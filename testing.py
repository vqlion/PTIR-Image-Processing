import os
import argparse
import sys

from test_descriptors_distance import descriptors_distance
from test_keypoints_distance import keypoints_distance
from score_sorter import score_sorter

parser = argparse.ArgumentParser(description='Launch the tests')

parser.add_argument(
    '--path', type=str, required=True,
    help='path of the directory containing the npz files'
)

parser.add_argument(
    '--main_file', type=str, default="1.ppm.npz",
    help='name of the main file (compared to all other npz files in directory)'
)

args = parser.parse_args()
main_file_name = args.main_file
main_file = ""
path = args.path
extension = '.npz'
number_of_points = 127

for root, dirs, files in os.walk(path):
    for file in files:
        if file == main_file_name:
            main_file = file
            print(main_file)

    if not main_file:
        print("No main file was found in this directory.")
        sys.exit()
    
    absolute_main_file_name = main_file[0]
    main_file_path = os.path.join(root, main_file_name)
    score_sorter(main_file_path, number_of_points)

    for file in files:
        if file == main_file or os.path.splitext(file)[-1].lower() != extension:
            continue

        absolute_file_name = file[0]
        matrix_file = f"H_{absolute_main_file_name}_{absolute_file_name}"

        if not(matrix_file in files):
            print("No matrix file was found for", file)
            continue
        
        file_path = os.path.join(root, file)
        matrix_file_path = os.path.join(root, matrix_file)

        print(main_file, file, matrix_file)

        score_sorter(file_path, number_of_points)

        print(keypoints_distance(main_file_path, file_path, matrix_file_path, 1))
        print(descriptors_distance(main_file_path, file_path, matrix_file_path, 1, 2))


        
        

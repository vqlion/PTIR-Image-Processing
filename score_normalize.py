import numpy as np
import argparse

# parser = argparse.ArgumentParser(description='Sort by scores')

# parser.add_argument(
#     '--path', type=str, required=True,
#     help='path to a file containing the keypoints and descriptors of the first image'
# )

# parser.add_argument(
#     '--nb', type=str, required=True,
#     help='number of points to keep'
# )

# args = parser.parse_args()

def score_sorter(path, nb):
    
    # Importation
    keypoints = np.load(path)['keypoints']
    scores = np.load(path)['scores']
    descriptors = np.load(path)['descriptors']
    number_of_points = int(nb)
    
    # Convert from colmap format to pixels and normalize descriptors 
    keypoints = np.round(keypoints).astype(int)
    descriptors /= np.mean(descriptors)

    # On range les keypoints et descriptors par ordre décroissant de scores
    indices = np.argsort(scores)[::-1]
    sorted_keypoints = [keypoints[i].tolist() for i in indices]
    sorted_descriptors = [descriptors[i].tolist() for i in indices]
    scores = sorted(scores, reverse=True)

    with open(path, 'wb') as output_file:
        np.savez(
            output_file,
            keypoints=sorted_keypoints[:number_of_points],
            scores=scores[:number_of_points],
            descriptors=sorted_descriptors[:number_of_points]
        )

# score_sorter(args.path, args.nb)

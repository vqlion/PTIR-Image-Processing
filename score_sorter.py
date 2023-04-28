import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Sort by scores')

parser.add_argument(
    '--path', type=str, required=True,
    help='path to a file containing the keypoints and descriptors of the first image'
)

parser.add_argument(
    '--nb', type=str, required=True,
    help='number of points to keep'
)

args = parser.parse_args()

keypoints = np.load(args.path)['keypoints']
scores = np.load(args.path)['scores']
descriptors = np.load(args.path)['descriptors']

number_of_points = int(args.nb)

indices = np.argsort(scores)[::-1]
sorted_keypoints = [keypoints[i].tolist() for i in indices]
sorted_descriptors = [descriptors[i].tolist() for i in indices]
scores = sorted(scores, reverse=True)

with open(args.path, 'wb') as output_file:
    np.savez(
        output_file,
        keypoints=sorted_keypoints[:number_of_points],
        scores=scores[:number_of_points],
        descriptors=sorted_descriptors[:number_of_points]
    )

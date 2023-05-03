import numpy as np
import argparse
import glob

# parser = argparse.ArgumentParser(description='Keypoints scores computing script')

# parser.add_argument(
#     '--folder', type=str, required=False,
#     help='folder containing the files keypoints and descriptors'
# )

# parser.add_argument(
#     '--threshold', type=int, default=1,
#     help='the maximum distance between two descriptors to be considered as a pair'
# )

# args = parser.parse_args()

def distance(a, b):
    sum = 0
    for i in range(len(a)):
        sum += abs(a[i] - b[i])
    return sum

def scores(folder_path, threshold):

    # Importation des .npz
    files_path = glob.glob(f"{folder_path}/*.npz")
    files_path.sort()
    pts = []
    descs = []
    for f in files_path:
        pts.append(np.load(f)['keypoints'])
        descs.append(np.load(f)['descriptors'])

    # Importation des matrices de transformation
    matrices_path = glob.glob(f"{folder_path}/H_1*")
    matrices_path.sort()
    matrices = []
    for f in matrices_path:
        matrices.append(np.loadtxt(f))

    scores = [0.0] * len(pts[0])
    for img2 in range(1, len(pts)):
        new_pts0 = []

        # On applique la matrice de transformation aux points de l'image 1
        for p in pts[0]:
            new_p = np.append(p, 1)
            new_pts0.append(np.matmul(new_p, matrices[img2 - 1]))

        # On ajoute 1 dans le score du point i si les conditions sont remplies
        for i in range(len(new_pts0)):
            stop = False
            j = 0
            while not stop and j < len(pts[img2]):
                if distance(new_pts0[i][:2], pts[img2][j]) <= threshold and distance(descs[0][i], descs[img2][j]) <= threshold:
                    scores[i] += 1
                    stop = True
                j += 1
        
    # Normalisation
    for i in range(len(scores)):
        scores[i] = round(scores[i] / (len(pts) - 1) * 100, 2)

    return(scores)

# print(scores(args.folder, args.threshold))

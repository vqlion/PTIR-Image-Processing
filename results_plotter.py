import numpy as np
import argparse
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', type=int, required=True,
    help='id of the dataset you want to display'
)

args = parser.parse_args()

json_files = []
with open('message5.json', 'r') as file:
    json = json.load(file)
    json_files.append(json)
    json_files.append(json)
    json_files.append(json)

# Titre du graphique
fig, axs = plt.subplots(3, len(json_files), figsize=(18, 10))
fig.suptitle(json_files[0][args.dataset]["image_folder"], fontsize=25)
plt.subplots_adjust(hspace=0.3)

# Pour le dataset choisi, on crée un graph pour chaque modèle et on trace une courbe dessus pour chacune des comparaisons des images
for model in range(len(json_files)):
    dataset = json_files[model][args.dataset]
    for j in range(len(dataset["images_tests"])):
        for i in range(3):
            axs[i][model].set_xlabel('Threshold')
            axs[i][model].grid(True)

        axs[0][model].set_xlim(0, 50)
        axs[0][model].set_ylim(0, 100)
        axs[0][model].set_xlabel('Threshold')
        axs[0][model].set_title(dataset["model"], fontsize=20)
        axs[0][model].plot(dataset["images_tests"][j]["keypoints_distances"])
        axs[0][model].set_ylabel('Keypoints success rate')

        axs[1][model].set_xlim(0, 50)
        axs[1][model].set_ylim(0, 100)
        axs[1][model].set_xlabel('Threshold')
        axs[1][model].plot(dataset["images_tests"][j]["descriptors_distances"])
        axs[1][model].set_ylabel('Descriptors success rate')

        axs[2][model].set_ylabel('Scores regressions')
        axs[2][model].set_xlabel('Keypoints')
        for k in np.arange(0, len(dataset["scores"]), 3):
            coefficients = np.polyfit(range(50), dataset["scores"][k], 1)
            regression = np.poly1d(coefficients)
            axs[2][model].plot(regression(range(50)), lw=1)
    plt.draw()
plt.show()



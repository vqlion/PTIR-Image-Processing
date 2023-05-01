import os
import argparse

def loop_through_dir(path, extension):
    tmp = ''
    for filename in os.listdir(path):
        filepath = os.path.join(filename, path)
        if os.path.isdir(filepath):
            loop_through_dir(filepath)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filepath)[-1].lower()
            if ext == extension:
                tmp += filepath + '\r\n'
                print(filepath)
    return tmp

    
parser = argparse.ArgumentParser(description='Get path to all images in directory in a txt file.')

parser.add_argument(
    '--path', type=str, required=True,
    help='path of the directory'
)

parser.add_argument(
    '--type', type=str, required=True,
    help='type of image files'
)

args = parser.parse_args()

directory = args.path
filetype = args.type

res = loop_through_dir(directory, filetype)

with open ('image_list.txt', 'w') as file:  
    file.write(res)  
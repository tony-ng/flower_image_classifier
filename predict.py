import argparse
import json
from utilities import *
from model_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("img_path", type = str, help = "path of the image")
parser.add_argument("checkpoint", type = str, help = "directory to load checkpoint")
parser.add_argument("--top_k", type = int, default = 5, help = "the number of most likely classes returned")
parser.add_argument("--category_names", type = str, default = "cat_to_name.json", help = "a mapping of categories to real names")
parser.add_argument("--gpu", action = "store_true", help = "use gpu for predict")

in_args = parser.parse_args()

cat_to_name = None
with open(in_args.category_names, 'r') as f:
    cat_to_name = json.load(f)

if cat_to_name == None:
    print('Fail to load category mapping file')
    exit()

model = load_checkpoint(in_args.checkpoint)

np_img = process_image(in_args.img_path)

probs, classes = predict(np_img, model, in_args.top_k, in_args.gpu)

for i in range(len(classes)):
    class_name = cat_to_name[classes[i]]
    print("{} with a probability of {:.4f}".format(class_name, probs[i]))

    

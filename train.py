import argparse
from utilities import *
from model_functions import *

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type = str, help = "directory of the training data")
parser.add_argument("--save_dir", type = str, default = "checkpoint.pth", help = "directory to save checkpoint")
parser.add_argument("--arch", type = str, default = "vgg16", help = "a pre-trained network used for training")
parser.add_argument("--learning_rate", type = float, default = 0.001, help = "directory of the training data")
parser.add_argument("--hidden_units", type = int, default = 8192, help = "The number of units on the hidden layer")
parser.add_argument("--epochs", type = int, default = 4, help = "number of repeated training")
parser.add_argument("--gpu", action = "store_true", help = "use gpu for training")

in_args = parser.parse_args()

train_dataset, valid_dataset = load_datasets(in_args.data_dir)
train_dataloader = load_dataloader(train_dataset, True)
valid_dataloader = load_dataloader(valid_dataset)

model = load_model(in_args.arch)
if model == None:
    exit()
    
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = create_classifier(in_args.arch, in_args.hidden_units)

model = train_model(model, train_dataloader, valid_dataloader, in_args.learning_rate, in_args.epochs, in_args.gpu)

save_checkpoint(model, in_args.save_dir, in_args.arch, in_args.hidden_units, train_dataset)






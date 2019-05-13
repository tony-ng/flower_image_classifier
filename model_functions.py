import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from workspace_utils import active_session

vgg_input_size = 25088
densenet_input_size = 2208

def load_model(arch):
    """
    Load the pre-trained model
    Parameters:
    arch - the model architecture, either vgg or densenet
    Returns:
    model - the pre-trained model
    """
    model = None
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
    else:
        print("Sorry! The model is not supported")
        
    return model
        
class Detect_Flower_Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        """
        Define the model structure
        Parameters:
        input_size - the size of input layer
        output_size - the size of output layer
        hidden_layers - the size of each hidden layer
        drop_p - the dropout rate of each weight
        """
        super().__init__()
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        hidden_layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in hidden_layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
            
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    
def create_classifier(arch, hidden_units):
    """
    Create a new network for detecting flower
    Parameters:
    arch - the model architecture, either vgg or densenet
    hidden_units - the number of units on the hidden layer
    Returns:
    classifier - the network newly created
    """
    classifier = None
    input_size = None
    if arch == "vgg16":
        input_size = vgg_input_size
    elif arch == "densenet161":
        input_size = densenet_input_size
        
    classifier = Detect_Flower_Network(input_size, 102, [hidden_units], drop_p=0.5)
        
    return classifier

def train_model(model, train_dataloader, valid_dataloader, learning_rate, epochs, gpu):
    """
    Train the model and the training loss, validation loss, and validation accuracy are printed out
    Parameters:
    model - the model to be trained
    train_dataloader - the dataloader for training dataset
    valid_dataloader - the dataloader for validation dataset
    learning_rate - the learning rate of the training
    epochs - the number of repeated time for training
    gpu - use gpu if True; otherwise use cpu
    Returns:
    model - the trained model
    """
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print_every = 20
    steps = 0

    with active_session():
        if gpu:
            model.to('cuda')
            
        print("Start training")
        for e in range(epochs):
            model.train()
            running_loss = 0
            for inputs, labels in iter(train_dataloader):
                steps += 1

                if gpu:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()

                    with torch.no_grad():
                        test_loss, accuracy = validate(model, criterion, valid_dataloader, gpu)

                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.4f}.. ".format(running_loss/print_every),
                          "Validate Loss: {:.4f}.. ".format(test_loss/len(valid_dataloader)),
                          "Validate Accuracy: {:.4f}.. ".format(accuracy/len(valid_dataloader)))

                    running_loss = 0
                    model.train()
                    
        print("Finish training")
        return model                    
                    
def validate(model, criterion, valid_dataloader, gpu):
    """
    Validate the loss and accuracy of the model
    Parameters:
    model - the model to be validated
    criterion - the function to calculate the loss
    valid_dataloader - the validation dataloader
    gpu - use gpu if True; otherwise use cpu
    Returns:
    test_loss - the total loss of the model for all validation batches
    accuracy - the total accuracy in proportion for all validation batches
    """
    test_loss = 0
    accuracy = 0
    for images, labels in iter(valid_dataloader):
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy

def save_checkpoint(model, save_dir, arch, hidden_units, train_dataset):
    """
    Save the checkpoint
    Parameters:
    model - the model to be saved
    save_dir - the directory to save the checkpoint, e.g. 'checkpoints/'
    arch - the model architecture
    hidden_units - the size of each hidden layer
    train_dataset - the dataset for training
    """
    input_size = None
    if arch == "vgg16":
        input_size = vgg_input_size
    elif arch == "densenet161":
        input_size = densenet_input_size
        
    checkpoint = {"arch": arch,
              "input_size": input_size,
              "hidden": [hidden_units],
              "output_size": 102,
              "state_dict": model.classifier.state_dict(),
              "class_to_idx": train_dataset.class_to_idx}

    torch.save(checkpoint, save_dir)
    print("Finish saving checkpoint")
    
def load_checkpoint(checkpoint_path):
    """
    Load the checkpoint
    Parameters:
    checkpoint_path - the path of the checkpoint file
    Returns:
    model - the model created from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    
    model = load_model(checkpoint['arch'])
    lc_model = Detect_Flower_Network(checkpoint['input_size'],
                                     checkpoint['output_size'],
                                     checkpoint['hidden'])
    lc_model.load_state_dict(checkpoint['state_dict'])
    
    model.classifier = lc_model
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def predict(np_img, model, topk, gpu):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    Parameters:
    np_img - the image in numpy array
    model - the trained deep learning model
    topk - the number of most likely classes returned
    gpu - use gpu if True; otherwise use cpu
    Returns:
    probs - list of probabilities of classes returned
    classes - list of classes returned
    '''
    img = torch.from_numpy(np_img)
    
    with active_session():
        model.eval()
        if gpu:
            model, img = model.to('cuda'), img.to('cuda')

        with torch.no_grad():
            img.unsqueeze_(0)
            img = img.float()
            output = model.forward(img)
            ps = torch.exp(output)
            
        probs, index = ps.topk(topk)
        if gpu:
            probs = probs.cpu()
            index = index.cpu()
            
        classes = list()
        index_list = index.numpy()[0]
        for i in range(topk):
            idx = index_list[i]
            for img_class, img_idx in model.class_to_idx.items():                                
                if idx == img_idx:
                    classes.append(img_class)
                    break
            
        return probs.numpy()[0], classes
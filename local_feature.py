from mlxtend.frequent_patterns import fpgrowth
import pandas as pd
import torch 
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


# Load an image using PIL
image_path = r'C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg'  # Replace with your image path

# Load a pretrained VGG-16 model
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

# Set up image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and transform an image
image = Image.open(image_path)
image = transform(image).unsqueeze(0)

# Pass the image through the VGG-16 model
with torch.no_grad():
    features = vgg16.features(image)

# Obtain the salient object by performing a mask operation
features = features.squeeze(0)
A = features.sum(dim=0)
threshold = A.mean()
mask = (A > threshold).float()
mask = mask.unsqueeze(0).unsqueeze(0)
salient_object = features * mask
salient_object = salient_object.squeeze(0)
# Convert feature maps and activated positions into transactions and items
transactions = []
for i in range(salient_object.shape[0]):
    feature_map = salient_object[i]
    activated_positions = (feature_map > 0).nonzero(as_tuple=True)
    items = [f'({x},{y})' for x, y in zip(*activated_positions)]
    transactions.append(items)

# Mine frequent patterns using FPM
minsupp = 2 # minimum support threshold for FPM
I = sorted(set(item for transaction in transactions for item in transaction))
df = pd.DataFrame([[int(item in transaction) for item in I] for transaction in transactions], columns=I)
frequent_itemsets = fpgrowth(df, min_support=minsupp/len(transactions), use_colnames=True)

# Extract the local feature fL from the frequent patterns
patterns = torch.zeros_like(salient_object)
for itemset in frequent_itemsets['itemsets']:
    for item in itemset:
        x, y = map(int, item.strip('()').split(','))
        patterns[:, x, y] = 1

fL_max_pooling = nn.functional.adaptive_max_pool2d(patterns, (1, 1)).view(-1)
fL_avg_pooling = nn.functional.adaptive_avg_pool2d(patterns, (1, 1)).view(-1)
fL = torch.cat((fL_max_pooling, fL_avg_pooling), dim=0)




class LocalFeature:
    def __init__(self, vgg16):
        self.vgg16 = vgg16

    def extract(self, image):
        with torch.no_grad():
            features = self.vgg16.features(image)

        # Obtain the salient object by performing a mask operation
        features = features.squeeze(0)
        A = features.sum(dim=0)
        threshold = A.mean()
        mask = (A > threshold).float()
        mask = mask.unsqueeze(0).unsqueeze(0)
        salient_object = features * mask
        salient_object = salient_object.squeeze(0)

        # Convert feature maps and activated positions into transactions and items
        transactions = []
        for i in range(salient_object.shape[0]):
            feature_map = salient_object[i]
            activated_positions = (feature_map > 0).nonzero(as_tuple=True)
            items = [f'({x},{y})' for x, y in zip(*activated_positions)]
            transactions.append(items)

        # Mine frequent patterns using FPM
        minsupp = 2 # minimum support threshold for FPM
        I = sorted(set(item for transaction in transactions for item in transaction))
        df = pd.DataFrame([[int(item in transaction) for item in I] for transaction in transactions], columns=I)
        frequent_itemsets = fpgrowth(df, min_support=minsupp/len(transactions), use_colnames=True)

        # Extract the local feature fL from the frequent patterns
        patterns = torch.zeros_like(salient_object)
        for itemset in frequent_itemsets['itemsets']:
            for item in itemset:
                x, y = map(int, item.strip('()').split(','))
                patterns[:, x, y] = 1

        fL_max_pooling = nn.functional.adaptive_max_pool2d(patterns, (1, 1)).view(-1)
        fL_avg_pooling = nn.functional.adaptive_avg_pool2d(patterns, (1, 1)).view(-1)
        fL = torch.cat((fL_max_pooling, fL_avg_pooling), dim=0)

        return fL

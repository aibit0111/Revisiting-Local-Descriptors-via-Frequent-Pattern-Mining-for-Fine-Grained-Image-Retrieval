import os
from PIL import Image
from torchvision import transforms
import torch
from torchvision.models import vgg16
from global_feature import GlobalFeature
from local_feature import LocalFeature

# Define paths to query and database image folders
query_image_folder = r"C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\code\query_image"
database_image_folder = r"C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\code\database_image"


vgg16_model = vgg16(pretrained=True)


# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load query images
query_images = []
for filename in os.listdir(query_image_folder):
    image = Image.open(os.path.join(query_image_folder, filename))
    image = transform(image).unsqueeze(0)
    query_images.append(image)
query_images = torch.stack(query_images)

# Load database images
database_images = []
for filename in os.listdir(database_image_folder):
    image = Image.open(os.path.join(database_image_folder, filename))
    image = transform(image)
    database_images.append(image)
database_images = torch.stack(database_images)

# Define CNN model
model = ... # define your CNN model here

# Extract global and local features for query images
query_global_features = []
query_local_features = []

for image in query_images:
    # Extract global and local features for the current image
    global_extractor = GlobalFeature(vgg16_model)
    global_feature = global_extractor.extract(image)
    local_extractor = LocalFeature(vgg16_model)
    local_feature = global_extractor.extract(image)
    query_global_features.append(global_feature)
    query_local_features.append(local_feature)
query_global_features = torch.stack(query_global_features)
query_local_features = torch.stack(query_local_features)

# Extract global and local features for database images
database_global_features = []
database_local_features = []

for image in database_images:
    # Extract global and local features for the current image
    global_extractor = GlobalFeature(vgg16_model)
    global_feature = global_extractor.extract(image)
    local_extractor = LocalFeature(vgg16_model)
    local_feature = global_extractor.extract(image)
    query_global_features.append(global_feature)
    query_local_features.append(local_feature)
    database_global_features.append(global_feature)
    database_local_features.append(local_feature)
database_global_features = torch.stack(database_global_features)
database_local_features = torch.stack(database_local_features)

# Aggregate global and local features for query and database images
# define alpha here
alpha = 0.1 
query_features = query_global_features + alpha * query_local_features
database_features = database_global_features + alpha * database_local_features

# Calculate similarity scores for all possible combinations of query and database images
similarity_scores = torch.matmul(query_features, database_features.T)
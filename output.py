import os
from PIL import Image
from torchvision import transforms
import torch
from torchvision.models import vgg16
from global_feature import GlobalFeature
from local_feature import LocalFeature

# Define paths to query and database image folders
query_image_folder = r"C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\code\query_image"
relevant_image_folder =  r"C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\code\database_image\001.Black_footed_Albatross"
database_image_folder = r"C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\CUB_200_2011\images"

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
relevant_images = []
database_images = []
database_filenames = []




# Define image extensions
image_extensions = ['jpg', 'png', 'jpeg']

# Iterate over all subdirectories and files in the root directory
for subdir, dirs, files in os.walk(database_image_folder):
    for filename in files:
        # Check if the file is an image
        if filename.split('.')[-1].lower() in image_extensions:
            # Append the filename to the filenames list
            database_filenames.append(filename)
            # Open and transform the image, then append it to the images list
            image = Image.open(os.path.join(subdir, filename))
            image = transform(image).unsqueeze(0)
            database_images.append(image)

# Convert the list of images into a torch tensor
database_images = torch.stack(database_images)




def get_image_names_from_folder(root_folder):
    image_extensions = ['jpg', 'png', 'gif', 'jpeg']
    image_names = []

    for ext in image_extensions:
        image_paths = glob.glob(f'{root_folder}/**/*.{ext}', recursive=True)
        for path in image_paths:
            image_name = os.path.basename(path)
            image_names.append(image_name)

    return image_names

relevant_images = get_image_names_from_folder(relevant_image_folder)


# for filename in os.listdir(database_image_folder):
#     database_filenames.append(filename)  
#     image = Image.open(os.path.join(database_image_folder, filename))
#     image = transform(image)
#     database_images.append(image)
# database_images = torch.stack(database_images)

# Extract global and local features for query images
query_global_features = []
query_local_features = []

for image in query_images:
    # Extract global and local features for the current image
    global_extractor = GlobalFeature(vgg16_model)
    global_feature = global_extractor.extract(image)
    local_extractor = LocalFeature(vgg16_model)
    local_feature = local_extractor.extract(image)
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
    local_feature = local_extractor.extract(image)
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


k = 4
top_k_indices = similarity_scores.topk(k=k, dim=1).indices.tolist()
top_k_results = [[database_filenames[i] for i in indices] for indices in top_k_indices]


import os
import glob

def get_image_names_from_folder(root_folder):
    image_extensions = ['jpg', 'png', 'gif', 'jpeg']
    image_names = []

    for ext in image_extensions:
        image_paths = glob.glob(f'{root_folder}/**/*.{ext}', recursive=True)
        for path in image_paths:
            image_name = os.path.basename(path)
            image_names.append(image_name)

    return image_names

root_folder = r'C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\CUB_200_2011\images'

image_names = get_image_names_from_folder(root_folder)


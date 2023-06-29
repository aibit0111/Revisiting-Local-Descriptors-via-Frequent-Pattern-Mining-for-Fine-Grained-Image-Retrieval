import os
from PIL import Image
from torchvision import transforms
import torch
from torchvision.models import vgg16
from global_feature import GlobalFeature
from local_feature import LocalFeature
import glob
from similarity import similarity_score
from evaluation import average_precision, mean_average_precision
import random

# Define paths to query and database image folders
database_image_folder = r"C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\code\database_image"

vgg16_model = vgg16(pretrained=True)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda image: image.convert('RGB') if image.mode != 'RGB' else image),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

alpha = 0.1 # hyperparamter 
k = 5 # top K result
no_random_images = 3



def load_and_transform_images(database_image_folder, transform):
    # Define image extensions
    image_extensions = ['jpg', 'png', 'jpeg']

    database_filenames = []
    database_images = []
    database_file_path = []

    # Iterate over all subdirectories and files in the root directory
    for subdir, dirs, files in os.walk(database_image_folder):
        for filename in files:
            # Check if the file is an image
            if filename.split('.')[-1].lower() in image_extensions:
                # Append the filename to the filenames list
                database_filenames.append(filename)
                # Open and transform the image, then append it to the images list
                file_path = os.path.join(subdir, filename)
                database_file_path.append(file_path)
                with Image.open(os.path.join(subdir, filename)) as image:
                    try:
                        tensor_image = transform(image)
                        database_images.append(tensor_image)
                    except Exception as e:
                        print(f"Error transforming image {filename}: {str(e)}")

    # Convert the list of images into a torch tensor
    try:
        database_images = torch.stack(database_images)
    except Exception as e:
        print(f"Error in stacking images: {str(e)}")

    return database_filenames, database_images, database_file_path

database_filenames, database_images, database_file_path  = load_and_transform_images(database_image_folder, transform)


def get_random_image(number_of_random_images, database_filenames, database_images, database_file_path):
    
    all_average_precision = []

    for i in range(number_of_random_images):

        query_image_path = random.choice(database_file_path)
        relevant_image_folder = os.path.dirname(query_image_path)
        query_image = Image.open(query_image_path)
        query_image = transform(query_image)

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


        
        global_extractor = GlobalFeature(vgg16_model)
        query_global_feature = global_extractor.extract(query_image)
        local_extractor = LocalFeature(vgg16_model)
        query_local_feature = global_extractor.extract(query_image)

        query_features = query_global_feature + alpha * query_local_feature


        all_similarity_score = []

        for i,j in zip(database_filenames, database_images):
            file_name = i
            image = j
            global_feature = global_extractor.extract(image)
            local_feature = local_extractor.extract(image)
            database_feature =  global_feature + alpha * local_feature
            get_similarity = similarity_score(database_feature, query_features)
            all_similarity_score.append(get_similarity)


        
        def get_top_k_indices(input_list, k):
            return sorted(range(len(input_list)), key=lambda i: input_list[i], reverse=True)[:k]

        top_k_indices = get_top_k_indices(all_similarity_score, k)

        top_k_results = [database_filenames[indices]  for indices in top_k_indices]

        one_average_precision = average_precision(top_k_results , relevant_images )


        all_average_precision.append(one_average_precision)

    return all_average_precision

ap = get_random_image(no_random_images,database_filenames, database_images, database_file_path )

map_ = mean_average_precision(ap)

print(map_)










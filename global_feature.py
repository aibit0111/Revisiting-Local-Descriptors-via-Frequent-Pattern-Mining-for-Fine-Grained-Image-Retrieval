import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models, transforms
from PIL import Image
from skimage import measure


   

class GlobalFeature:
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

        # Retain the largest connected component using the flood fill algorithm
        labels = measure.label(mask.cpu().numpy())
        largest_label = labels.max()
        if largest_label > 0:
            largest_area = 0
            for i in range(1, largest_label + 1):
                area = (labels == i).sum()
                if area > largest_area:
                    largest_area = area
                    largest_component = i

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if labels[i, j] != largest_component:
                    mask[i, j] = 0

        mask = mask.unsqueeze(0).unsqueeze(0)
        salient_object = features * mask

        # Extract the global feature fG from the salient object
        fG_max_pooling = nn.functional.adaptive_max_pool2d(salient_object, (1, 1)).view(-1)
        fG_avg_pooling = nn.functional.adaptive_avg_pool2d(salient_object, (1, 1)).view(-1)
        fG = torch.cat((fG_max_pooling, fG_avg_pooling), dim=0)

        return fG

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

# Retain the largest connected component using the flood fill algorithm
labels = measure.label(mask.cpu().numpy())
largest_label = labels.max()
if largest_label > 0:
    largest_area = 0
    for i in range(1, largest_label + 1):
        area = (labels == i).sum()
        if area > largest_area:
            largest_area = area
            largest_component = i

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if labels[i, j] != largest_component:
            mask[i, j] = 0



mask = mask.unsqueeze(0).unsqueeze(0)
salient_object = features * mask

# Extract the global feature fG from the salient object
fG_max_pooling = nn.functional.adaptive_max_pool2d(salient_object, (1, 1)).view(-1)
fG_avg_pooling = nn.functional.adaptive_avg_pool2d(salient_object, (1, 1)).view(-1)
fG = torch.cat((fG_max_pooling, fG_avg_pooling), dim=0)





# """
# import matplotlib.pyplot as plt

# # Load an image using PIL
# image_path = r'C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg'  # Replace with your image path
# image = Image.open(image_path)

# # Generate a heat map of the global feature using the mask tensor
# heatmap = mask.squeeze().cpu().numpy()

# # Display the input image and the heat map
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(image)
# ax2.imshow(heatmap, cmap='hot')
# plt.show()


# import matplotlib.pyplot as plt

# # Load an image using PIL
# image_path = r'C:\Users\mohit\OneDrive\Desktop\remove_background\entropy\CUB_200_2011\images\001.Black_footed_Albatross\Black_Footed_Albatross_0001_796111.jpg'  # Replace with your image path
# image = Image.open(image_path)

# # Generate a heat map of the global feature using the summed features tensor A
# heatmap = A.cpu().numpy()

# # Display the input image and the heat map
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(image)
# ax2.imshow(heatmap, cmap='hot')
# plt.show()




# """


# """
# the authors visualize the global feature as a heatmap overlaid on the original image to show which parts of the image are most important for fine-grained image retrieval. This can be done by using the aggregation map A that is calculated during the mask operation to obtain the salient object.


# import matplotlib.pyplot as plt
# from PIL import Image

# # Load and display the original image
# image = Image.open(image_path)
# plt.imshow(image)

# # Calculate the aggregation map A
# A = features.sum(dim=0)

# # Resize A to match the size of the original image
# A = nn.functional.interpolate(A.unsqueeze(0).unsqueeze(0), size=image.size, mode='bilinear', align_corners=False)
# A = A.squeeze().numpy()

# # Display A as a heatmap overlaid on the original image
# plt.imshow(A, cmap='jet', alpha=0.5)
# plt.show()


# the code I provided for visualizing the global feature as a heatmap overlaid on the original image does not use the previously calculated fG. Instead, it uses the aggregation map A that is calculated during the mask operation to obtain the salient object.

# The reason for this is that fG is a 1D vector that represents the most important information extracted from the salient object in an image, while A is a 2D matrix that represents the importance of each pixel in the image. Since we want to visualize the global feature as a heatmap overlaid on the original image, it makes more sense to use A, which has the same spatial dimensions as the image, rather than fG, which does not have any spatial information.

# In other words, A shows us which parts of the image are most important for fine-grained image retrieval, while fG is a more compact representation of this information that can be used for tasks such as similarity measurement and retrieval.
# """


"""
 The retrieval procedure of the proposed method is as follows:

When given a query image, the global and local features of the query image are extracted.
Similarly, for each image in the database, its global and local features are extracted.
The aggregated feature F is calculated for both the query image and each image in the database by F = fG + αfL, where fG is the global feature, fL is the local feature and α is a weight balancing the effect of different features.
A similarity score is calculated between the query image and each image in the database using the formula S(FQ, FD) = FQ^T * FD / (|FQ| * |FD|), where FQ and FD denote the aggregated feature of the query image and the aggregated feature of a database image, respectively.
The candidate images are returned based on their similarity scores.

"""


"""
Q. How KNN is used?
A - the proposed method aims to search for the top K nearest neighbors 
    from the image database when given a query image. 
    The K nearest neighbors are determined based on their 
    similarity scores with the query image. 
    The similarity score between the query image and each image 
    in the database is calculated using the formula S(FQ, FD) = FQ^T * FD / (|FQ| * |FD|), 
    where FQ and FD denote the aggregated feature of the query image 
    and the aggregated feature of a database image respectively. 
    The top K images with the highest similarity scores are returned 
    as the K nearest neighbors of the query image.

"""
import torch
def similarity_score(feature1, feature2):
    """
    Calculates the similarity score between two aggregated feature tensors.
    
    Args:
        feature1 (torch.Tensor): First aggregated feature tensor.
        feature2 (torch.Tensor): Second aggregated feature tensor.

    Returns:
        float: Similarity score between the two aggregated feature tensors.
    """
    score = torch.dot(feature1, feature2) / (torch.norm(feature1) * torch.norm(feature2))
    return score.item()
"""
The performance of the 
proposed method was evaluated using the mean average precision (mAP) score. 
The mAP score is a commonly used evaluation metric in information retrieval 
that measures the quality of the ranked list of retrieved items.

The mAP score is calculated by first computing the average precision (AP) 
score for each query and then taking the mean of these AP scores over all queries. 
The AP score for a single query is calculated as the average of the precision 
values obtained for each relevant item in the ranked list of retrieved items. 
The precision at a given rank k is defined as the number of relevant items 
among the top k retrieved items divided by k.

In this paper, the mAP score was used to evaluate the performance of the 
proposed method on five fine-grained datasets. For each dataset, a ranked 
list of retrieved images was generated for each query image using the proposed 
method. The mAP score was then calculated based on these ranked lists to 
measure the quality of the retrieval results.
        
"""


"""
relevant_items and retrieved_items are both input arguments to the 
mean_average_precision function that I provided in my previous response.

relevant_items is a dictionary that maps each query to a set of relevant items. 
For each query, this set contains the items that are considered relevant to the query. 
For example, if we have two queries query1 and query2, and the set of relevant items 
for query1 is {item1, item2} and the set of relevant items for query2 is {item3, item4}, 
then relevant_items would be a dictionary like this: {query1: {item1, item2}, 
query2: {item3, item4}}.

retrieved_items is also a dictionary that maps each query to a ranked list of retrieved 
items. For each query, this list contains the items that are returned by the retrieval 
system in response to the query, sorted by their relevance to the query. For example, 
if we have two queries query1 and query2, and the ranked list of retrieved items for 
query1 is [item1, item5, item2] and the ranked list of retrieved items for query2 is 
[item4, item3], then retrieved_items would be a dictionary like this: 
{query1: [item1, item5, item2], query2: [item4, item3]}

""" 



def average_precision(retrieved_items, relevant_items):
    rel_count = 0
    precisions = []

    for i, item in enumerate(retrieved_items, start=1):
        if item in relevant_items:
            rel_count += 1
            precision_at_i = rel_count / i
            precisions.append(precision_at_i)

    if precisions:
        avg_precision = sum(precisions) / len(precisions)
    else:
        avg_precision = 0.0

    return avg_precision



def mean_average_precision(ap_scores):
    
    """
    Calculates the mean average precision (mAP) score for a set of queries.
    
    Args:
        queries (list): List of queries.
        relevant_items (dict): Dictionary mapping each query to a set of relevant items.
        retrieved_items (dict): Diconaryti mapping each query to a ranked list of retrieved items.

    Returns:
        float: Mean average precision (mAP) score for the set of queries.
    """

    map_score = sum(ap_scores) / len(ap_scores)
    return map_score


def calculate_anmrr(retrieved_items, relevant_items, max_rank=None):
    ranks = []
    for item in relevant_items:
        rank = retrieved_items.index(item) + 1  # Adding 1 because index is 0-based
        ranks.append(rank)
    
    if max_rank is None:
        max_rank = len(retrieved_items)
    
    mrr = sum([1.0/rank if rank <= max_rank else 1.0/max_rank for rank in ranks])/len(relevant_items)
    max_mrr = 1.0
    min_mrr = 1.0/max_rank
    
    anmrr = 1 - ((mrr - min_mrr) / (max_mrr - min_mrr))
    
    return anmrr

retrieved_items = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
relevant_items = ['a', 'c', 'e', 'g', 'i']

anmrr = calculate_anmrr(retrieved_items, relevant_items)
print(f"The ANMRR is: {anmrr}")


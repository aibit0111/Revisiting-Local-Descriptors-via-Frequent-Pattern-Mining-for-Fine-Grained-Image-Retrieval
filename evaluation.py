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



def average_precision(relevant_items, retrieved_items):
    """
    Calculates the average precision (AP) score for a single query.
    
    Args:
        relevant_items (set): Set of relevant items for the query.
        retrieved_items (list): Ranked list of retrieved items for the query.

    Returns:
        float: Average precision (AP) score for the query.
    """
    num_relevant = 0
    sum_precision = 0
    for i, item in enumerate(retrieved_items):
        if item in relevant_items:
            num_relevant += 1
            precision = num_relevant / (i + 1)
            sum_precision += precision
    ap = sum_precision / len(relevant_items)
    return ap



def mean_average_precision(queries, relevant_items, retrieved_items):
    """
    Calculates the mean average precision (mAP) score for a set of queries.
    
    Args:
        queries (list): List of queries.
        relevant_items (dict): Dictionary mapping each query to a set of relevant items.
        retrieved_items (dict): Diconaryti mapping each query to a ranked list of retrieved items.

    Returns:
        float: Mean average precision (mAP) score for the set of queries.
    """
    ap_scores = []
    for query in queries:
        ap = average_precision(relevant_items[query], retrieved_items[query])
        ap_scores.append(ap)
    map_score = sum(ap_scores) / len(ap_scores)
    return map_score

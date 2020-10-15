from scipy.stats import pearsonr
from collabfilter.utils import *


def nearest_neighbour(critic, metric='Manhattan'):
    """
    Returns a sorted list of the nearest neighbours of the given critic.
    Parameters:
        critic: str,
        metric: str; distance metric either 'Manhattan' or 'Euclidean'.
    """
    assert metric in ['Manhattan', 'Euclidean'],\
        "Unsupported metric. Please, use 'Manhattan' or 'Euclidean'"
    check_critic(critic)

    critics_list = critics()
    distances = []

    if metric == 'Manhattan':
        distances = [(d_manhattan(critic_, critic), critic_)
                     for critic_ in critics_list if critic_ != critic]
    elif metric == 'Euclidean':
        distances = [(d_euclidean(critic_, critic), critic_)
                     for critic_ in critics_list if critic_ != critic]

    distances.sort()
    return distances


def s_prime(item, critic, metric='Manhattan'):
    """
    Return s'(item) for the specified critic.
    Parameters:
        item: str,
        critic: str,
        metric: str; distance metric either 'Manhattan' or 'Euclidean'.
    """
    assert metric in ['Manhattan', 'Euclidean'],\
        "Unsupported metric. Please, use 'Manhattan' or 'Euclidean'"
    check_critic(critic)
    check_item(item)

    ratings_df = ratings()
    critic_list = critics()

    # get item ratings
    item_ratings = ratings_df.loc[:, item]
    # critics who rated the item
    critics_idx = np.where(item_ratings != 'unavailable')[0]

    # computing the quantities
    total = 0
    s = 1
    if metric == 'Manhattan':
        total = sum([item_ratings[idx] / (1 + d_manhattan(critic, critic_list[idx]))
                     for idx in critics_idx])
        s = sum([1 / (1 + d_manhattan(critic, critic_list[idx])) for idx in critics_idx])

    elif metric == 'Euclidean':
        total = sum([item_ratings[idx] / (1 + d_euclidean(critic, critic_list[idx]))
                     for idx in critics_idx])
        s = sum([1 / (1 + d_euclidean(critic, critic_list[idx])) for idx in critics_idx])

    return total / s


def s_prime_exp(item, critic, metric='Manhattan'):
    """
    Return s'_exp(item) for the specified critic.
    Parameters:
        item: str,
        critic: str,
        metric: str; distance metric either 'Manhattan' or 'Euclidean'.
    """
    assert metric in ['Manhattan', 'Euclidean'],\
        "Unsupported metric. Please, use 'Manhattan' or 'Euclidean'"
    check_critic(critic)
    check_item(item)

    ratings_df = ratings()
    critic_list = critics()

    # get item ratings
    item_ratings = ratings_df.loc[:, item]
    # critics who rated the item
    critics_idx = np.where(item_ratings != 'unavailable')[0]

    # computing the quantities
    total = 0  # to avoid referencing before assigning
    s = 1
    if metric == 'Manhattan':
        total = sum([item_ratings[idx] * np.exp(-d_manhattan(critic, critic_list[idx]))
                     for idx in critics_idx])
        s = sum([np.exp(-d_manhattan(critic, critic_list[idx])) for idx in critics_idx])

    elif metric == 'Euclidean':
        total = sum([item_ratings[idx] * np.exp(-d_euclidean(critic, critic_list[idx]))
                     for idx in critics_idx])
        s = sum([np.exp(-d_euclidean(critic, critic_list[idx])) for idx in critics_idx])

    return total / s


def pearson(critic_1, critic_2):
    """Returns the Pearson coefficient between two critics."""

    ratings_1, ratings_2 = common_items_ratings(critic_1, critic_2)
    if len(ratings_1) < 2:
        pearson_c = 0
    else:
        pearson_c, _ = pearsonr(ratings_1, ratings_2)

    return pearson_c


def pearson_score(critic, item):
    """Returns pearson score for the given item and critic."""
    ratings_df = ratings()
    critic_list = critics()

    check_critic(critic)
    check_item(item)

    # get item ratings
    item_ratings = ratings_df.loc[:, item]
    # critics who rated the item
    critics_idx = np.where(item_ratings != 'unavailable')[0]
    # computing pearson coefficient for each neighbour
    total_pearson = sum([item_ratings[idx] * (2 + pearson(critic, critic_list[idx]))
                         for idx in critics_idx])

    return total_pearson


def cosine(critic_1, critic_2):
    """Returns the cosine of two critics."""
    ratings_1, ratings_2 = common_items_ratings(critic_1, critic_2)
    if len(ratings_1) >= 1:
        cos = ratings_1.dot(ratings_2) / (np.linalg.norm(ratings_1) * np.linalg.norm(ratings_2))
    else:
        cos = 0
    return cos


def cosine_score(critic, item):
    """Returns cosine score for the given item and critic."""
    ratings_df = ratings()
    critic_list = critics()

    check_critic(critic)
    check_item(item)

    # get item ratings
    item_ratings = ratings_df.loc[:, item]
    # critics who rated the item
    critics_idx = np.where(item_ratings != 'unavailable')[0]
    # computing cosine coefficient for each neighbour
    total_cosine = sum([item_ratings[idx] * (1 + cosine(critic, critic_list[idx]))
                        for idx in critics_idx])

    return total_cosine


if __name__ == '__main__':
    pass

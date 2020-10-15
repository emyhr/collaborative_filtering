from collabfilter.scores_coefficients import *


def recommend_nearest(critic, metric='Manhattan'):
    """
    Returns a item list recommended by the nearest neighbour.
    Parameters:
        critic: str,
        metric: str; distance metric either 'Manhattan' or 'Euclidean'.
    """
    ratings_df = ratings()
    item_list = items()

    # get the nearest neighbour
    neighbour = nearest_neighbour(critic, metric)[0][1]
    # extracting ratings of the critics
    critic_ratings = ratings_df.loc[ratings_df.name == critic].values[0][1:]
    neighbour_ratings = ratings_df.loc[ratings_df.name == neighbour].values[0][1:]

    # get item ratings which critic has not seen yet but neighbour has
    critic_unseen_ratings_idx = np.where(critic_ratings == 'unavailable')[0]
    neighbour_seen_ratings_idx = np.where(neighbour_ratings != 'unavailable')[0]

    # indices of items that have been rated by both of the critics
    recommended_items_idx = list(set(critic_unseen_ratings_idx) & set(neighbour_seen_ratings_idx))
    # extracting item names and ratings neighbour gave to recommended items
    recommended_items = [(item_list[idx], neighbour_ratings[idx]) for idx in recommended_items_idx]

    return recommended_items


def s_prime_recommend(item_list, critic, metric='Manhattan'):
    """Returns the best recommended item from the list of items with the given metric."""
    # dict{item: s'(item)}
    s_prime_scores = {item: s_prime(item, critic, metric) for item in item_list}
    # list of the items with the max s'
    best_items = [key for key, value in s_prime_scores.items() if value == max(s_prime_scores.values())]

    return best_items


def s_prime_exp_recommend(item_list, critic, metric='Manhattan'):
    """Returns the best recommended item using exp from the list of items with the given metric."""
    # dict{item: s'_exp(item)}
    s_exp_scores = {item: s_prime_exp(item, critic, metric) for item in item_list}
    # list of the items with the max s'
    best_items = [key for key, value in s_exp_scores.items() if value == max(s_exp_scores.values())]

    return best_items


def pearson_recommend(critic, item_list):
    """Returns the best recommended items by pearson score."""
    # dict{item: pearson_item(item)}
    pearson_scores = {item: pearson_score(critic, item) for item in item_list}
    # list of the items with the max pearson_score
    best_items = [key for key, value in pearson_scores.items() if value == max(pearson_scores.values())]

    return best_items


def cosine_recommend(critic, item_list):
    # dict{item: pearson_item(item)}
    cosine_scores = {item: cosine_score(critic, item) for item in item_list}
    # list of the items with the max cosine_score
    best_items = [key for key, value in cosine_scores.items() if value == max(cosine_scores.values())]

    return best_items


if __name__ == '__main__':
    pass

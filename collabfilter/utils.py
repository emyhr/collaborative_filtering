import pandas as pd
import numpy as np
from lorem import get_word
import random
from constants import FILE_PATH


def generate_data(n=7, m=7, file_path='new_data.csv'):
    """Generates rating dataset with n critics and m items."""

    rates = np.arange(1, 5.5, 0.5)  # possible rates
    # initial dataset
    data = np.array([[None for j in range(m)] for i in range(n)])
    # synthetic names
    names = [str(get_word(1)) + str(random.randint(0, 99))
             for i in range(n)]
    # synthetic titles
    titles = [str(get_word(1)) + str(random.randint(0, 99))
              for j in range(m)]

    # generating random data
    for i in range(n):
        random.shuffle(list(rates))  # shuffle rates just in case
        # randomly choose number of rates given
        number_of_not_nulls = random.randint(0, m+1)
        # randomly choose items with the rates
        not_nulls = np.random.randint(0, m, number_of_not_nulls)
        # assign generated values
        for j in not_nulls:
            data[i][j] = random.choice(rates)

    # create a df with titles from data
    data_df = pd.DataFrame(data, columns=titles)
    # add critics' names
    data_df = pd.concat([pd.DataFrame(names, columns=['name']), data_df], axis=1)
    data_df.set_index('name')
    data_df.to_csv(file_path,
                   index=False)


def ratings():
    """Returns a dataframe of item ratings."""
    ratings_df = pd.read_csv(FILE_PATH, header=0, index_col=False)
    ratings_df.fillna('unavailable', inplace=True)

    return ratings_df


def critics():
    """Returns a list of critics."""
    ratings_df = ratings()
    critics_list = ratings_df.name.values  # critics' names

    return critics_list


def items():
    """Returns a list of items."""
    ratings_df = ratings()
    item_list = ratings_df.columns.values[1:]  # item titles

    return item_list


def check_critic(critic):
    """Checks if the critic is in the data."""
    if critic not in critics():
        raise ValueError(f"Unknown critic: {critic}")


def check_item(item):
    """Checks if the item is in the data."""
    if item not in items():
        raise ValueError(f"Unknown item: {item}")


def common_items_ratings(critic_1, critic_2):
    """Returns the ratings of the items rated by both of the critics."""
    ratings_df = ratings()

    check_critic(critic_1)
    check_critic(critic_2)

    # extracting ratings of the critics
    ratings_1 = ratings_df.loc[ratings_df.name == critic_1].values[0][1:]
    ratings_2 = ratings_df.loc[ratings_df.name == critic_2].values[0][1:]
    # they might have not rated some items;
    # those items must be excluded from calculations
    existing_ratings_1 = np.where(ratings_1 != 'unavailable')[0]
    existing_ratings_2 = np.where(ratings_2 != 'unavailable')[0]

    # items that have been rated by both of the critics
    common_items = list(set(existing_ratings_1) & set(existing_ratings_2))

    return ratings_1[common_items], ratings_2[common_items]


def d_manhattan(critic_1, critic_2):
    """Calculates the Manhattan distance between the two given critics"""
    check_critic(critic_1)
    check_critic(critic_2)

    ratings_1, ratings_2 = common_items_ratings(critic_1, critic_2)

    return sum(abs(ratings_1 - ratings_2))


def d_euclidean(critic_1, critic_2):
    """Calculates the Euclidean distance between the two given critics"""
    check_critic(critic_1)
    check_critic(critic_2)

    ratings_1, ratings_2 = common_items_ratings(critic_1, critic_2)

    return sum((ratings_1 - ratings_2)**2)


if __name__ == '__main__':
    pass

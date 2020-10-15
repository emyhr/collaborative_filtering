from collabfilter.recommenders import *


def condition1(n=7, m=7):
    """
    Generates a dataset where a critic has
    the same recommendation given by s_prime
    and s_prime_exp but a different one by
    pearson_recommend.
    """

    while True:
        titles = items()
        names = critics()
        for critic in names:
            s = s_prime_recommend(titles, critic)
            s_exp = s_prime_exp_recommend(titles, critic)
            p = pearson_recommend(critic, titles)
            if s == s_exp and s != p:
                return critic

        generate_data(n, m)


def condition2(n=10, m=10):
    """
    Generates a dataset where a critic has
    different recommendations given by
    at least 4 similarity measures.
    """

    while True:
        titles = items()
        names = critics()
        for critic in names:
            # computing recommendations
            s = s_prime_recommend(titles, critic)
            s_exp = s_prime_exp_recommend(titles, critic)
            p = pearson_recommend(critic, titles)
            c = cosine_recommend(critic, titles)
            list_ = s + s_exp + p + c  # making a list of recommendations
            set_ = set(list_)  # selecting only unique recommendations
            if len(set_) == len(list_):
                return critic

        generate_data(n, m)


if __name__ == '__main__':
    pass

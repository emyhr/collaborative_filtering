from collabfilter.conditions import *


def main():
    films = items()
    critic = 'Hailey'
    print('s_prime: ', s_prime_recommend(films, critic))
    print('s_prime_exp: ', s_prime_exp_recommend(films, critic))
    print('Pearson coefficient: ', pearson_recommend(critic, films))
    print('Cosine coefficient: ', cosine_recommend(critic, films))


if __name__ == '__main__':
    main()

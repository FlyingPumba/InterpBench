import os


def is_running_in_circleci():
    in_circleci = 'CIRCLECI' in os.environ

    if in_circleci:
        print('Running on CircleCI')

    return in_circleci
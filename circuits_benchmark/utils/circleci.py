import os


def is_running_in_circleci() -> bool:
    in_circleci = 'CIRCLECI' in os.environ

    if in_circleci:
        print('Running on CircleCI')

    return in_circleci


def get_circleci_cases_percentage() -> float:
    return 0.25

# Inspiriation from:
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0
# https://github.com/orgs/python-poetry/discussions/1879#discussioncomment-2255728

FROM python:3.10-buster as base



ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VERSION=1.7.1

RUN pip install pipx
RUN pipx install "poetry==$POETRY_VERSION"

WORKDIR /circuit-benchmark

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

FROM python:3.10-slim-buster as runtime

ENV VIRTUAL_ENV=/circuit-benchmark/.venv \
    PATH="/circuit-benchmark/.venv/bin:$PATH"

# TODO: maybe we do want to run `poetry install --without dev` here? (Because we have the `--no-root` above).
# https://medium.com/@albertazzir/blazing-fast-python-docker-builds-with-poetry-a78a66f5aed0

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY . ./circuit-benchmark

# ENTRYPOINT ["python", "-m", "annapurna.main"]

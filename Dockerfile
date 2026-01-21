# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml pyproject.toml
COPY LICENSE LICENSE
COPY README.md README.md
COPY agent_skills/* agent_skills/

RUN pip install -e .

RUN pip uninstall -y pycrdt datalayer_pycrdt
RUN pip install datalayer_pycrdt==0.12.17

CMD ["python", "-m", "agent_skills.server"]

FROM continuumio/miniconda3

WORKDIR /src/ibicus

COPY environment.yml /src/ibicus/

RUN conda install -c conda-forge gcc python=3.10 \
    && conda env update -n base -f environment.yml

COPY . /src/ibicus

RUN pip install --no-deps -e .

FROM nvidia/cudagl:10.2-devel-ubuntu18.04

WORKDIR /parrot-photogrammetry
COPY . /parrot-photogrammetry

ARG DEBIAN_FRONTEND=noninteractive
RUN install_dependencies.sh


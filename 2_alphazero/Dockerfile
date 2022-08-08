FROM continuumio/miniconda3:latest

ADD condarc /root/.condarc
# RUN /opt/conda/bin/conda update --all
RUN /opt/conda/bin/conda install -y \
    python==3.10.5 pytorch torchvision torchaudio cudatoolkit=11.3 opencv -c pytorch -c conda-forge \
    && /opt/conda/bin/conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete

RUN pip install ray ray[default] gym numba

ENV LANG C.UTF-8

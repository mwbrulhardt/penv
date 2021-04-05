FROM rayproject/ray-ml:1.1.0

RUN pip install --upgrade pip
RUN pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl
RUN pip install tensortrade==1.0.1b0

WORKDIR /app

FROM public.ecr.aws/lambda/python:3.11

RUN yum install -y gcc gcc-c++ make zlib-devel libjpeg-devel libpng-devel freetype-devel
RUN pip install -U pip
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --skip-lock

COPY ["lambda_function.py", "./" ]

CMD ["lambda_function.lambda_handler"]
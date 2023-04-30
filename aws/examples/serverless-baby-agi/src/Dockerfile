ARG FUNCTION_DIR="/function"

FROM ubuntu:22.04

ARG FUNCTION_DIR
ARG CHROME_DRIVER_DIR

RUN apt update && \
    apt install -y unzip \
                   bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   libglib2.0-dev \
                   libnss3 \
                   libmagic-dev \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

# Create function directory
RUN mkdir -p ${FUNCTION_DIR}/

WORKDIR ${FUNCTION_DIR}/

# Copy handler function
COPY . ${FUNCTION_DIR}/

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install -r requirements.txt

# the unstrucutred python package requires nltk packges
# Pre-installing them to the function dir as lambdas do not have disk write acces
# The NLTK_DATA parameter will be set to the function dir in app.py
RUN python3 -m nltk.downloader -d ${FUNCTION_DIR}/ punkt
RUN python3 -m nltk.downloader -d ${FUNCTION_DIR}/ averaged_perceptron_tagger

ENTRYPOINT [ "python3", "-m", "awslambdaric" ]

# This will get replaced by the proper handler by the CDK script
CMD [ "lambda_function.lambda_handler" ]

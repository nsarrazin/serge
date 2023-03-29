FROM gcc:11 as llama_builder 

WORKDIR /tmp

RUN git clone https://github.com/ggerganov/llama.cpp.git --branch master-d5850c5

RUN cd llama.cpp && \
    make && \
    mv main llama

# Copy over rest of the project files

FROM ubuntu:22.04 as deployment

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

WORKDIR /usr/src/app

# install pip
RUN apt update
RUN apt-get install -y python3-pip curl wget
RUN pip install --upgrade pip

# install nodejs
RUN curl -sL https://deb.nodesource.com/setup_19.x | bash
RUN apt-get install nodejs

# MongoDB
RUN wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add -
RUN echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list

RUN wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.17_amd64.deb
RUN dpkg -i libssl1.1_1.1.1f-1ubuntu2.17_amd64.deb


# install requirements
COPY ./api/requirements.txt api/requirements.txt
RUN pip install -r ./api/requirements.txt

# copy llama binary from llama_builder
COPY --from=llama_builder /tmp/llama.cpp/llama /usr/bin/llama

# copy built webserver from web_builder
COPY ./web/package.json web/
COPY ./web/package-lock.json web/
# Install Node modules
RUN cd web && npm install

COPY ./web /usr/src/app/web/

#copy api folder
COPY ./api /usr/src/app/api

# get the deploy script with the right permissions
COPY deploy.sh /usr/src/app/deploy.sh
RUN chmod +x /usr/src/app/deploy.sh

CMD /usr/src/app/deploy.sh

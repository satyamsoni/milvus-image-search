# Milvus Vector Image Search
Vector Database Setup, Image ingestion and Search - Small Sample

## Purpose : 
A small sample repo to learn 
1. How to setup Milvus standalone in local machine
2. Create collection and ingest images as embeddings in Vector Database using pretrained AI models.
3. Search near match of an image in the database.

## How to setup Milvus
1. Clone the repo "git clone git@github.com:satyamsoni/vector_image_search.git"
2. Create virtual environment "python3 -m venv venv"
3. Activate the virtual environment "source venv/bin/activate"
4. Install python lib pymilvus, transformers, torch, numpy, Pillow,tqdm by "pip install <package_name>"
5. Install docker if not not already installed ref : https://docs.docker.com/engine/install/ubuntu
6. run milvus standalone "docker-compose up -d"
7. You can verify if it is running by "docker ps"


## How to Ingest Photos
run "python ImageIngest.py ./images"

## How to Search Photos
run "python SearchImage.py search.jpg"
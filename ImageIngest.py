import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import MilvusClient, DataType
import time
import uuid
import hashlib
import struct
import sys


class ImageIngest:
	
	collection="image_collection"
	milvus_host="http://127.0.0.1:19530"
	dimension = 512
	image_folder="./images"

	def __init__(self,img_path):
		print("Starting Image Ingestion in Milvus")
		self.client = MilvusClient(ImageIngest.milvus_host)
		image_folder=img_path
		# Create Collection if not available 
		self.prepCollection()
		self.Ingest()
		
	def Ingest(self):
		# Load CLIP model
		model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
		processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
		# Load Collection 
		self.client.load_collection(collection_name=ImageIngest.collection)
		# Loop through images
		for filename in os.listdir(ImageIngest.image_folder):
			if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
				image_path = os.path.join(ImageIngest.image_folder, filename)
				try:
					image = Image.open(image_path).convert("RGB")
					# Preprocess and extract features
					inputs = processor(images=image, return_tensors="pt")
					with torch.no_grad():
						image_features = model.get_image_features(**inputs)
					# Normalize vector (optional but recommended)
					image_features = image_features / image_features.norm(dim=-1, keepdim=True)
					#Vector Data 
					vector = image_features.squeeze().tolist()
					# Generate ID
					vector_id = self.vector_md5(vector)
					existing = self.client.query(
						collection_name=ImageIngest.collection,
						filter=f'id == "{vector_id}"',
						output_fields=["id"]
					)
					if existing:
						print(f"Image already ingested.")
					else:
						self.client.insert( 
							collection_name=ImageIngest.collection,
							data=[
								{
									"id":self.vector_md5(image_features.squeeze().tolist()),
									"filename":filename,
									"embedding":image_features.squeeze().tolist(),
									"timestamp":int(time.time())
								}
							])
						print(f"Ingested - {filename}")
				except Exception as e:
					print(f"Error processing {filename}: {e}")


	def prepCollection(self):
		# Run to Drop 
		#self.client.drop_collection(collection_name=ImageIngest.collection)
		if not self.client.has_collection(ImageIngest.collection):
			print(f"Creating collection '{ImageIngest.collection}'.")
			schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
			schema.add_field(
				field_name="id",
				datatype=DataType.VARCHAR,
				is_primary=True,
				max_length=32,
			)
			schema.add_field(
				field_name="filename",
				datatype=DataType.VARCHAR,
				max_length=256
			)
			schema.add_field(
				field_name="embedding",
				datatype=DataType.FLOAT_VECTOR,
				dim=ImageIngest.dimension
			)
			schema.add_field(
				field_name="timestamp",
				datatype=DataType.INT64,
			)
			self.client.create_collection(
				collection_name=ImageIngest.collection,
				schema=schema
			)
			index_params = self.client.prepare_index_params()
			index_params.add_index(
				field_name="embedding",
				index_type="IVF_FLAT",
				index_name="emb_index",
				metric_type="L2",
				params={"nlist": 1024}
			)
			self.client.create_index(
				collection_name=ImageIngest.collection,
				index_params=index_params,
				sync=False
			)
			print("Collection created.")
		else:
			print(f"Collection '{ImageIngest.collection}' already exists.")
	
	def vector_md5(self,vector):
		vector_bytes = bytearray()
		for num in vector:
			vector_bytes.extend(bytearray(struct.pack("f", num)))
		return hashlib.md5(vector_bytes).hexdigest()


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python ImageIngest.py <images_path>")
	else:
		app = ImageIngest(sys.argv[1])
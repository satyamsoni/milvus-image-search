from pymilvus import MilvusClient
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import sys

class SearchImage:

	collection="image_collection"
	milvus_host="http://127.0.0.1:19530"
	vector_field = "embedding"

	def __init__(self,img_path):
		# Connecting and initializing 
		self.client = MilvusClient(SearchImage.milvus_host)
		self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
		self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
		self.search(img_path)

	def search(self,img_path):
		print(f"Searching similar images for: {img_path}")
		try:
			image = Image.open(img_path).convert("RGB")
		except Exception as e:
			print(f"Failed to open image: {e}")
			return

		# Generate vector
		inputs = self.processor(images=image, return_tensors="pt")
		with torch.no_grad():
			image_features = self.model.get_image_features(**inputs)
		image_vector = image_features.squeeze().tolist()

		# Search in Milvus
		results = self.client.search(
			collection_name=SearchImage.collection,
			data=[image_vector],
			limit=3,
			output_fields=["id", "filename", "timestamp"]
		)
		print(f"\nTop 3 similar images:\n")
		for hit in results[0]:
			print(f"ID: {hit['id']}, Filename: {hit['filename']}, Score: {hit.score:.4f}")


if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python SearchImage.py <image_path>")
	else:
		app = SearchImage(sys.argv[1])
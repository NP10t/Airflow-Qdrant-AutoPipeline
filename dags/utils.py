from transformers import AutoModel
import torch
from torchvision import transforms
from PIL import Image
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid
import pandas as pd

def getRootPath():
    return "./data_DE/"

def getFramesPath():
    return os.path.join(getRootPath(), "Dataframes/")

def getImagesPath():
    return os.path.join(getRootPath(), "Images/")

def getCSVFile(city_name):
    return os.path.join(getFramesPath(), f"{city_name}.csv")

def getImagesDir(city_name):
    return os.path.join(getImagesPath(), city_name)

def init_model():
    model_ckpt = "nateraw/vit-base-beans"
    model = AutoModel.from_pretrained(model_ckpt)
    return model

def get_img_name(row):
    '''Given a row from the dataframe
    return the corresponding image name'''

    city = row['city_id']

    # there are at most 99999 places in each city
    # we remove any added prefix
    place_id = row.name % 10**5  # since place_id is the index of the CSV, this is the only way to get it
    place_id = str(place_id).zfill(7) # place_if is encoded on 7 positions in each filename

    panoid = row['panoid']
    year = str(row['year']).zfill(4)
    month = str(row['month']).zfill(2)
    northdeg = str(row['northdeg']).zfill(3)
    lat, lon = str(row['lat']), str(row['lon'])
    name = city+'_'+place_id+'_'+year+'_'+month+'_' + \
        northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
    return name

# Embed MỘT ảnh và metadata của nó

def embed_and_metadata(row, model, img_dir):
    """
    Embed a single image and store it in Qdrant.
    
    :param row: DataFrame row containing image information
    :param model: PyTorch model for embedding
    :param img_dir: Directory containing the images
    :param qdrant_client: Initialized Qdrant client
    :param collection_name: Name of the Qdrant collection
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_name = get_img_name(row)
    img_path = os.path.join(img_dir, img_name)
    
    try:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(image_tensor).last_hidden_state[:, 0].cpu().numpy()[0]

        metadata = {
            'place_id': int(row.name),
            'city_id': row['city_id'],
            'panoid': row['panoid'],
            'year': int(row['year']),
            'month': int(row['month']),
            'northdeg': float(row['northdeg']),
            'lat': float(row['lat']),
            'lon': float(row['lon'])
        }
        
        return embedding, metadata 

        
    except Exception as e:
        print(f"Error processing image {img_name}: {str(e)}")
        return None, None
    
    

def define_qdrant_collection(qdrant_client, collection_name, vector_size):
    """
    Define and create a Qdrant collection.
    
    :param qdrant_client: Initialized Qdrant client
    :param collection_name: Name of the collection to create
    :param vector_size: Size of the embedding vectors
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Collection '{collection_name}' created successfully.")
    
    
    
def store_embedding_in_qdrant(embedding, metadata, collection_name, qdrant_client):

    unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{metadata['place_id']}_{metadata['panoid']}"))
    
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            {
                "id": unique_id,
                "vector": embedding,
                "payload": metadata
            }
        ]
    )



def query_similar_images(qdrant_client, collection_name, query_vector, top_k=5):
    """
    Query Qdrant for similar images based on a given embedding.
    
    :param qdrant_client: Initialized Qdrant client
    :param collection_name: Name of the collection to query
    :param query_vector: The embedding vector to use for the query
    :param top_k: Number of similar images to retrieve
    :return: List of similar images with their metadata and scores
    """
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )
    
    return [
        {
            "score": hit.score,
            "metadata": hit.payload,
            "id": hit.id
        } for hit in search_result
    ]
    
def restore_deleted_files_in_directory(directory):
    # Duyệt qua toàn bộ các file trong thư mục
    for filename in os.listdir(directory):
        # Kiểm tra nếu file có hậu tố _deleted.csv
        if filename.endswith("_deleted.csv"):
            # Tạo đường dẫn đầy đủ cho file cũ và file mới
            old_file_path = os.path.join(directory, filename)
            new_file_path = old_file_path.replace("_deleted.csv", ".csv")
            
            # Đổi tên file từ _deleted.csv thành .csv
            os.rename(old_file_path, new_file_path)
            print(f"Đã đổi tên file từ {old_file_path} thành {new_file_path}")
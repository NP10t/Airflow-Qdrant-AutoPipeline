from utils import *

ROOT_PATH = "./data_DE/"
city_name = 'Phoenix'

model = init_model()

qdrant_client = QdrantClient("http://localhost:6333")

restore_deleted_files_in_directory(getFramesPath())

city_df = pd.read_csv(getCSVFile(city_name))
city_df = city_df.set_index('place_id')

place_id, image_1_info = next(iter(city_df.iterrows()))

embedding, metadata = embed_and_metadata(image_1_info, model, getImagesDir(city_name))')

similar_images = query_similar_images(qdrant_client, 'Cities', embedding)

print("Similar images:")
for img in similar_images:
    print(f"Score: {img['score']:.4f}")
    print(f"Place ID: {img['metadata']['place_id']}")
    print(f"City ID: {img['metadata']['city_id']}")
    print(f"Coordinates: {img['metadata']['lat']}, {img['metadata']['lon']}")
    print("---")
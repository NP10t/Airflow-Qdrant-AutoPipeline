from utils import *

model = init_model()

qdrant_client = QdrantClient("http://localhost:6333")

collection_name = 'Cities'

define_qdrant_collection(qdrant_client, collection_name, model.config.hidden_size)

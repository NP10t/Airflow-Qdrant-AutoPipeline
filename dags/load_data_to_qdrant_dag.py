from utils import *

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.sensors.python import PythonSensor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
    
def load_city_data_to_qdrant(city_name, **kwargs):
    
    model = init_model()
    
    qdrant_client = QdrantClient("http://qdrant:6333")
    
    collection_name = "Cities"
    
    city_df = pd.read_csv(getCSVFile(city_name))
    city_df = city_df.set_index('place_id')
    
    for _, row in city_df.head(5).iterrows():
        embedding, metadata = embed_and_metadata(row, model, getImagesDir(city_name))
        store_embedding_in_qdrant(embedding, metadata, collection_name, qdrant_client)
    
    new_csv_name = os.path.join(getFramesPath(), f"{city_name}_deleted.csv")
    os.rename(getCSVFile(city_name), new_csv_name)


def detect_new_city(**kwargs):

    for city_name in os.listdir(getImagesPath()): # liệt kê vừa thư mục vừa file
        
        if os.path.isdir(getImagesDir(city_name)) and \
            os.path.isfile(getCSVFile(city_name)):
            
            kwargs['ti'].xcom_push(key='new_city', value=city_name)
            return city_name
    return None


with DAG('Auto_Load_Data_to_Qdrant',
         default_args=default_args,
         schedule_interval='*/5 * * * *',  # Run every 5 minutes
         catchup=False) as dag:

    check_new_city_task = PythonSensor(
        task_id='detect_new_city',
        python_callable=detect_new_city,
        poke_interval=300,  # Check every 5 minutes
        timeout=24 * 60 * 60,  # Timeout after 24 hours
        mode='poke'
    )

    def process_new_city(**kwargs):
        ti = kwargs['ti']
        city_name = ti.xcom_pull(task_ids='detect_new_city', key='new_city')
        if city_name:
            load_city_data_to_qdrant(city_name, **kwargs)

    process_city_task = PythonOperator(
        task_id='process_new_city',
        python_callable=process_new_city,
        dag=dag
    )

    # define_collection_task >> check_new_city_task >> process_city_task
    check_new_city_task >> process_city_task

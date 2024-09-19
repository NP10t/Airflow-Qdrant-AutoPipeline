## Image Embedding and Similarity Search Project
This project automatically embeds images into vectors and pushes them to Qdrant using Apache Airflow. It also implements similarity search functionality.

## Project Structure
  ```bash
  .
  ├── dags/
  ├── scripts/
  ├── data_DE/
  │   ├── Images/
  │   │   ├── WashingtonDC/
  │   │   └── Rome/
  │   └── DataFrames/
  │       ├── WashingtonDC.csv
  │       └── Rome.csv
  ├── docker-compose.yml
  ├── Dockerfile
  └── requirements.txt
```

## Dataset
The project uses the GSV-Cities dataset, which can be downloaded from Kaggle. https://www.kaggle.com/datasets/amaralibey/gsv-cities 

## Requirements
- WSL 2 (Windows Subsystem for Linux 2)
- Qdrant
- Apache Airflow
- Docker and Docker Compose

## Setup

1. Install WSL 2 on your Windows machine if not already installed.
2. Install Docker and Docker Compose.
3. Clone this repository and navigate to the project directory.
4. Download the GSV-Cities dataset from the Kaggle link provided above and place the images in the appropriate folders under data_DE/Images/.
5. Ensure the CSV files are placed in the data_DE/DataFrames/ directory.
6. Build and start the Docker containers:


  ```
  docker-compose up -d
```

## Usage
1. Once the containers are up and running, access the Airflow webserver at http://localhost:8080.
2. Enable the DAGs for this project in the Airflow UI.
3. The DAGs will automatically process the images, embed them into vectors, and push them to Qdrant.
4. Use the provided scripts to perform similarity searches on the embedded images.

## Dependencies
All project dependencies are listed in the requirements.txt file. These will be installed in the Docker container during the build process.

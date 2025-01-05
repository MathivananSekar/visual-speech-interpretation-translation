#!/bin/bash

# Define the base project directory
BASE_DIR="visual-speech-interpretation-translation"

# Create the main project directory
mkdir -p $BASE_DIR

# Create subdirectories
echo "Creating project structure..."

mkdir -p $BASE_DIR/{data/{raw,processed,annotations,datasets},notebooks,src/{lip_reading,nlp_translation,utils,pipeline},experiments/{experiment_1,experiment_2},tests,docs,scripts,models/{lip_reading,translation},results/{logs,visualizations}}

# Create placeholder files
echo "Creating placeholder files..."

touch $BASE_DIR/{README.md,.gitignore,requirements.txt}
touch $BASE_DIR/docs/{abstract.md,metrics_definition.md,dataset_details.md,usage_guide.md}
touch $BASE_DIR/scripts/{run_training.sh,run_evaluation.sh,setup_env.sh}
touch $BASE_DIR/notebooks/{data_preprocessing.ipynb,model_training.ipynb,evaluation.ipynb}
touch $BASE_DIR/tests/{test_preprocessing.py,test_models.py,test_pipelines.py}
touch $BASE_DIR/src/lip_reading/{preprocessing.py,feature_extraction.py,cnn_model.py,evaluation.py}
touch $BASE_DIR/src/nlp_translation/{text_generation.py,translation_model.py,post_processing.py}
touch $BASE_DIR/src/utils/{logger.py,config.py,helpers.py}
touch $BASE_DIR/src/pipeline/{training_pipeline.py,evaluation_pipeline.py,end_to_end_pipeline.py}

# Print completion message
echo "Project structure created successfully in the '$BASE_DIR' directory!"
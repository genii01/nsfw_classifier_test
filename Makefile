# Variables
POETRY := poetry

# Paths
SRC_DIR := .
DATA_DIR := dataset
MODELS_DIR := saved_models
CSV_DIR := $(DATA_DIR)

# Files to clean
CLEAN_PATTERNS := *.pyc __pycache__ .pytest_cache .coverage *.pth

.PHONY: all setup clean clean-all train create-dataset test help convert-model predict predict-batch serve load-test load-test-ui

# Default target
all: setup create-dataset train

# Help command
help:
	@echo "Available commands:"
	@echo "  make setup         - Install dependencies and create virtual environment"
	@echo "  make train         - Run model training"
	@echo "  make create-dataset- Create dataset CSV file"
	@echo "  make clean         - Remove Python cache files and saved models"
	@echo "  make clean-all     - Remove all generated files including virtual environment"
	@echo "  make test          - Run tests"
	@echo "  make all           - Setup environment and run full training pipeline"
	@echo "  make convert-model - Convert PyTorch model to ONNX"
	@echo "  make predict       - Run inference"
	@echo "  make predict-batch - Run batch inference"
	@echo "  make serve         - Run API server"
	@echo "  make load-test     - Run load test"
	@echo "  make load-test-ui  - Start load test UI"

# Setup environment and dependencies
setup: pyproject.toml
	@echo "Installing dependencies with Poetry..."
	@$(POETRY) install
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(DATA_DIR)

# Create dataset
create-dataset:
	@echo "Creating dataset CSV file..."
	@$(POETRY) run python create_dataframe.py

# Train model
train:
	@echo "Starting model training..."
	@$(POETRY) run python train.py

# Run tests
test:
	@echo "Running tests..."
	@$(POETRY) run pytest tests/

# Clean Python cache files and models
clean:
	@echo "Cleaning cache files and models..."
	@for pattern in $(CLEAN_PATTERNS); do \
		find . -type d -name "$$pattern" -exec rm -rf {} +; \
		find . -type f -name "$$pattern" -delete; \
	done
	@rm -rf $(MODELS_DIR)/*
	@echo "Cleaned!"

# Clean everything including virtual environment
clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@rm -rf $(CSV_DIR)/*.csv
	@echo "Everything cleaned!"

# Create directories if they don't exist
$(MODELS_DIR):
	@mkdir -p $(MODELS_DIR)

$(DATA_DIR):
	@mkdir -p $(DATA_DIR)

# Convert model to ONNX
convert-model:
	@echo "Converting PyTorch model to ONNX..."
	@$(POETRY) run python tools/convert_to_onnx.py \
		--model-path saved_models/best_model.pth \
		--config-path config/train_config.yaml \
		--output-path saved_models/model.onnx \
		--rtol 1e-3 \
		--atol 1e-5 \
		--num-samples 100

# Run inference
predict:
	@echo "Running inference..."
	@$(POETRY) run python inference/predictor.py

# Run batch inference
predict-batch:
	@echo "Running batch inference..."
	@$(POETRY) run python inference/batch_predictor.py

# Run API server
serve:
	@echo "Starting API server..."
	@$(POETRY) run python -m app.main

# Load testing
load-test:
	@echo "Running load test..."
	@$(POETRY) run python tests/locust/run_load_test.py --headless

load-test-ui:
	@echo "Starting load test UI..."
	@$(POETRY) run python tests/locust/run_load_test.py
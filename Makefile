help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


build-env: ## Build virtual environment and install required packages
	pip install virtualenv
	virtualenv -p python3 venv
	venv/bin/python -m pip install --upgrade pip
	venv/bin/pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
	venv/bin/pip install -Ur requirements.txt
	venv/bin/pip install tensorboard


install-flake8: ## Install flake8 code linter
	venv/bin/pip install -U flake8


lint: install-flake8 ## Lint code using flake8
	venv/bin/flake8


install-pytest: ## Install pytest for unit testing
	venv/bin/pip install -U pytest


test: install-pytest ## Test code using pytest
	venv/bin/pytest


data: build-env ## Download data to ./data
	. venv/bin/activate; python ./utils/download_data.py


train: ## Train model
	. venv/bin/activate; python main.py


tb: ## See Tensorboard results. Usage : make tb log_id="your_log_id"
	. venv/bin/activate; tensorboard --logdir="./saved/logs/$(log_id)"


clean-env: ## Clean environment
	rm -rf venv
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pth' -delete


.PHONY: help build-env lint test train tb clean-env
.DEFAULT_GOAL := help

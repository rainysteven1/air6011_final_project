REPO ?= RoboTwin
REPO_URL ?= https://github.com/TianxingChen/RoboTwin.git

.PHONY: help sync upgrade

help:  ## Display targets with category headers
	@awk 'BEGIN { \
		FS = ":.*##"; \
		printf "\n\033[1;34mUsage:\033[0m\n  make \033[36m<target>\033[0m\n\n\033[1;34mTargets:\033[0m\n"; \
	} \
	/^##@/ { \
		header = substr($$0, 5); \
		printf "\n\033[1;33m%s\033[0m\n", header; \
	} \
	/^[a-zA-Z_-]+:.*?##/ { \
		printf "  \033[36m%-20s\033[0m \033[90m%s\033[0m\n", $$1, $$2; \
	}' $(MAKEFILE_LIST)

##@ Git Submodule

add: ## Add submodule if not exists
	git submodule add -f -b main $(REPO_URL) lib/$(REPO)

remove:
	git submodule deinit -f lib/$(REPO) && git rm -f lib/$(REPO) && \
	git commit -m "delete $(REPO) submodule"

sync: ## Synchronize Git submodules
	git submodule update --init --recursive --remote

upgrade: ## Upgrade Git submodules to the latest main branch
	git submodule foreach 'git checkout main && git pull origin main --verbose'

##@ Docker
CMD_1 = cd ./GR-MG && bash goal_gen/train_ip2p.sh goal_gen/config/train.json
CMD_2 = cd ./GR-MG && bash policy/main.sh policy/config/pretrain.json
CMD_3 = cd ./GR-MG && bash policy/main.sh policy/config/train.json

build: ## Build GR-MG container
	docker build -t gr_mg:latest .

run: ## Run GR-MG container (CMD=1: goal generation, CMD=2: policy pretraining, CMD=3: policy training)
	@if [ -z "$(CMD)" ]; then \
        echo "Error: Please specify a command (CMD=1|2|3)"; \
        echo "  CMD=1: Train goal generation model (train_ip2p.sh)"; \
        echo "  CMD=2: Policy pretraining (pretrain.json)"; \
        echo "  CMD=3: Policy training (train.json)"; \
        exit 1; \
    fi
	@if [ "$(CMD)" = "1" ]; then \
        echo "Executing goal generation model training..."; \
        COMMAND="$(CMD_1)"; \
    elif [ "$(CMD)" = "2" ]; then \
        echo "Executing policy pretraining..."; \
        COMMAND="$(CMD_2)"; \
    elif [ "$(CMD)" = "3" ]; then \
        echo "Executing policy training..."; \
        COMMAND="$(CMD_3)"; \
    else \
        echo "Error: Invalid CMD value: $(CMD)"; \
        echo "Please use CMD=1|2|3"; \
        exit 1; \
    fi; \
    docker run -d --rm --gpus all --name gr_mg \
        --shm-size=8g \
    	-v $$(pwd)/config/GR-MG/goal_gen/train.json:/app/GR-MG/goal_gen/config/train.json \
    	-v $$(pwd)/config/GR-MG/policy/pretrain.json:/app/GR-MG/policy/config/pretrain.json \
    	-v $$(pwd)/config/GR-MG/policy/train.json:/app/GR-MG/policy/config/train.json \
    	-v $$(pwd)/resources:/app/resources \
        -v /data/FinalProject/data:/app/data \
    	-v /data/FinalProject/results:/app/results \
        -v /data/docker_tmp:/tmp \
        -e WANDB_API_KEY=$$(cat wandb_key.txt) \
        -e WANDB_CONSOLE=wrap \
        -e WANDB_START_METHOD=thread \
        -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        -e PYTHONPATH=/app \
    	gr_mg:latest /bin/bash -c "$$COMMAND"

##@ Download 
MODEL1 = timbrooks/instruct-pix2pix
MODEL2 = google-t5/t5-base

download: ## Download Model
	@if [ -z "$(CMD)" ]; then \
        echo "Error: Please specify a model to download (CMD=1|2)"; \
        echo "  CMD=1: Download $(MODEL1)"; \
        echo "  CMD=2: Download $(MODEL2)"; \
        exit 1; \
    fi
	@if command -v jq >/dev/null 2>&1; then \
        USERNAME=$$(jq -r '.username' script/hfd_config.json); \
        TOKEN=$$(jq -r '.token' script/hfd_config.json); \
    else \
        USERNAME=$$(python -c "import json; print(json.load(open('script/hfd_config.json'))['username'])"); \
        TOKEN=$$(python -c "import json; print(json.load(open('script/hfd_config.json'))['token'])"); \
    fi
	@if [ "$(CMD)" = "1" ]; then \
        echo "Downloading $(MODEL1)..."; \
        cd script && ./hfd.sh $(MODEL1) --hf_username $$USERNAME --hf_token $$TOKEN; \
    elif [ "$(CMD)" = "2" ]; then \
        echo "Downloading $(MODEL2)..."; \
        cd script && ./hfd.sh $(MODEL2) --hf_username $$USERNAME --hf_token $$TOKEN; \
    else \
        echo "Error: Invalid CMD value: $(CMD)"; \
        echo "Please use CMD=1|2"; \
        exit 1; \
    fi


##@ Script
output-img: ## Extract RGB images from PKL files
	python script/extract_pkl_img.py --input_dir /data/RoboTwin --output_dir /data/FinalProject/data/RoboTwin_output

output-dataset: ## Output Dataset
	python script/format_data.py --input_dir /data/FinalProject/data/RoboTwin_output \
                     --output_dir /data/FinalProject/data/RoboTwin_format \
                     --workers 16 \
                     --batch_size 8 \
                     --buffer_size 100
                     
output-meta: ## Output meta.json
	python script/output_meta.py --input_dir /data/FinalProject/data/RoboTwin_format --train_samples 20 --val_samples 5
TASK ?= block_hammer_beat
GPU  ?= 1
IMAGE_NAME ?= robotwin:latest

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
	git submodule add -b main https://github.com/TianxingChen/RoboTwin.git data/RoboTwin

remove:
	git config -f .gitmodules --remove-section submodule."data/RoboTwin" 2>/dev/null || true && \
	git rm --cached data/RoboTwin 2>/dev/null || true

sync: ## Synchronize Git submodules
	git submodule update --init --recursive --remote

upgrade: ## Upgrade Git submodules to the latest main branch
	git submodule foreach 'git checkout main && git pull origin main --verbose'

##@ Download 
download-IP2P: ## instruct-pix2pix
	@if command -v jq >/dev/null 2>&1; then \
        USERNAME=$$(jq -r '.username' script/hfd_config.json); \
        TOKEN=$$(jq -r '.token' script/hfd_config.json); \
    else \
        USERNAME=$$(python -c "import json; print(json.load(open('script/hfd_config.json'))['username'])"); \
        TOKEN=$$(python -c "import json; print(json.load(open('script/hfd_config.json'))['token'])"); \
    fi; \
    cd script && ./hfd.sh timbrooks/instruct-pix2pix --hf_username $$USERNAME --hf_token $$TOKEN

##@ Script

output-img:
	python script/extract_pkl_img.py --input_dir /data/RoboTwin --output_dir /data/RoboTwin_output

output-npy:
	python script/extract_depth.py --input_dir /data/RoboTwin --output_dir /data/RoboTwin_output
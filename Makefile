.PHONY: install
install:
	pip install -r requirements.txt
	pip install -U rich


.PHONY: download_models_ssh
download_models_ssh:
	dvc get git@gitlab.com:oliyyaa/cvr-hw2-segmdet.git weights/ -o models/


.PHONY: clean_models
clean_models:
	mv models/weights/*.* models
	rm -r models/weights


.PHONY: download_models
download_models:
	make download_models_ssh
	make clean_models


.PHONY: download_models_dvc
download_detector:
	dvc pull -R models/


.PHONY: prepare_data
prepare_data:
	python src/preprocess/prepare_data.py data/barcodes-annotated-gorai data/barcodes-annotated-gorai_detected


.PHONY: train
train:
	python src/train.py --log --logger clearml


.PHONY: lint
lint:
	flake8 src/
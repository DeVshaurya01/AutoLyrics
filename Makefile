.PHONY: install prepare train eval demo test clean

install:
	pip install -e . -r requirements.txt

prepare:
	python scripts/prepare_data.py

train:
	python scripts/train.py

eval:
	python scripts/evaluate.py

demo:
	python app.py

test:
	pytest tests/ -v

clean:
	rm -rf outputs/checkpoints/* outputs/logs/* outputs/wandb/* outputs/reports/*

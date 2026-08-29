.PHONY: install test lint api train drift docker

install:
	python -m pip install -r requirements-dev.txt

test:
	PYTHONPATH=. python -m pytest -q

lint:
	ruff check api tests src monitoring

api:
	PYTHONPATH=. uvicorn api.app:app --reload --port 8000

train:
	PYTHONPATH=. python src/train.py --data data/BankChurners.csv --output-dir artifacts

drift:
	PYTHONPATH=. python monitoring/drift.py --reference artifacts/reference_profile.json --predictions prediction_logs/predictions.jsonl

docker:
	docker build -f api/Dockerfile -t churn-api .

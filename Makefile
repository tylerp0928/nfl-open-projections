SEASONS?=2019-2025

.PHONY: setup run run-ext predict
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	python -m src.pipelines.run_all --seasons $(SEASONS)

run-ext:
	python -m src.pipelines.run_extended --seasons $(SEASONS)

predict:
	python -m src.models.predict_game_week --season $(SEASON) --week $(WEEK)

docker-build:
	docker build -t nfl-open-proj -f docker/Dockerfile .

docker-run:
	docker run --rm -it -v ${PWD}/data:/app/data nfl-open-proj

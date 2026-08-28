.PHONY: help setup data osm notebooks app test lint format screenshot clean

help:                ## show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | column -t -s "$$(printf '\t')"

setup:               ## create the environment and register the Jupyter kernel
	uv sync
	uv run python -m ipykernel install --user --name london-rent --display-name "London Rent (uv)"

data:                ## download the Inside Airbnb London snapshot into data/raw
	mkdir -p data/raw
	@base=https://data.insideairbnb.com/united-kingdom/england/london/2026-06-19 ; \
	curl -sS -o data/raw/listings.csv.gz          $$base/data/listings.csv.gz ; \
	curl -sS -o data/raw/calendar.csv.gz          $$base/data/calendar.csv.gz ; \
	curl -sS -o data/raw/neighbourhoods.geojson   $$base/visualisations/neighbourhoods.geojson

osm:                 ## refresh the OpenStreetMap extracts (stations, parks, food)
	uv run python -m londonrent.geo

notebooks:           ## execute all notebooks in order
	uv run jupyter nbconvert --to notebook --execute --inplace \
		notebooks/01_eda.ipynb notebooks/02_geo_features.ipynb \
		notebooks/03_price_model.ipynb notebooks/04_text.ipynb

app:                 ## run the Streamlit demo
	uv run streamlit run app/streamlit_app.py

screenshot:          ## regenerate docs/app.png from the running app
	uv run python scripts/screenshot.py

test:                ## run the test suite
	uv run pytest

lint:                ## ruff check + format check
	uv run ruff check .
	uv run ruff format --check .

format:              ## auto-format
	uv run ruff format .

clean:               ## remove caches and generated data (keeps raw + OSM)
	rm -rf data/interim/* data/processed/* cache/ .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

PROJECT := ibicus
CONDA := conda
CONDAFLAGS :=
COV_REPORT := html

default: qa unit-tests type-check

qa:
	pre-commit run --all-files

unit-tests:
	python -m pytest -vv --cov=. --cov-report=$(COV_REPORT)

type-check:
	python -m mypy .

conda-env-update:
	$(CONDA) env update $(CONDAFLAGS) -f environment.yml

docker-build:
	docker build -t $(PROJECT) .

docker-run:
	docker run --rm -ti -v $(PWD):/srv $(PROJECT)

docs-build:
	cd docs && rm -fr _api && make clean && make html

# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW

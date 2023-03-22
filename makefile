.DEFAULT_GOAL := all
code_folder = GSSP_utils
isort = isort $(code_folder)
black = black $(code_folder)

# install:
# 	pip install -e .

# .PHONY: install-dev-requirements
# install-dev-requirements:
# 	pip install -r tests/requirements.txt
# 	pip install -r tests/requirements-linting.txt

.PHONY: format
format:
	$(isort)
	$(black)

.PHONY: lint-python
lint-python:
	ruff $(code_folder)
	$(isort) --check-only --df
	$(black) --check --diff


.PHONY: lint
lint: lint-python

.PHONY: mypy
mypy:
	mypy $(code_folder)



.PHONY: all
all: lint mypy

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -f `find . -type f -name '*.cpython-*' `
	rm -rf dist
	rm -rf build
	rm -rf target
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf *.egg-info
	rm -rf .ruff*
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
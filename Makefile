.PHONY: build
build:
	pip install -U pip
	pip install Cython
	pip install pybind11
	pip install wheel setuptools pip --upgrade
	pip install -r requirements.txt

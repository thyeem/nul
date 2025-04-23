.PHONY: build
build:
	pip install -U pip
	pip install Cython
	pip install pybind11
	pip install wheel setuptools pip --upgrade
	pip install -U foc ouch
	pip install -r requirements.txt

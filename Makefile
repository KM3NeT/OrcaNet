PKGNAME=orcanet
ALLNAMES = $(PKGNAME)
ALLNAMES += orcanet_contrib
ALLNAMES += examples

install:
	pip install .

install-dev: dependencies
	pip install -e .

clean:
	python setup.py clean --all
	rm -f -r build/

test:
	py.test --junitxml=./reports/junit.xml -o junit_suite_name=$(PKGNAME) $(PKGNAME)

retest:
	py.test --junitxml=./reports/junit.xml -o junit_suite_name=$(PKGNAME) $(PKGNAME) --last-failed

test-cov:
	py.test --cov ./ --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage $(ALLNAMES)

flake8:
	py.test --flake8

pep8: flake8

docstyle:
	py.test --docstyle

lint:
	py.test --pylint

dependencies:
	pip install -Ur requirements.txt
	pip install -Ur requirements_dev.txt

.PHONY: yapf
yapf:
	yapf -i -r $(PKGNAME)
	yapf -i setup.py

.PHONY: all clean build install install-dev test retest test-nocov flake8 pep8 dependencies docstyle

# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* ui_customerchurn/*.py

black:
	@black scripts/* ui_customerchurn/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr ui_customerchurn-*.dist-info
	@rm -fr ui_customerchurn.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# Docker run commands ----------------------------------------------

docker_build:
	@docker build -t eu.gcr.io/le-wagon-cathal/ui-customerchurn1 .

docker_run:
	@docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/le-wagon-cathal/ui-customerchurn1

docker_push:
	@docker push eu.gcr.io/le-wagon-cathal/ui-customerchurn1

docker_deploy:
	@gcloud run deploy --image eu.gcr.io/le-wagon-cathal/ui-customerchurn1 --platform managed --region europe-west1 --timeout=600 --memory=2Gi

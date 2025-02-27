UID=$(shell id -u)
PWD=$(shell pwd)
CLARIN_KEY=$(shell echo $$CLARIN_KEY)

build:
	docker build --build-arg UID=$(UID) --tag local-law-rag .

black:
	docker run --user "$(UID)" --rm -v $(PWD):/app local-law-rag black ./scripts

run_app:
	docker run --user "$(UID)" --rm -v $(PWD):/app -p 8501:8501 -e CLARIN_KEY="$(CLARIN_KEY)" local-law-rag streamlit run app.py

dvc_status:
	docker run --user "$(UID)" --rm -v $(PWD):/app local-law-rag dvc status

dvc_repro:
	docker run --user "$(UID)" --rm -v $(PWD):/app local-law-rag dvc repro

dvc_stage_list:
	docker run --user "$(UID)" --rm -v $(PWD):/app local-law-rag dvc stage list

dvc_checkout:
	docker run --user "$(UID)" --rm -v $(PWD):/app local-law-rag dvc checkout
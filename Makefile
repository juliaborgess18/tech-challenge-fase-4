# Atalhos (Linux/macOS/Git Bash). No Windows sem `make`, rode os comandos diretamente.
PY ?= python

.PHONY: install data kb index all app ask eval test

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

kb:
	$(PY) scripts/build_knowledge_base.py

index:
	$(PY) scripts/build_index.py

all: kb index

app:
	streamlit run app/streamlit_app.py

ask:
	$(PY) scripts/ask.py "$(Q)" --trace

eval:
	$(PY) evaluation/run_eval.py

test:
	$(PY) -m pytest -q

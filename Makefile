.PHONY: up data ingest eval load test down

up:            ## start db, redis, api, worker
	cp -n .env.example .env || true
	docker compose up -d --build

data:          ## build 7,600-doc corpus + 500 labeled questions (SQuAD)
	docker compose run --rm api python scripts/prepare_data.py --docs 7600 --eval 500 --out data

ingest:        ## push corpus through the API; workers embed + index
	docker compose run --rm -v $$PWD/data:/srv/data api python scripts/ingest_bulk.py --api http://api:8000

eval:          ## recall@k / MRR / nDCG for all 4 modes + answer faithfulness on 100 Qs
	docker compose run --rm -v $$PWD/data:/srv/data -v $$PWD/reports:/srv/reports api \
	  python -m eval.run_eval --answers 100

load:          ## 50 concurrent users, 60s, skewed query mix
	docker compose run --rm -v $$PWD/data:/srv/data api python scripts/loadtest.py --api http://api:8000 --users 50 --duration 60

test:          ## pure-logic unit tests (no services needed)
	python -m pytest -q tests

down:
	docker compose down

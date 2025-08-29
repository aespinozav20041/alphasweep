.PHONY: dev test live-paper live-real

dev:
	docker compose up --build
test:
	docker compose run --rm app pytest -q
live-paper:
	LIVE_MODE=paper docker compose up app worker db
live-real:
	LIVE_MODE=real docker compose up app worker db

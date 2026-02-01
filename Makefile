.PHONY: help reset-db crawl fresh-crawl dev backend install

help:
	@echo "Available commands:"
	@echo "  make install      - Install all dependencies"
	@echo "  make reset-db     - Reset the database (drop and recreate articles table)"
	@echo "  make crawl        - Run the article crawl script"
	@echo "  make fresh-crawl  - Reset database and run crawl"
	@echo "  make dev          - Start the Next.js frontend dev server"
	@echo "  make backend      - Start the FastAPI backend server"

install:
	npm install
	pip install -r backend/requirements.txt

reset-db:
	cd backend && python reset_db.py

crawl:
	cd backend && python crawl_articles.py

fresh-crawl: reset-db crawl

dev:
	npm run dev

backend:
	cd backend && uvicorn main:app --reload --port 8000

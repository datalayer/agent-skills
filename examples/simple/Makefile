SHELL=/bin/bash

.DEFAULT_GOAL := help

.PHONY: help run clean

help:
	@echo "Targets:"
	@echo "  run    Run the skills example"
	@echo "  clean  Remove generated skill directories"

run:
	python skills_example.py

clean:
	rm -rf example_skills example_simple_skills

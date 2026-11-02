.PHONY: install data test test-all lint all report
install:  ; pip install -e ".[dev]"
data:     ; adlift fetch
test:     ; pytest -m "not slow" -q
test-all: ; pytest -q
lint:     ; ruff check src tests
all:      ; adlift all
report:   ; adlift report

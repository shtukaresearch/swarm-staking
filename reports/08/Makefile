TARGET ?= ./target
INVOKER ?= podman run --rm -v $$(pwd):/tex -w /tex paber

all: $(TARGET)/swarm-staking-report-08.pdf

$(TARGET):
	mkdir -p $@

$(TARGET)/swarm-staking-report-08.pdf: swarm-staking-report-08.tex document.tex macro.tex | $(TARGET)
	$(INVOKER) lualatex -synctex=1 -interaction=batchmode -output-directory=$(TARGET) swarm-staking-report-08

build-once: | $(TARGET)
	$(INVOKER) lualatex -synctex=1 -interaction=batchmode -output-directory=$(TARGET) swarm-staking-report-08

debug: | $(TARGET)
	$(INVOKER) lualatex -synctex=1 -output-directory=$(TARGET) swarm-staking-report-08

clean: | $(TARGET)
	rm $(TARGET)/*

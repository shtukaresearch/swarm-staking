TARGET ?= ./target

all: $(TARGET)/swarm-staking-report-08.pdf

$(TARGET):
	mkdir -p $@

$(TARGET)/swarm-staking-report-08.pdf: swarm-staking-report-08.tex document.tex macro.tex | $(TARGET)
	xelatex -synctex=1 -interaction=batchmode -output-directory=$(TARGET) swarm-staking-report-08

build-once: | $(TARGET)
	xelatex -synctex=1 -interaction=batchmode -output-directory=$(TARGET) swarm-staking-report-08

debug: | $(TARGET)
	xelatex -synctex=1 -output-directory=$(TARGET) swarm-staking-report-08

clean: | $(TARGET)
	rm $(TARGET)/*

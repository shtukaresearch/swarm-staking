TARGET ?= ./target
NAME = swarm-staking-report-full
ENV_NAME = swarm-staking-build
INVOKER ?= podman run --rm -v $$(pwd):/tex -v $$(pwd)/../common:/tex/common -w /tex $(ENV_NAME)

all: $(TARGET)/$(NAME).pdf

$(TARGET):
	mkdir -p $@

$(TARGET)/$(NAME).pdf: $(NAME).tex document.tex $(TARGET)/$(NAME).bbl | $(TARGET)
	$(INVOKER) lualatex -synctex=1 -interaction=batchmode -output-directory=$(TARGET) $(NAME)

$(TARGET)/$(NAME).bbl: $(TARGET)/$(NAME).bcf swarm-staking.bib
	$(INVOKER) biber $(basename $<) --quiet

$(TARGET)/$(NAME).bcf: $(NAME).tex
	$(INVOKER) lualatex -synctex=1 -interaction=batchmode -output-directory=$(TARGET) $(NAME)

build-once: | $(TARGET)
	$(INVOKER) lualatex -synctex=1 -interaction=batchmode -output-directory=$(TARGET) $(NAME)

debug: | $(TARGET)
	$(INVOKER) lualatex -synctex=1 -output-directory=$(TARGET) $(NAME)

clean: | $(TARGET)
	rm $(TARGET)/*
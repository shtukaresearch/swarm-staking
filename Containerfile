FROM registry.fedoraproject.org/fedora:40
RUN dnf install -y texlive
# default install includes:
# texlive-geometry texlive-amsmath texlive-fancyhdr texlive-fontspec
# texlive-amsfonts: amsfonts, amssymb
# texlive-xcolor texlive-hyperref texlive-bookmark texlive-biblatex
# texlive-tools: verbatim, calc, 
# texlive-graphics texlive-microtype texlive-babel
RUN dnf install -y texlive-libertine

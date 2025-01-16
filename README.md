# Swarm staking economic analysis

A [Shtuka Research](https://shtuka.io) study on incentives and revenue sharing in the [Swarm Network](https://www.ethswarm.org/).

> The object of study in this analysis is the Swarm Protocol revenue sharing mechanism. 
> We aim to define quantitative economic objectives, evaluate mechanism design choices against these objectives, and suggest avenues for optimisation or further research. 
> *Full report, p.2*

## Repo contents

* `reports/` TeX source for the monthly and final reports.
* `notebooks/` Jupyter notebook for risk calculations.
* `marimonb/` Marimo notebook for observational study of competitive staking behaviour.
* `sow.md` Original Scope of Work created at the start of this project.

## How to build

We recommend reading the built PDFs on the Release page.

If you wish to build the reports yourself, you will need Podman and Make installed.

1. Run `make environment` in the root directory (here) to generate a build container with the correct LaTeX installation.
2. Navigate to the root directory of the report you want to build (probably `/reports/full/`).
3. Run `make`. LaTeX commands are invoked using the build container generated in step 1. LaTeX outputs will be saved to `./target/`. 

Run `make debug` to print LaTeX logs to console, and `make shell` to open a shell into the build container for debugging purposes.
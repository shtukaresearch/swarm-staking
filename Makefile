ENV_NAME = swarm-staking-build

environment: env.json

env.json: Containerfile
	podman build -t $(ENV_NAME) . && podman image inspect $(ENV_NAME) > env.json

update-env:
	podman build -t $(ENV_NAME) . && podman image inspect $(ENV_NAME) > env.json

clean-env:
	podman rmi $(ENV_NAME)
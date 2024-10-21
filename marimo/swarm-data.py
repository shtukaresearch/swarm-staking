import marimo

__generated_with = "0.9.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import subprocess as sp
    return json, sp


@app.cell
def __():
    STAKE_UPDATED_TOPIC_v0_4_0 = "0x61e979698346a2aa868a3a9f08d30c846174841dc9b074bbf2a82d20554bc682"
    STAKE_REGISTRY_v0_4_0_ADDRESS = "0x781c6D1f0eaE6F1Da1F604c6cDCcdB8B76428ba7"
    STAKE_REGISTRY_v0_4_0_DEPLOYED_BLOCK = 25527075
    STAKE_REGISTRY_v0_4_0_PAUSED_BLOCK = 35963617

    STAKE_REGISTRY_v0_9_1_ADDRESS = "0xBe212EA1A4978a64e8f7636Ae18305C38CA092Bd"
    STAKE_REGSITRY_v0_9_1_DEPLOYED_BLOCK = 35961749

    BLOCKS_IN_12_H = 12*60*12
    return (
        BLOCKS_IN_12_H,
        STAKE_REGISTRY_v0_4_0_ADDRESS,
        STAKE_REGISTRY_v0_4_0_DEPLOYED_BLOCK,
        STAKE_REGISTRY_v0_4_0_PAUSED_BLOCK,
        STAKE_REGISTRY_v0_9_1_ADDRESS,
        STAKE_REGSITRY_v0_9_1_DEPLOYED_BLOCK,
        STAKE_UPDATED_TOPIC_v0_4_0,
    )


@app.cell
def __():
    STAKE_REGISTRY_v0_4_0 = {
        "address": "0x781c6D1f0eaE6F1Da1F604c6cDCcdB8B76428ba7",
        "deployed-block": 25527075,
        "paused-block": 35963617,
        "events": {
            "stake-updated": "0x61e979698346a2aa868a3a9f08d30c846174841dc9b074bbf2a82d20554bc682"
        }
    }

    STAKE_REGISTRY_v0_9_1 = {
        "address": "0xBe212EA1A4978a64e8f7636Ae18305C38CA092Bd",
        "deployed-block": 35961749,
        "events": {
            "stake-updated": "0xf3fb57c8e9287d05c0b3c8612031896c43149edcf7ca6f1b287ac836b4b5d569"
        }
    }
    return STAKE_REGISTRY_v0_4_0, STAKE_REGISTRY_v0_9_1


@app.cell
def __(STAKE_UPDATED_TOPIC_v0_4_0):
    def get_cast_command(start, end, topic=STAKE_UPDATED_TOPIC_v0_4_0):
        return ["cast", "logs", "-j", "--from-block", str(start), "--to-block", str(end), topic]
    return (get_cast_command,)


@app.cell
def __(get_cast_command, json, sp):
    # rewrite using asyncio.create_subprocess_shell
    # https://docs.python.org/3/library/asyncio-subprocess.html

    def fetch_logs(start, end, chunk_size=10_000):
        logs = []
        for offset in range(start, end, chunk_size):
            end_offset = min(end, offset+chunk_size)
            command = get_cast_command(offset, end_offset)
            p = sp.run(command, capture_output=True)
            chunk = json.loads(p.stdout.decode())
            logs.extend(chunk)
            print(len(chunk))
            err = p.stderr.decode()
            if err:
                print(err)
        return logs
    return (fetch_logs,)


@app.cell
def __(
    STAKE_REGISTRY_v0_4_0_DEPLOYED_BLOCK,
    STAKE_REGISTRY_v0_4_0_PAUSED_BLOCK,
    fetch_logs,
):
    # takes around 30 minutes
    data = fetch_logs(STAKE_REGISTRY_v0_4_0_DEPLOYED_BLOCK, STAKE_REGISTRY_v0_4_0_PAUSED_BLOCK)
    return (data,)


@app.cell
def __():
    def decode_stake_updated_event_data(log):
        data = log["data"][2:]
        overlay = log["topics"][1]
        amount = int(data[0:64], 16)
        owner = "0x" + data[108:128]
        return {"overlay": overlay, "amount": amount, "owner": owner}
    return (decode_stake_updated_event_data,)


@app.cell
def __(data, decode_stake_updated_event_data):
    data_processed = []
    for row in data:
        data_processed.append(decode_stake_updated_event_data(row))
    return data_processed, row


@app.cell
def __(data_processed, json):
    with open("stake-registry-v0_4_0-logs.json", "w") as h:
        json.dump(data_processed, h)
    return (h,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

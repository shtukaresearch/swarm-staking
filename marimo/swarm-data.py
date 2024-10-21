import marimo

__generated_with = "0.9.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import subprocess as sp
    import altair as alt
    return alt, json, sp


@app.cell
def __():
    import datetime

    def now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H_%M_%S")
    return datetime, now


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
def __(STAKE_REGISTRY_v0_4_0):
    def get_cast_command(start, end, topic=STAKE_REGISTRY_v0_4_0["events"]["stake-updated"]):
        return ["cast", "logs", "-j", "--from-block", str(start), "--to-block", str(end), topic]
    return (get_cast_command,)


@app.cell
def __(STAKE_REGISTRY_v0_4_0, get_cast_command, json, sp):
    # rewrite using asyncio.create_subprocess_shell
    # https://docs.python.org/3/library/asyncio-subprocess.html

    from aiostream import stream
    import asyncio

    async def run_cast(start, end, topic=STAKE_REGISTRY_v0_4_0["events"]["stake-updated"]):
        proc = await asyncio.create_subprocess_shell(" ".join(get_cast_command(start, end, topic)), 
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        stdout, stderr = await proc.communicate()

        if proc.returncode:
            print(f"Failed to fetch logs from blocks {start} to {end}.")
            err = start, end, stderr.decode()
            print(err[2])
            return [], err
        else:
            return json.loads(stdout.decode()), None

    async def async_fetch_logs(start, end, chunk_size=10_000, max_concurrency=10):
        chunks = stream.range(start, end, chunk_size) # use stream.chunks API?

        async def run_cast_closure(start):
            return await run_cast(start, min(start+chunk_size,end))
        
        runs = stream.map(chunks, run_cast_closure, task_limit=max_concurrency)
        return await stream.list(runs)



    # WORKS (but is slow)

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
    return async_fetch_logs, asyncio, fetch_logs, run_cast, stream


@app.cell
def __(STAKE_REGISTRY_v0_4_0, async_fetch_logs):
    fetch_stream = async_fetch_logs(STAKE_REGISTRY_v0_4_0["deployed-block"], STAKE_REGISTRY_v0_4_0["paused-block"])
    return (fetch_stream,)


@app.cell(disabled=True)
async def __(fetch_stream):
    # 9 minutes with a few (~10) errors
    output = await fetch_stream
    return (output,)


@app.cell
def __(json, now, output):
    logs = []
    for r, e in output:
        print(e)
        logs.extend(r)

    with open(f"staking-logs-{now()}.json", "w") as fh:
        json.dump(logs, fh)
    return e, fh, logs, r


@app.cell
def __():
    return


app._unparsable_cell(
    r"""
    takes around 30 minutes sequentially
    data = fetch_logs(STAKE_REGISTRY_v0_4_0[\"deployed-block\"], STAKE_REGISTRY_v0_4_0[\"paused-block\"])
    """,
    name="__",
    column=None, disabled=True, hide_code=False
)


@app.cell
def __():
    def decode_stake_updated_event_data(log):
        data = log["data"][2:]
        overlay = log["topics"][1]
        amount = int(data[0:64], 16)
        owner = "0x" + data[108:128]
        block_number = int(log["blockNumber"], 16)
        return {"overlay": overlay, "amount": amount, "owner": owner, "block-number": block_number}
    return (decode_stake_updated_event_data,)


@app.cell
def __():
    #data_processed = []
    #for row in data:
    #    data_processed.append(decode_stake_updated_event_data(row))
    return


@app.cell
def __(json):
    with open("stake-registry-v0_4_0-logs.json") as h:
        data = json.load(h)
    return data, h


@app.cell
def __():
    def generate_overlay_histories(logs):
        registry = {}
        for log in logs:
            if log["overlay"] in registry:
                registry[log["overlay"]].append(log["amount"])
            else:
                registry[log["overlay"]] = [log["amount"]]
            
        return registry
        
    def generate_transition_history(logs):
        transitions = []
        registry = {}
        for log in logs:
            if log["overlay"] in registry:
                transition = (registry[log["overlay"]], log["amount"])
                registry[log["overlay"]] += log["amount"]
            else:
                transition = (0, log["amount"])
                registry[log["overlay"]] = log["amount"]
            transitions.append(transition)
        return transitions, registry
    return generate_overlay_histories, generate_transition_history


@app.cell
def __(data, generate_overlay_histories):
    registry = generate_overlay_histories(data)
    return (registry,)


@app.cell
def __(registry):
    max([len(row) for row in registry.values()])
    return


@app.cell
def __(registry):
    for k, v in registry.items():
        if len(v) > 1:
            print(f"{k}\t{len(v)}")
    return k, v


@app.cell
def __(registry):
    registry["0x12645eb4b51ef3ecf0c62c1b1c76af62b01bf1071c6628459569b5b3ec7cb46f"]
    return


@app.cell
def __():
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import numpy as np
    return mpl, np, plt


@app.cell
def __(data, np):
    amounts =  [row["amount"] / 10000_0000_0000_0000 for row in data]
    amounts = np.array(amounts, dtype="float64")
    return (amounts,)


@app.cell
def __(amounts, plt):
    stuff = plt.hist(amounts, bins=88, range=(12,100))
    plt.gcf().set_size_inches(15,5)
    return (stuff,)


@app.cell
def __(stuff):
    stuff[2]
    return


@app.cell
def __(amounts):
    amounts[((amounts % 100000000) != 0)]
    return


@app.cell
def __(amounts):
    amounts.max() // 100000000
    return


@app.cell
def __(amounts, np):
    hist, bins = np.histogram(amounts, bins=320)
    return bins, hist


@app.cell
def __(hist):
    hist
    return


@app.cell
def __(bins):
    bins
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()

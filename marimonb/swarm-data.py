import marimo

__generated_with = "0.9.14"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import subprocess as sp
    import numpy as np
    import scipy


    import altair as alt
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    import marimo as mo
    return alt, json, mo, mpl, np, plt, scipy, sp


@app.cell
def __():
    import datetime
    DT_FMT = "%Y-%m-%dT%H_%M_%S"

    def now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime(DT_FMT)
        
    BLOCK_25_000_000_DT = datetime.datetime(2022,11,15,22,33,55,tzinfo=datetime.timezone.utc)
    BLOCK_TD = datetime.timedelta(seconds=5)

    def block_number_to_datetime(n: int) -> datetime.datetime:
        n -= 25_000_000
        if n < 0:
            raise ValueError("Only works for blocks past 25,000,000")
        return BLOCK_25_000_000_DT + (n * BLOCK_TD)

    BLOCKS_IN_12_H = datetime.timedelta(days=1) / (2*BLOCK_TD)
    return (
        BLOCKS_IN_12_H,
        BLOCK_25_000_000_DT,
        BLOCK_TD,
        DT_FMT,
        block_number_to_datetime,
        datetime,
        now,
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

        # Possibly would be slicker to handle errors with a custom Exception type
        # and explicit Solidity event JSON schema
        # then return a Union[SolidityEvent, Exception]

        if proc.returncode:
            print(f"Failed to fetch logs from blocks {start} to {end}.")
            err = start, end, stderr.decode()
            print(err[2])
            return [], err
        else:
            return json.loads(stdout.decode()), None

    # This should be rewritten to return a set of failed jobs for manual intervention
    # And the use of aiostream seems superfluous; just use a asyncio.Semaphore and asyncio.gather
    # In fact aiostream just seems to be doing sequential batches?

    async def async_fetch_logs(start, end, chunk_size=10_000, max_concurrency=5):
        chunks = stream.range(start, end, chunk_size) # use stream.chunks API?

        async def run_cast_closure(start):
            result, err = await run_cast(start, min(start+chunk_size,end))
            if err:
                backoff = 1
            else:
                print("Yes")
            while err:
                backoff *= 2
                print(f"Job failed, retrying in {backoff}...")
                await asyncio.sleep(backoff)
                result, err = await run_cast(start, min(start+chunk_size,end))
                if backoff > 10:
                    print("Gave up!")
                    break
                if not err:
                    print("Job succeeded after retry.")
            return result, err

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


@app.cell(disabled=True)
async def __(STAKE_REGISTRY_v0_4_0, async_fetch_logs):
    # Sequential fetch takes about 30'
    # 20 mins with max_concurrency = 3
    # 10 mins with max_concurrency = 10
    # 9'15" with max_concurrency = 20

    output = await async_fetch_logs(STAKE_REGISTRY_v0_4_0["deployed-block"], STAKE_REGISTRY_v0_4_0["paused-block"], max_concurrency=8)
    return (output,)


@app.cell(disabled=True)
def __(json, now, output):
    def strip_log(log):
        return {
            "address": log["address"],
            "topics": log["topics"],
            "data": log["data"],
            "blockNumber": log["blockNumber"]
        }

    logs = []
    for r, e in output:
        if e: print(e)
        logs.extend((strip_log(log) for log in r))

    with open(f"staking-logs-{now()}.json", "w") as fh:
        json.dump(logs, fh)
    return e, fh, logs, r, strip_log


@app.cell(disabled=True)
def __(output):
    logs_full = []
    for batch, _ in output:
        logs_full.extend(batch)
    return batch, logs_full


@app.cell
def __(json):
    with open("staking-logs-2024-10-24T06_18_46.json") as fh:
        logs_full = json.load(fh)
    return fh, logs_full


@app.cell
def __(mo):
    mo.md(r"""## Strategic behaviour""")
    return


@app.cell
def __(logs_full, np):
    # Sort events by replication neighbourhood
    bins = [[] for _ in range(1024)]
    for log in logs_full:
        if int(log["blockNumber"], 16) < 31_000_000: # skip the big Chinese node event
            continue
        overlay_hex = log["topics"][1]
        overlay_bin = format(int(overlay_hex, 16), "0256b")
        prefix = int(overlay_bin[:10], 2) # most significant 10 bits
        bins[prefix].append(log)

    print(f"Found {len(bins)} replication neighbourhoods.")
    print("")

    events_per_bin = np.array([len(v) for v in bins])
    print(f"Most events in a single bin: {events_per_bin.max()} in bin {format(events_per_bin.argmax(), "010b")}.")
    print(f"Least events in a single bin: {events_per_bin.min()} in bin {format(events_per_bin.argmin(), "010b")}.")
    print(f"Mean events per bin: {events_per_bin.mean()}.")
    print(f"Standard deviation of events per bin: {events_per_bin.std()}.")
    print("")

    entities = [set(e["topics"][1] for e in bin) for bin in bins]
    entities_per_bin = np.array([len(e) for e in entities])
    print(f"Most entities in a single bin: {entities_per_bin.max()} in bin {format(entities_per_bin.argmax(), "010b")}.")
    print(f"Least entities in a single bin: {entities_per_bin.min()} in bin {format(entities_per_bin.argmin(), "010b")}.")
    print(f"Mean entities per bin: {entities_per_bin.mean()}.")
    print(f"Standard deviation of entities per bin: {entities_per_bin.std()}.")

    print(f"Total number of entities: {len(set.union(*entities))}.")
    return (
        bins,
        entities,
        entities_per_bin,
        events_per_bin,
        log,
        overlay_bin,
        overlay_hex,
        prefix,
    )


@app.cell
def __(bins, entities_per_bin, events_per_bin):
    # call a bin "weakly contested" if there are more events than entities (note this includes the case of 1 entity)
    # and "contested" if it is weakly contested and the event chain, considered chronologically
    # leaves and revisits an entity at least once

    weakly_contested = entities_per_bin < events_per_bin
    weakly_contested_idx = weakly_contested.nonzero()[0]
    print(f"Number of weakly contested bins: {len(weakly_contested_idx)}.")

    def is_contested(bin):
        seen_entities = set()
        last_seen_entity = None
        for event in bin:
            current_entity = event["topics"][1]
            # If current entity has been seen before but is not the last seen entity, we have a contest
            if current_entity in seen_entities:
                if last_seen_entity is not None and current_entity != last_seen_entity:
                    return True
                else:
                    last_seen_entity = current_entity
            else:
                seen_entities.add(current_entity)

    contested_idx = [i for i in weakly_contested_idx if is_contested(bins[i])]
    print(f"Number of contested bins: {len(contested_idx)}.")
    return (
        contested_idx,
        is_contested,
        weakly_contested,
        weakly_contested_idx,
    )


@app.cell
def __():
    def events_by_overlay(events):
        """
        Convert list of events into event plot format
        """
        entities = {}
        for e in events:
            overlay = e["topics"][1]
            if overlay in entities:
                entities[overlay].append(int(e["blockNumber"],16))
            else:
                entities[overlay] = [int(e["blockNumber"],16)]
        return entities
    return (events_by_overlay,)


@app.cell
def __(bins, events_by_overlay, np):
    # Statistics of number of events per address
    events_d = {}
    for bin in bins:
        events_d.update(events_by_overlay(bin))

    events_per_address = {k: len(v) for k, v in events_d.items()}
    events_per_address_k, events_per_address_v = zip(*events_per_address.items())
    events_per_address_v = np.array(events_per_address_v)

    print(f"Most events for a single address: {events_per_address_v.max()} for address \
           {events_per_address_k[events_per_address_v.argmax()]}.")
    print(f"Mean events per address: {events_per_address_v.mean()}.")
    print(f"Standard deviation of events per address: {events_per_address_v.std()}.")
    return (
        bin,
        events_d,
        events_per_address,
        events_per_address_k,
        events_per_address_v,
    )


@app.cell
def __(bins, block_number_to_datetime, contested_idx):
    # Let's have a look at one of these things
    for event in bins[contested_idx[7]]:
        dt = block_number_to_datetime(int(event["blockNumber"], 16))
        ol = event["topics"][1][:6]
        amt = int(event["data"][2:66], 16) / 1_0000_0000_0000_0000
        # note this is the new running total amount, not the amount added.
        print(f"{dt}\t{ol}\t{amt}")
    return amt, dt, event, ol


@app.cell
def __(bins, contested_idx, events_by_overlay, plt):
    fig, axs = plt.subplots(9,6, layout="tight")
    fig.set_size_inches(9,13)
    #fig.tight_layout()
    # may also use plt.subplots_adjust() or subplots(constrained_layout=True)

    for n, i in zip(range(54), contested_idx):
        current_ax = axs[n//6][n%6]

        # event plot
        events = events_by_overlay(bins[i])
        current_ax.eventplot(
            events.values(), 
            colors=[f'C{i}' for i in range(len(events))], 
            linelengths=0.6
        )

        # y axis tick labels
        keys = [k[:6] for k in events.keys()]
        current_ax.set_yticks(range(len(keys)), labels=keys, fontsize=6)

        # x axis tick labels
        xticks = [31000000, 33_000_000, 35_000_000]
        xlabels = ["Nov.", "Mar.", "July"]
        current_ax.set_xticks(xticks, labels=xlabels, fontsize=8)


    plt.savefig("contested.eps")
    plt.show()
    return axs, current_ax, events, fig, i, keys, n, xlabels, xticks


@app.cell
def __(mo):
    mo.md(r"""## Active blocks""")
    return


@app.cell
def __(logs_full, np):
    print(len(logs_full))
    block_numbers = [int(log["blockNumber"],16) for log in logs_full]
    block_number_bins = np.bincount(block_numbers)
    print(f"Most events in one block: {block_number_bins.max()} in block {block_number_bins.argmax()}.")
    print(f"Most events in one recent block: {block_number_bins[30_000_000:].max()} in block {block_number_bins[30_000_000:].argmax() + 30_000_000}.")
    print(f"Number of blocks with multiple events: {len(block_number_bins[block_number_bins>1])}")
    print(f"Mean number of events per block: {block_number_bins.mean()}.")
    return block_number_bins, block_numbers


@app.cell
def __(STAKE_REGISTRY_v0_4_0, scipy):
    print("Assume arrivals follow a Poisson process with rate given by the sample mean.")
    p = scipy.stats.poisson.cdf(1,0.0008561035056467292) # probability of
    print(f"Probability of never seeing more than 1 event in a block: {p**(STAKE_REGISTRY_v0_4_0["paused-block"] -  STAKE_REGISTRY_v0_4_0["deployed-block"])}")
    p = scipy.stats.poisson.cdf(2,0.0008561035056467292) # probability of
    print(f"Probability of never seeing more than 2 events in a block: {p**(STAKE_REGISTRY_v0_4_0["paused-block"] -  STAKE_REGISTRY_v0_4_0["deployed-block"])}")
    return (p,)


@app.cell
def __(block_number_bins):
    active_block_indices = (block_number_bins>1).nonzero()[0]
    #for idx in active_block_indices:
    #    events = [log for log in logs_full if int(log["blockNumber"],16) == idx]
    #    print(len(events))
    #    entities = [log["topics"][1] for log in events]
    #    print(entities)
    #    print("")
    return (active_block_indices,)


@app.cell
def __():
    # EVENT SCHEMA
    #{
    #    "address": "0x781c6d1f0eae6f1da1f604c6cdccdb8b76428ba7",
    #    "topics": [],
    #    "data": "0x",
    #    "blockHash": "0xee4fc6235cc56e0be316d10998b6319355aa1d1a2ff02197e9be2b602e39fe71",
    #    "blockNumber": "0x185c82a",
    #    "transactionHash": "0x173784b8484dc38e6e4e56aaa6d0699487814775a36be4ad65af4e97a2f41a68",
    #    "transactionIndex": "0x0",
    #    "logIndex": "0x2",
    #    "transactionLogIndex": "0x2",
    #    "removed": False
    #}
    # we only care about address, topics, data, and blockNumber
    return


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


if __name__ == "__main__":
    app.run()

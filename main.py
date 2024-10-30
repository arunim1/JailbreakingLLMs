from pair import pair
import asyncio
import pandas as pd
from dotdict import DotDict as dotdict
import wandb
import copy
from loggers import WandBLogger

async def main():
    args = dotdict()

    args.wandb = True
    args.quiet = True
    # const args 
    args.attack_model = "gpt-4o"
    args.target_model = "claude-3-haiku-20240307"
    args.judge_model = "gpt-4o"

    args.attack_max_n_tokens = 1000
    args.max_n_attack_attempts = 5
    args.target_max_n_tokens = 250
    args.judge_max_n_tokens = 100
    args.judge_temperature = 0

    args.n_streams = 20  # number of parallel threads
    args.keep_last_n = 3
    args.n_iterations = 3  # max depth of search

    if args.wandb:
        args.logger = WandBLogger(args, project_name = "baseline-pair-full", entity = "arunim_a", name = f"{args.attack_model}-vs-{args.target_model}")

    data = pd.read_csv("data/harmful_behaviors_custom.csv")

    tasks = []
    full_out = []
    # var args
    for i, row in data.iterrows():
        args.index = row["Original index"]
        args.goal = row["goal"]
        args.target_str = row["target"]
        args.category = row["category"]

        out, q2jb = await pair(args)
        full_out.append((args.index, out, q2jb))
    #     tasks.append(pair(args))
    # await asyncio.gather(*tasks)
    print(f"<main.py_output>{full_out}</main.py_output>")

    args.logger.logger.finish()



if __name__ == "__main__":
    asyncio.run(main())
"""
Generate training data based on conversations

Usage: python -m AIZone.data.generate_data --in-file sharegpt_gpt4.jsonl --tokenizer-name HF_REPO_NAME --out-dir .
"""

import argparse
import os
import gc
import random

import ray
import orjson
import pyarrow
from pyarrow import parquet


PAD_TOKEN_ID = 0


def _split(a, n):
    # Split list a to n chunks
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m): (i+1)*k+min(i+1, m)] for i in range(n)]


def truncate_trailing_zero_weighted(tokens, weights):
    non_zero_index = len(weights) - 1
    while non_zero_index >= 0 and weights[non_zero_index] == 0:
        non_zero_index -= 1

    return tokens[:non_zero_index + 1], weights[:non_zero_index + 1]


def add_single_conv(output, tokens, weights):
    # truncate trailing zero weighted tokens
    tokens, weights = truncate_trailing_zero_weighted(tokens, weights)
    if not tokens:
        return

    # labels
    length = len(tokens)
    labels = [(t if w != 0 else PAD_TOKEN_ID) for t, w in zip(tokens, weights)]

    # populate results
    results = {
        "total_length": length,

        "seqlens": [length],
        "nz_input_ids": tokens,
        "nz_position_ids": list(range(length)),

        "nz_shifted_label_ids":    labels[1:]  + [PAD_TOKEN_ID],
        "nz_shifted_loss_weights": weights[1:] + [0.0]
    }
    results["num_seqs"] = sum(results["nz_shifted_loss_weights"])

    for k, v in results.items():
        output[k].append(v)


@ray.remote
def convert_conversation_batch(model_type: str, model_path: str, batch: list, schema: pyarrow.Schema, per_sequence_loss: bool): # type ignore
    # Tokenization
    # من المفترض أن يتم الاستيراد من AIZone.config إذا كان موجوداً
    # from AIZone.config import MODEL_CONFIG_MAP, Conversation
    pass  # أضف هنا الكود المناسب إذا توفرت التعريفات المطلوبة


def generate_epoch(seed: int, model_type: str, model_path: str, in_filename: str, out_filename: str, per_sequence_loss: bool):
    # schema
    metadata = {
        "model_type": model_type
    }
    schema = [
        pyarrow.field("total_length", pyarrow.int32()),
        pyarrow.field("num_seqs", pyarrow.float32()),

        pyarrow.field(f"seqlens", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_input_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_position_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_label_ids", pyarrow.list_(pyarrow.int32())),
        pyarrow.field(f"nz_shifted_loss_weights", pyarrow.list_(pyarrow.float32()))
    ]

    schema = pyarrow.schema(schema, metadata={"metadata_json": orjson.dumps(metadata)})

    # Load data
    with open(in_filename, "rb") as f:
        batches = f.readlines()

        random.seed(seed)  # Randomized load balancing
        random.shuffle(batches)

        batches = _split(batches, int(ray.available_resources()["CPU"]))

    # launch remote workers
    handles = [convert_conversation_batch.remote(
        model_type=model_type,  # type: ignore
        model_path=model_path,
        batch=batch,
        schema=schema,
        per_sequence_loss=per_sequence_loss
    ) for batch in batches]

    # write
    parquet.write_table(pyarrow.concat_tables([ray.get(handle) for handle in handles]), out_filename)


def generate_dataset(model_type, model_path, in_prefix, out_prefix, per_sequence_loss, seed):
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())

    # Load epochs and tokenize
    epoch = 0
    while True:
        in_filename = f"{in_prefix}.{epoch}.jsonl"
        if not os.path.exists(in_filename):
            break

        out_filename = f"{out_prefix}.{epoch}.parquet"
        generate_epoch(
            seed=seed + epoch,
            model_type=model_type,
            model_path=model_path,
            in_filename=in_filename,
            out_filename=out_filename,
            per_sequence_loss=per_sequence_loss
        )
        gc.collect()

        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)

    parser.add_argument("--in-prefix", type=str, required=True)
    parser.add_argument("--out-prefix", type=str, required=True)

    parser.add_argument("--per-sequence-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(**vars(args))

import argparse
import datasets
import logging
from math import ceil
import os
from preprocessing import utils as preproc_utils

import tempfile
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt
import torch.distributed as dist
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    get_scheduler,
)
import torch_xla.distributed.parallel_loader as pl
from torch_xla.amp.syncfree import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from threading import Thread
from transformers.modeling_utils import xla_fsdp_wrap
from models.llama import get_wrapped_llama_from_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
os.environ["PT_XLA_DEBUG"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--num_cores", type=int)
parser.add_argument("--dataset", type=str)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--train_split", type=str, default="train")
parser.add_argument("--train_steps", type=int, default=1000)
parser.add_argument("--block_size", type=int, default=1024)
parser.add_argument("--enable_profiling", action="store_true", default=False)
parser.add_argument("--profile_steps", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--report_steps", type=int)
parser.add_argument("--logging_steps", type=int)
args = parser.parse_args()


def main(index):

    print(f"Ordinal: {xm.get_ordinal()}, Local Ordinal: {xm.get_local_ordinal()}, World Size: {xm.xrt_world_size()}")

    if args.enable_profiling:
        server = xp.start_server(9012)
        logger.info(f"Profiling server started: {str(server)}")

    def print_info(step, loss, tracker):
        loss_value = loss.item()
        xm.master_print(
            f"step: {step}, loss: {loss_value}, rate: {tracker.rate()}, global rate: {tracker.global_rate()}"
        )

    dev = xm.xla_device()

    # Tokenizer
    if args.tokenizer is None:
        args.tokenizer = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Training dataset
    dataset = datasets.load_dataset(args.dataset)
    column_names = list(dataset[args.train_split].features)
    preproc_func = preproc_utils.get_preprocessed_dataset(args.dataset)
    tokenized_dataset = dataset.map(
        lambda x: preproc_func(x, tokenizer),
        batched=True,
        num_proc=args.num_cores,
        remove_columns=column_names,
    )
    packed_dataset = tokenized_dataset.map(
        lambda x: preproc_utils.group_texts(x, block_size=args.block_size),
        batched=True,
        num_proc=args.num_cores,
    )
    train_dataset = packed_dataset.with_format("torch")

    # Data loader
    data_loader = DataLoader(train_dataset[args.train_split], batch_size=args.batch_size)
    data_loader = pl.MpDeviceLoader(data_loader, dev)

    # Model
    logging.info("loading model")
    config = LlamaConfig.from_pretrained(args.config)
    model = get_wrapped_llama_from_config(config)
    model.train()

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_epochs*ceil(len(train_dataset)/args.batch_size),
    )

    tracker = xm.RateTracker()

    # Training loop
    for epoch in range(0, args.num_epochs):
        with tqdm(data_loader) as tepoch:
            for step, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step() # do not reduce gradients on sharded params
                lr_scheduler.step()
                tracker.add(args.batch_size)

                if args.logging_steps is not None and step % args.logging_steps == 0:
                    xm.add_step_closure(print_info, (step, loss, tracker))

                if args.report_steps is not None and step % args.report_steps == 0:
                    xm.master_print(met.metrics_report())

                if args.enable_profiling:
                    if step % args.profile_steps == 0 and xm.is_master_ordinal():
                        logger.info("start profiling")
                        trace = lambda: xp.trace('127.0.0.1:9012', tempfile.mkdtemp(), 20000)
                        Thread(target=trace).start()

    # For full report that includes all metrics.
    xm.master_print(met.metrics_report())


if __name__ == "__main__":
    xmp.spawn(main, args=(), nprocs=args.num_cores)

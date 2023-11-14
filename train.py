import argparse
import logging
from math import ceil
import os
from preprocessing import utils as preproc_utils
from utils.weights import reshard_and_save_weights

import tempfile
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
from transformers import (
    AutoTokenizer,
    get_scheduler,
)
import torch_xla.distributed.parallel_loader as pl
from torch_xla.amp.syncfree import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from threading import Thread
from utils import (
    datasets as dataset_utils,
    models as model_utils,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"
# os.environ["PT_XLA_DEBUG"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--presharded_checkpoints", type=str)
parser.add_argument("--num_cores", type=int)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--dataset_config_name", type=str)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--tokenizer", type=str, required=True)
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

    logger.info(
        f"ordinal: {xm.get_ordinal()}, "
        f"local ordinal: {xm.get_local_ordinal()}, "
        f"world size: {xm.xrt_world_size()}"
    )

    if args.enable_profiling:
        server = xp.start_server(9012)
        logging.info(f"Profiling server started: {str(server)}")

    dev = xm.xla_device()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # training dataset
    dataset = dataset_utils.load_and_process_dataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        tokenizer=tokenizer,
        detect_columns_from_split=args.train_split,
        block_size=args.block_size,
    )

    # data loader
    data_loader = DataLoader(dataset[args.train_split], batch_size=args.batch_size)
    data_loader = pl.MpDeviceLoader(data_loader, dev)

    # model
    model = model_utils.load_model(args.config, args.presharded_checkpoints)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = args.num_epochs * ceil(
        len(dataset[args.train_split]) / args.batch_size
    )
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    def print_info(step, loss, tracker):
        loss_value = loss.item()
        xm.master_print(
            f"Step: {step}, Loss: {loss_value}, Rate: {tracker.rate()}, Global Rate: {tracker.global_rate()}"
        )

    # training loop
    tracker = xm.RateTracker()
    xm.master_print(model)
    for epoch in range(0, args.num_epochs):
        with tqdm(data_loader) as tepoch:
            for step, batch in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch+1}")
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()  # do not reduce gradients on sharded params
                lr_scheduler.step()
                tracker.add(args.batch_size)

                if args.logging_steps is not None and (
                    step % args.logging_steps == 0
                    or num_training_steps == step * (epoch + 1)
                ):
                    xm.add_step_closure(print_info, (step, loss, tracker))

                if args.report_steps is not None and (
                    step % args.report_steps == 0
                    or num_training_steps == step * (epoch + 1)
                ):
                    xm.master_print(met.metrics_report())

                if args.enable_profiling:
                    if step % args.profile_steps == 0 and xm.is_master_ordinal():
                        logger.info("start profiling")
                        trace = lambda: xp.trace(
                            "127.0.0.1:9012", tempfile.mkdtemp(), 20000
                        )
                        Thread(target=trace).start()

    # save and consolidate checkpoints
    model_utils.save_model(model, optimizer, args.output_dir)


if __name__ == "__main__":
    xmp.spawn(main, args=(), nprocs=args.num_cores)

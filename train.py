import argparse
import logging
from math import ceil

import tempfile
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
    weights as weight_utils,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)
parser.add_argument("--resharded_checkpoint_dir", type=str)
parser.add_argument("--huggingface_model_dir", type=str)
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
parser.add_argument("--enable_gradient_checkpointing", action="store_true", default=False)
parser.add_argument("--profile_steps", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--report_steps", type=int)
parser.add_argument("--logging_steps", type=int)
parser.add_argument("--save_steps", type=int)
parser.add_argument("--max_steps", type=int)
parser.add_argument("--enable_checkpoint_consolidation", action="store_true", default=False)
parser.add_argument("--reshard_checkpoint_on_master_only", action="store_true", default=False)
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

    # reshard model for checkpoint loading by layer
    if args.huggingface_model_dir:
        if not args.resharded_checkpoint_dir:
            raise ValueError("--resharded_checkpoint_dir must be set if --huggingface_model_dir is set!")
        if args.reshard_checkpoint_on_master_only:
            if xm.is_master_ordinal(local=False):
                logger.info("Resharding model checkpoint to allow FSDP wrapping per-layer on master ONLY")
                logger.info("Please ensure that --resharded_checkpoint_dir is readable from ALL workers for this to work!")
                weight_utils.reshard_and_save_weights(args.huggingface_model_dir, args.resharded_checkpoint_dir)
            xm.rendezvous("reshard_model")
        else:
            logging.info("Resharding model checkpoint to allow FSDP wrapping per-layer")
            weight_utils.reshard_and_save_weights(args.huggingface_model_dir, args.resharded_checkpoint_dir)

    # model
    model = model_utils.load_model(
        config_file_path=args.config,
        resharded_checkpoint_dir=args.resharded_checkpoint_dir,
        use_grad_checkpoint=args.enable_gradient_checkpointing,
    )

    if xm.is_master_ordinal(local=False):
        logger.debug(model)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_training_steps = args.num_epochs * ceil(
        len(dataset[args.train_split]) / args.batch_size
    )

    # learning rate scheduler
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
    global_step = 0
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
                        logger.info("Start profiling")
                        trace = lambda: xp.trace(
                            "127.0.0.1:9012", tempfile.mkdtemp(), 20000
                        )
                        Thread(target=trace).start()

                global_step += 1

                if args.save_steps and global_step % args.save_steps:
                    ckpt_name = f"ckpt_step_{global_step}"
                    logger.info(f"Saving checkpoint {ckpt_name}")
                    model_utils.checkpoint_model(
                        model=model,
                        optimizer=optimizer,
                        output_dir=args.output_dir,
                        ckpt_name=ckpt_name,
                    )
                
                if global_step == args.max_steps:
                    logger.info("Stopping training due to max_steps reached")
                    break

            if args.save_steps is None and args.max_steps is None:
                ckpt_name = f"ckpt_epoch_{epoch+1}"
                logger.info(f"Saving checkpoint {ckpt_name}")
                model_utils.checkpoint_model(
                    model=model,
                    optimizer=optimizer,
                    output_dir=args.output_dir,
                    ckpt_name=ckpt_name,
                )

    # save and consolidate checkpoints
    model_utils.save_model(
        model=model,
        optimizer=optimizer,
        output_dir=args.output_dir,
        consolidate_checkpoint=args.enable_checkpoint_consolidation,
    )


if __name__ == "__main__":
    xmp.spawn(main, args=(), nprocs=args.num_cores)

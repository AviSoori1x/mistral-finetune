import contextlib
import dataclasses
import datetime
import logging
import time
from typing import Optional, Protocol

import torch

logger = logging.getLogger("utils")


@dataclasses.dataclass
class TrainState:
    max_steps: int
    # total_samples: int
    step: int = 0
    elapsed_time: float = 0.0
    n_seen_tokens: int = 0
    n_seen_samples: int = 0
    this_step_time: float = 0.0
    begin_step_time: float = 0.0
    this_eval_runtime: Optional[float] = None
    this_eval_perplexity: Optional[float] = None
    this_eval_steps_per_second: Optional[float] = None
    this_eval_samples_per_second: Optional[float] = None
    this_eval_loss: Optional[float] = None
    grad_norm: Optional[float] = None


    def start_step(self):
        self.step += 1
        self.begin_step_time = time.time()

    def end_step(self, n_batch_tokens: int, n_batch_samples: int):
        self.this_step_time = time.time() - self.begin_step_time
        self.this_step_tokens = n_batch_tokens
        self.this_step_samples = n_batch_samples

        self.elapsed_time += self.this_step_time
        self.n_seen_tokens += self.this_step_tokens
        self.n_seen_samples += self.this_step_samples

        self.begin_step_time = time.time()

    # @property
    # def epochs_completed(self) -> float:
    #     return self.n_seen_samples / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def samples_per_second(self):
        return self.n_seen_samples / self.elapsed_time if self.elapsed_time > 0 else 0

    @property
    def wps(self):
        return self.this_step_tokens / self.this_step_time

    @property
    def avg_wps(self):
        return self.n_seen_tokens / self.elapsed_time
    

    @property
    def eta(self):
        steps_left = self.max_steps - self.step
        avg_time_per_step = self.elapsed_time / self.step

        return steps_left * avg_time_per_step
    
    #New
    @property
    def steps_per_second(self):
        return self.step / self.elapsed_time
    


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class Closable(Protocol):
    def close(self):
        pass


@contextlib.contextmanager
def logged_closing(thing: Closable, name: str):
    """
    Logging the closing to be sure something is not hanging at exit time
    """
    try:
        setattr(thing, "wrapped_by_closing", True)
        yield
    finally:
        logger.info(f"Closing: {name}")
        try:
            thing.close()
        except Exception:
            logger.error(f"Error while closing {name}!")
            raise
        logger.info(f"Closed: {name}")


def now_as_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# From https://www.adamcasson.com/posts/transformer-flops
def openai_flops_per_token(n_layers, n_heads, d_model, n_ctx, n_vocab, ff_ratio=4):
    """Open AI method for forward pass FLOPs counting of decoder-only Transformer
    """
    d_attn = d_model // n_heads
    d_ff = d_model * ff_ratio
 
    embeddings = 4 * d_model
    attn_qkv = 2 * n_layers * d_model * 3 * (d_attn * n_heads)
    attn_mask = 2 * n_layers * n_ctx * (d_attn * n_heads)
    attn_project = 2 * n_layers * (d_attn * n_heads) * d_model
    ff = 2 * n_layers * 2 * d_model * d_ff
    logits = 2 * d_model * n_vocab
 
    return embeddings + attn_qkv + attn_mask + attn_project + ff + logits

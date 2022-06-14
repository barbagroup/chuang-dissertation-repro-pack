#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Custom solvers.
"""
import logging as _logging
import pathlib as _pathlib
import time as _time
import torch as _torch
from lzma import open as _lzmaopen
from io import BytesIO as _BytesIO
from datetime import datetime as _datetime
from datetime import timezone as _timezone
from dataclasses import dataclass
from typing import Optional as _Optional
from termcolor import colored as _colored
from hydra.core.config_store import ConfigStore as _ConfigStore
from modulus.hydra.optimizer import OptimizerConf as _OptimizerConf
from modulus.hydra.utils import instantiate_agg as _instantiate_agg
from modulus.hydra.utils import instantiate_optim as _instantiate_optim
from modulus.hydra.utils import add_hydra_run_path as _add_hydra_run_path
from modulus.continuous.domain.domain import Domain as _Domain
from modulus.distributed.manager import DistributedManager as _DistributedManager
from omegaconf import DictConfig as _DictConfig


class TimerCuda:
    """A data holder for Cuda profiler.
    """

    def __init__(self):
        self.start_event = _torch.cuda.Event(enable_timing=True)
        self.end_event = _torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def elapsed_time(self):
        self.end_event.record()
        self.end_event.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event)  # in milliseconds
        return elapsed_time

    def reset(self):
        self.start_event.record()


class TimerWalltime:
    """A data holder for profiling w/ walltime.
    """

    def __init__(self):
        self.tbg = _time.time()
        self.ted = None

    def elapsed_time(self):
        self.ted = _time.time()
        elapsed_time = (self.ted - self.tbg) * 1.0e3  # in milliseconds
        return elapsed_time

    def reset(self):
        self.tbg = _time.time()


class SolverBase:
    """A simplified base solver class.
    """

    cfg = None
    network_dir = property(lambda self: self.cfg.network_dir)
    initialization_network_dir = property(lambda self: self.cfg.initialization_network_dir)
    max_steps = property(lambda self: self.cfg.training.max_steps)
    grad_agg_freq = property(lambda self: self.cfg.training.grad_agg_freq)
    save_network_freq = property(lambda self: self.cfg.training.save_network_freq)
    print_stats_freq = property(lambda self: self.cfg.training.print_stats_freq)
    summary_freq = property(lambda self: self.cfg.training.summary_freq)
    amp = property(lambda self: self.cfg.training.amp)
    save_filetypes = property(lambda self: self.cfg.save_filetypes)
    summary_histograms = property(lambda self: self.cfg.summary_histograms)

    def __init__(self, cfg: _DictConfig, domain: _Domain):

        # check if removed features are called
        assert not cfg.training.amp, f"AMP featrure was removed from {self.__class__}"

        # save the reference to the Config and Domain object
        self.cfg = cfg
        self.domain = domain

        # initialize step counter
        self.step = -1
        self.initial_step = -1

        # make logger and tensorboard writer
        self.log = _logging.getLogger(__name__)
        self.writer = _torch.utils.tensorboard.SummaryWriter(self.network_dir, purge_step=self.summary_freq+1)

        # Set distributed manager
        self.manager = _DistributedManager()

        # set device
        self.device = self.manager.device

        # create global model for restoring and saving
        self.saveable_models = self.domain.get_saveable_models()
        self.global_optimizer_model = self.domain.create_global_optimizer_model()

        # initialize optimizer
        self.optimizer = _instantiate_optim(self.cfg, self.global_optimizer_model)

        # initialize aggregator from hydra
        self.aggregator = _instantiate_agg(cfg, self.global_optimizer_model.parameters(), self.domain.get_num_losses())
        assert len(list(self.aggregator.parameters())) == 0, "Not yet supported."

        if self.cfg.jit:
            self.aggregator = _torch.jit.script(self.aggregator)

        # get the shape of each parameter group
        self.shapes = []
        for param in self.global_optimizer_model.parameters():
            self.shapes.append(param.shape)

        # make directory
        if self.manager.rank == 0:
            _pathlib.Path(self.network_dir).mkdir(exist_ok=True)

        # distributed barrier before starting the train loop
        if self.manager.distributed:
            _torch.distributed.barrier(device_ids=[self.manager.local_rank] if self.manager.cuda else None)

    def get_flattened_params(self):
        """Return the flattenend and concatenated parameters.
        """
        params = []
        with _torch.no_grad():
            for i, param in enumerate(self.global_optimizer_model.parameters()):
                assert self.shapes[i] == param.shape
                params.append(param.data.reshape(-1))
            params = _torch.concat(params)
        return params

    def set_flattened_params(self, x):
        """Given a ndarray of flattened and concatenated parameters, copy them into the model's parameters.
        """
        self.global_optimizer_model.zero_grad()
        beg = 0
        with _torch.no_grad():
            for i, param in enumerate(self.global_optimizer_model.parameters()):
                val = x[beg:beg+param.numel()].copy()
                val = _torch.tensor(val, dtype=param.dtype).to(self.device)  # in case x is not a torch.Tensor
                param.copy_(val.reshape(self.shapes[i]))
                beg += param.numel()

    def load_network(self):
        """Load network.
        """
        # attempt to restrore from initialization network dir
        if self.initialization_network_dir != "":
            for i_dir in self.initialization_network_dir.split(","):
                for model in self.saveable_models:
                    filename = _pathlib.Path(i_dir).joinpath(model.checkpoint_filename)
                    if filename.is_file():
                        try:
                            model.load(i_dir, map_location=self.device)
                            self.log.info(f"{_colored('Success loading model:', 'green')} {filename}")
                        except Exception:
                            self.log.error(f"{_colored('Fail loading model:', 'red')} {filename}")
                    else:
                        self.log.warning(f"model {model.checkpoint_filename} not found in {i_dir}")

        # attempt to restore optimizer
        filename = _pathlib.Path(self.network_dir).joinpath("optim_checkpoint.pth")
        if filename.is_file():
            try:
                checkpoint = _torch.load(filename, map_location=self.device)
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.aggregator.load_state_dict(checkpoint["aggregator_state_dict"])
                self.initial_step = checkpoint["step"]
                self.log.info(f"{_colored('Success loading optimizer:', 'green')} {filename}")
            except Exception:
                self.initial_step = 0
                self.log.info(f"{_colored('Fail loading optimizer:', 'red')} {filename}")
        else:
            self.initial_step = 0
            self.log.warning("optimizer checkpoint not found")

        # attempt to restore models
        for model in self.saveable_models:
            filename = _pathlib.Path(self.network_dir).joinpath(model.checkpoint_filename)
            if filename.is_file():
                try:
                    model.load(self.network_dir, map_location=self.device)
                    self.log.info(f"{_colored('Success loading model:', 'green')} {filename}")
                except Exception:
                    self.log.info(f"{_colored('Fail loading model:', 'red')} {filename}")
            else:
                self.log.warning("model " + model.checkpoint_filename + " not found")

        # logging initial step
        self.log.info(f"Training will start from step {self.initial_step}")

    def save_checkpoint(self):
        """Save a checkpoint.
        """

        # no need to save checkpoint
        if self.step % self.save_network_freq != 0:
            return

        msg = _colored("[step: {:10d}] saved checkpoint to {}", "green")

        if self.manager.rank == 0:
            # save models
            for model in self.saveable_models:
                model.save(self.network_dir)

            # save step, optimizer, aggregator, and scaler
            _torch.save(
                {
                    "step": self.step,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "aggregator_state_dict": self.aggregator.state_dict(),
                },
                self.network_dir+"/optim_checkpoint.pth",
            )

            self.log.info(msg.format(self.step, _add_hydra_run_path(self.network_dir)))

        # wait if necessarry
        if self.manager.distributed:
            _torch.distributed.barrier(device_ids=[self.manager.local_rank] if self.manager.cuda else None)

    def solve(self):
        """Training loop wrapper.
        """

        # load network, which will also set self.initial_step
        self.load_network()

        # solver implementation
        self._solve()

    def _solve(self):
        """Actual training loop.
        """
        raise NotImplementedError

    def _logging_info(self, loss: float):
        """A helper to log info to files.
        """

        msg = _colored("[step: {:10d}] saved {} results to {} in {:10.3e}s", "blue")

        # rank 0: write train / inference / validation datasets to tensorboard and file
        if self.manager.rank == 0:
            if self.step % self.cfg.training.rec_constraint_freq == 0:
                timer = TimerWalltime()
                self.domain.rec_constraints(self.network_dir)
                self.log.debug(msg.format(self.step, "constraint", self.network_dir, timer.elapsed_time()))

            if self.step % self.cfg.training.rec_validation_freq == 0 and bool(self.domain.validators):
                timer = TimerWalltime()
                self.domain.rec_validators(self.network_dir, self.writer, self.save_filetypes, self.step)
                self.log.debug(msg.format(self.step, "validation", self.network_dir, timer.elapsed_time()))

            if self.step % self.cfg.training.rec_inference_freq == 0 and bool(self.domain.inferencers):
                timer = TimerWalltime()
                self.domain.rec_inferencers(self.network_dir, self.writer, self.save_filetypes, self.step)
                self.log.debug(msg.format(self.step, "inferencer", self.network_dir, timer.elapsed_time()))

            if self.step % self.cfg.training.rec_monitor_freq == 0 and bool(self.domain.monitors):
                timer = TimerWalltime()
                self.domain.rec_monitors(self.network_dir, self.writer, self.step)
                self.log.debug(msg.format(self.step, "monitor", self.network_dir, timer.elapsed_time()))

        # wait for rank 0 to finish the job
        if self.manager.distributed:
            _torch.distributed.barrier(device_ids=[self.manager.local_rank] if self.manager.cuda else None)

    def _print_stdout_info(self, loss: float, timer):
        """Print some info to stdout.
        """

        if self.step % self.print_stats_freq != 0:
            return

        # string template
        msg = "[step: {:10d}] loss={:10.3e}, lr={:10.3e}, time/iter={:10.3e}"

        # synchronize and get end time
        elapsed_time = timer.elapsed_time() / self.print_stats_freq

        # Reduce loss across all GPUs
        if self.manager.distributed:

            # loss
            _torch.distributed.reduce(loss, 0, op=_torch.distributed.ReduceOp.SUM)
            loss = loss.cpu().detach().numpy() / self.manager.world_size

            # elapsed_time
            elapsed_time = _torch.tensor(elapsed_time).to(self.device)
            _torch.distributed.reduce(elapsed_time, 0, op=_torch.distributed.ReduceOp.SUM)
            elapsed_time = elapsed_time.cpu().numpy() / self.manager.world_size

        try:
            _last_lr = [group['lr'] for group in self.optimizer.param_groups][0]
        except AttributeError as err:
            if "'NoneType' object has no attribute" in str(err):
                _last_lr = float("NaN")
            else:
                raise

        if self.manager.rank == 0:
            self.log.info(msg.format(self.step, loss, _last_lr, elapsed_time))

        # reset timer
        timer.reset()


class LBFGSSolver(SolverBase):
    """Using L-BFGS + SWA.

    Notes
    -----
    1. No AMP support.
    2. No porfiler support.
    3. No tensorboard support.
    4. No minibatch.
    """
    # nullify the property from the parent class
    max_steps = None

    def __init__(self, cfg: _DictConfig, domain: _Domain):
        super().__init__(cfg, domain)
        assert cfg.optimizer._target_ == "torch.optim.LBFGS", "This solver can only be used with L-BFGS optimizer"

        self.swa_model = _torch.optim.swa_utils.AveragedModel(self.global_optimizer_model)
        self.swa_start = cfg.batch_size.nbatches

        # only train for 2 epochs; the 1st epoch using L-BFGS, the second using L-BFGS + SWA
        self.max_steps = 2 * self.cfg.batch_size.nbatches

        if self.cfg.training.max_steps != self.max_steps:
            self.log.warn(_colored(
                f"max iter is set to {self.cfg.training.max_steps}, "
                f"but we will train only up to {self.max_steps}",
                "red"
            ))

    def objective_function(self):
        """Given a set of parameters, compute the loss and gradients wrt. parameters.
        """

        # ZERO the gradients
        self.optimizer.zero_grad()

        # calculate total loss
        loss = 0.0
        losses = self.domain.compute_losses(self.step)
        loss = self.aggregator(losses, self.step)
        loss.backward()

        # for distributed optimization, gradients are averaged across ranks, so should the loss
        if self.manager.distributed:
            with _torch.no_grad():
                _torch.distributed.all_reduce(loss, _torch.distributed.ReduceOp.SUM)
                loss /= self.manager.world_size

        return loss

    def _solve(self):
        """Actual training loop.
        """
        # initialize timer
        timer = TimerCuda() if self.manager.cuda else TimerWalltime()

        # save the initial states
        self.step = self.initial_step
        loss = self.objective_function()
        self._logging_info(loss)
        self.save_checkpoint()
        self._print_stdout_info(loss, timer)

        # self.step means how many times the model is trained at the end of each iteration, so starts from 1
        for self.step in range(self.initial_step+1, self.max_steps+1):

            # use a new L-BFGS solver every time to discard states silently stored in the optimizer
            self.optimizer = _instantiate_optim(self.cfg, self.global_optimizer_model)

            # train against the current data batch with L-BFGS
            loss = self.optimizer.step(self.objective_function)
            assert not _torch.isnan(loss), "NaN for loss"

            # SWA moving average
            if self.step >= self.swa_start:
                self.swa_model.update_parameters(self.global_optimizer_model)

            # save states
            self._logging_info(loss)
            self.save_checkpoint()
            self._print_stdout_info(loss, timer)

        # training complete
        else:
            if self.manager.rank == 0:
                self.log.info(f"[step: {self.step:10d}] finished training!")

    def load_network(self):
        """Load network.
        """

        # attempt to swa model
        filename = _pathlib.Path(self.network_dir).joinpath("swa-model.pth")
        if filename.is_file():
            try:
                self.swa_model.load_state_dict(_torch.load(filename, map_location=self.device))
                self.log.info(f"{_colored('Success loading swa-model:', 'green')} {filename}")
            except Exception:
                self.log.info(f"{_colored('Fail loading swa-model:', 'red')} {filename}")
        else:
            self.log.warning("swa-model checkpoint not found")

        # load usual things
        SolverBase.load_network(self)

    def save_checkpoint(self):
        """Save checkpoints.
        """

        # no need to save checkpoint
        if self.step % self.save_network_freq != 0:
            return

        # save usual things
        SolverBase.save_checkpoint(self)

        # save swa model
        if self.step >= self.swa_start and self.manager.rank == 0:
            filename = _pathlib.Path(self.network_dir).joinpath("swa-model.pth")
            _torch.save(self.swa_model.state_dict(), filename)
            msg = _colored("[step: {:10d}] saved swa-model checkpoint to {}", "green")
            self.log.info(msg.format(self.step, filename))

        # wait if necessarry
        if self.manager.distributed:
            _torch.distributed.barrier(device_ids=[self.manager.local_rank] if self.manager.cuda else None)

    def _logging_info(self, loss: float):
        """A helper to log info to files.
        """

        # doing regular things
        super()._logging_info(loss)

        # outputing trasient swa model following inferecing frequencies
        msg = _colored("[step: {:10d}] saved swa-model snapshot to {} in {} ms", "green")
        if self.manager.rank == 0:
            if self.step >= self.swa_start and self.step % self.cfg.training.rec_inference_freq == 0:

                # filename
                filename = _pathlib.Path(self.network_dir).joinpath("inferencers")
                filename.mkdir(exist_ok=True)
                filename = filename.joinpath(f"swa-model-{self.step:07d}.pth")

                # timestamp
                time = _datetime.utcnow().replace(tzinfo=_timezone.utc).isoformat()

                # timer to measure performance
                timer = TimerWalltime()

                # in-memory dump
                with _BytesIO() as mem:
                    _torch.jit.save(_torch.jit.script(self.swa_model.module), mem)
                    mem.seek(0)

                    # write memory content to file
                    with _lzmaopen(filename, "wb") as fobj:
                        _torch.save({
                            "step": self.step, "time": time, "model": mem.read(),
                            "n_averaged": self.swa_model.n_averaged
                        }, fobj)

                self.log.info(msg.format(self.step, filename, timer.elapsed_time()))

        # wait for rank 0 to finish the job
        if self.manager.distributed:
            _torch.distributed.barrier(device_ids=[self.manager.local_rank] if self.manager.cuda else None)


@dataclass
class LBFGSConf(_OptimizerConf):
    _target_: str = "torch.optim.LBFGS"
    lr: float = 1.0
    max_iter: int = 20
    max_eval: _Optional[float] = None
    tolerance_grad: float = 1e-07
    tolerance_change: float = 1e-07
    history_size: int = 50
    line_search_fn: _Optional[str] = "strong_wolfe"


def register_optimizer_configs() -> None:
    """Register for custom optimizers.
    """
    _ConfigStore.instance().store(
        group="optimizer",
        name="lbfgs",
        node=LBFGSConf,
    )

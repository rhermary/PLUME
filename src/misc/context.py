"""Utility context managers."""

from typing import Generator, Callable, Optional, List, Union, Tuple, Any
from unittest.mock import patch
import contextlib

from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard._pytorch_graph import (
    parse,
)
import torch


@contextlib.contextmanager
# Typing from https://peps.python.org/pep-0484/#annotating-generator-functions-and-coroutines
def patch_over_check_trace() -> Generator[Callable, None, None]:
    """
    Allowing `check_trace` argument for `torch.trace` to be `False` to avoid
    errors with non-deterministic nodes.
    The patching is messy but does not interfere with anything; with this
    approach, the redefinition of two functions are required for the changes to
    work.

    Yields:
        Callable: patch over the
        `torch.utils.tensorboard.writer.SummaryWriter.add_graph` method
    """

    @contextlib.contextmanager
    def _set_model_to_eval(model: torch.nn.Module) -> Any:
        """A context manager to temporarily set the training mode of ``model`` to eval.

        Taken from a newest version of `PyTorch`:
        https://github.com/pytorch/pytorch/blob/4582ceb2c4a7316f09afd471a97d910347d21f01/torch/utils/tensorboard/_pytorch_graph.py#L367

        Args:
            model (torch.nn.Module): the model to trace

        Yields:
            None

        """
        if not isinstance(model, torch.jit.ScriptFunction):
            originally_training = model.training
            model.train(False)
            try:
                yield
            finally:
                model.train(originally_training)
        else:
            # Do nothing for ScriptFunction
            try:
                yield
            finally:
                pass

    def graph(
        model: torch.nn.Module,
        args: Optional[Union[torch.Tensor, List[torch.Tensor]]],
        verbose: bool = False,
        use_strict_trace: bool = True,
    ) -> Tuple[GraphDef, RunMetadata]:
        """
        Redefining `torch.utils.tensorboard.writer._pytorch_graph.graph`

        Raises:
            RuntimeError: if an error occurred during tracing.`

        """
        # pylint: disable=protected-access,invalid-name
        with _set_model_to_eval(model):
            try:
                trace = torch.jit.trace(
                    model, args, check_trace=False, strict=use_strict_trace
                )
                graph = trace.graph
                torch._C._jit_pass_inline(graph)
            except RuntimeError as e:
                print(e)
                print("Error occurs, No graph saved")
                raise e

        if verbose:
            print(graph)
        list_of_nodes = parse(graph, trace, args)
        stepstats = RunMetadata(
            step_stats=StepStats(
                dev_stats=[DeviceStepStats(device="/device:CPU:0")]
            )
        )
        return (
            GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)),
            stepstats,
        )

    def add_graph(
        self: SummaryWriter,
        model: torch.nn.Module,
        input_to_model: Optional[
            Union[torch.Tensor, List[torch.Tensor]]
        ] = None,
        verbose: bool = False,
        use_strict_trace: bool = True,
    ) -> None:
        # pylint: disable=protected-access
        """
        Redefining `torch.utils.tensorboard.writer.SummaryWriter.add_graph`

        Raises:
            NotImplementedError:  if the forward method of the given model is
                is not implemented.

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_graph")
        if not hasattr(model, "forward"):
            raise NotImplementedError("Behavior for Caffe not patched.")

        # A valid PyTorch model should have a 'forward' method
        self._get_file_writer().add_graph(
            graph(model, input_to_model, verbose, use_strict_trace)
        )

    with patch(
        "torch.utils.tensorboard.writer.SummaryWriter.add_graph", add_graph
    ) as context:
        yield context

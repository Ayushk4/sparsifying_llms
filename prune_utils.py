"""Iteratively Prune a model based on the magnitude of weights.

Pytorch only supports x86/ARM for quantization.


"""
import collections # pylint: disable=syntax-error
import copy

from typing import Union, List # pylint: disable=syntax-error
import torch
from torch.nn.utils import prune as torch_prune # pylint: disable=wrong-import-position


def _get_attr_from_custom_pattern(layer, attr_pattern: Union[str, List[str]], idx=0):
    """Get the attribute of the layer that matches the regex.

    Only regex "*" pattern are supported.
    Here

    For example:
        model.decoder.layers.*.self_attn.in_proj_weight:
        Here * is a wildcard that matches all the keys for one attr.
        Assumes that model.decoder.layers is iterable.
    """
    if isinstance(attr_pattern, str):
        attr_pattern = attr_pattern.split(".")
        idx = 0
    if len(attr_pattern) <= idx:
        assert idx == len(attr_pattern), "Invalid index {idx} for pattern {attr_pattern}"
        return [layer]
    if "*" in attr_pattern[idx]:
        assert attr_pattern[idx] == "*", "Only * is supported as a wildcard for entire layer."
        assert isinstance(layer, collections.abc.Iterable), "Layer is not iterable."
        return [param for iterate in layer
                for param in _get_attr_from_custom_pattern(iterate, attr_pattern, idx + 1)]
    return _get_attr_from_custom_pattern(
        getattr(layer, attr_pattern[idx]), attr_pattern, idx + 1)

def log_prune_statistics(parameters_to_prune):
    """Logs the prune statistics."""
    total_num_pruned = 0
    total_num_params = 0
    for param_layer, _ in parameters_to_prune:
        this_layer_pruned = float(torch.sum(param_layer.weight == 0)) # pylint: disable=no-member
        this_num_params = float(param_layer.weight.nelement())
        total_num_pruned += this_layer_pruned
        total_num_params += this_num_params

        sparsity_percent = round(100. * this_layer_pruned / this_num_params, 3)
        print(f"Sparsity in {param_layer}: {sparsity_percent}%")

    print(f"Global sparsity: {round(100. * total_num_pruned / total_num_params, 3)}%")

class DummyPruner:
    """Dummy pruner that does nothing."""

    def __init__(self, *args, **kwargs): # pylint: disable=unused-argument
        self.times_pruned = 0

    def done_pruning(self):
        """Whether pruning is done."""
        return False

    def should_prune(self, *args, **kwargs): # pylint: disable=unused-argument
        """Dummy function to be overwritten by the actual pruner."""
        return False

    def prune_model(self, *args, **kwargs): # pylint: disable=unused-argument
        """Dummy function to be overwritten by the actual pruner."""
        return

    def prune_if_its_time(self, *args, **kwargs): # pylint: disable=unused-argument
        """Dummy function to be overwritten by the actual pruner."""
        return

    def update_iteration_count_and_performance(self, *args, **kwargs): # pylint: disable=unused-argument
        """Dummy function to be overwritten by the actual pruner."""
        return

class Pruner(DummyPruner):
    """Class to Prune a model based on the pruning recipe.

    prune_recipe = {"layer_patterns": [...],
                    # "method": "L1Unstructured",
                    "amounts": [...],
                    "breakpoints": [...],
                    "scorer": [...],
                    "num_datapoints_score": 1000,
                   }
    Scorer has values from "magnitude", "firstOrderOptimizer", "secondOrder"

    """

    def __init__(self, prune_recipe, iter_steps=0):
        super().__init__()
        self.prune_recipe = prune_recipe
        self.layer_patterns = self.prune_recipe.layer_patterns
        self.breakpoints = self.prune_recipe.breakpoints
        self.amounts = self.prune_recipe.amounts
        # self.method = self.prune_recipe.method
        self.scorer = self.prune_recipe.scorer
        self.num_datapoints_score = self.prune_recipe.num_datapoints_score

        assert len(self.amounts) == len(self.breakpoints), (
            "Amounts and breakpoints should be of same length")
        assert len(self.amounts) == len(self.scorer), (
            "Amounts and scorer should be of same length")
        for scorer in self.scorer:
            assert scorer in ["magnitude", "firstOrderOptimizer", "secondOrder"], (
                "Scorer should be one of magnitude, firstOrderOptimizer, secondOrder")

        self.iter_steps = iter_steps

        self.times_pruned = 0
        while self.breakpoints[self.times_pruned] < self.iter_steps:
            self.times_pruned += 1
        self.model_initialized = False

    def init_prune(self, model):
        """Initializes the model for pruning, if it already has zero weights, then its pruned."""
        parameters_to_prune = [(layer, 'weight') for layer_pattern in self.layer_patterns
                               for layer in _get_attr_from_custom_pattern(model, layer_pattern)]
        num_prune_params = [torch.sum(layer.weight.data == 0) # pylint: disable=no-member
                            for layer, _ in parameters_to_prune]
        fraction_init_prune = sum(num_prune_params) / sum(
            [layer.weight.data.nelement() for layer, _ in parameters_to_prune]) # pylint: disable=no-member
        fraction_init_prune = fraction_init_prune.item()

        if self.times_pruned != 0:
            assert abs(fraction_init_prune - self.breakpoints[self.times_pruned - 1]) < 0.01, (
                "Pruning recipe is not consistent with the model. "
                "The model is already pruned to a different level.")
            print("Initial sparsity:", fraction_init_prune)
        assert 0 <= fraction_init_prune <= 1, "Initial sparsity should be between 0 and 1."

        if fraction_init_prune > 0.01:
            torch_prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch_prune.L1Unstructured,
                amount=fraction_init_prune)
        self.model_initialized = True
        return model

    def update_iter_steps(self, iter_steps):
        """Updates the iteration steps."""
        self.iter_steps = iter_steps

    def is_it_time_to_prune(self):
        """Decides whether to prune the model or not and accordingly prunes."""
        if self.times_pruned >= len(self.breakpoints):
            return False
        if self.iter_steps < self.breakpoints[self.times_pruned]:
            return False
        assert self.model_initialized, "Model not initialized for pruning."
        return True

    def prune_model(self, model, dataloader, optimizer, *args, **kwargs):
        """Prunes the model if called."""
        if not self.is_it_time_to_prune():
            return

        print(type(model))
        parameters_to_prune = [(layer, "weight") for layer_pattern in self.layer_patterns
                               for layer in _get_attr_from_custom_pattern(model, layer_pattern)]
        parameters_to_prune = tuple(parameters_to_prune)
        print([(type(x[0]), x[1]) for x in parameters_to_prune])
        print("Sparsity before pruning:")
        log_prune_statistics(parameters_to_prune)

        if self.scorer[self.times_pruned] == "magnitude":
            # This will be handled by the default pruning method
            pass
        elif self.scorer[self.times_pruned] == "firstOrderOptimizer":
            # We use the optimizer's state to estimate the gradient
            # and compute sensitivity by multiplying the gradient with the weight.
            # First, we make a copy of model and set it weights equal to the sensitivity.
            # Then use torch_prune.global_unstructured to prune the copy of model.
            # Then use the mask of the copy of model to mask the real model.
            sensitivity_model = copy.deepcopy(model)
            sensitivity_model = sensitivity_model.to(next(model.parameters()).device)
            copy_parameters_to_prune = [
                (layer, "weight") for layer_pattern in self.layer_patterns
                for layer in _get_attr_from_custom_pattern(sensitivity_model, layer_pattern)]
            for layer, _ in copy_parameters_to_prune:
                layer.weight.data = layer.weight.data * optimizer.state[layer]["exp_avg"]
            torch_prune.global_unstructured(
                copy_parameters_to_prune,
                pruning_method=torch_prune.L1Unstructured,
                amount=self.amounts[self.times_pruned])
            for layer, _ in parameters_to_prune:
                torch_prune.remove(layer, 'weight')
            # Now we mask the real model to lower the connections masked in the copy.
            for (layer, _), (copy_layer, _) in zip(parameters_to_prune, copy_parameters_to_prune):
                layer.weight.data = copy_layer.weight.data
        elif self.scorer[self.times_pruned] == "secondOrder":
            raise NotImplementedError("Second order pruning not implemented yet.")
            model.train() # pylint: disable=unreachable
            idx = 0
            for _ in dataloader:
                idx += 1
                if idx > self.num_datapoints_score:
                    break
        torch_prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch_prune.L1Unstructured,
            amount=self.amounts[self.times_pruned])

        print("Sparsity after pruning:")
        log_prune_statistics(parameters_to_prune)

        self.times_pruned += 1

    def remove_prune(self, model):
        """Removes the pruning from the model."""
        for layer_pattern in self.layer_patterns:
            for layer in _get_attr_from_custom_pattern(model, layer_pattern):
                torch_prune.remove(layer, 'weight')
        return model

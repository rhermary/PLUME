"""MLFlow wrappers to ease analysis process."""

from typing import List, Iterator
from contextlib import contextmanager

import pandas as pd
import mlflow
from mlflow.entities import Experiment as MLFExperiment


class Runs(pd.DataFrame):
    """ "Wraps a list of runs to add callable methods."""

    def filter_attributes_(self, attribute: str) -> List[str]:
        """
        Filter the runs columns names to only those part of the `attribute`
        section.
        """
        return [
            column_name
            for column_name in self.columns
            if attribute in column_name
        ]

    def metrics(self) -> List[str]:
        """Returns the names of the values in metrics section."""
        return self.filter_attributes_("metrics.")

    def params(self) -> List[str]:
        """Returns the names of the values in the params section."""
        return self.filter_attributes_("params.")

    def with_normal_class(self, normal_class: int) -> "Runs":
        """Returns the runs only with the given normal class."""
        return Runs(self[self["params.normal_class"] == str(normal_class)])

    def filter_columns(self, columns: List[str]) -> "Runs":
        """Filter the columns to only those asked."""
        return Runs(self[columns])

    def format_log_paths(self) -> "Runs":
        """Clean the saved paths."""
        runs = self
        columns = list(
            set(["params.TB_folder", "params.best_model_path"])
            & set(runs.columns)
        )
        for column in columns:
            runs[column] = runs[column].apply(
                lambda x: x.split("lightning_logs/")[1]
            )

        return runs


class Experiment:
    """Wrapper around MLFlow experiments."""

    def __init__(self, experiment_name: str, root_dir: str = "") -> None:
        self.experiment_name = experiment_name
        self.root_dir = root_dir

        with self.stable_tracking_context():
            self.experiment_ = mlflow.get_experiment_by_name(experiment_name)
            self.runs_ = Runs(
                mlflow.search_runs(self.experiment_.experiment_id)
            )

    @property
    def experiment(self) -> MLFExperiment:
        """The loaded experiment."""
        return self.experiment_

    @property
    def runs(self) -> Runs:
        """The experiment runs."""
        return self.runs_

    @contextmanager
    def stable_tracking_context(self) -> Iterator[None]:
        """Used not to mess up mlflow's environment variables.

        Yields:
            None

        """
        previous_uri = mlflow.get_tracking_uri()
        try:
            mlflow.set_tracking_uri(self.root_dir)
            yield
        finally:
            mlflow.set_tracking_uri(previous_uri)

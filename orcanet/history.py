import numpy as np
import os
from orcanet.utilities.visualization import plot_history
from orcanet.in_out import get_subfolder


class HistoryHandler:
    """
    For reading and plotting data from summary and train log files.

    """
    def __init__(self, main_folder):
        self.main_folder = main_folder
        self.summary_filename = "summary.txt"

    @property
    def summary_file(self):
        main_folder = self.main_folder
        if not main_folder[-1] == "/":
            main_folder += "/"
        return main_folder + self.summary_filename

    @property
    def train_log_folder(self):
        return get_subfolder(self.main_folder, "train_log")

    def plot_metric(self, metric_name, **kwargs):
        """
        Plot the training and validation history of a metric.

        This will read out data from the summary file, as well as
        all training log files, and plot them over the epoch.

        Parameters
        ----------
        metric_name : str
            Name of the metric to be plotted over the epoch. This name is what
            was written in the head line of the summary.txt file, except without
            the train_ or val_ prefix.
        kwargs
            Keyword arguments for the plot_history function.

        """
        summary_data = self.get_summary_data()
        full_train_data = self.get_train_data()
        summary_label = "val_" + metric_name

        if metric_name not in full_train_data.dtype.names:
            raise ValueError(
                "Train log metric name {} unknown, must be one of {}".format(
                    metric_name, self.get_metrics()))
        if summary_label not in summary_data.dtype.names:
            raise ValueError(
                "Summary metric name {} unknown, must be one of {}".format(
                    summary_label, self.get_metrics()))

        if summary_data["Epoch"].shape == (0,):
            # When no lines are present in the summary.txt file.
            val_data = None
        else:
            val_data = [summary_data["Epoch"], summary_data[summary_label]]

        if full_train_data["Batch_float"].shape == (0,):
            # When no lines are present
            raise ValueError("Can not make summary plot: Training log files "
                             "contain no data!")
        else:
            train_data = [full_train_data["Batch_float"],
                          full_train_data[metric_name]]

        # if no validation has been done yet
        if np.all(np.isnan(val_data)[1]):
            val_data = None

        if "y_label" not in kwargs:
            kwargs["y_label"] = metric_name

        plot_history(train_data, val_data, **kwargs)

    def plot_lr(self, **kwargs):
        """
        Plot the learning rate over the epochs.

        Parameters
        ----------
        kwargs
            Keyword arguments for the plot_history function.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The plot.

        """
        summary_data = self.get_summary_data()

        epoch = summary_data["Epoch"]
        lr = summary_data["LR"]
        # plot learning rate like val data (connected dots)
        val_data = (epoch, lr)

        if "y_label" not in kwargs:
            kwargs["y_label"] = "Learning rate"
        if "legend" not in kwargs:
            kwargs["legend"] = False

        plot_history(train_data=None, val_data=val_data, logy=True,
                     y_lims=None, **kwargs)

    def get_metrics(self):
        """
        Get the name of the metrics from the first line in the summary file.

        This will be the actual name of the metric, i.e. "loss" and not
        "train_loss" or "val_loss".

        Returns
        -------
        all_metrics : List
            A list of the metrics.

        """
        summary_data = self.get_summary_data()
        all_metrics = []
        for keyword in summary_data.dtype.names:
            if keyword == "Epoch" or keyword == "LR":
                continue
            if "train_" in keyword:
                keyword = keyword.split("train_")[-1]
            else:
                keyword = keyword.split("val_")[-1]
            if keyword not in all_metrics:
                all_metrics.append(keyword)
        return all_metrics

    def get_summary_data(self):
        """
        Read out the summary file in the output folder.

        Returns
        -------
        summary_data : ndarray
            Numpy structured array with the column names as datatypes.
            Its shape is the number of lines with data.

        """
        summary_data = self._load_txt(self.summary_file)
        if summary_data.shape == ():
            # When only one line is present
            summary_data = summary_data.reshape(1,)
        return summary_data

    def get_best_epoch_info(self, metric="val_loss", mini=True):
        """
        Get the line in the summary file where the given metric is best, i.e.
        either minimal  or maximal.

        Parameters
        ----------
        metric : str
            Which metric to look up.
        mini : bool
            If true, look up the minimum. Else the maximum.

        Raises
        ------
        ValueError
            If there is no best line (e.g. no validation has been done).

        """
        summary_data = self.get_summary_data()
        metric_data = summary_data[metric]

        if all(np.isnan(metric_data)):
            raise ValueError("Can not find best epoch in summary.txt")

        if mini:
            opt_loss = np.nanmin(metric_data)
        else:
            opt_loss = np.nanmax(metric_data)

        best_index = np.where(metric_data == opt_loss)[0]
        # if multiple epochs with same loss, take first
        best_index = min(best_index)
        best_line = summary_data[best_index]
        return best_line

    def get_best_epoch_fileno(self, metric="val_loss", mini=True):
        """
        Get the epoch/fileno tuple where the given metric is smallest.
        """
        best_line = self.get_best_epoch_info(metric=metric, mini=mini)
        best_epoch_float = best_line["Epoch"]
        epoch, fileno = self._transform_epoch(best_epoch_float)
        return epoch, fileno

    def _transform_epoch(self, epoch_float):
        """
        Transfrom the epoch_float read from a file to a tuple.

        (By just counting the number of lines in the given epoch).
        TODO Hacky, (epoch, filno) should probably be written to the summary.
        """
        summary_data = self.get_summary_data()

        epoch = int(np.floor(epoch_float - 1e-8))
        # all lines in the epoch of epoch_float
        indices = np.where(np.floor(summary_data["Epoch"] - 1e-8) == epoch)[0]
        lines = summary_data[indices]
        fileno = int(np.where(lines["Epoch"] == epoch_float)[0]) + 1
        epoch += 1
        return epoch, fileno

    def get_column_names(self):
        """
        Get the str in the first line in each column.

        Returns
        -------
        tuple : column_names
            The names in the same order as they appear in the summary.txt.

        """
        summary_data = self.get_summary_data()
        column_names = summary_data.dtype.names
        return column_names

    def get_train_data(self):
        """
        Read out all training logfiles in the output folder.

        Read out the data from the summary.txt file, and from all training
        log files in the train_log folder, which is in the same directory
        as the summary.txt file.

        Returns
        -------
        summary_data : numpy.ndarray
            Structured array containing the data from the summary.txt file.
            Its shape is the number of lines with data.

        """
        # list of all files in the train_log folder of this model
        files = os.listdir(self.train_log_folder)
        train_file_data = []
        for file in files:
            if not (file.startswith("log_epoch_") and file.endswith(".txt")):
                continue
            filepath = os.path.join(self.train_log_folder, file)
            if os.path.getsize(filepath) == 0:
                continue

            # file is sth like "log_epoch_1_file_2.txt", extract epoch & fileno:
            epoch, file_no = [int(file.split(".")[0].split("_")[i]) for i in [2, 4]]
            file_data = self._load_txt(filepath)
            train_file_data.append([[epoch, file_no], file_data])

        if len(train_file_data) == 0:
            raise OSError(f"No train files found in {self.train_log_folder}!")

        # sort so that earlier epochs come first
        train_file_data.sort()
        full_train_data = train_file_data[0][1]
        for [epoch, file_no], file_data in train_file_data[1:]:
            full_train_data = np.append(full_train_data, file_data)

        if full_train_data.shape == ():
            # When only one line is present
            full_train_data = full_train_data.reshape(1,)
        return full_train_data

    def get_state(self):
        """
        Get the state of a training.

        For every line in the summary logfile, get a dict with the epoch
        as a float, and is_trained and is_validated bools.

        Returns
        -------
        state_dicts : List
            List of dicts.

        """
        summary_data = self.get_summary_data()
        state_dicts = []
        names = summary_data.dtype.names

        for line in summary_data:
            val_losses, train_losses = {}, {}
            for name in names:
                if name.startswith("val_"):
                    val_losses[name] = line[name]
                elif name.startswith("train_"):
                    train_losses[name] = line[name]
                elif name not in ["Epoch", "LR"]:
                    raise NameError(
                        "Invalid summary file: Invalid column name {}: must be "
                        "either Epoch, LR, or start with val_ or train_".format(name))
            # if theres any not-nan entry, consider it completed
            is_trained = any(~np.isnan(tuple(train_losses.values())))
            is_validated = any(~np.isnan(tuple(val_losses.values())))

            state_dicts.append({
                "epoch": line["Epoch"],
                "is_trained": is_trained,
                "is_validated": is_validated, })

        return state_dicts

    @staticmethod
    def _load_txt(filepath):
        # TODO suboptimal that n/a gets replaced by np.nan, because this
        #  means that legitamte, not availble cells can not be distinguished
        #  from failed 'nan' metric values produced by training.
        file_data = np.genfromtxt(
            filepath,
            names=True,
            delimiter="|",
            autostrip=True,
            comments="--",
            missing_values="n/a",
            filling_values=np.nan
        )
        # replace inf with nan so it can be plotted
        for column_name in file_data.dtype.names:
            x = file_data[column_name]
            x[np.isinf(x)] = np.nan
        return file_data

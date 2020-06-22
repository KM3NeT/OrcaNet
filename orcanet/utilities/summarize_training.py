import matplotlib.pyplot as plt
import argparse
import warnings
from orcanet.history import HistoryHandler
from orcanet.utilities.visualization import TrainValPlotter


class Summarizer:
    """
    Summarize one or more trainings by giving their orcanet folder(s).

    - Plot the training and validation curves in a single plot and show them
    - Print info about the best and worst epochs

    Parameters
    ----------
    folders : str or List, optional
        Path to a orcanet folder, or to multiple folder as a list.
        [default: CWD].
    metric : str or List, optional
        The metric to plot [default: 'loss'].
        If its a list: Same length as folders. Plot a different metric for
        each folder.
    smooth : int, optional
        Apply gaussian blur to the train curve with given sigma.
    labels : List, optional
        Labels for each folder.
    noplot : bool
        Dont plot the train/val curves [default: False].
    width : float
        Scaling of the width of the curves and the marker size [default: 1].

    """
    def __init__(self, folders,
                 metric="loss",
                 smooth=None,
                 labels=None,
                 noplot=False,
                 width=1.):
        if not folders:
            folders = ["./"]
        elif isinstance(folders, str):
            folders = [folders]
        self.folders = folders

        if isinstance(metric, str):
            metric = [metric]
        if len(metric) == 1:
            self.metrics = metric * len(self.folders)
            self._unique_metrics = False
        else:
            if len(metric) != len(folders):
                raise ValueError("Need to give exactly one metric per folder!")
            self.metrics = metric
            self._unique_metrics = True

        if labels is None:
            self.labels = self.folders
        else:
            self.labels = labels

        self.smooth = smooth
        self.noplot = noplot
        self.width = width
        self._tvp = None

    def summarize(self):
        if not self.noplot:
            self._tvp = TrainValPlotter()

        min_stats, max_stats = [], []
        print("Reading stats of {} trainings...".format(len(self.folders)))
        for folder_no in range(len(self.folders)):
            try:
                min_stat, max_stat = self._summarize_folder(folder_no)
                if min_stat is not None:
                    min_stats.append(min_stat)
                if max_stat is not None:
                    max_stats.append(max_stat)
            except Exception as e:
                print(
                    f"Warning: Can not summarize {self.folders[folder_no]}"
                    f", skipping... ({e})"
                )

        if self._unique_metrics:
            column_title, y_label = ("combined metrics",) * 2
        else:
            column_title, y_label = self._full_metrics[0], self._metric_names[0]

        if len(min_stats) > 0:
            min_stats.sort()
            print("\nMinimum\n-------")
            print("{}   \t{}\t{}\t{}".format(" ", "Epoch", column_title, "name"))
            for i, stat in enumerate(min_stats, 1):
                print("{} | \t{}\t{}\t{}".format(i, stat[2], stat[0], stat[1]))

        if len(max_stats) > 0:
            max_stats.sort(reverse=True)
            print("\nMaximum\n-------")
            print("{}   \t{}\t{}\t{}".format(" ", "Epoch", column_title, "name"))
            for i, stat in enumerate(max_stats, 1):
                print("{} | \t{}\t{}\t{}".format(i, stat[2], stat[0], stat[1]))

        if not self.noplot:
            self._tvp.apply_layout(
                x_label="Epoch",
                y_label=y_label,
                grid=True,
                legend=True,
            )
            plt.show()

    @property
    def _metric_names(self):
        """ E.g. [loss, ...] """
        metric_names = []
        for metric in self.metrics:
            if metric.startswith("train_"):
                m = metric[6:]
            elif metric.startswith("val_"):
                m = metric[4:]
            else:
                m = metric
            metric_names.append(m)
        return metric_names

    @property
    def _full_metrics(self):
        """ E.g. [val_loss, ...] """
        full_metrics = []
        for metric in self.metrics:
            if metric.startswith("train_") or metric.startswith("val_"):
                full_metrics.append(metric)
            else:
                full_metrics.append("val_" + metric)
        return full_metrics

    def _summarize_folder(self, folder_no):
        label = self.labels[folder_no]
        folder = self.folders[folder_no]

        hist = HistoryHandler(folder)
        val_data, min_stat, max_stat = None, None, None
        # read data from summary file
        try:
            smry_met_name = self._full_metrics[folder_no]
            max_line = hist.get_best_epoch_info(
                metric=smry_met_name, mini=False)
            min_line = hist.get_best_epoch_info(
                metric=smry_met_name, mini=True)
            min_stat = [min_line[smry_met_name], label, min_line["Epoch"]]
            max_stat = [max_line[smry_met_name], label, max_line["Epoch"]]

            summary_data = hist.get_summary_data()
            val_data = [summary_data["Epoch"],
                        summary_data[self._full_metrics[folder_no]]]

        except OSError:
            print(f"Warning: No summary file found for {folder}")

        except ValueError as e:
            print(f"Error reading summary file {hist.summary_file} ({e})")

        # read data from training files
        full_train_data = hist.get_train_data()
        train_data = [full_train_data["Batch_float"],
                      full_train_data[self._metric_names[folder_no]]]

        if not self.noplot:
            if len(self.labels) == 1:
                train_label, val_label = "training", "validation"
            elif val_data is None:
                train_label, val_label = label, None
            else:
                train_label, val_label = None, label

            self._tvp.plot_curves(train_data=train_data,
                                  val_data=val_data,
                                  train_label=train_label,
                                  val_label=val_label,
                                  smooth_sigma=self.smooth,
                                  tlw=0.5*self.width,
                                  vlw=0.5*self.width,
                                  vms=3*self.width**0.5)
        return min_stat, max_stat

    def summarize_dirs(self):
        """
        Get the best and worst epochs of all given folders as a dict.

        Returns
        -------
        minima : dict
            Keys : Name of folder.
            Values : [Epoch, metric] of where the metric is lowest.
        maxima : dict
            As above, but for where the metric is highest.

        """
        minima, maxima = {}, {}
        for folder_no, folder in enumerate(self.folders):
            hist = HistoryHandler(folder)
            smry_met_name = self._full_metrics[folder_no]
            try:
                max_line = hist.get_best_epoch_info(metric=smry_met_name,
                                                    mini=False)
                min_line = hist.get_best_epoch_info(metric=smry_met_name,
                                                    mini=True)
            except OSError as e:
                warnings.warn(str(e))
                continue

            minima[folder] = [min_line["Epoch"], min_line[smry_met_name]]
            maxima[folder] = [max_line["Epoch"], max_line[smry_met_name]]
        return minima, maxima


def main():
    parser = argparse.ArgumentParser(
        description=str(Summarizer.__doc__),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('folders', type=str, nargs='*')
    parser.add_argument('--metric', type=str, nargs="*")
    parser.add_argument('--smooth', nargs="?", type=int)
    parser.add_argument('--width', nargs="?", type=float)
    parser.add_argument('--labels', nargs="*", type=str)
    parser.add_argument('--noplot', action="store_true")
    args = vars(parser.parse_args())
    for key in list(args.keys()):
        if args[key] is None:
            args.pop(key)

    Summarizer(**args).summarize()


if __name__ == '__main__':
    main()

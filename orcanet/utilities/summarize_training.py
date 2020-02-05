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

    Attributes
    ----------
    folders : str or List, optional
        Path to a orcanet folder, or to multiple folder as a list.
        Default: CWD.
    metric : str
        The metric to plot. Default: loss.
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
        self.folders = folders
        self.metric = metric
        self.smooth = smooth
        self.labels = labels
        self.noplot = noplot
        self.width = width
        self._tvp = None

    def summarize(self):
        if not self.noplot:
            self._tvp = TrainValPlotter()

        min_stats, max_stats = [], []
        print("Reading stats of {} trainings...".format(len(self._folders)))
        for folder_no in range(len(self._folders)):
            try:
                min_stat, max_stat = self._summarize_folder(folder_no)
                if min_stat is not None:
                    min_stats.append(min_stat)
                if max_stat is not None:
                    max_stats.append(max_stat)
            except Exception as e:
                print(
                    f"Warning: Can not summarize {self._folders[folder_no]}"
                    f", skipping... ({e})"
                )
        if len(min_stats) > 0:
            min_stats.sort()
            print("\nMinimum\n-------")
            print("{}   \t{}\t{}\t{}".format(" ", "Epoch", self._full_metric, "name"))
            for i, stat in enumerate(min_stats, 1):
                print("{} | \t{}\t{}\t{}".format(i, stat[2], stat[0], stat[1]))

        if len(max_stats) > 0:
            max_stats.sort(reverse=True)
            print("\nMaximum\n-------")
            print("{}   \t{}\t{}\t{}".format(" ", "Epoch", self._full_metric, "name"))
            for i, stat in enumerate(max_stats, 1):
                print("{} | \t{}\t{}\t{}".format(i, stat[2], stat[0], stat[1]))

        if not self.noplot:
            self._tvp.apply_layout(
                x_label="Epoch",
                y_label=self._metric_name,
                grid=True,
                legend=True,
            )
            plt.show()

    @property
    def _metric_name(self):
        """ E.g. loss """
        if self.metric.startswith("train_"):
            metric = self.metric[6:]
        elif self.metric.startswith("val_"):
            metric = self.metric[4:]
        else:
            metric = self.metric
        return metric

    @property
    def _full_metric(self):
        """ E.g. val_loss """
        if not (self.metric.startswith("train_") or
                self.metric.startswith("val_")):
            full_metric = "val_" + self.metric
        else:
            full_metric = self.metric
        return full_metric

    @property
    def _folders(self):
        """ Get a list of folders. """
        if not self.folders:
            folders = "./"
        else:
            folders = self.folders

        if isinstance(folders, str):
            folders = [folders]
        return folders

    @property
    def _labels(self):
        """ Get a list of labels. """
        if self.labels is None:
            return self._folders
        else:
            return self.labels

    def _summarize_folder(self, folder_no):
        label = self._labels[folder_no]
        folder = self._folders[folder_no]

        hist = HistoryHandler(folder)
        val_data, min_stat, max_stat = None, None, None
        # read data from summary file
        try:
            smry_met_name = self._full_metric
            max_line = hist.get_best_epoch_info(
                metric=smry_met_name, mini=False)
            min_line = hist.get_best_epoch_info(
                metric=smry_met_name, mini=True)
            min_stat = [min_line[smry_met_name], label, min_line["Epoch"]]
            max_stat = [max_line[smry_met_name], label, max_line["Epoch"]]

            summary_data = hist.get_summary_data()
            val_data = [summary_data["Epoch"],
                        summary_data[self._full_metric]]

        except OSError:
            print(f"Warning: No summary file found for {folder}")

        except ValueError as e:
            print(f"Error reading summary file {hist.summary_file} ({e})")

        # read data from training files
        full_train_data = hist.get_train_data()
        train_data = [full_train_data["Batch_float"],
                      full_train_data[self._metric_name]]

        if not self.noplot:
            if len(self._labels) == 1:
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
        for folder in self._folders:
            hist = HistoryHandler(folder)
            smry_met_name = self._full_metric
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
    parser.add_argument('-metric', type=str, nargs="?")
    parser.add_argument('-smooth', nargs="?", type=int)
    parser.add_argument('-width', nargs="?", type=float)
    parser.add_argument('-labels', nargs="*", type=str)
    parser.add_argument('-noplot', action="store_true")
    args = vars(parser.parse_args())
    for key in list(args.keys()):
        if args[key] is None:
            args.pop(key)

    Summarizer(**args).summarize()


if __name__ == '__main__':
    main()

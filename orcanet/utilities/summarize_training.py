import matplotlib.pyplot as plt
import argparse
import warnings
from orcanet.history import HistoryHandler
from orcanet.utilities.visualization import TrainValPlotter


class Summarizer:
    def __init__(self, folders,
                 metric="loss",
                 ksize=10,
                 labels=None,
                 no_plot=False):
        """
        Summarize one or more trainings.

        - Plot the training and validation curves in a single plot and show them
        - Print info about the best and worst epochs

        Parameters
        ----------
        folders : str or List, optional
            Path to a folder, or to multiple folder as a list.
            Default: CWD.
        metric : str
            The metric to plot. Default: loss.
        ksize : int, optional
            Size of the moving average window for smoothing the train line.
        labels : List, optional
            Labels for each folder.
        no_plot : bool
            Dont plot the train/val curves.

        """
        self.folders = folders
        self.metric = metric
        self.ksize = ksize
        self.labels = labels
        self.no_plot = no_plot
        self._tvp = None

    def __call__(self):
        if not self.no_plot:
            self._tvp = TrainValPlotter()

        min_stats, max_stats = [], []
        print("Reading stats of {} trainings...".format(len(self._folders)))
        for folder_no in range(len(self._folders)):
            try:
                min_stat, max_stat = self._summarize_folder(folder_no)
                min_stats.append(min_stat)
                max_stats.append(max_stat)
            except OSError:
                warnings.warn("Can not summarize {}, skipping..."
                              "".format(self._folders[folder_no]))

        min_stats.sort()
        print("\nMinimum\n-------")
        print("{}   \t{}\t{}\t{}".format(" ", "Epoch", self._full_metric, "name"))
        for i, stat in enumerate(min_stats, 1):
            print("{} | \t{}\t{}\t{}".format(i, stat[2], stat[0], stat[1]))

        max_stats.sort(reverse=True)
        print("\nMaximum\n-------")
        print("{}   \t{}\t{}\t{}".format(" ", "Epoch", self._full_metric, "name"))
        for i, stat in enumerate(max_stats, 1):
            print("{} | \t{}\t{}\t{}".format(i, stat[2], stat[0], stat[1]))

        if not self.no_plot:
            self._tvp.apply_layout(x_label="Epoch",
                                   y_label=self._metric_name,
                                   grid=True,
                                   legend=True)
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
        if len(self._labels) == 1:
            train_label, val_label = "training", "validation"
        else:
            train_label, val_label = None, label

        hist = HistoryHandler(folder)
        summary_data = hist.get_summary_data()
        full_train_data = hist.get_train_data()

        train_data = [full_train_data["Batch_float"],
                      full_train_data[self._metric_name]]
        val_data = [summary_data["Epoch"],
                    summary_data[self._full_metric]]

        smry_met_name = self._full_metric
        max_line = hist.get_best_epoch_info(metric=smry_met_name,
                                            mini=False)
        min_line = hist.get_best_epoch_info(metric=smry_met_name, mini=True)

        min_stat = [min_line[smry_met_name], label, min_line["Epoch"]]
        max_stat = [max_line[smry_met_name], label, max_line["Epoch"]]

        if not self.no_plot:
            self._tvp.plot_curves(train_data=train_data,
                                  val_data=val_data,
                                  train_label=train_label,
                                  val_label=val_label,
                                  train_smooth_ksize=self.ksize)
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
        description=Summarizer.__init__.__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('folder', type=str, nargs='*', default=None)
    parser.add_argument('-metric', type=str, nargs="?", default="loss")
    parser.add_argument('-ksize', nargs="?", type=int, default=None)
    parser.add_argument('-labels', nargs="*", type=str, default=None)
    parser.add_argument('-no_plot', action="store_true")

    args = parser.parse_args()
    Summarizer(folders=args.folder,
               metric=args.metric,
               ksize=args.ksize,
               labels=args.labels,
               no_plot=args.no_plot)()


if __name__ == '__main__':
    main()

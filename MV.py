import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix as confusion_matrix_metric

from yellowbrick.style import find_text_color
from yellowbrick.style.palettes import color_sequence
from yellowbrick.utils import div_safe

from sklearn.metrics import auc, roc_curve
from yellowbrick.style.palettes import LINE_COLOR

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics._classification import _check_targets
from yellowbrick.draw import bar_stack
from yellowbrick.classifier.base import ClassificationScoreVisualizer
from yellowbrick.exceptions import ModelError, YellowbrickValueError, NotFitted

class MetricsVisualizer():
    def __init__(self, model, test_data, y_true, classes, name, path_to_save="./"):
        self.model = model
        self.test_data = test_data
        y_pred = model.predict(test_data)
        if len(y_true.shape) > 1:
            self.y_true = np.argmax(y_true, axis=1)
            self.y_pred = np.argmax(y_pred, axis=1)
        else:
            self.y_true = y_true
            self.y_pred = y_pred
        self.classes = classes
        self.name = name
        self.path_to_save = path_to_save
        self.cmap = color_sequence("YlOrRd")
        self.cmap.set_over(color="w")
        self.cmap.set_under(color="#2a7d4f")
        self._edgecolors = []
        self.fontsize = None

    def ClassificationReportViz(self, support=True):
        displayed_scores = [key for key in ("precision", "recall", "f1", "support")]

        results = precision_recall_fscore_support(self.y_true, self.y_pred)
        scores = map(lambda s: dict(zip(self.classes, s)), results)
        scores_ = dict(zip(tuple(displayed_scores), scores))

        if not support:
            displayed_scores.remove("support")
            scores_.pop("support")

        # Create display grid
        cr_display = np.zeros((len(self.classes), len(displayed_scores)))

        # For each class row, append columns for precision, recall, f1, and support
        for idx, cls in enumerate(self.classes):
            for jdx, metric in enumerate(displayed_scores):
                cr_display[idx, jdx] = scores_[metric][cls]

        # Set up the dimensions of the pcolormesh
        # NOTE: pcolormesh accepts grids that are (N+1,M+1)
        X, Y = (
            np.arange(len(self.classes) + 1),
            np.arange(len(displayed_scores) + 1),
        )

        fig, ax = plt.subplots(ncols=1, nrows=1)

        ax.set_ylim(bottom=0, top=cr_display.shape[0])
        ax.set_xlim(left=0, right=cr_display.shape[1])

        # Set data labels in the grid, enumerating over class, metric pairs
        # NOTE: X and Y are one element longer than the classification report
        # so skip the last element to label the grid correctly.
        for x in X[:-1]:
            for y in Y[:-1]:

                # Extract the value and the text label
                value = cr_display[x, y]
                svalue = "{:0.3f}".format(value)
                
                if y == 3:
                    value = cr_display[x, y]
                    svalue = "{:0.0f}".format(value)
                        
                # Determine the grid and text colors
                base_color = self.cmap(value)
                text_color = find_text_color(base_color)

                # Add the label to the middle of the grid
                cx, cy = x + 0.5, y + 0.5
                ax.text(cy, cx, svalue, va="center", ha="center", color=text_color)

        # Draw the heatmap with colors bounded by the min and max of the grid
        # NOTE: I do not understand why this is Y, X instead of X, Y it works
        # in this order but raises an exception with the other order.
        g = ax.pcolormesh(
            Y, X, cr_display, vmin=0, vmax=1, cmap=self.cmap, edgecolor="w"
        )

        # Add the color bar
        plt.colorbar(g, ax=ax)  # TODO: Could use fig now

        # Set the title of the classifiation report
        ax.set_title("Classification Report for {}".format(self.name))
        
        # Set the tick marks appropriately
        ax.set_xticks(np.arange(len(displayed_scores)) + 0.5)
        ax.set_yticks(np.arange(len(self.classes)) + 0.5)

        ax.set_xticklabels(displayed_scores, rotation=45)
        ax.set_yticklabels(self.classes)

        fig.tight_layout()
        fig.savefig(self.path_to_save+"/ClassificationReport_"+self.name+".pdf")
        # Return the axes being drawn on
        return ax

    def ConfusionMatrixViz(self, percent=True):
        """
        Renders the classification report; must be called after score.
        """
        labels = [0,1]
        confusion_matrix_ = confusion_matrix_metric(self.y_true, self.y_pred, labels=labels)
        class_counts_ = dict(zip(*np.unique(self.y_true, return_counts=True)))

        # Make array of only the classes actually being used.
        # Needed because sklearn confusion_matrix only returns counts for
        # selected classes but percent should be calculated on all classes
        selected_class_counts = []
        for c in labels:
            try:
                selected_class_counts.append(class_counts_[c])
            except KeyError:
                selected_class_counts.append(0)
        class_counts_ = np.array(selected_class_counts)

        # Perform display related manipulations on the confusion matrix data
        cm_display = confusion_matrix_

        # Convert confusion matrix to percent of each row, i.e. the
        # predicted as a percent of true in each class.
        if percent is True:
            # Note: div_safe function returns 0 instead of NAN.
            cm_display = div_safe(
                confusion_matrix_, class_counts_.reshape(-1, 1)
            )
            cm_display = np.round(cm_display * 100, decimals=0)

        # Y axis should be sorted top to bottom in pcolormesh
        cm_display = cm_display[::-1, ::]

        # Set up the dimensions of the pcolormesh
        n_classes = len(self.classes)
        X, Y = np.arange(n_classes + 1), np.arange(n_classes + 1)

        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.set_ylim(bottom=0, top=cm_display.shape[0])
        ax.set_xlim(left=0, right=cm_display.shape[1])

        # Fetch the grid labels from the classes in correct order; set ticks.
        xticklabels = self.classes
        yticklabels = self.classes[::-1]
        ticks = np.arange(n_classes) + 0.5

        ax.set(xticks=ticks, yticks=ticks)
        ax.set_xticklabels(
            xticklabels, rotation="vertical", fontsize=self.fontsize
        )
        ax.set_yticklabels(yticklabels, fontsize=self.fontsize)

        # Set data labels in the grid enumerating over all x,y class pairs.
        # NOTE: X and Y are one element longer than the confusion matrix, so
        # skip the last element in the enumeration to label grids.
        for x in X[:-1]:
            for y in Y[:-1]:

                # Extract the value and the text label
                value = cm_display[x, y]
                svalue = "{:0.0f}".format(value)
                if percent:
                    svalue += "%"

                # Determine the grid and text colors
                base_color = self.cmap(value / cm_display.max())
                text_color = find_text_color(base_color)

                # Make zero values more subtle
                if cm_display[x, y] == 0:
                    text_color = "0.75"

                # Add the label to the middle of the grid
                cx, cy = x + 0.5, y + 0.5
                ax.text(
                    cy,
                    cx,
                    svalue,
                    va="center",
                    ha="center",
                    color=text_color,
                    fontsize=self.fontsize,
                )

                # Add a dark line on the grid with the diagonal. Note that the
                # tick labels have already been reversed.
                lc = "k" if xticklabels[x] == yticklabels[y] else "w"
                self._edgecolors.append(lc)

        # Draw the heatmap with colors bounded by vmin,vmax
        vmin = 0.00001
        vmax = 99.999 if percent is True else cm_display.max()
        ax.pcolormesh(
            X,
            Y,
            cm_display,
            vmin=vmin,
            vmax=vmax,
            edgecolor=self._edgecolors,
            cmap=self.cmap,
            linewidth="0.01",
        )
        ax.set_title("Confusion Matrix for {}".format(self.name))
        ax.set_ylabel("True Class")
        ax.set_xlabel("Predicted Class")

        # Call tight layout to maximize readability
        fig.tight_layout()
        fig.savefig(self.path_to_save+"/ConfusionMatrix_"+self.name+".pdf")
        # Return the axes being drawn on
        return ax
    
    def ROCAUCViz(self):
        # Target Type Constants
        BINARY = "binary"
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()
        if len(self.y_pred.shape) == 2 and self.y_pred.shape[1] == 2:
                        self.fpr[BINARY], self.tpr[BINARY], _ = roc_curve(self.y_true, self.y_pred[:, 1])
        else:
            # decision_function returns array of shape (n,), so plot it directly
            self.fpr[BINARY], self.tpr[BINARY], _ = roc_curve(self.y_true, self.y_pred)
        self.roc_auc[BINARY] = auc(self.fpr[BINARY], self.tpr[BINARY])
        fig, ax = plt.subplots(ncols=1, nrows=1)
        ax.plot(self.fpr[BINARY],self.tpr[BINARY], label="ROC for binary decision, AUC = {:0.3f}".format(self.roc_auc[BINARY]))
        # Plot the line of no discrimination to compare the curve to.
        ax.plot([0, 1], [0, 1], linestyle=":", c=LINE_COLOR)
        # Set the title and add the legend
        ax.set_title("ROC Curves for {}".format(self.name))
        ax.legend(loc="lower right", frameon=True)
        # Set the limits for the ROC/AUC (always between 0 and 1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        # Set x and y axis labels
        ax.set_ylabel("True Positive Rate")
        ax.set_xlabel("False Positive Rate")
        fig.savefig(self.path_to_save+"/ROCAUC_"+self.name+".pdf")
        return ax
    
    def ClassPredictionErrorViz(self):
        y_type, y_true, y_pred = _check_targets(self.y_true, self.y_pred)
        if y_type not in ("binary", "multiclass"):
            raise YellowbrickValueError("{} is not supported".format(y_type))
        # Get the indices of the unique labels
        indices = unique_labels(self.y_true, self.y_pred)
        labels = self.classes
        predictions_ = np.array([
                [(self.y_pred[self.y_true == label_t] == label_p).sum() for label_p in indices]
                for label_t in indices
        ])
        fig, ax = plt.subplots(ncols=1, nrows=1)
        legend_kws = {"bbox_to_anchor": (1.04, 0.5), "loc": "center left"}
        bar_stack(predictions_,ax,labels=list(self.classes),ticks=self.classes,legend_kws=legend_kws,)
        # Set the title
        ax.set_title("Class Prediction Error for {}".format(self.name))
        # Set the axes labels
        ax.set_xlabel("Actual Class")
        ax.set_ylabel("Number of Predicted Class")
        # Compute the ceiling for the y limit
        cmax = max([sum(predictions) for predictions in predictions_])
        ax.set_ylim(0, cmax + cmax * 0.1)
        # Ensure the legend fits on the figure
        fig.tight_layout(rect=[0, 0, 0.90, 1])
        fig.savefig(self.path_to_save+"/ClassPredictionError_"+self.name+".pdf")
        return ax
    
"""
This module provides functionality for doing dimensional stacking of
uncertain factors in order to reveal patterns in the values for a single
outcome of interests. It is inspired by the work reported `here <https://www.onepetro.org/conference-paper/SPE-174774-MS>`_
with one deviation.

Rather than using association rules to identify the
uncertain factors to use, this code uses random forest based feature scoring
instead. It is also possible to use the code provided here in combination
with any other feature scoring or factor prioritization technique instead, or
by simply selecting uncertain factors in some other manner.


"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import functools as ft
import itertools as it
import operator as op
import seaborn as sns
from typing import Union, Tuple, List, Dict, Set

from . import feature_scoring
from ..util import get_module_logger

# Created on Nov 13, 2015
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ["create_pivot_plot"]
_logger = get_module_logger(__name__)


def discretize(data, nbins=3, with_labels=False):
    """ Discretize the data, using the number of bins specified.

    Parameters
    ----------
    data : DataFrame
    nbins : int, optional
            the number of bins to use (default is 3)
    with_labels : bool, optional

    Returns
    -------
    discretized
        the discretized data frame

    note:: nbins is currently a constant for all float and integer columns.
           Categorical data is not discretized. If the number of integers is
           lower than the number of bins, the integer variable is also not
           discretized.


    """
    discretized = data.copy()

    for i, entry in enumerate(data.dtypes):
        column = data.columns[i]
        column_data = data[column]
        n = nbins

        if entry.name == "category":
            n_unique = column_data.unique().shape[0]
            n = n_unique
            column_data = column_data.cat.rename_categories(
                [x for x in range(1, n + 1)]
            )
            indices = column_data

        else:
            if issubclass(entry.type, np.integer):
                n_unique = column_data.unique().shape[0]
                if n_unique <= n:
                    n = n_unique

            if with_labels:
                indices = pd.cut(column_data, n, precision=2, retbins=True)[0]
            else:
                indices = pd.cut(
                    column_data, n, retbins=False, labels=False, precision=2
                )

        discretized[column] = indices

    return discretized


def dim_ratios(axis, figsize):
    """Get the proportions of the figure taken up by each axes

    adapted from seaborn
    """
    figdim = figsize[axis]
    # Get resizing proportion of this figure for the dendrogram and
    # colorbar, so only the heatmap gets bigger but the dendrogram stays
    # the same size.
    dendrogram = min(2.0 / figdim, 0.2)

    # add the colorbar
    colorbar_width = 0.8 * dendrogram
    colorbar_height = 0.2 * dendrogram
    if axis == 0:
        ratios = [colorbar_width, colorbar_height]
    else:
        ratios = [colorbar_height, colorbar_width]

    # Add the ratio for the heatmap itself
    ratios += [0.8]

    return ratios


def plot_line(ax, axis, i, length, lw):
    """
    Helper function for plotting lines separating bins in the hierarchical index
    
    Parameters
    ----------
    ax
        axis object
    axis : int
        0 for horizontal axis, 1 for vertical axis
    i : Union[float, int]
        position along the axis
    lw : Union[float, int]
        line width
    length : Union[float, int]
        length of the line (orthogonal to the axis)

    """

    if axis == 0:
        ax.plot([i, i], [length, 1], lw=lw, color="grey")
    else:
        ax.plot([length, 1], [i, i], lw=lw, color="grey")


def plot_category(ax, axis, i, label, pos, level):
    """
    helper function for plotting labels
    
    Parameters
    ----------
    ax
        axis object
    axis
        axis number (0, 1)
    i
        ??
    label
        text to draw
    pos
        ???
    level
        dataframe index level

    Returns
    -------

    """

    if isinstance(label, float):
        label = f"{label:g}"
    else:
        label = f"{label}"

    if axis == 0:
        # horizontal axis
        ax.text(
            x=i,
            y=pos,
            s=label,
            ha="center",
            va="center",
            rotation="vertical"
            if (level > 0) & (len(str(label)) > 0)
            else "horizontal",
            fontsize=8,
        )

    else:
        # vertical axis
        ax.text(
            x=pos,
            y=i,
            s=label,
            ha="center",
            va="center",
            rotation="vertical"
            if (level == 0) & (len(str(label)) > 0)
            else "horizontal",
            fontsize=8,
        )


def plot_index(
    ax, ax_plot, axis, index, plot_labels=True, plot_cats=True, hide_lines=False
):
    """helper function for visualizing the hierarchical index

    Parameters
    ----------

    ax : Axes instance
         the axes on which to plot the hierarchical index
    ax_plot : Axes instance
         the axes on which the table itself is displayed
    axis : int {0, 1}
        indicates whether we are plotting rows or columns
    plot_labels : bool, optional
        if true, also plot names of uncertain factors
    plot_cats : bool, options
        if true, plot category names for uncertain factors

    """

    for entry in ["bottom", "top", "right", "left"]:
        ax.spines["bottom"].set_color("grey")

    ####################################################
    # axis labels
    ####################################################
    if axis == 0:
        # draw vertical axis labels
        names = index.names
        ax.spines["top"].set_color("white")
        ax.spines["top"].set_linewidth(1.0)
        ax.spines["bottom"].set_color("white")
        ax.spines["bottom"].set_linewidth(1.0)

        ax.invert_yaxis()
        ax.yaxis.tick_right()
        ax.set_xticks([])

        if plot_labels:
            tick_locs = np.linspace(
                1 / (2 * len(names)), 1 - 1 / (2 * len(names)), len(names)
            )
            ax.set_yticks(tick_locs)
            ax.set_yticklabels(names)
        else:
            ax.set_yticks([])

    else:
        # draw horizontal axis labels
        index = index[::-1]
        names = index.names
        ax.set_yticks([])

        ax.spines["left"].set_color("white")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["right"].set_color("white")
        ax.spines["right"].set_linewidth(1.0)

        if plot_labels:
            tick_locs = np.linspace(
                1 / (2 * len(names)), 1 - 1 / (2 * len(names)), len(names)
            )
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(names, rotation="vertical")
        else:
            ax.set_xticks([])

    ####################################################
    # axis ticks
    ####################################################
    try:
        # number of index levels
        nr_levels = len(index.levels)

        # pd.FrozenList of pd.Index of unique index level values
        levels = index.levels

        # np.array of hierarchical tuples tuples
        indices = index.values

    except AttributeError:
        nr_levels = 1
        levels = [index.values.tolist()]
        indices = list(zip(index.values))

    if axis == 1:
        # plotting ticks on the vertical axis, putting origin to top left
        indices = indices[::-1]

    multiplicity = [1] + list(it.accumulate(map(len, levels), op.mul))

    for i, level in enumerate(levels):
        for j, label in enumerate(
            it.chain.from_iterable(it.repeat(level, multiplicity[i]))
        ):
            if plot_cats:
                plot_category(
                    ax=ax,
                    axis=axis,
                    i=j * len(index) / multiplicity[i + 1]
                    + 1 / 2 / multiplicity[i + 1] * len(index),
                    pos=1 / (2 * nr_levels) + i / nr_levels,
                    label=label,
                    level=i,
                )

            if j % len(level):
                length = i / nr_levels
                plot_line(
                    ax,
                    axis,
                    j * len(index) / multiplicity[i + 1],
                    length=length,
                    lw=1 - length,
                )

            if not hide_lines:
                if axis:
                    ax_plot.axhline(i, c="w", lw=lw)
                else:
                    ax_plot.axvline(i, c="w", lw=lw)


def plot_pivot_table(
    table, plot_labels=True, plot_cats=True, figsize=(10, 10), cmap="viridis", **kwargs
):
    """ visualize a pivot table using colors

    Parameters
    ----------
    table : Pandas DataFrame
    plot_labels : bool, optional
                 if true, display uncertain factor names
    plot_cats : bool, optional
                 if true, display category labels for each uncertain factor
    fig_size : tuple of 2 ints, optional
               size of the figure to create
    cmap : matplotlib colormap name or object, optional
           default is viridis (requires matplotlib 1.5 or higher)
    kwargs : other keyword arguments
             All other keyword arguments are passed to ax.pcolormesh.

    Returns
    -------
    Figure

    """

    hide_lines = kwargs.pop("hide_lines", False)

    with sns.axes_style("white"):

        fig = plt.figure(figsize=figsize)

        width_ratios = dim_ratios(figsize=figsize, axis=1)
        height_ratios = dim_ratios(figsize=figsize, axis=0)

        gs = mpl.gridspec.GridSpec(
            3,
            3,
            wspace=0.01,
            hspace=0.01,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
        )

        ax_plot = fig.add_subplot(gs[2, 2])
        ax_rows = fig.add_subplot(gs[2, 0:2], facecolor="white")
        ax_cols = fig.add_subplot(gs[0:2, 2], facecolor="white")
        cax = fig.add_subplot(gs[0, 0])

        # actual plotting
        plot_data = table.values
        # sns.heatmap(plot_data, ax=ax_plot, cbar_ax=cax, cmap=cmap,
        #             vmin=0, vmax=10, **kwargs)
        sns.heatmap(plot_data, ax=ax_plot, cbar_ax=cax, cmap=cmap, **kwargs)

        # set the tick labels
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])

        # plot row labeling
        ax_rows.set_ylim(ax_plot.get_ylim())
        ax_rows.set_xlim(0, 1)
        plot_index(
            ax_rows,
            ax_plot,
            axis=1,
            index=table.index,
            plot_labels=plot_labels,
            plot_cats=plot_cats,
            hide_lines=hide_lines,
        )

        # plot column labeling
        ax_cols.set_xlim(ax_plot.get_xlim())
        ax_cols.set_ylim(0, 1)
        plot_index(
            ax_cols,
            ax_plot,
            axis=0,
            index=table.columns,
            plot_labels=plot_labels,
            plot_cats=plot_cats,
            hide_lines=hide_lines,
        )

    return fig


def _prepare_experiments(experiments):
    """
    transform the experiments structured array into a numpy array.

    Parameters
    ----------
    experiments :DataFrame

    Returns
    -------
    ndarray, list

    """
    try:
        experiments = experiments.drop("scenario", axis=1)
    except KeyError:
        pass

    x = experiments.copy()

    x_nominal = x.select_dtypes(exclude=np.number)
    x_nominal_columns = x_nominal.columns.values

    for column in x_nominal_columns:
        if np.unique(x[column]).shape == (1,):
            x = x.drop(column, axis=1)
            _logger.info(
                ("{} dropped from analysis " "because only a single category").format(
                    column
                )
            )
        else:
            x[column] = x[column].astype("category")

    return x


def create_pivot_plot(
    x, y, nr_levels=3, labels=True, categories=True, nbins=3, bin_labels=False
):
    """ convenience function for easily creating a pivot plot

    Parameters
    ----------
    x : DataFrame
    y : 1d ndarray
    nr_levels : int, optional
                the number of levels in the pivot table. The number of
                uncertain factors included in the pivot table is two
                times the number of levels.
    labels : bool, optional
             display names of uncertain factors
    categories : bool, optional
                 display category names for each uncertain factor
    nbins : int, optional
            number of bins to use when discretizing continuous uncertain
            factors
    bin_labels : bool, optional
                 if True show bin interval / name, otherwise show
                 only a number

    Returns
    -------
    Figure


    This function performs feature scoring using random forests, selects
    a number of high scoring factors based on the specified number of
    levels, creates a pivot table, and visualizes the table. This is a
    convenience  function. For more control over the process, use the
    code in this function as a template.

    """
    x = _prepare_experiments(x)
    scores = feature_scoring.get_ex_feature_scores(x, y)[0]
    x = x[scores.index]

    n = nr_levels * 2

    scores = scores.index.tolist()
    rows = [entry for entry in scores[0:n:2]]
    columns = [entry for entry in scores[1:n:2]]

    discretized_x = discretize(x, nbins=nbins, with_labels=bin_labels)

    ooi_label = "y"
    ooi = pd.DataFrame(y[:, np.newaxis], columns=[ooi_label])

    x_y_concat = pd.concat([discretized_x, ooi], axis=1)
    pvt = pd.pivot_table(
        x_y_concat, values=ooi_label, index=rows, columns=columns, dropna=False
    )

    fig = plot_pivot_table(pvt, plot_labels=labels, plot_cats=categories)

    return fig

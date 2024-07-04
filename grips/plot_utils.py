import matplotlib.pyplot as plt
import numpy as np

"""Plots a 3d wireframe for the given data

    Parameters
    ----------
    xs : np.ndarray
        x-coordinates (no duplication)
    ys : np.ndarray
        y-coordinates (no duplication)
    data : np.ndarray (2-dimensional)
        data[x][y] will be the z-coordinate at (x, y)
    """


def plot_wireframe(xs: np.ndarray, ys: np.ndarray, data: np.ndarray, title: str, xlabel: str, ylabel: str, zlabel: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.azim = -130
    x, y = np.meshgrid(xs, ys)

    ax.plot_wireframe(x, y, data.T)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    return fig


"""Plots a 3d scatter plot for the given data

    Parameters
    ----------
    xs : np.ndarray
        x-coordinates (no duplication)
    ys : np.ndarray
        y-coordinates (no duplication)
    data : np.ndarray (2-dimensional)
        data[x][y] will be the z-coordinate at (x, y)
    """


def plot_3d_scatter(xs: np.ndarray, ys: np.ndarray, data: np.ndarray, title: str, xlabel: str, ylabel: str, zlabel: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.azim = -130

    x_scatter = np.zeros(len(xs) * len(ys))
    y_scatter = np.zeros(len(xs) * len(ys))
    data_scatter = np.zeros(len(xs) * len(ys))
    i = 0
    for x, y in np.ndindex((len(xs), len(ys))):
        x_scatter[i] = xs[x]
        y_scatter[i] = ys[y]
        data_scatter[i] = data[x][y]
        i += 1

    ax.scatter(x_scatter, y_scatter, data_scatter)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    return fig


"""Plots a 2d projection (parallel projection onto y=0) of a 3d scatter plot for the given data

    Parameters
    ----------
    xs : np.ndarray
        x-coordinates (no duplication)
    ys : np.ndarray
        y-coordinates (no duplication)
    data : np.ndarray (2-dimensional)
        data[x][y] will be the z-coordinate at (x, y)
    """


def plot_color_projection(xs: np.ndarray, ys: np.ndarray, data: np.ndarray, title: str, xlabel: str, zlabel: str) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot()

    for y in range(1, len(ys) + 1):
        data_x_2d = np.zeros(len(xs))
        data_2d = np.zeros(len(xs))
        for x in range(1, len(xs) + 1):
            data_x_2d[x - 1] = x
            data_2d[x - 1] = data[x - 1][y - 1]

        c = int(y * 510.0 / len(ys))
        r = 255 - max(0, c - 255)
        b = min(c, 255)
        rgb = [r, 0, b]
        rgb_str = "#" + "".join("{:02X}".format(a) for a in rgb)

        ax.scatter(data_x_2d, data_2d, color=rgb_str)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(zlabel)
    ax.set_title(title)
    return fig


"""Plots a heat map for the given data

    Parameters
    ----------
    xs : np.ndarray
        x-coordinates (no duplication)
    ys : np.ndarray
        y-coordinates (no duplication)
    data : np.ndarray (2-dimensional)
        data[x][y] will be the heat at (x, y)
    """


def plot_heat_map(xs: np.ndarray, ys: np.ndarray, data: np.ndarray, title: str, xlabel: str, ylabel: str, num_labels: int = 10) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Create the heatmap
    cax = ax.imshow(data, aspect="auto")

    # Determine the number of labels and the step size
    x_step = len(xs) // num_labels
    y_step = len(ys) // num_labels

    # Select equally spaced indices
    x_indices = np.arange(0, len(xs), x_step)
    y_indices = np.arange(0, len(ys), y_step)

    # Handle cases where the last label might be skipped
    if x_indices[-1] != len(xs) - 1:
        x_indices = np.append(x_indices, len(xs) - 1)
    if y_indices[-1] != len(ys) - 1:
        y_indices = np.append(y_indices, len(ys) - 1)

    # Set the ticks and tick labels
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{xs[i]:.2f}" for i in x_indices])
    ax.set_yticks(y_indices)
    ax.set_yticklabels([f"{ys[i]:.2f}" for i in y_indices])

    # Adds a color bar
    fig.colorbar(cax)
    return fig

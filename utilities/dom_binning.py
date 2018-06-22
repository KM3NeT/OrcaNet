# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
mpl.rcParams.update({'font.size': 22})
z_layer_no = 5  # number of z layer to plot xy from, bottom to top
geo_file = "/home/woody/capn/mppi033h/Code/HPC/h5ToHisto/ORCA_Geo_115lines.txt"
n_bins = (11, 13, 18)

# [id,x,y,z]
geo = np.loadtxt(geo_file)

x = geo[:, 1]
y = geo[:, 2]
z = geo[:, 3]

which_z_layer = z == np.unique(z)[z_layer_no]
geo_reduced = geo[which_z_layer, :]
x_red = geo_reduced[:, 1]
y_red = geo_reduced[:, 2]


def calculate_bin_edges(n_bins, geo):
    """
    Calculates the bin edges for the later np.histogramdd actions based on the number of specified bins.
    This is performed in order to get the same bin size for each event regardless of the fact if all bins have a hit or not.
    :param tuple n_bins: contains the desired number of bins for each dimension. [n_bins_x, n_bins_y, n_bins_z]
    :param str fname_geo_limits: filepath of the .txt ORCA geometry file.
    :return: ndarray(ndim=1) x_bin_edges, y_bin_edges, z_bin_edges: contains the resulting bin edges for each dimension.
    """
    # Gefittete offsets: x,y,factor: factor*(x+x_off)
    # [6.19, 0.064, 1.0128]

    # print "Reading detector geometry in order to calculate the detector dimensions from file " + fname_geo_limits
    # geo = np.loadtxt(fname_geo_limits)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]
    geo_limits = np.nanmin(geo, axis=0), np.nanmax(geo, axis=0)
    # print ('Detector dimensions [[first_OM_id, xmin, ymin, zmin], [last_OM_id, xmax, ymax, zmax]]: ' + str(geo_limits))

    x_bin_edges = np.linspace(geo_limits[0][1] - 9.95, geo_limits[1][1] + 9.95, num=n_bins[
                                                                                        0] + 1)  # try to get the lines in the bin center 9.95*2 = average x-separation of two lines
    y_bin_edges = np.linspace(geo_limits[0][2] - 9.75, geo_limits[1][2] + 9.75, num=n_bins[1] + 1)  # Delta y = 19.483
    z_bin_edges = np.linspace(geo_limits[0][3] - 4.665, geo_limits[1][3] + 4.665, num=n_bins[2] + 1)  # Delta z = 9.329

    # offset_x, offset_y, scale = [6.19, 0.064, 1.0128]
    # x_bin_edges = (x_bin_edges + offset_x )*scale
    # y_bin_edges = (y_bin_edges + offset_y )*scale

    # calculate_bin_edges_test(geo, y_bin_edges, z_bin_edges) # test disabled by default. Activate it, if you change the offsets in x/y/z-bin-edges

    return x_bin_edges, y_bin_edges, z_bin_edges


x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges(n_bins, geo)


def calculate_bin_stats(x_one_layer, y_one_layer, x_bin_edges, y_bin_edges):
    hist_xy = np.histogram2d(x_one_layer, y_one_layer, bins=(x_bin_edges, y_bin_edges))[0]
    unique, counts = np.unique(hist_xy.flatten(), return_counts=True)
    return unique, counts


def scan_for_optimum(x_one_layer, y_one_layer, n_bins, geo):
    x_edges, y_edges, z_edges = calculate_bin_edges(n_bins, geo)
    current_unique, current_counts = calculate_bin_stats(x_one_layer, y_one_layer, x_edges, y_edges)

    result_matrix = np.zeros((3, 10, 10), dtype=int)

    for x_off in np.arange(-5, 5, 1):
        for y_off in np.arange(-5, 5, 1):
            unique, counts = calculate_bin_stats(x_one_layer, y_one_layer, x_edges + x_off, y_edges + y_off)
            result_matrix[0, x_off + 5, y_off + 5] = counts[0]
            result_matrix[1, x_off + 5, y_off + 5] = counts[1]
            result_matrix[2, x_off + 5, y_off + 5] = counts[2] if len(counts) >= 3 else 0
            print(x_off, y_off, "\t", counts)
    #for i in range(len(result_matrix[0])):
        #for j in range(len(result_matrix[0, 0])):
            #print (result_matrix[:, i, j], end="")
        #print("")


def get_distance_to_edges(x_edges, y_edges, x_one_layer, y_one_layer):
    # Return the distance of all doms

    # The size of the bins:
    box_length_x = x_edges[1] - x_edges[0]
    box_length_y = y_edges[1] - y_edges[0]

    # The position of the doms in the unit box from (0,0) to (box_length_x,box_length_y)
    x_unit_cell = (x_one_layer - np.min(x_edges)) % box_length_x
    y_unit_cell = (y_one_layer - np.min(y_edges)) % box_length_y

    center = [box_length_x / 2, box_length_y / 2]
    # euclidean distance to center
    distances = np.sqrt((x_unit_cell - center[0]) ** 2 + (y_unit_cell - center[1]) ** 2)
    # l1 distance to center
    # distances = np.abs(x_unit_cell - center[0]) + np.abs(y_unit_cell - center[1])

    # Minimum distance to edge
    # distances = -1 * np.min([np.min(x_unit_cell), np.min(box_length_x-x_unit_cell), np.min(y_unit_cell), np.min(box_length_y-y_unit_cell)] )

    return np.mean(distances)


def scan_for_highest_distance(x_one_layer, y_one_layer, x_bin_edges, y_bin_edges, x_offsets, y_offsets, multiply=[1],
                              efficient=False):
    # Shift the grid around by x/y offsets in meters, and get the distance to the bin center
    # Also returns the highest number of doms per bin:
    # return [[x_off],[y_off],[distance],[max_doms_per_bin]]

    i = 0
    total_offsets_to_check = len(x_offsets) * len(y_offsets) * len(multiply)
    if efficient == False:
        results = np.zeros((5, total_offsets_to_check))  # [[x_off],[y_off],[distance],[max_doms_per_bin], [factor]
    else:
        results = np.array([0.0, 0.0, 1000.0, 0.0, 0.0])

    for x_offset in x_offsets:
        for y_offset in y_offsets:
            for factor in multiply:
                x_edges = factor * (x_bin_edges + x_offset)
                y_edges = factor * (y_bin_edges + y_offset)

                hist_xy = np.histogram2d(x_one_layer, y_one_layer, bins=(x_edges, y_edges))[0]
                max_bins_per_dom = hist_xy.flatten().max()
                distance = get_distance_to_edges(x_edges, y_edges, x_one_layer, y_one_layer)

                if efficient == False:
                    results[:, i] = (x_offset, y_offset, distance, max_bins_per_dom, factor)
                else:
                    if distance < results[2] and max_bins_per_dom == 1:
                        results = (x_offset, y_offset, distance, max_bins_per_dom, factor)

                # if i % 5000 == 0:
                #     print(i, "/", total_offsets_to_check)
                i += 1
    return results


def plot_3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_2d(x_bin_edges, y_bin_edges, x_one_layer, y_one_layer):
    # xy_one_layer: x and y coordinates of doms of one z layer
    box_length_x = x_bin_edges[1] - x_bin_edges[0]
    box_length_y = y_bin_edges[1] - y_bin_edges[0]
    hist_xy = np.histogram2d(x_one_layer, y_one_layer, bins=(x_bin_edges, y_bin_edges))[0]
    #print("Doms per bin:", calculate_bin_stats(x_one_layer, y_one_layer, x_bin_edges, y_bin_edges))

    # max_doms_inside=hist_xy.max() #2
    # min_doms_inside=hist_xy.min() #0

    fig = plt.figure(figsize=(8, 13))
    ax = fig.add_subplot(111)
    #ax = plt.subplot(111, adjustable='box-forced')


    for x_bin_edge in x_bin_edges:
        ax.plot([x_bin_edge, x_bin_edge], [y_bin_edges.min(), y_bin_edges.max()], color="black", ls="-", zorder=-1)
    for y_bin_edge in y_bin_edges:
        ax.plot([x_bin_edges.min(), x_bin_edges.max()], [y_bin_edge, y_bin_edge], color="black", ls="-", zorder=-1)

    for bin_no_x, x_bin_edge in enumerate(x_bin_edges[:-1]):
        for bin_no_y, y_bin_edge in enumerate(y_bin_edges[:-1]):
            alpha_max = 0.3
            doms_inside = hist_xy[bin_no_x, bin_no_y]
            alpha = doms_inside * alpha_max / 2
            # alpha = (doms_inside-min_doms_inside) * alpha_max/max_doms_inside
            ax.add_patch(
                Rectangle([x_bin_edge, y_bin_edge], box_length_x, box_length_y, fc="blue", alpha=alpha, zorder=-2))

    plt.rcParams.update({'font.size': 16})
    ax.scatter(x_one_layer, y_one_layer, c='r', marker='o', label="DOM lines", zorder=1)
    ax.set_xlabel('X (m)')
    ax.minorticks_on()
    ax.set_ylabel('Y (m)')
    ax.set_aspect("equal")

    new_tick_locations_x = x_bin_edges[:-1] + 0.5 * box_length_x
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(new_tick_locations_x)
    ax2.set_xticklabels(np.arange(1, n_bins[0] + 1, 1))
    ax2.set_xlabel("x bin no.")
    ax2.set_aspect("equal")

    new_tick_locations_y = y_bin_edges[:-1] + 0.5 * box_length_y
    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    ax3.set_yticks(new_tick_locations_y)
    ax3.set_yticklabels(np.arange(1, n_bins[1] + 1, 1))
    ax3.set_ylabel("y bin no.")
    ax3.set_aspect("equal")

    legend = ax.legend(loc="lower right")
    legend.get_frame().set_alpha(1)
    fig.suptitle("Dom locations and Binning in XY direction")
    #ax.set_ylim(-200, 200)
    #plt.setp(ax, aspect=1.0, adjustable='box-forced')
    #plt.show()
    plt.savefig('binning.pdf')


# scan_for_optimum(x_red,y_red, n_bins, geo)

def maximize_distance():
    x_offsets = np.linspace(-10, 10, 200 + 1)
    y_offsets = np.linspace(-2, 6, 2000 + 1)
    # multiply=np.linspace(1.01,1.02, 50+1)
    multiply = [1]
    x_offsets = [6.19]

    efficient = True

    distances = scan_for_highest_distance(x_red, y_red, x_bin_edges, y_bin_edges, x_offsets, y_offsets, multiply,
                                          efficient)

    # [[x_off],[y_off],[distance],[max_doms_per_bin]] ... (4, ... )

    def surface_plot():
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        """
        for i in range(1,int(distances[3].max())+1):
            i_max_doms_per_bin = distances[:,distances[3]==i]
            surf=ax.scatter(i_max_doms_per_bin[0], i_max_doms_per_bin[1], i_max_doms_per_bin[2], label=str(i))
        """

        ax.plot_trisurf(distances[0], distances[1], distances[2])

        ax.set_xlabel('X offset (m)')
        ax.set_ylabel('Y offset (m)')
        ax.set_zlabel('Extremal distance to edges')
        plt.legend()
        plt.show()

    # surface_plot()
    if efficient == False:
        one_dom_per_bin = distances[:, distances[3] == 1]
        best_x, best_y, best_dist, doms_per_bin, best_factor = one_dom_per_bin[:, np.argmin(one_dom_per_bin[2])]
    else:
        best_x, best_y, best_dist, doms_per_bin, best_factor = distances

    # print("Best x-offset:", best_x, "Best y-offset", best_y, "Best factor: ", best_factor, "Distance at this point:",
    #       best_dist)
    return (best_x, best_y, best_factor)


def show_distance_of_bins(x_one_layer, y_one_layer, x_bin_edges, y_bin_edges, offsets):
    for offset in offsets:
        x_edges = offset[2] * (x_bin_edges + offset[0])
        y_edges = offset[2] * (y_bin_edges + offset[1])

        box_length_x = x_edges[1] - x_edges[0]
        box_length_y = y_edges[1] - y_edges[0]

        x_unit_cell = (x_one_layer - np.min(x_edges)) % box_length_x
        y_unit_cell = (y_one_layer - np.min(y_edges)) % box_length_y

        distances = np.sort(
            np.min([x_unit_cell, box_length_x - x_unit_cell, y_unit_cell, box_length_y - y_unit_cell], axis=0))
        plt.plot(distances, label="x: " + str(offset[0]) + " y: " + str(offset[1]) + " k: " + str(offset[2]))

    plt.xlabel("Dom number")
    plt.ylabel("Minimal distance to bin edge (m)")
    plt.xlim([0, 114])
    plt.ylim([0, box_length_x / 2])
    plt.grid()
    plt.legend()
    plt.show()


def get_distance_to_edges_minimizer(x, args):
    # Return the distance of all doms
    x_offset, y_offset, factor = x
    x_bin_edges, y_bin_edges, x_one_layer, y_one_layer = args

    x_edges = factor * (x_bin_edges + x_offset)
    y_edges = factor * (y_bin_edges + y_offset)

    # The size of the bins:
    box_length_x = x_edges[1] - x_edges[0]
    box_length_y = y_edges[1] - y_edges[0]

    # The position of the doms in the unit box from (0,0) to (box_length_x,box_length_y)
    x_unit_cell = (x_one_layer - np.min(x_edges)) % box_length_x
    y_unit_cell = (y_one_layer - np.min(y_edges)) % box_length_y

    distance = -1 * np.min([np.min(x_unit_cell), np.min(box_length_x - x_unit_cell), np.min(y_unit_cell),
                            np.min(box_length_y - y_unit_cell)])

    return distance


# res = minimize( get_distance_to_edges_minimizer, x0=[0,0,1], args=[x_bin_edges, y_bin_edges, x_red, y_red], method="Powell")


# best_x, best_y, best_factor = maximize_distance()


# gute Offsets:
# 6.45,-4.25     Distance: -2.7355
# x,y,factor
offset_array = [[0, 0, 1], [6.45, -4.25, 1], [6.19, 0.064, 1.0128], [-2.2, -0.1, 1]]

# show_distance_of_bins(x_red, y_red, x_bin_edges, y_bin_edges, offset_array )

plot_offset = offset_array[2]
# plot_offset=[best_x, best_y, best_factor]
plot_2d(plot_offset[2] * (x_bin_edges + plot_offset[0]), plot_offset[2] * (y_bin_edges + plot_offset[1]), x_red, y_red)

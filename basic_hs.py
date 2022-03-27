import numpy as np
import scipy.ndimage as spn
import math
import time
import flowpy
from PIL import Image
from matplotlib import pyplot as plt
from math import ceil, isnan


def build_kern(du):  # du is an experimental delta changing the stencil, should be kept at du = 1 logically though

    dim = 2 * du + 1  # o . x . o
    half = int(dim / 2)
    kernel = np.zeros((dim, dim))
    kernel[0, 0] = 1 / 12
    kernel[0, half] = 1 / 6
    kernel[0, dim - 1] = 1 / 12
    kernel[half, 0] = 1 / 6
    kernel[half, dim - 1] = 1 / 6
    kernel[dim - 1, 0] = 1 / 12
    kernel[dim - 1, half] = 1 / 6
    kernel[dim - 1, dim - 1] = 1 / 12

    return kernel


def set_up(images, c):  # store intensity values of images in a 3D matrix, where c is an intensity scaling factor

    width, height = images[0].size
    imgs_I = np.zeros((2, width, height))

    for i, image in enumerate(images):
        for j in range(width):
            for k in range(height):
                imgs_I[i][j][k] = image.getpixel((j, k))

    return c * imgs_I


def gradient_calcs(img1, img2, c):  # calculates intensity gradients based on Horn and Schuncks work in 1981 ahead of iteration

    width, height = img1.size

    frame_a = img1.convert('L')  # convert to greyscale
    frame_b = img2.convert('L')

    images = [frame_a, frame_b]
    I = set_up(images, c)  # Intensity I has as components in this order: time, x pos (width dir), y pos (height dir)

    print("Startup")

    i = 0
    I_x = np.zeros((width, height))  # brightness gradients
    I_y = np.zeros((width, height))
    I_t = np.zeros((width, height))

    for j in range(width):  # x value
        l = j  # used to still properly assign value (left hand side below)
        if j + 1 > width - 1:
            j -= 1  # used to make neumann boundary conditions work
        for k in range(height):  # y value
            m = k
            if k + 1 > height - 1:
                k -= 1  # used to make neumann boundary conditions work

            I_x[l][m] = 1 / 4 * (I[i][j + 1][k] - I[i][j][k] + I[i][j + 1][k + 1] - I[i][j][k + 1]
                                 + I[i + 1][j + 1][k] - I[i + 1][j][k] + I[i + 1][j + 1][k + 1] - I[i + 1][j][
                                     k + 1])
            I_y[l][m] = 1 / 4 * (I[i][j][k + 1] - I[i][j][k] + I[i][j + 1][k + 1] - I[i][j + 1][k]
                                 + I[i + 1][j][k + 1] - I[i + 1][j][k] + I[i + 1][j + 1][k + 1] - I[i + 1][j + 1][
                                     k])
            I_t[l][m] = 1 / 4 * (I[i + 1][j][k] - I[i][j][k] + I[i + 1][j][k + 1] - I[i][j][k + 1]
                                 + I[i + 1][j + 1][k] - I[i][j + 1][k] + I[i + 1][j + 1][k + 1] - I[i][j + 1][k + 1])

    print("Derivatives computed")

    return I_x, I_y, I_t


def default_hs(I_x, I_y, I_t, u_init, v_init, a, it, eps):  # default hs scheme as described in Horn and Schuncks work in 1981

    width = len(u_init)
    height = len(u_init[0])
    u_values = [u_init]
    v_values = [v_init]
    du_values = []  # delta values needed for convergence graph
    dv_values = []

    steps = 0
    print("Starting iteration...")
    print("Relative difference from last iteration: (u dir), (v dir)")
    start = time.time()
    while True:
        steps += 1
        print(steps)
        iter_u = np.zeros((width, height))  # velocities in direction u of this iteration
        iter_v = np.zeros((width, height))  # velocities in direction v of this iteration

        kernel = build_kern(du=1) # build kernel

        iter_u_bar = spn.convolve(u_values[-1], kernel) # use kernel to convolve with previous flow
        iter_v_bar = spn.convolve(v_values[-1], kernel)

        for j in range(width):
            for k in range(height):

                if eps != 0: # use weight to combat contrast dependence when choosing a value for epsilon
                    omega = np.sqrt(np.sqrt(I_x[j][k] ** 2 + I_y[j][k] ** 2 + I_t[j][k] ** 2) + eps ** 2)

                else: # dont use weight
                    omega = 1

                iter_u[j][k] = iter_u_bar[j][k] \
                               - I_x[j][k] *  (
                                           I_x[j][k] * iter_u_bar[j][k] + I_y[j][k] * iter_v_bar[j][k] + I_t[j][k]) \
                               / (omega * a ** 2 + I_x[j][k] ** 2 + I_y[j][k] ** 2)
                iter_v[j][k] = iter_v_bar[j][k] \
                               - I_y[j][k] *  (
                                           I_x[j][k] * iter_u_bar[j][k] + I_y[j][k] * iter_v_bar[j][k] + I_t[j][k]) \
                               / (omega * a ** 2 + I_x[j][k] ** 2 + I_y[j][k] ** 2)

        u_values.append(iter_u)
        v_values.append(iter_v)

        if steps != 1: # convergence information
            delta_u = np.linalg.norm((np.array(u_values)[-2] - np.array(u_values)[-1]), "fro") / np.linalg.norm(
                np.array(u_values)[-2], "fro")
            delta_v = np.linalg.norm((np.array(v_values)[-2] - np.array(v_values)[-1]), "fro") / np.linalg.norm(
                np.array(v_values)[-2], "fro")
            du_values.append(delta_u)
            dv_values.append(delta_v)

            print(f"{round(delta_u, 3)}, {round(delta_v, 3)}")

        if steps == it: # computation time
            end = time.time()
            print(f"\nComputation time elapsed: {round(end - start, 2)} s\n")
            break

    return np.array(u_values[-1]), np.array(
        v_values[-1])  # return if needed for multiscale, du dv values for convergence graph


def vis_plot(u_values, v_values, div, a, it):  # u_values can take multiple instances of flows, for comparison reasons

    width, height = np.array(u_values).shape
    red_w = ceil(width / div)  # reduced width based on div
    red_h = ceil(height / div)

    u_all = np.zeros((red_w * red_h))
    v_all = np.zeros((red_w * red_h))
    x, y = np.meshgrid(np.linspace(0, width - 1, red_w), np.linspace(0, height - 1, red_h))
    vis = []

    # arrow vizualisation
    l = 0
    for j in range(red_h - 1, -1, -1):
        for i in range(0, red_w):
            u_all[l] = u_values[div * i][
                div * j]  # every div'tht entry is used i.e. div = 4, every 4th entry is used, improves visualisation
            v_all[l] = -v_values[div * i][div * j]
            l += 1

    # colorwheel vizualization
    add2 = []
    for i in range(width):
        add = list(zip(np.array(u_values[i]), np.array(v_values[i])))
        add2.append(add)

    vis = np.rot90(np.flip(np.array(add2), axis=1), 1)  # represent correctly

    print("Visualizing...")

    fig, ax = plt.subplots(1, 2)

    ax[0].quiver(x, y, u_all, v_all)
    title = "$\\alpha$ = " + str(a) + ", its = " + str(it) + ", div = " + str(div)

    ax[0].set_title(title)
    ax[1].imshow(flowpy.flow_to_rgb(np.array(vis)))

    plt.show()

    return vis


def comp_gt(flow_c, flow, a, it):  # comparison between computed flow and ground truth if avaiblabe; flow_c is 3 dimensionsal: rows, columns, (u,v)

    height, width, uv = np.array(flow_c).shape  # uv not used
    absm_v = []
    abssd_v = []
    angm_v = []
    angsd_v = []

    if height == flow.shape[0] and width == flow.shape[1]: #check if ground truth is applicable

        diff_abs = np.zeros((height, width))
        diff_ang = np.zeros((height, width))

        print("\nComparison to ground truth\n")

        fig, ax = plt.subplots(1, 2)

        for i in range(height):
            for j in range(width):
                if isnan(flow[i, j, 0]) or isnan(flow[i, j, 1]):  # if gt is not present, set diff_abs and diff_ang to nan
                    diff_abs[i, j] = math.nan
                    diff_ang[i, j] = math.nan
                else:
                    diff_abs[i, j] = np.sqrt((flow_c[i][j][0] - flow[i][j][0]) ** 2 + (
                            flow_c[i][j][1] - flow[i][j][
                        1]) ** 2)  # error calc as in middlebury: endpoint error

                    arg = np.inner(flow_c[i][j], flow[i][j]) / (np.sqrt(flow_c[i][j][0] ** 2 +
                                                                             flow_c[i][j][1] ** 2) * np.sqrt(
                        flow[i][j][0] ** 2 + flow[i][j][1] ** 2))
                    diff_ang[i, j] = np.arccos(arg)  # error calc as in middlebury: angular error

        absm = np.mean(np.ma.array(diff_abs, mask=np.isnan(diff_abs)))
        abssd = np.std(np.ma.array(diff_abs, mask=np.isnan(diff_abs)))
        angm = np.mean(np.ma.array(diff_ang, mask=np.isnan(diff_ang)))
        angsd = np.std(np.ma.array(diff_ang, mask=np.isnan(diff_ang)))
        absm_v.append(absm)
        abssd_v.append(abssd)
        angm_v.append(angm)
        angsd_v.append(angsd)

        print(
            f"Average difference in absolute value: {round(absm, 3)}\nAverage difference in angular value: {round(angm, 3)}\n"
            f"Standard Deviation in absolute value: {round(abssd, 3)}\nStandard Deviation in angular value: {round(angsd, 3)}\n")

        #visualization

        title = f"$\\alpha$ = {a}, its = {it}\n$\Delta$ to Ground Truth in Absolute Value"

        ax[0].set_title(title)
        ax[1].set_title(f"$\\alpha$ = {a}, its = {it}\n$\Delta$ to Ground Truth in Angular Value")
        im = ax[0].imshow(diff_abs)
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(diff_ang)
        plt.colorbar(im, ax=ax[1])

        plt.show()

    else:
        print("\nGround Truth Flow not compatible, comparison halted.")


if __name__ == '__main__':

    images = ["whale10.png", "whale11.png"]
    print(images)

    flow_gt = flowpy.flow_read("whale_flow.flo")
    frameA = Image.open(images[0])
    frameB = Image.open(images[1])
    # a_s = np.concatenate((np.arange(0.1,1,0.1),np.arange(1,10,1),np.arange(10,110,10)))  # DO NOT CHOOSE TWO ARRAYS WITH MORE THAN ONE ENTRY EACH!!!
    a = 10
    it = 25
    div = 2  # purely cosmetical in arrow representation
    c = 1  # grey value multiplicator (keep 1 naturally)
    eps = 0 # set 0 to compute unweighted

    width, height = frameA.size
    print(f"Image Dimensions: {width} x {height}\n")

    u_init = np.zeros((width, height))  # initial u
    v_init = np.zeros((width, height))  # initial v

    u = []
    v = []

    print(f"a = {a}")
    print(f"its = {it}")
    I_x, I_y, I_t = gradient_calcs(frameA, frameB, c)
    u, v = default_hs(I_x, I_y, I_t, u_init, v_init, a, it, eps)

    flow_c = vis_plot(u, v, div, a, it)  # return for comparison to gt
    plt.imshow(flowpy.flow_to_rgb(flow_c))
    plt.title(f"$\\alpha = {a}, its = {it}$")
    plt.show()

    comp_gt(flow_c, flow_gt, a, it)
import numpy as np
import scipy.ndimage as spn
import math
import flowpy
from PIL import Image
from matplotlib import pyplot as plt
import basic_hs, basic_hs_ms


def prepare_flowc_a(flow_gt,u,v): # adapted version; prepare the computed flow to fit the dimensions of the provided gt flow, in order to make comparisons possible; e.g. 256x256 -> 388x584

    height, width, uv = flow_gt.shape

    flow_c_u = u
    flow_c_v = v

    img_u = Image.fromarray(flow_c_u)
    img_v = Image.fromarray(flow_c_v)

    img_u_r = img_u.resize((width, height))
    img_v_r = img_v.resize((width, height))

    flow_c_u = np.array(img_u_r)
    flow_c_v = np.array(img_v_r)

    return flow_c_u, flow_c_v


def convergence_proof(gt,u,v):

    gt_u = np.ma.array(gt[:, :, 0], mask=np.isnan(gt[:, :, 0])) #mask nan
    gt_v = np.ma.array(gt[:, :, 1], mask=np.isnan(gt[:, :, 1])) #mask nan

    u_rs, v_rs = prepare_flowc_a(gt, np.rot90(np.flip(u, 1), 1), np.rot90(np.flip(v, 1), 1))

    err_u = np.linalg.norm(u_rs - gt_u, "fro")
    err_v = np.linalg.norm(v_rs - gt_v, "fro")

    return err_u, err_v


def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def warped_gradients(f, u, v): #compute the warped image intensities and warped image intesity derivatves

    width, height = u.shape

    fs = np.zeros((width,height))

    for i in range(width): # warp intensities as described
        for j in range(height):
            new_i = i + int(u[i][j])
            new_j = j + int(v[i][j])
            try:
                fs[i][j] = f[1][new_i][new_j]
            except:
                if new_i > width-1:
                    fs[i][j] = f[1][width-1][j]
                elif new_i < 0:
                    fs[i][j] = f[1][0][j]
                if new_j > height-1:
                    fs[i][j] = f[1][i][height-1]
                elif new_j < 0:
                    fs[i][j] = f[1][i][0]


    fs_x = np.zeros((width, height))
    fs_y = np.zeros((width, height))

    for i in range(width):
        for j in range(height):

            try:
                fs_y[i][j] = 1 / 2 * (fs[i][j + 1] - fs[i][j - 1])
                fs_x[i][j] = 1 / 2 * (fs[i + 1][j] - fs[i - 1][j])
            except: # image boundaries dealt with
                if (j + 1 > height-1):
                    fs_y[i][j] = 0
                elif (j - 1 < 0):
                    fs_y[i][j] = 0
                if (i + 1 > width-1):
                    fs_y[i][j] = 0
                elif (j - 1 < 0):
                    fs_y[i][j] = 0

    return fs_x, fs_y, fs  # fs_x = warped intensities derivatives; f[0] = first frame intensities


def incr_warping(f, u, v, a, its, eps, img1_rszd, img2_rszd):

    f_x, f_y, f_t = basic_hs.gradient_calcs(img1_rszd, img2_rszd, c=1) # needed for omega, weighting term

    kernel = basic_hs.build_kern(du=1) # set up kernel

    width, height = u.shape

    all_u = np.zeros((its+1, width, height))
    all_v = np.zeros((its+1, width, height))
    all_u[0] = u
    all_v[0] = v
    all_delta_u = []
    all_delta_v = []

    print("Starting iteration...")

    for it in range(its): # iterations in one pyramid stage

        print(it+1)
        u = all_u[it]  # all last iteration u's
        v = all_v[it]  # all last iteration v's
        fs_x, fs_y, fs = warped_gradients(f,u,v)

        if it == (its-1): #show warped image at the end of each stage to gauge warping performance
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(np.rot90(np.flip(fs, 1), 1))
            ax[0].set_title("warped image")
            ax[1].imshow(np.rot90(np.flip(f[0], 1), 1))
            ax[1].set_title("frame 1")
            ax[2].imshow(np.rot90(np.flip(f[1], 1), 1))
            ax[2].set_title("frame 2")
            plt.show()

        u_bar = spn.convolve(u, kernel) # use kernel to convolve with previous flow
        v_bar = spn.convolve(v, kernel)

        for i in range(width):
            for j in range(height):

                if eps > 0: # weighted
                    omega = np.sqrt(np.sqrt(f_x[i][j] ** 2 + f_y[i][j] ** 2 + f_t[i][j] ** 2) + eps ** 2)
                else: # un-weighted
                    omega = 1

                fs_xy = np.array([fs_x[i][j], fs_y[i][j]]) # gradients of warped image at given pixel i,j
                h = np.array([u[i][j], v[i][j]]) # take out only the last iteration u that is relevant to this pixel i,j
                h_bar = np.array([u_bar[i][j],v_bar[i][j]])

                A = np.outer(fs_xy, fs_xy.T) + omega * a**2 * np.identity(2) # matrix A
                b = (f[0][i][j] - fs[i][j] + np.inner(fs_xy,h)) * fs_xy + omega * a**2 * h_bar # right hand side b

                h = np.linalg.solve(A, b) # solve using LU-Factorisation
                all_u[it+1][i][j] = h[0] # store results
                all_v[it+1][i][j] = h[1]

        if it > 0: # observe convergence behaviour
            delta_u = np.linalg.norm(all_u[it] - all_u[it+1], "fro") / np.linalg.norm(all_u[it], "fro")
            delta_v = np.linalg.norm(all_v[it] - all_v[it+1], "fro") / np.linalg.norm(all_v[it], "fro")
            all_delta_u.append(delta_u)
            all_delta_v.append(delta_v)

            print(f"{round(delta_u, 3)}, {round(delta_v, 3)}")

    return all_u[it + 1], all_v[it + 1]


if __name__ == "__main__":

    nam = "woodbox"
    flo_name = nam + "_flow.flo"
    flow_gt = flowpy.flow_read(flo_name)
    img1 = "woodbox01.png"
    img2 = "woodbox02.png"
    image1 = Image.open(img1)
    image2 = Image.open(img2)

    width, height = image1.size
    print(f"Image Dimensions: {width} x {height}\n")

    div = 4 # visualization
    a = 10  # a in ld part
    a_hs = 1 # a in first stage (basic hs)
    it = 20  # iterations in ld part
    it_hs = 25  # iterations in first stage (basic hs)

    eps = 0 # include weight if a value is set
    cf = 2 # conversion factor for conversion of u,v to next scale

    pxs = [16,32,64,128] # side lengths of quadratic images in pyramid stages

    u_init = np.zeros((pxs[0], pxs[0])) # initial u
    v_init = np.zeros((pxs[0], pxs[0])) # initial v
    all_delta_u = []
    all_delta_v = []


    for cnt, px in enumerate(pxs):
        print(f"{px}x{px}")

        img1_rszd = image1.resize((px, px)) #resize the images
        img2_rszd = image2.resize((px, px))

        if cnt != 0: # conversion only necessary after first stage
            u_old = cf * basic_hs_ms.convert_init(u,cf) # convert previous resolution values to next resolution values
            v_old = cf * basic_hs_ms.convert_init(v,cf)

        if cnt == 0: # basic Horn and Schunck scheme employed in first stage
            I_x, I_y, I_t = basic_hs.gradient_calcs(img1_rszd, img2_rszd, c=1)
            u, v = basic_hs.default_hs(I_x, I_y, I_t, u_init, v_init, a_hs, it_hs, eps)

            flow_c = basic_hs.vis_plot(u, v, div, a_hs, it_hs)
            plt.imshow(flowpy.flow_to_rgb(flow_c))
            plt.show()
        else: # iterative incremental warping scheme in all of the other stages
            frame_a = img1_rszd.convert('L') #convert to greyscale
            frame_b = img2_rszd.convert('L')
            images = [frame_a, frame_b]
            f = basic_hs.set_up(images, c=1) # set up intensities

            u, v = incr_warping(f, u_old, v_old, a, it, eps, img1_rszd, img2_rszd)

            flow_c = basic_hs.vis_plot(u, v, div, a, it)

    if height == flow_gt.shape[0] and width == flow_gt.shape[1]:

        flow_c = basic_hs_ms.prepare_flowc(flow_gt, flow_c)
        plt.imshow(flowpy.flow_to_rgb(flow_c))
        plt.show()
        basic_hs.comp_gt(flow_c, flow_gt, a, it)  # u and v in brackets because the function usually deals with multiple instances

    else:
         print("\nGround Truth Flow not compatible, comparison halted.")

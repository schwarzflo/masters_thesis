import numpy as np
from math import floor
import time
import flowpy
from PIL import Image
from matplotlib import pyplot as plt
import basic_hs


def prepare_flowc(flow_gt,flow_c): # prepare the computed flow to fit the dimensions of the provided gt flow, in order to make comparisons possible; e.g. 256x256 -> 388x584

    height, width, uv = flow_gt.shape

    flow_c_u = flow_c[:, :, 0]
    flow_c_v = flow_c[:, :, 1]

    img_u = Image.fromarray(flow_c_u)
    img_v = Image.fromarray(flow_c_v)

    img_u_r = img_u.resize((width, height))
    img_v_r = img_v.resize((width, height))

    flow_c_u = np.array(img_u_r)
    flow_c_v = np.array(img_v_r)

    flow_c = np.zeros((height,width,2))

    for i in range(height):
        flow_c[i] = np.array(list(zip(flow_c_u[i], flow_c_v[i])))

    return flow_c


def convert_init(u_old,cf): # u_old is the u or v of the previous image size; px is the new size (px x px)

    px = cf*u_old.shape[0] # new u to fit new size px
    u_new = np.zeros((px,px))

    for i in range(px):
        for j in range(px):
            u_new[i][j] = u_old[floor(i / cf)][floor(j/cf)] # all the pixels which cover one pixel in the last size, get the same value
    return u_new


if __name__ == '__main__':

    images = ["brickbox01.png", "brickbox02.png"]
    print(images)

    flow_gt = flowpy.flow_read("brickbox_flow.flo")
    frameA = Image.open(images[0])
    frameB = Image.open(images[1])

    pxs = [32,64,128,256,512]
    div = 4
    a = 25
    it = 20
    c = 1
    eps = 0
    cf = 2

    width, height = frameA.size
    print(f"Image Dimensions: {width} x {height}\n")

    u = np.zeros((pxs[0],pxs[0]))
    v = np.zeros((pxs[0],pxs[0]))

    times = []

    for cnt, px in enumerate(pxs):

        frameA_rszd = frameA.resize((px, px), Image.ANTIALIAS)
        frameB_rszd = frameB.resize((px, px), Image.ANTIALIAS)

        start = time.time()
        I_x, I_y, I_t = basic_hs.gradient_calcs(frameA_rszd, frameB_rszd, c)
        u, v = basic_hs.default_hs(I_x, I_y, I_t, u, v, a, it, eps)
        end = time.time()
        times.append(end-start)
        flow_c = basic_hs.vis_plot(u, v, div, a, it)

        if cnt != len(pxs)-1:
            u = cf*convert_init(u,cf) #doubling the a values because image size is changed
            v = cf*convert_init(v,cf)

    if height == flow_gt.shape[0] and width == flow_gt.shape[1]:

        flow_c = prepare_flowc(flow_gt, flow_c)
        plt.imshow(flowpy.flow_to_rgb(flow_c))
        plt.title(f"$\\alpha = {a}, its = {it}$")
        plt.show()
        basic_hs.comp_gt(flow_c,flow_gt,a,it) #u and v in brackets because the function usually deals with multiple instances

    else:
        print("\nGround Truth Flow not compatible, comparison halted.")

# masters_thesis
Code concerning my master's thesis on optical flow

Besides a basic Horn-Schunck algorithm (basic_hs.py), a naive pyramidal approach (basic_hs_ms.py) and an iterative incremental scheme with warping, also pyramid-based (incr_warping_ms.py), was implemented.

For two given input frames, all algorithms produce a visual output as well as convergence behaviour of the algorithm. Further, given a ground-truth file of .flo format, a numerical comparison is made based on the measures of Endpoint Error and Angular Error, which is then also visualised. 

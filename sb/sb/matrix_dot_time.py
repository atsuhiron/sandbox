import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from time_stamp_print import tsprint


tsprint("initialize")


def poly(x, a, off):
    return a*x**3 + off


def gen_dataset(size: int, max_num: int, dtype, num: int):
    size = (num, 2, size, size)
    data = np.random.random(size) * max_num
    return data.astype(dtype)


dtypes = [np.int8, np.int16, np.int32, np.int64,
          np.uint8, np.uint16, np.uint32, np.uint64,
          np.float16, np.float32, np.float64]
dtn = len(dtypes)
dtypes_str = []

maxes = np.array([2**(4-1), 2**(8-1), 2**(16-1), 2**(16-1),
                  2**(4-1), 2**(8-1), 2**(16-1), 2**(16-1),
                  2**(8-1), 2**(16-1), 2**(16-1)])
maxes = np.sqrt(maxes).astype(np.int64)

color_seqs = ["Blues", "Greens", "Oranges"]
seq_nums = [4, 4, 3]
colors = [[plt.get_cmap(cseq)(nn*50 + 50) for nn in range(sn)] for cseq, sn in zip(color_seqs, seq_nums)]
colors = sum(colors, [])

sizes = np.array([6, 10, 20, 40, 80, 100, 160, 200])

epoch = 200
iter_num = 32
logger = np.zeros((dtn, len(sizes), epoch))

for dd in range(dtn):
    dataset = gen_dataset(sizes.max(), maxes[dd], dtypes[dd], iter_num)
    dtypes_str.append(dataset.dtype.name)
    tsprint("==== {} ====".format(dtypes_str[-1]))
    for ss in range(len(sizes)):
        tsprint("     {}".format(sizes[ss]))
        dataset_let = dataset[:, :, :sizes[ss], :sizes[ss]]
        for ii in range(epoch):
            s = time.time()
            for jj in range(iter_num):
                _ = np.dot(dataset_let[jj][0], dataset_let[jj][1])
            logger[dd, ss, ii] = time.time() - s

tsprint("summarize")
means = logger.mean(axis=2)
stdas = logger.std(axis=2)
paras = np.zeros((dtn, 2))
sigmas = np.zeros((dtn, 2))
residulas = np.zeros(dtn)

apx_x = np.linspace(sizes.min()*0.8, sizes.max()*1.2)

plt.figure(figsize=(16, 8))
plt.subplot(121)
fmt_dict = dict(fmt="o", markersize=8, capsize=2, ls="", alpha=0.8)
for dd in range(dtn):
    paras[dd], cov = so.curve_fit(poly, sizes, means[dd], p0=(0.1, 0.1), sigma=stdas[dd])
    sigmas[dd] = np.sqrt(np.diag(cov))
    residulas[dd] = np.sum((means[dd] - poly(sizes, *paras[dd]))**2)
    plt.plot(apx_x, poly(apx_x, *paras[dd]), color=colors[dd], alpha=0.8, lw=3)
    plt.errorbar(sizes, means[dd], stdas[dd], color=colors[dd], label=dtypes_str[dd], **fmt_dict)
plt.yscale("log")
plt.ylabel("Time s", fontsize=12)
#plt.ylim([8e-5, 2])
plt.xlabel("Matrix size n", fontsize=12)
plt.legend()
plt.title("Matrix dot production calculation time per {} times".format(iter_num))

plt.subplot(122)
paras_limits = np.array([paras.T.min(axis=1), paras.transpose().max(axis=1)]).T
#  np.array([[x_min, x_max],
#            [y_min, y_max]])
paras_range = np.array([axl[1] - axl[0] for axl in paras_limits])
paras_limits[:, 0] -= paras_range*0.1
paras_limits[:, 1] += paras_range*0.1

fmt_dict["markersize"] = 10
for dd in range(dtn):
    plt.errorbar(paras[dd][0], paras[dd][1], sigmas[dd][0], sigmas[dd][1],
                 color=colors[dd], label=dtypes_str[dd], **fmt_dict)
plt.title(r"fit parameter (to $y=ax^3 + b$) distribution")
plt.ylabel("b", fontsize=12)
plt.ylim(paras_limits[1])
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
plt.xlabel("a", fontsize=12)
plt.xlim(paras_limits[0])
plt.grid()
plt.legend()
tsprint("finished")
plt.tight_layout()
plt.show()



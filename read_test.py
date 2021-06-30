import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

conv = nn.Conv2d(3, 4, 3, bias_init='zeros')
input_data = Tensor(np.ones([1, 3, 5, 5]).astype(np.float32))
out1 = conv(input_data)
print(out1)
context.set_context(mode=context.PYNATIVE_MODE)

out2 = conv(input_data)
print("finish")

import os
from collections import OrderedDict
from copy import copy

import numpy as np
import torch
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes

from polygraphy.backend.trt import util as trt_util


class Engine:
    "High level wrapper for TensorRT engines"

    def __init__(
        self,
        model_name,
        engine_dir,
    ):
        self.engine_path = os.path.join(engine_dir, model_name + ".plan")
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [
            buf.free()
            for buf in self.buffers.values()
            if isinstance(buf, cuda.DeviceArray)
        ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def activate(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device="cuda"):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt_util.np_dtype_from_trt(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            # Workaround to convert np dtype to torch
            np_type_tensor = np.empty(shape=[], dtype=dtype)
            torch_type_tensor = torch.from_numpy(np_type_tensor)
            tensor = torch.empty(tuple(shape), dtype=torch_type_tensor.dtype).to(
                device=device
            )
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(
                ptr=tensor.data_ptr(), shape=shape, dtype=dtype
            )

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(
            bindings=bindings, stream_handle=stream.ptr
        )
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        return self.tensors

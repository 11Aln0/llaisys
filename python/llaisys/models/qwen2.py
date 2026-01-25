from typing import Sequence, List, Optional
from pathlib import Path
from ctypes import c_size_t, c_int, c_int64
import json

import safetensors

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    llaisysDeviceType_t,
    llaisysDataType_t,
)
from ..libllaisys.models import (
    llaisysQwen2Model_t,
    LlaisysQwen2Meta,
)
from ..tensor import Tensor


# Map torch_dtype string to DataType
DTYPE_MAP = {
    "float32": DataType.F32,
    "float16": DataType.F16,
    "bfloat16": DataType.BF16,
}


class Qwen2:

    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
        device_ids: List[int] = None,
    ):
        model_path = Path(model_path)
        if device_ids is None:
            device_ids = [0]

        # Load model config
        config_path = model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create meta from config
        self._meta = LlaisysQwen2Meta()
        self._meta.dtype = DTYPE_MAP.get(config.get("torch_dtype", "float32"), DataType.F32)
        self._meta.nlayer = config["num_hidden_layers"]
        self._meta.hs = config["hidden_size"]
        self._meta.nh = config["num_attention_heads"]
        self._meta.nkvh = config["num_key_value_heads"]
        self._meta.dh = config["hidden_size"] // config["num_attention_heads"]
        self._meta.di = config["intermediate_size"]
        self._meta.maxseq = config["max_position_embeddings"]
        self._meta.voc = config["vocab_size"]
        self._meta.epsilon = config.get("rms_norm_eps", 1e-6)
        self._meta.theta = config.get("rope_theta", 10000.0)
        self._meta.end_token = config.get("eos_token_id", 151643)

        # Create device_ids array
        _ndevice = len(device_ids)
        _device_ids = (c_int * _ndevice)(*device_ids)

        # Create model
        self._model: llaisysQwen2Model_t = LIB_LLAISYS.llaisysQwen2ModelCreate(
            self._meta,
            llaisysDeviceType_t(device),
            _device_ids,
            c_int(_ndevice),
        )

        # Load weights from safetensors
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                print(name_)
                # TODO: Load weights into model

    def __del__(self):
        if hasattr(self, "_model") and self._model is not None:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def get_weight(self, name: str) -> Optional[Tensor]:
        """Get model weight tensor by name."""
        name_bytes = name.encode('utf-8')
        tensor_handle = LIB_LLAISYS.llaisysQwen2ModelGetWeight(self._model, name_bytes)
        if tensor_handle is None:
            return None
        return Tensor(tensor=tensor_handle)

    def infer(self, token_ids: Sequence[int]) -> int:
        """Run inference on token_ids, return next token id."""
        _ntoken = len(token_ids)
        _token_ids = (c_int64 * _ntoken)(*token_ids)
        return int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, _token_ids, c_size_t(_ntoken)
            )
        )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> List[int]:
        """Generate tokens from input token ids."""
        # TODO: Implement full generate function with sampling
        token_ids = list(inputs)

        if max_new_tokens is None:
            max_new_tokens = 100

        for _ in range(max_new_tokens):
            next_token = self.infer(token_ids)
            if next_token == self._meta.end_token:
                break
            token_ids.append(next_token)

        return token_ids

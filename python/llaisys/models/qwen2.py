from typing import Sequence, List, Optional
from pathlib import Path
from ctypes import c_size_t, c_int, c_int64, c_void_p
import json
import safetensors.torch
import numpy as np
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
    LlaisysQwen2Weights,
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
        self._meta.maxseq = 4096
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

        # Fetch weight handles (C-side allocated, do NOT destroy them in Python)
        self._weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        self._weights: LlaisysQwen2Weights = self._weights_ptr.contents

        # Prepare dtype conversion for loading
        # target_dtype = DataType(int(self._meta.dtype))
        # np_dtype_map = {
        #     DataType.F32: np.float32,
        #     DataType.F16: np.float16,
        #     DataType.BF16: np.dtype("bfloat16"),
        # }
        # self._np_dtype = np_dtype_map.get(target_dtype, np.float32)

        # Load weights from safetensors
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                arr = data_.get_tensor(name_)
                ptr = arr.data_ptr()
                self._load_weight(name_, ptr)
                    
                
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
        input_token_ids = token_ids
        if max_new_tokens is None:
            max_new_tokens = 2048

        for _ in range(max_new_tokens):
            next_token = self.infer(input_token_ids)
            token_ids.append(next_token)
            input_token_ids = [next_token]  # Feed only the last token for next step
            if next_token == self._meta.end_token:
                break
            print(f"Generated token: {next_token}", flush=True)

        return token_ids

    # ---------------- Internal helpers -----------------
    def _tensor_handle(self, name: str):
        """Map HF weight name to llaisys tensor handle."""
        w = self._weights

        # Embedding + output
        if name == "model.embed_tokens.weight":
            return w.in_embed
        if name == "lm_head.weight":
            return w.out_embed
        if name == "model.norm.weight":
            return w.out_norm_w

        # Layered weights
        prefix = "model.layers."
        if not name.startswith(prefix):
            return None

        try:
            rest = name[len(prefix):]
            layer_str, sub = rest.split(".", 1)
            layer_id = int(layer_str)
        except Exception:
            return None

        # Guard layer index
        if layer_id < 0 or layer_id >= int(self._meta.nlayer):
            return None

        # Attn norm
        if sub == "input_layernorm.weight":
            return w.attn_norm_w[layer_id]

        # Attention projections
        if sub == "self_attn.q_proj.weight":
            return w.attn_q_w[layer_id]
        if sub == "self_attn.q_proj.bias":
            return w.attn_q_b[layer_id]
        if sub == "self_attn.k_proj.weight":
            return w.attn_k_w[layer_id]
        if sub == "self_attn.k_proj.bias":
            return w.attn_k_b[layer_id]
        if sub == "self_attn.v_proj.weight":
            return w.attn_v_w[layer_id]
        if sub == "self_attn.v_proj.bias":
            return w.attn_v_b[layer_id]
        if sub == "self_attn.o_proj.weight":
            return w.attn_o_w[layer_id]

        # MLP / FFN
        if sub == "post_attention_layernorm.weight":
            return w.mlp_norm_w[layer_id]
        if sub == "mlp.gate_proj.weight":
            return w.mlp_gate_w[layer_id]
        if sub == "mlp.up_proj.weight":
            return w.mlp_up_w[layer_id]
        if sub == "mlp.down_proj.weight":
            return w.mlp_down_w[layer_id]

        return None

    def _load_weight(self, name: str, ptr):
        handle = self._tensor_handle(name)
        if handle is None:
            # Unknown weight, skip silently
            return
        # print(f"Loading weight: {name}")
        # # Ensure dtype/contiguity match target
        # arr = np_arr.astype(self._np_dtype, copy=False)
        # if not arr.flags["C_CONTIGUOUS"]:
        #     arr = np.ascontiguousarray(arr)

        # Copy into backend tensor
        LIB_LLAISYS.tensorLoad(handle, ptr)

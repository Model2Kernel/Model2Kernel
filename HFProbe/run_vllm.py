from framework import *
from utils import *
from datetime import datetime, timezone
import json
import shutil
import resource
import traceback
import jsonpickle

from vllm import LLM, SamplingParams
from vllm.config import LoadFormat
from huggingface_hub import snapshot_download, list_repo_files


CUTOFF_DATE = datetime(2025, 6, 1, tzinfo=timezone.utc)

X_COUNT = 3
GB = 1024 ** 3
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
new_soft = min(4 * GB, hard)
resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
oom_models = []
failed_models = []
batch_size_configs = [1, 3, 5]
seq_lens_configs = [1, 7, 17]

def clean(modelId):
    # --- Cleanup HuggingFace cache ---
    try:
        model_cache_dir = snapshot_download(modelId, local_files_only=True, ignore_patterns=["*.bin", "*.safetensors"])
        shutil.rmtree(model_cache_dir)
        print(f"[{modelId}] Cache removed: {model_cache_dir}")
    except Exception as e:
        print(f"[{modelId}] Cache cleanup error:", e)
  
def handleVLLMModel(modelId, configs={}, suffix=None, outdir="./vllmout", loadDir="./vllm-load", dataDir="./data", is_op_suffix=False):
    global batch_size_configs
    global seq_lens_configs
    global calls_map
    global tensor_calls
    global failed_models, oom_models
    
    os.makedirs(outdir, exist_ok=True)
    if not is_op_suffix:
        outPath = outdir+"/"+modelId.replace('/', '_')
        if suffix:
            outPath+="_"+suffix
        outPath+=".json"
    else:
        outdir = os.path.join(outdir, modelId.replace('/', '_'))
        os.makedirs(outdir, exist_ok=True)
        outPath = os.path.join(outdir, suffix+".json")
        
    if os.path.exists(outPath):
        return
        
    print("Running model ", modelId, "...")
    if "max_num_seqs" not in configs:
        configs["max_num_seqs"] = 20
    if "max_model_len" not in configs:
        configs["max_model_len"] = 514
    if "block_size" not in configs:
        configs["block_size"] = 512
    if "num_gpu_blocks_override" not in configs:
        configs["num_gpu_blocks_override"] = 30
    if os.environ.get("VLLM_USE_V1", "0") != "1":
        configs["preemption_mode"] = "swap"
    if "compilation_config" not in configs:
        configs["enforce_eager"] = True
    if "enable_chunked_prefill" not in configs:
        configs["enable_chunked_prefill"] = False
    
    os.makedirs(loadDir, exist_ok=True)
    if not is_op_suffix:
        loadOutPath = loadDir+"/"+modelId.replace('/', '_')
        if suffix:
            loadOutPath+="_"+suffix
        loadOutPath+=".json"
    else:
        loadDir = os.path.join(loadDir, modelId.replace('/', '_'))
        os.makedirs(loadDir, exist_ok=True)
        loadOutPath = os.path.join(loadDir, suffix+".json")
    
    try:
        with fast_dummy_init(mode=os.getenv("BOS_FAST_DUMMY_INIT", "empty")):
            with kv_probe("init"):
                with enable_thin_kv():
                    llm = LLM(
                        model=modelId,
                        load_format=LoadFormat.DUMMY,
                        device="cuda",
                        gpu_memory_utilization=1.0,
                        trust_remote_code=True,
                        hf_token=HF_TOKEN,
                        **configs
                    )
                    if not os.path.exists(loadOutPath):
                        tmp_calls = tensor_calls.copy()
                        with open(loadOutPath, "w") as wf:
                            json.dump(tmp_calls, wf)
    except Exception as e: 
        traceback.print_exc()
        print(e)
        clean(modelId)
        tensor_calls.clear()
        calls_map.clear()
        failed_models.append(modelId)
        return
    except MemoryError:
        print("Caught MemoryError - likely trying to allocate too much RAM")
        clean(modelId)
        tensor_calls.clear()
        calls_map.clear()
        oom_models.append(modelId)
        return
    
    sampling_params = SamplingParams(
        temperature=0
    )
    tokenizer = llm.get_tokenizer()
    
    tensor_calls.clear()
    calls_map.clear()
    # data = []
    total_calls_map = {}
    
    with kv_probe("gen"):
        for batch_size in batch_size_configs:
            for seq_len in seq_lens_configs:
                single_prompt = "word " * seq_len
                single_prompt = single_prompt.strip() 
                tokens = tokenizer(single_prompt)["input_ids"]
                seq_len_real = len(tokens)
                print(f"batch_size={batch_size}, seq_len={seq_len_real} ...")
            
                with enable_thin_kv():
                    with torch.inference_mode():
                        out = llm.generate([single_prompt]*batch_size, sampling_params)

                for func_name in calls_map:
                    if func_name not in total_calls_map:
                        total_calls_map[func_name] = {}
                    for call_stack in calls_map[func_name]:
                        if call_stack not in total_calls_map[func_name]:
                            total_calls_map[func_name][call_stack] = {}
                        total_calls_map[func_name][call_stack][(batch_size, seq_len_real)] = calls_map[func_name][call_stack].copy()

                tensor_calls.clear()
                calls_map.clear()
    
    os.makedirs(dataDir, exist_ok=True)
    if not is_op_suffix:
        dataOutPath = dataDir+"/"+modelId.replace('/', '_')
        if suffix:
            dataOutPath+="_"+suffix
        dataOutPath+=".json"
    else:
        dataDir = os.path.join(dataDir, modelId.replace('/', '_'))
        os.makedirs(dataDir, exist_ok=True)
        dataOutPath = os.path.join(dataDir, suffix+".json")
        
    with open(dataOutPath, "w") as wf:
        wf.write(jsonpickle.encode(total_calls_map, indent=2))
    
    computeSymbolicArgsWithMap(total_calls_map, outPath)
    clean(modelId)

def copy_config_to_modules_if_needed(cache_dir):
    module_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    hash_number = cache_dir.split("/")[-1]
    module_dir = os.path.join(module_dir, hash_number)
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            # if file.endswith(".json") or file.endswith(".yaml"):
            if not os.path.exists(module_dir):
                os.makedirs(module_dir)
            src = os.path.join(root, file)
            relative_path = os.path.relpath(src, cache_dir)
            dst = os.path.join(module_dir, relative_path)
            dst_dir = os.path.dirname(dst)
            if not os.path.exists(dst):
                print(f"Copying {src} to {dst}")
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(src, dst)

def createEmptyModelBin(modelId, cache_dir):
    files = list_repo_files(modelId)
    
    hasModelBin = False
    for f in files:
        if f == "pytorch_model.bin":
            hasModelBin = True
            break
        
    if not hasModelBin:
        return
    
    dst_path = os.path.join(cache_dir, "pytorch_model.bin")
    print(f"Creating dummy file: {dst_path}")
    with open(dst_path, "wb") as f:
        f.write(b"")

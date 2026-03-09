from agents import Agent, Runner, ModelSettings, RunConfig, function_tool, set_default_openai_key, FileSearchTool, WebSearchTool
from agents.memory import Session
from agents.exceptions import MaxTurnsExceeded
from pydantic import BaseModel, Field
from openai import OpenAI
import json, os
from typing import Dict, Any, List, Optional
import HFProbe.run_vllm as run_vllm
import tiktoken
import traceback
import re
from datetime import datetime
import time

set_default_openai_key("OPENAI_KEY")
VLLM_PATH = "vllm_path"

MODEL_PRICING = {
    "gpt-4o":       {"input": 0.0025,  "output": 0.01},   # per 1K tokens
    "gpt-5":        {"input": 0.00125,   "output": 0.01},     
    "gpt-5.1":        {"input": 0.00125,   "output": 0.01},     
}

def count_tokens(model: str, text: str) -> int:
    """Safely count tokens for any GPT model; handles lists/dicts."""
    # --- Normalize text ---
    if text is None:
        normalized = ""
    elif isinstance(text, str):
        normalized = text
    elif isinstance(text, list):
        parts = []
        for sub in text:
            if isinstance(sub, dict):
                ttype = sub.get("type")
                if ttype == "text":
                    parts.append(sub.get("text", ""))
                else:
                    # Skip tool_use, tool_result, images, etc.
                    continue
            else:
                parts.append(str(sub))
        normalized = " ".join(parts)
    elif isinstance(text, dict):
        # Flatten key-value dict content
        # Only count textual dicts (rare case)
        if text.get("type") == "text":
            normalized = text.get("text", "")
        else:
            normalized = ""
    else:
        normalized = str(text)
        
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base (used by GPT-4/5-class models)
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(normalized))

class MyCustomSession(Session):
    def __init__(self, session_id: str, model: str = "gpt-5"):
        self.session_id = session_id
        self.model = model
        self.items: List[Dict[str, str]] = []  # conversation history
        self.prices = MODEL_PRICING.get(model)
        self.token_usage = {"input": 0, "output": 0, "cost": 0.0}

    async def get_items(self, limit: Optional[int] = None) -> List[Dict]:
        """Return the current conversation items (or last N if limit given)."""
        if limit is not None:
            return self.items[-limit:]
        return list(self.items)
    
    async def add_items(self, items: List[Dict]) -> None:
        """Add new conversation turns."""
        for item in items:
            role = item.get("role")
            content = item.get("content", "")
            self.items.append({"role": role, "content": content})

            # Token & cost accounting
            tokens = count_tokens(self.model, content)
            if role in ("system", "user"):
                self.token_usage["input"] += tokens
                self.token_usage["cost"] += tokens / 1000 * self.prices["input"]
            elif role == "assistant":
                self.token_usage["output"] += tokens
                self.token_usage["cost"] += tokens / 1000 * self.prices["output"]
                
    async def pop_item(self) -> Optional[Dict]:
        """Remove and return the most recent conversation item."""
        return self.items.pop() if self.items else None

    async def clear_session(self) -> None:
        """Clear session history and token counters."""
        self.items.clear()
        self.token_usage = {"input": 0, "output": 0, "cost": 0.0}


structure_configs = []
resCon = {}
out_path = ""

@function_tool
def saveRes(config: str):
    """
    Save the result config into a json file.
    
    Args:
        config: The model config in JSON string format.
    """
    global resCon
    global out_path
    with open(out_path, "w") as f:
        resCon = json.loads(config)
        json.dump(resCon, f)
    
testAgent = Agent(
    name="LLMConfigAgent",
    instructions=(
        "You are a precise LLM analyzer. You can search on the web to retrieve the model code, framework code and config.json. Then you can analyze the code and config. You check all the conditions in the call chain and match with config fields. Your task is to find the config values that are need to invoke the custom kernel. If this operator cannot be trigger with any config, answer 'No, it cannot be triggered.' If no new config is needed, answer 'No, current config is enough.' Otherwise, you should answer the new config.json and invoke saveRes tool to store the new config.\n"
    ),
    model="gpt-5",
    tools=[WebSearchTool(), saveRes],
)

def runAgent(prompt):
    try:
        session = MyCustomSession("my_session")
        result = Runner.run_sync(
            testAgent,
            prompt,
            max_turns=30,
            session=session
        )
    except MaxTurnsExceeded as e:
        print(f"[⚠️ Warning] Agent exceeded max_turns ({e}).")

        # Access partial run data from e.partial_result if available
        if hasattr(e, "partial_result") and e.partial_result:
            result = e.partial_result
            print("=== Partial Result Before Abort ===")
            print(result.final_output or "No final message yet.")
            print("\nTrace of completed tool calls:")
            for t in result.traces:
                print(f"{t.tool_name}: {t.output[:200]}")
        else:
            print("No partial result captured.")

    print(result.final_output)
    print(f"input token: {session.token_usage['input']}, output token: {session.token_usage['output']}, cost: ${session.token_usage['cost']}")
    return result.final_output, session.token_usage              

def generate_vllm(model_structure, op_name, config_example, framework_config=None, code_snippet=None):
    global out_path
    out_dir = f"./vllm-exp/config/{model_structure}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{op_name}.json")
    if os.path.exists(out_path):
        return None, None
    
    prompt_vllm = f"For models using {model_structure}, analyze the python code in vllm. Generate model config to trigger operator {op_name}, so that I can use this config in vllm framework to test it. The repo url is https://github.com/vllm-project/vllm.git, use branch releases/v0.9.0."
    
    if framework_config:
        vllm_config = None
        if "vllm_config" in framework_config:
            vllm_config = framework_config["vllmconfig"]
        envs_config = None
        if "envs" in framework_config:
            envs_config = framework_config["envs"]
        config_str = ""
        if vllm_config:
            config_str = f" The vllm config is set as: {vllm_config}."
        if envs_config:
            envs_str = ""
            for key in envs_config:
                envs_str += f"envs.{key}={envs_config[key]}, "
            envs_str = envs_str[:-2]
            config_str += f" The environment variables are set as: {envs_str}."
    
        if config_str:
            prompt_vllm += config_str
    
    prompt_vllm += f' Below is an example of config.json for the model config. If this operator cannot be trigger with any config values, answer "No, it cannot be triggered." If no new config is needed, answer "No, current config is enough." Otherwise, you should answer the new config.json.\n{config_example}'
    if code_snippet:
        prompt_vllm += f"\nrelated code in vllm:\n{code_snippet}"
        
    print("prompt:", prompt_vllm)
    print("****************************************")
    return runAgent(prompt_vllm)

def read_code_snippet(filePath, start_line, end_line):
    if filePath.startswith("vllm"):
        filePath = os.path.join(VLLM_PATH, filePath)
        
    code_snippet = ""
    with open(filePath) as f:
        lines = f.readlines()
        for i in range(start_line-1, end_line):
            code_snippet += lines[i]
    return code_snippet

def copy_config_file():
    with open("./vllm_text_model_structures_map.json") as f:
        model_structures_map = json.load(f)
    
    dest_dir = "./vllm-configs-examples"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for model_id in model_structures_map:
        model_structure = model_structures_map[model_id]
        new_dir = model_id.replace('/', '_')
        src_path = f"../exp/data/vllm_models/{new_dir}/config.json"
        dest_path = f"{dest_dir}/{model_structure}.json"
        if os.path.exists(dest_path):
            continue
        
        with open(src_path, "r") as src_file:
            config_data = json.load(src_file)
        with open(dest_path, "w") as dest_file:
            json.dump(config_data, dest_file, indent=4)
        print(f"Copying config for {model_id} from {src_path} to {dest_path}")
    

def findAllOps(filePath, call_op_map):
    ops = set()
    with open(filePath) as f:
        call_graph = json.load(f)
        
    for key in call_graph:
        value = call_graph[key]
        for callee in value.get("callees", []):
            if callee["function"] and callee["function"].startswith("torch.ops._"):
                func_name = callee["function"].split(".")[-1]
                ops.add(func_name)
                
        if "unknown" in value and value["unknown"]:
            for call in value["unknown"]:
                if call["type"]:
                    func = call["function"].split(".")[-1]
                    if call["type"]+"-"+func in call_op_map:
                        for sub_class in call_op_map[call["type"]+"-"+func]:
                            ops.update(call_op_map[call["type"]+"-"+func][sub_class])
                    elif os.path.isdir("./cgout/"+call["type"]):
                        for fname in os.listdir("./cgout/"+call["type"]):
                            if fname.endswith(func+".json"):
                                sub_ops = findAllOps("./cgout/"+call["type"]+"/"+fname, call_op_map)
                                if call["type"]+"-"+func not in call_op_map:
                                    call_op_map[call["type"]+"-"+func] = {}
                                call_op_map[call["type"]+"-"+func][fname] = sub_ops
                                ops.update(sub_ops)
    return ops

def findTriggeredOps(model_id, op_name=None):
    triggered_ops = set()
    outPath = "./vllm-exp/out/"+model_id.replace('/', '_')
    if op_name:
        outPath+="/"+op_name
    outPath+=".json"
    
    loadPath = "./vllm-exp/load/"+model_id.replace('/', '_')
    if op_name:
        loadPath+="/"+op_name
    loadPath+=".json"
    
    if os.path.exists(outPath):
        with open(outPath) as f:
            out_data = json.load(f)
        for key in out_data:
            triggered_ops.add(key.split(".")[-1])
            
    if os.path.exists(loadPath):        
        with open(loadPath) as f:
            load_data = json.load(f)        
        for item in load_data:
            triggered_ops.add(item["name"].split(".")[-1])
            
    return triggered_ops

def extract_diff_config(old, new):
    res = {}
    for key in new:
        if key in old:
            if new[key] != old[key]:
                res[key] = new[key]
        else:
            res[key] = new[key]
    return res

def save_diff_config(op, config):
    if not config:
        return
    
    outFilePath = "./vllm-exp/config/diff.json"
    data = {}
    if os.path.exists(outFilePath):
        with open(outFilePath) as f:
            data = json.load(f)
    
    if op in data:
        if config not in data[op]:
            data[op].append(config)
    else:
        data[op] = [config]
    
    with open(outFilePath, "w") as wf:
        json.dump(data, wf)
        
def run_vllm_config(framework_config, model_config, model_id, op_name):
    env_old = os.environ.copy()
    config = {}
    if framework_config:
        if "envs" in framework_config:
            for k in framework_config["envs"]:
                os.environ[k] = framework_config["envs"][k]
        
        if "vllmconfig" in framework_config:
            config = framework_config["vllmconfig"]
        if "batch_size" in framework_config:
            run_vllm.batch_size_configs = framework_config["batch_size"]
        if "seq_len" in framework_config:
            run_vllm.seq_lens_configs = framework_config["seq_len"]  
            
    if model_config:
        if "architectures" in model_config:
            model_config.pop("architectures")
        if "rope_scaling" in model_config:
            if "rope_type" not in model_config["rope_scaling"] and "type" in model_config["rope_scaling"]:
                model_config["rope_scaling"]["rope_type"] = model_config["rope_scaling"]["type"]

        config["hf_overrides"] = model_config
        if "quantization_config" in model_config:
            if "quant_method" in model_config["quantization_config"] and "quantization" not in config:
                quant_method = model_config["quantization_config"]["quant_method"]
                config["quantization"] = quant_method
                if op_name == "moe_wna16_gemm":
                    config["quantization"] = "moe_wna16"
            else:
                run_vllm.batch_size_configs = [1, 3, 5]
                run_vllm.seq_lens_configs = [1, 7, 17]
                print(f"{op_name} config invalid.")
                return False, None
    
    config["dtype"] = "float16"
    if "max_model_len" not in config:
        config["max_model_len"] = max(max(run_vllm.seq_lens_configs)+10, 100)
    tmp_triggered_ops = None
    try:
        run_vllm.handleVLLMModel(model_id, config, op_name, "./vllm-exp/out", "./vllm-exp/load", "./vllm-exp/data", True)
        tmp_triggered_ops = findTriggeredOps(model_id, op_name)
    except:
        traceback.print_exc()
        pass
        
    if framework_config:
        os.environ = env_old
        run_vllm.batch_size_configs = [1, 3, 5]
        run_vllm.seq_lens_configs = [1, 7, 17]
    
    if tmp_triggered_ops and (op_name in tmp_triggered_ops or "vllm."+op_name in tmp_triggered_ops):
        return True, tmp_triggered_ops
    return False, tmp_triggered_ops

def main_vllm():
    global structure_configs
    with open("vllm_text_model_structures_map.json") as mf:
        structure_model_map = json.load(mf)
    
    with open("vllm_other_model_structures_map.json") as mf:
        other_structure_model_map = json.load(mf)
        structure_model_map.update(other_structure_model_map)
    
    for model_id in structure_model_map:      
        structure_configs = []
        structure = structure_model_map[model_id]
        if os.path.exists(f"./vllm-exp/config/{structure}/cost.json"):
            continue
        test_one(model_id, structure)
      
def test_one(model_id, structure):
    global out_path
    global structure_configs
    
    start_time = time.time()
    tmp_config = {"dtype": "float16"}
    try:
        run_vllm.handleVLLMModel(model_id, tmp_config, None, "./vllm-exp/out", "./vllm-exp/load", "./vllm-exp/data")
    except:
        traceback.print_exc()
        return
    
    triggered_ops = findTriggeredOps(model_id)
    if not triggered_ops:
        return
    print("triggered_ops", triggered_ops)
    
    if not os.path.exists(f"./opout/{structure}.json"):
        return
    with open(f"./opout/{structure}.json") as opf:
        call_op_info = json.load(opf)    
    
    with open("./framework_config.json", "r") as ff:
        framework_configs = json.load(ff) 
    
    config_example_path = os.path.join("./vllm-configs-examples", structure+".json")
    with open(config_example_path, "r") as f:
        config_example = json.load(f)
    
    generated_configs_path = "./vllm-exp/config/diff.json"
    generated_configs = {}
    if os.path.exists(generated_configs_path):
        with open(generated_configs_path) as f:
            generated_configs = json.load(f)
    
    cost_path = f"./vllm-exp/config/{structure}/cost.json"
    cost_map = {}
    if os.path.exists(cost_path):
        with open(cost_path) as f:
            cost_map = json.load(f)
    
    cannot_tri_path = f"./vllm-exp/config/{structure}/no_trigger.json"
    cannot_tri = []
    if os.path.exists(cannot_tri_path):
        with open(cannot_tri_path) as nf:
            cannot_tri = json.load(nf)
    print("ops cannot be triggered:", cannot_tri)
    
    final_res_path = f"./vllm-exp/config/{structure}/result.json"
    out_config_dir = f"./vllm-exp/config/{structure}"
    os.makedirs(out_config_dir, exist_ok=True)
    
    targets_ops = {"advance_step_flashattn", "advance_step_flashinfer", "swap_blocks", "copy_blocks", "copy_blocks_mla"}
    targets_ops.update(set(call_op_info.keys()))
    initial_trigger = triggered_ops.copy()
    total_count = len(call_op_info) - len(triggered_ops)
    gpt_failed = []
    input_token_num, out_token_num, money_cost = 0, 0, 0.00
    for op_name in targets_ops:
        
        if op_name in triggered_ops:
            print(f"Op name: {op_name} has been triggered.")
            continue
        
        if op_name in cannot_tri:
            print(f"Op name: {op_name} can not be triggered.")
            continue
        
        execute_res_path = f"./vllm-exp/out/{model_id.replace('/', '_')}/{op_name}.json"
        if os.path.exists(execute_res_path):
            print(f"Op name: {op_name} already handled.")
            continue
         
        print(f"Processing operator {op_name}...")
        out_path = os.path.join(out_config_dir, f"{op_name}.json")
        
        fcon = None
        if op_name in framework_configs:
            fcon = framework_configs[op_name]
        
        triggered = False    
        if op_name in generated_configs and not os.path.exists(out_path):
            old_con = generated_configs[op_name][-1]
            time0 = time.time()
            triggered, tmp_triggered_ops = run_vllm_config(fcon, old_con, model_id, op_name)
            time1 = time.time()
            if triggered:
                cost_map[op_name] = {"execute_time": time1 - time0}
                new_config = config_example.copy()
                new_config.update(old_con)
                with open(out_path, "w") as wf:
                    json.dump(new_config, wf)
                    
                print(f"Operator {op_name} is successfully triggered with old configs, execute_time: {time1 - time0}")
                triggered_ops.update(tmp_triggered_ops)
                continue
            else:
                if os.path.exists(execute_res_path):
                    os.remove(execute_res_path)
        
        if triggered:
            continue
        
        if fcon and op_name not in generated_configs:
            if "vllmconfig" in fcon or "envs" in fcon:
                time0 = time.time()
                triggered, tmp_triggered_ops = run_vllm_config(fcon, {}, model_id, op_name)
                time1 = time.time()
                if triggered:
                    cost_map[op_name] = {"execute_time": time1 - time0}
                    print(f"Operator {op_name} is successfully triggered with framework config, execute_time: {time1 - time0}")
                    triggered_ops.update(tmp_triggered_ops)
                    continue
                else:
                    if os.path.exists(execute_res_path):
                        os.remove(execute_res_path)
                    else:
                        continue

        if op_name not in call_op_info:
            continue
        
        code_sinppet = None
        if op_name in call_op_info:
            filePath = call_op_info[op_name]["filePath"]
            start_line = call_op_info[op_name]["lines"][0]
            end_line = call_op_info[op_name]["lines"][1]
            code_sinppet = read_code_snippet(filePath, start_line, end_line)
        
        time0 = time.time()
        output, token_usage = generate_vllm(structure, op_name, config_example, fcon, code_sinppet) 
        time1 = time.time()
        if not output:
            continue
        
        time_cost = time1 - time0
        input_token_num += token_usage["input"]
        out_token_num += token_usage["output"]
        money_cost += token_usage["cost"]
        cost_map[op_name] = {"input_token": token_usage["input"], "output_token": token_usage["output"], "money_cost": token_usage["cost"], "gpt_time_cost": time_cost}
        print("cost record", cost_map[op_name])
        
        if "No, it cannot be triggered.".lower() in output.lower():
            gpt_failed.append(op_name)
            cannot_tri.append(op_name)
            print(f"Op name: {op_name} cannot be triggered.")
            continue
        
        config_data = {}
        mode_config_path = f"./vllm-exp/config/{structure}/{op_name}.json"
        if os.path.exists(mode_config_path):
            with open(mode_config_path) as cf:
                config_data = json.load(cf)
        
        if not config_data and not fcon:
            continue
        
        if not config_data and fcon:
            if "vllmconfig" in fcon or "envs" in fcon:
                continue
        
        if config_data and not fcon and config_data in structure_configs:
            continue
        
        if config_data and config_data not in structure_configs:
            structure_configs.append(config_data)
        
        time0 = time.time()        
        triggered, tmp_triggered_ops = run_vllm_config(fcon, config_data, model_id, op_name)
        time1 = time.time()
        cost_map[op_name]["execute_time"] = time1 - time0
        diff_con = extract_diff_config(config_example, config_data)
        if tmp_triggered_ops:
            for incre_op in tmp_triggered_ops:
                if incre_op not in triggered_ops:
                    triggered_ops.add(incre_op)
                    save_diff_config(incre_op, diff_con)
        
        print("cost record", cost_map[op_name])        
        if triggered:
            print(f"Op name: {op_name} is successfully triggered.")
    
    end_time = time.time()
    with open(cannot_tri_path, "w") as nwf:
        json.dump(cannot_tri, nwf)    
        
    cost_map["total"] = {"input_token": input_token_num, "output_token": out_token_num, "money_cost": money_cost, "time_cost": end_time-start_time}
    with open(cost_path, "w") as nwf:
        json.dump(cost_map, nwf)   
    
    solved = len(triggered_ops - initial_trigger)
    final_result = {"initial": list(initial_trigger), "initial_num": len(initial_trigger), "new": list(triggered_ops-initial_trigger), "inre_len": solved}
    with open(final_res_path, "w") as resf:
        json.dump(final_result, resf)
        
    print(f"Total ops: {total_count}, Solved: {solved}")
    print(f"total input token: {input_token_num}, total output token: {out_token_num}, total cost: ${money_cost}")


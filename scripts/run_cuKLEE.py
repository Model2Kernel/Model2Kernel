import os, re
import json
import argparse
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Set up logging to record exceptions if needed
logging.basicConfig(filename='klee_exceptions.log', level=logging.ERROR)
TIMEOUT_LIMIT = 6 * 60 * 60

def run_z3_and_get_model(file_path):
    # Run Z3
    result = subprocess.run(
        ["z3", "-smt2", file_path],
        capture_output=True,
        text=True
    )

    output = result.stdout
    model = {}

    pattern = re.compile(
        r"\(define-fun\s+(\S+)\s*\(\)\s+\S+\s+([\s\S]+?)\)",
        re.MULTILINE
    )

    for var, value in pattern.findall(output):
        # Clean up whitespace/newlines
        value = value.strip().replace("\n", " ")
        model[var] = value

    return model

def store_bs_res(out_dir):
    for subdir in os.listdir(out_dir):
        res_path = os.path.join(out_dir, subdir, "result.json")
        if os.path.exists(res_path):
            continue

        res = {}
        for dirname in os.listdir(os.path.join(out_dir, subdir)):
            if not dirname.startswith("klee-out-"):
                continue
            if "jindex" in dirname:
                index = dirname.split("-")[-2]
            else:
                index = dirname.split("-")[-1]
            
            res[index] = {}
            for filename in os.listdir(os.path.join(out_dir, subdir, dirname)):
                if not ("_" in filename and filename.endswith(".txt")):
                    continue

                solution = run_z3_and_get_model(os.path.join(out_dir, subdir, dirname, filename))
                key = filename.split(".")[0]
                res[index][key] = {"batch_size": int(solution["batch_size"]), "seq_len": int(solution["seq_len"])}
        
        with open(res_path, "w") as outf:
            json.dump(res, outf, indent=4)

def run_klee_on_file(filepath, logDir, outputdir, useDirName=False):
    one_timeout = 3600
    try:
        os.makedirs(logDir, exist_ok=True)
        
        if useDirName:
            dir_name = os.path.basename(os.path.dirname(filepath))
            log_file = os.path.join(logDir, dir_name + '_klee_output.log')
            outputdir = os.path.join(outputdir, dir_name)
            os.makedirs(outputdir, exist_ok=True)
        else:
            log_file = os.path.join(logDir, os.path.splitext(os.path.basename(filepath))[0] + '_klee_output.log')
        if os.path.exists(log_file):
            return True         
        
        with open(log_file, 'w') as output_file:
            subprocess.run(['Model2Kernel', f"--timeout={one_timeout}", f"--output-dir={outputdir}", filepath], stdout=output_file, stderr=output_file, timeout=TIMEOUT_LIMIT, check=True)
        
        with open(log_file, 'r') as output_file:
            log_content = output_file.read()
            store_bs_res(outputdir)
            if "KLEE: done: completed paths =" not in log_content:
                print(f"KLEE not complete on {filepath}. See log for details.")
                logging.error(f"KLEE not complete on {filepath}.")
                return False
            else:
                print(f"Successfully ran KLEE on {filepath}. Output saved to {outputdir}")
                return True
    
    except subprocess.TimeoutExpired:
        store_bs_res(outputdir)
        # Handle the timeout error
        logging.error(f"KLEE run on {filepath} timed out after {TIMEOUT_LIMIT} seconds.")
        print(f"KLEE run on {filepath} timed out after {TIMEOUT_LIMIT} seconds. See log for details.")
        return False

    except subprocess.CalledProcessError as e:
        store_bs_res(outputdir)
        # Log the error if KLEE throws an exception
        logging.error(f"Error running KLEE on {filepath}: {str(e)}")
        print(f"Error running KLEE on {filepath}. See log for details.")
        return False

def main_multiple_threads(directory, logDir, outputdir, isJson=True, max_processes=5, json_files=None, useDirName=False):
    failed_files = []
    if not json_files:
        if isJson:
            json_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.json')]
        else:
            json_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('_combined.bc')]
    total = len(json_files)
    
    with ProcessPoolExecutor(max_processes) as executor:
        future_to_file = {executor.submit(run_klee_on_file, json_file, logDir, outputdir, useDirName): json_file for json_file in json_files} 
        
        for future in as_completed(future_to_file):
            json_file = future_to_file[future]
            try:
                success = future.result()
                if not success:
                    failed_files.append(os.path.basename(json_file))
            except Exception as e:
                logging.error(f"Exception occurred while processing {json_file}: {str(e)}")
                failed_files.append(os.path.basename(json_file))
    
    print(failed_files)
    print("total:", total, "failed:", len(failed_files))

def main_single_thread(directory, logDir, outputdir, isJson=True, useDirName=False):
    failed_files = []
    if isJson:
        json_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.json')]
    else:
        json_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('_combined.bc')]
    
    total = len(json_files)

    for f in json_files:
        success = run_klee_on_file(f, logDir, outputdir, useDirName)
        if not success:
            failed_files.append(f)
    
    print(failed_files)
    print("total:", total, "failed:", len(failed_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc", type=str)
    parser.add_argument("--json", type=str)
    parser.add_argument("--bc-dir", type=str)
    parser.add_argument("--json-dir", type=str)
    parser.add_argument("--out-dir", type=str)
    parser.add_argument("--threads", type=int, default=5)

    args = parser.parse_args()
    max_threads = args.threads
    out_dir = None
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_dir = args.out_dir

    if args.bc_dir:
        dirname = Path(args.bc_dir).name
        log_dir = os.path.dirname(__file__) + "/logs/" + dirname
        os.makedirs(log_dir, exist_ok=True)
        if not out_dir:
            out_dir = os.path.dirname(__file__) + "/out/" + dirname
            os.makedirs(out_dir, exist_ok=True)
        main_multiple_threads(args.bc_dir, log_dir, out_dir, False, max_threads)

    elif args.json_dir:
        dirname = Path(args.bc_dir).name
        log_dir = os.path.dirname(__file__) + "/logs/" + dirname
        os.makedirs(log_dir, exist_ok=True)
        if not out_dir:
            out_dir = os.path.dirname(__file__) + "/out/" + dirname
            os.makedirs(out_dir, exist_ok=True)
        main_multiple_threads(args.json_dir, log_dir, out_dir, True, max_threads)

    elif args.bc:
        out_dir = os.path.dirname(__file__) + "/out"
        os.makedirs(out_dir, exist_ok=True)
        log_dir = os.path.dirname(__file__) + "/logs"
        os.makedirs(log_dir, exist_ok=True)
        run_klee_on_file(args.bc, log_dir, out_dir)

    elif args.json:
        out_dir = os.path.dirname(__file__) + "/out"
        os.makedirs(out_dir, exist_ok=True)
        log_dir = os.path.dirname(__file__) + "/logs"
        os.makedirs(log_dir, exist_ok=True)
        run_klee_on_file(args.json, log_dir, out_dir)


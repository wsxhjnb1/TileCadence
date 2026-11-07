"""CPU-only isolated real-model strided native source emission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DECODE=(("decode_v6",0,128,"native_v6_qwen3_decode_build_v1"),("decode_v61",1,128,"native_v61_qwen3_decode_build_v1"),
    ("decode_v62",2,64,"native_v62_qwen3_decode_build_v1"),("decode_m16n128",3,128,"native_v7_qwen3_decode_m16n128w1_cg_build_v1"),
    ("decode_m16n32",4,32,"native_v7_qwen3_decode_m16n32w1_cg_build_v1"))
PREFILL=(("prefill_natural64",100,"native_v6_qwen3_prefill_native64x64_natural_build_v1"),
    ("prefill_headfast64",101,"native_v6_qwen3_prefill_native64x64_head_fast_build_v1"),
    ("prefill_share64",102,"native_v6_qwen3_prefill_shareq64x64_head_fast_build_v1"),
    ("prefill_reload32",103,"native_v7_qwen3_prefill_m64n32w4_reload_cg_build_v1"))
RESOURCE_KEYS=("effective_splits","effective_query_rows","effective_query_heads","original_gqa_ratio","stages","main_grid_ctas","combine_grid_ctas",
    "workspace_o_float_elements","workspace_lse_float_elements","block_m","block_n","kv_repeat_factor",
    *["main_"+name for name in ("threads","dynamic_shared_bytes","static_shared_bytes","registers_per_thread","local_bytes_per_thread","binary_version","ptx_version","theoretical_active_ctas_per_sm")],
    *["combine_"+name for name in ("threads","dynamic_shared_bytes","static_shared_bytes","registers_per_thread","local_bytes_per_thread","binary_version","ptx_version","theoretical_active_ctas_per_sm")],
    "phase_code","query_length","key_length","query_heads","kv_heads","causal_effective","family_code",
    "q_batch_stride","q_row_stride","q_head_stride","k_batch_stride","k_row_stride","k_head_stride","v_batch_stride","v_row_stride","v_head_stride",
    "effective_q_row_stride","effective_q_head_stride")
ABI={"forward_out":["q","k","v","out","lse","workspace_o","workspace_lse","num_splits","mapping","scale","causal"],
    "workspace_spec":["q","k","v","num_splits","mapping","causal"],"workspace_spec_keys":["effective_splits","O_float_elements","LSE_float_elements"],
    "resources":["q","k","v","num_splits","mapping","causal"]}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def need(ok,message):
    if not ok:raise ValueError(message)

def emit(out):
    need(not out.exists(),"Retain previous emission attempts");out.mkdir(parents=True)
    vendor=ROOT/"vendor_spark_pin_v6/vendor";vendor_pin=read(vendor/"manifest.json")
    need(vendor_pin["commit"]=="49774b31c8d1172a5b28f72e9aed0f322bf4ccba","Wrong FA2 pin")
    records=[]
    for phase,entries in (("decode",DECODE),("prefill",PREFILL)):
        for entry in entries:
            family,code,parent=entry[0],entry[1],entry[-1];target=out/family;target.mkdir()
            for name in (phase+".cu","common.cuh","binding.cpp"):shutil.copyfile(HERE/name,target/name)
            shutil.copyfile(ROOT/"native_qwen3_v7/v7_traits.cuh",target/"v7_traits.cuh")
            if phase=="decode":
                traits=(ROOT/"native_qwen3_v10/family_traits.cuh").read_text().replace('"config.cuh"','"config.cuh"').replace("V10_FAMILY_CODE","MODEL_FAMILY_CODE").replace("V10_CANDIDATE_BLOCK_N","MODEL_CANDIDATE_BLOCK_N")
                (target/"decode_traits.cuh").write_text(traits)
                config=f"#pragma once\n#define MODEL_FAMILY_CODE {code}\n#define MODEL_CANDIDATE_BLOCK_N {entry[2]}\n"
            else:
                config=(ROOT/"spark"/parent/"source/config.cuh").read_text()
                config+=f"\n#define MODEL_FAMILY_CODE {code}\n#define MODEL_PREFILL_V7_TRAITS {int(code==103)}\n"
            (target/"config.cuh").write_text(config)
            origin=ROOT/"spark"/parent/"source"/(phase+".cu")
            manifest=dict(schema="real-qwen3-strided-native-model-family-v11",state="source_only_not_compiled_or_GPU_validated",phase=phase,family=family,family_code=code,
                parent_family=parent,parent_source_sha256=sha(origin),upstream_commit=vendor_pin["commit"],vendor_manifest_sha256=sha(vendor/"manifest.json"),
                source_sha256={p.name:sha(p) for p in target.iterdir() if p.is_file()},abi=ABI,resource_keys=RESOURCE_KEYS,
                layout="Actual safe strided BNHD, lastdim1, positive nonoverlapping aligned strides; no packing, expansion or copies",
                native_default=dict(mapping=0,num_splits=0,force_split_exposed=False),candidate_mapping=1,
                cache_semantics="HF DynamicCache update already complete before attention; all supplied K/V positions visible; NONAPPEND; no cache mutation",
                scope="BF16 Hq32/Hkv8/D128; full equal-length no-past causal prefill or q1 decode; caller scale passed unchanged asfloat",
                output="Common contiguous BNHD O plus complete float32 natural-log LSE[B,32,Tq]",no_hidden_D2H=True,
                actual_compiled_resources_required=True,no_inherited_latency_validation_claim=True)
            (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            records.append(dict(family=family,phase=phase,family_code=code,parent_family=parent,source=str(target.resolve()),manifest_sha256=sha(target/"manifest.json")))
    mapping=dict(schema="real-qwen3-strided-native-source-map-v11",families=records,abi=ABI,resource_keys=RESOURCE_KEYS,
        generator_sha256=sha(Path(__file__)),GPU_launched=False,compiler_launched=False)
    (out/"mapping.json").write_text(json.dumps(mapping,indent=2)+"\n");return mapping
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--out",required=True,type=Path)
    print(json.dumps(emit(ap.parse_args().out)),flush=True)

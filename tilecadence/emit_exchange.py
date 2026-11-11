"""V5.2 exact4KiB ready-mask/last-commit-overlap CUDA/CuTe lowering.

CPU generation imports no Torch/CUDA runtime and launches no device work.
The initial backend is exactly N64/D128 with two independently live Dv64
accumulators. Cost units and source liveness are never called GPU completion.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "cuda" / "exchange"
SLAB = 4096
ALLOWED = {"copy_g2s", "load_s2r", "store_r2s", "qk", "stats_first", "stats_next",
           "recover_p", "pv_first", "rescale_o", "pv_next", "normalize_store", "store_lse"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapping(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text())
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("Expected a typed problem/result, JSON path, or mapping")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate(problem, result):
    from . import model
    p, r = _mapping(problem), _mapping(result)
    typed = model.problem_from_dict(p)
    checked = model.replay(typed, r)
    if r.get("status") not in ("OPTIMAL", "FEASIBLE"):
        raise ValueError("A feasible retained V5 ready-aware solver result is required")
    shape = typed.provenance.get("shape")
    if shape != {"M": 64, "N": 32, "D": 128, "Dv_stripe": 64, "warps": 4, "kv_tiles": 2}:
        raise ValueError("Native ready-aware backend is fixed to M64/N32/D128/two Dv64 stripes")
    sizes = {"q": (16384, 16384), "k": (8192, 32768), "v": (4096, 16384),
             "s": (8192, 8192), "p": (4096, 4096), "m": (256, 1024),
             "l": (256, 1024), "alpha": (256, 1024)}
    for value in typed.values:
        expected=(8192,32768) if value.role=="v" and value.stripe is None else sizes.get(value.role)
        if expected is None or (value.logical_bytes, value.register_bytes) != expected:
            raise ValueError("Declared operand geometry/replication disagrees with this CuTe backend")
    if typed.provenance["backend_spec"]["row_sum_representation"] != "full_sum_quad_replicated":
        raise ValueError("Native ready-aware backend requires full replicated row sums")
    if typed.provenance["backend_spec"].get("copy_semantics") != "async_issue_ready_mask_wait":
        raise ValueError("Native V5.1 requires exact ready-cutoff/read-mask copy semantics")
    copies = {c["id"]: dict(c) for c in r["copies"]}
    events = sorted(r["events"], key=lambda e: (e["start"], e["id"]))
    if any(e["kind"] not in ALLOWED for e in events):
        raise ValueError("Unsupported ready-aware action kind")
    states = {state["stripe"]: state for state in r["output_states"]}
    if set(states) != {0, 1} or any(s["bytes"] != 16384 for s in states.values()):
        raise ValueError("Both independently live 16KiB O stripes are required")
    for event in events:
        work=[copies[x] for x in event.get("workspace_copies",[])]
        requested=event.get("workspace_smem_bytes",0)
        if event["kind"]=="normalize_store":
            if requested not in (0,8192) or len(work)!=(1 if requested else 0):
                raise ValueError("O close requires no scratch or exactly one 8KiB shared workspace")
        elif requested or work:
            raise ValueError("Only O close has an implemented native shared workspace")
        for c in work:
            if not c.get("workspace") or c["role"]!="workspace" or c["tier"]!="smem" or c["occupied_bytes"]!=8192:
                raise ValueError("Invalid explicit O close workspace binding")
            if c["offset_bytes"]%128 or c["start"]!=event["start"] or c["end"]!=event["end"]:
                raise ValueError("O close workspace must be aligned and scoped to its source action")
    return p, r, typed, copies, events, checked


def source_for(problem, result, *, opaque_recovery=True, completion_guard="none",conversion="legacy",output_cache="default"):
    """Return CUDA source + manifest without compiling or launching anything.

    opaque_recovery adds a real volatile load of a caller-owned zero nonce and
    bitwise-XOR identity before repeated QK/pure P recovery. Its overhead must
    be included in that mode's prices and confirmed by binary inspection.
    trap_nan is an optional result-consuming diagnostic, not a production fence
    proof. It is explicitly unaudited until a matching compiled audit exists.
    """
    if not isinstance(opaque_recovery,bool) or completion_guard!="none":
        raise ValueError("V5.1 exact ready-mask domain requires no diagnostic/recovery completion guard")
    if conversion not in ("legacy","packed_rn") or output_cache not in ("default","streaming_cs"):
        raise ValueError("Declare explicit ordinary-output conversion and cache policies")
    p, r, typed, copies, events, checked = validate(problem, result)
    names = {identity: f"r{i}" for i, identity in enumerate(copies)}
    smem = {identity: f"s{i}" for i, identity in enumerate(copies) if copies[identity]["tier"] == "smem" and not copies[identity].get("workspace")}
    opaque = [i for i, e in enumerate(events) if opaque_recovery and
              (e["kind"] == "recover_p" or (e["kind"] == "qk" and e["instance"] > 0))]
    lines = [
        "// Generated from V3 solver events, concrete copy bindings and shared byte offsets.",
        "// Initial numerical domain: complete N64 attention, D128, two Dv64 output stripes.",
        '#include <ATen/ATen.h>', '#include <c10/cuda/CUDAException.h>',
        '#include <c10/cuda/CUDAStream.h>',
        f'#define FA2_V5_OUTPUT_PACKED_CVT {int(conversion=="packed_rn")}',
        f'#define FA2_V5_OUTPUT_STREAMING_STORE {int(output_cache=="streaming_cs")}',
        '#include "backend.cuh"',
        "using namespace cute;", "using namespace fa2_solver_striped_v5_ready;", "",
        "namespace TORCH_EXTENSION_NAME {",
        "template<class Element, bool Causal>",
        "__global__ void solver_attention(const Element* Q,const Element* K,const Element* V,",
        "    Element* O,float* LSE,const uint32_t* NONCE,int H) {",
        "    static_assert(sizeof(Element)==2);",
        "    extern __shared__ __align__(128) unsigned char arena[];",
        "    const int batch=blockIdx.y, head=blockIdx.z;",
        "    const int row_stride=H*128;",
        "    const int64_t head_base=(int64_t(batch)*64*H+head)*128;",
        "    const Element* q_base=Q+head_base;",
        "    const Element* k_base=K+head_base;",
        "    const Element* v_base=V+head_base;",
        "    Element* o_base=O+head_base;",
        "    float* lse_base=LSE+(int64_t(batch)*H+head)*64;",
        "    typename Backend<Element>::Mma mma;",
        "    auto thread=mma.get_thread_slice(threadIdx.x);",
    ]
    for identity, name in smem.items():
        c = copies[identity]
        role, offset = c["role"], c["offset_bytes"]
        scalar = "float" if role == "s" else "Element"
        layout = {"q": "SmemQ", "k": "SmemK", "v": "SmemV", "s": "Checkpoint", "p": "Checkpoint"}.get(role)
        if role=="v" and c["stripe"] is None:layout="SmemV128"
        if layout is None:
            raise ValueError("Row snapshots are RF-only in this initial native backend")
        lines.append(f"    auto {name}=make_tensor(make_smem_ptr(reinterpret_cast<{scalar}*>(arena+{offset})),typename Backend<Element>::{layout}{{}});")
    event_lines = {}
    guard_arguments = {}
    primitives = {}
    opened = set()
    unreleased_reads=[]
    pending_groups=0
    max_pending_groups=0
    ready_cutoff=0
    hmma_gap=ex2_gap=0
    site_weights={'qk':(32,0),'pv_first':(16,0),'pv_next':(16,0),'stats_first':(0,16),'stats_next':(0,18)}
    by_event={e["id"]:e for e in events}
    oracle={row["id"]:row for row in r["lowering_trace"]}

    def shared_span(c):
        return (c["offset_bytes"],c["offset_bytes"]+c["occupied_bytes"])

    def overlaps(a,b):
        return a[0]<b[1] and b[0]<a[1]

    def read_mask():
        result=0
        for identity,(lo,hi) in unreleased_reads:
            if lo%SLAB or hi%SLAB:raise ValueError("V5 read span must use exact physical slabs")
            result |= ((1<<((hi-lo)//SLAB))-1) << (lo//SLAB)
        return result

    def emit(text):
        lines.append("    " + text)

    for index, event in enumerate(events):
        kind = event["kind"]
        inputs = [copies[x] for x in event["inputs"]]
        outputs = [copies[x] for x in event["outputs"]]
        ins = [names[c["id"]] for c in inputs]
        outs = [names[c["id"]] for c in outputs]
        event_lines[event["id"]] = {"comment_line": len(lines)+1}
        emit("// EVENT " + event["id"])
        consumed = []
        primitives[event["id"]]={"kind":kind,"workspace_copies":event.get("workspace_copies",[]),
            "read_mask_before":read_mask(),"ready_cutoff_before":ready_cutoff,
            "last_commit_hmma_before":hmma_gap,"last_commit_ex2_before":ex2_gap,
            "wait_before_load":False,"acquire_before_load":False}
        writes=[c for c in outputs if c["tier"]=="smem"] if kind in ("copy_g2s","store_r2s") else []
        if kind=="normalize_store":writes.extend(copies[x] for x in event.get("workspace_copies",[]))
        hazards=[[c["id"],old_id] for c in writes for old_id,old_span in unreleased_reads
                 if overlaps(shared_span(c),old_span)]
        if hazards:
            emit("release_shared_reads();  // actual overlapping shared write after earlier distributed reads")
            unreleased_reads.clear()
        primitives[event["id"]].update(release_before_write=bool(hazards),shared_read_write_hazards=hazards)
        if kind == "copy_g2s":
            src, dst = inputs[0], outputs[0]
            role, tile, stripe = dst["role"], dst["tile"], dst["stripe"]
            if role == "q":
                emit(f"issue_g2s<Element,64,128>(q_base,row_stride,{smem[dst['id']]});")
            elif role == "k":
                emit(f"issue_g2s<Element,32,128>(k_base+{tile*32}*row_stride,row_stride,{smem[dst['id']]});")
            elif role == "v":
                if stripe is None:
                    emit(f"issue_g2s<Element,32,128>(v_base+{tile*32}*row_stride,row_stride,{smem[dst['id']]});")
                else:
                    emit(f"issue_g2s<Element,32,64>(v_base+{tile*32}*row_stride+{stripe*64},row_stride,{smem[dst['id']]});")
            else:
                raise ValueError("Only actual immutable input Q/K/V have global copies")
            primitives[event["id"]].update(primitive="cp_async_issue_commit",consumer_completion_included=False)
            pending_groups+=1
            max_pending_groups=max(max_pending_groups,pending_groups)
            if pending_groups>8:
                raise ValueError("More than eight outstanding copy groups: backend action graph unsupported")
        elif kind == "load_s2r":
            src, dst = inputs[0], outputs[0]
            role = dst["role"]
            producer=by_event[src["producer"]]
            checkpoint = role in ("s", "p")
            if checkpoint:
                if producer["kind"] != "store_r2s":
                    raise ValueError("Checkpoint requires its synchronized store")
                wait = False
            else:
                if producer["kind"] != "copy_g2s":
                    raise ValueError("Operand requires its earlier committed input copy")
                wait = producer["end"] > ready_cutoff
            if role in ("q", "k"):
                emit(f"auto {outs[0]}=load_operand_ready<Element,{str(role=='q').lower()},{str(wait).lower()}>({smem[src['id']]},mma);")
            elif role == "v":
                if src["stripe"] is None:
                    emit(f"auto {outs[0]}=load_v_parent_ready<Element,{dst['stripe']},{str(wait).lower()}>({smem[src['id']]},mma);")
                else:
                    emit(f"auto {outs[0]}=load_v_ready<Element,{str(wait).lower()}>({smem[src['id']]},mma);")
            elif role in ("s", "p"):
                scalar = "float" if role == "s" else "Element"
                emit(f"auto {outs[0]}=load_checkpoint<{scalar}>({smem[src['id']]},mma);")
            else:
                raise ValueError("Unsupported shared load role")
            if wait:
                unreleased_reads.clear()
                pending_groups=0
                ready_cutoff=event["end"]
            read_id=src['id'];read_span=shared_span(src)
            if role=='v' and src['stripe'] is None:
                read_id+=':view'+str(dst['stripe'])
                start=src['offset_bytes']+dst['stripe']*4096
                read_span=(start,start+4096)
            if read_id not in [identity for identity,span in unreleased_reads]:
                unreleased_reads.append((read_id,read_span))
            primitive = "load_checkpoint" if checkpoint else ("ldsm_wait0_acquire" if wait else "ldsm_ready_without_acquire")
            primitives[event["id"]].update(primitive=primitive,
                wait_all_older_groups=wait,wait_before_load=wait,acquire_before_load=wait,unconditional_release_cta_barrier=False)
            consumed = outs
        elif kind == "store_r2s":
            emit(f"store_checkpoint({ins[0]},{smem[outputs[0]['id']]},mma);")
            unreleased_reads.clear()  # actual store_checkpoint CTA barrier
            primitives[event["id"]].update(primitive="store_checkpoint_with_cta_barrier")
            consumed = ins
        elif kind == "qk":
            qarg = ins[0]
            if index in opaque:
                emit(f"auto opaque_q_{index}=opaque_half_operand({qarg},NONCE,{index});")
                qarg = f"opaque_q_{index}"
            emit(f"auto {outs[0]}=qk(mma,{qarg},{ins[1]});")
            consumed = outs
        elif kind in ("stats_first", "stats_next"):
            for c, name in zip(outputs, outs):
                if c["role"] == "p":
                    emit(f"auto {name}=make_tensor<Element>({ins[0]}.layout());")
                else:
                    emit(f"auto {name}=make_tensor<float>(Shape<_2>{{}});")
            tile = event["tile"]
            if kind == "stats_first":
                emit(f"stats_first<Causal,{tile}>({ins[0]},{outs[0]},{outs[1]},{outs[2]},mma);")
            else:
                emit(f"stats_next<Causal,{tile}>({ins[0]},{ins[1]},{ins[2]},{outs[0]},{outs[1]},{outs[2]},{outs[3]},mma);")
            consumed = outs
        elif kind == "recover_p":
            sarg = ins[0]
            if index in opaque:
                emit(f"auto opaque_s_{index}=opaque_score_operand({sarg},NONCE,{index});")
                sarg = f"opaque_s_{index}"
            emit(f"auto {outs[0]}=make_tensor<Element>({ins[0]}.layout());")
            emit(f"recover_p<Causal,{event['tile']}>({sarg},{ins[1]},{outs[0]},mma);")
            consumed = outs
        elif kind in ("pv_first", "pv_next"):
            stripe = event["stripe"]
            if kind == "pv_first":
                if stripe in opened:
                    raise ValueError("An O stripe may only be opened once")
                emit(f"auto o{stripe}=partition_fragment_C(mma,Shape<_64,_64>{{}});")
                emit(f"cute::clear(o{stripe});  // real zero-C first PV; O lifetime begins here")
                opened.add(stripe)
            elif stripe not in opened:
                raise ValueError("PV update before O allocation")
            emit(f"pv(mma,{ins[0]},{ins[1]},o{stripe});")
            consumed = [f"o{stripe}"]
        elif kind == "rescale_o":
            emit(f"rescale(o{event['stripe']},{ins[0]});")
            consumed = [f"o{event['stripe']}"]
        elif kind == "normalize_store":
            stripe = event["stripe"]
            work=event.get("workspace_copies",[])
            if work:
                scratch=copies[work[0]]
                emit(f"normalize_store_coalesced128<Element,{stripe}>(o{stripe},{ins[0]},o_base,row_stride,mma,arena+{scratch['offset_bytes']});")
                primitives[event["id"]].update(primitive="close_shared128",scratch_offset_bytes=scratch["offset_bytes"],scratch_bytes=8192)
                unreleased_reads.clear()  # actual shared writes -> CTA barrier
                unreleased_reads.append((scratch["id"],shared_span(scratch)))
            else:
                emit(f"normalize_store_packed32<Element,{stripe}>(o{stripe},{ins[0]},o_base,row_stride,mma);")
                primitives[event["id"]].update(primitive="close_raw_packed32",scratch_bytes=0)
            emit(f"// O{stripe} is closed: no later source action may consume its accumulator.")
            consumed = [f"o{stripe}"]
        elif kind == "store_lse":
            emit(f"store_lse({ins[0]},{ins[1]},lse_base);")
            consumed = ins
        else:
            raise ValueError("Unsupported action")
        guard_arguments[event["id"]] = consumed
        if completion_guard == "trap_nan" and consumed:
            for name in consumed:
                emit(f"diagnostic_consume_non_nan({name});")
            emit("__syncthreads();  // diagnostic request only; compiled completion must be audited")
            unreleased_reads.clear()
        event_lines[event["id"]]["last_line"] = len(lines)
        entry=primitives[event["id"]]
        wait=entry['wait_before_load']
        entry.update(last_commit_wait_hmma_sites=hmma_gap if wait else 0,last_commit_wait_ex2_sites=ex2_gap if wait else 0)
        if kind=='copy_g2s' or wait:hmma_gap=ex2_gap=0
        else:
            hmma,ex2=site_weights.get(kind,(0,0));hmma_gap+=hmma;ex2_gap+=ex2
        entry.update(last_commit_hmma_after=hmma_gap,last_commit_ex2_after=ex2_gap)
        entry.update(read_mask_after=read_mask(),ready_cutoff_after=ready_cutoff,pending_copy_groups_after=pending_groups)
        expected=oracle[event["id"]]
        if any(entry[key]!=expected[key] for key in ("read_mask_before","read_mask_after","ready_cutoff_before",
                "ready_cutoff_after","wait_before_load","acquire_before_load","release_before_write","shared_read_write_hazards",
                'last_commit_hmma_before','last_commit_hmma_after','last_commit_ex2_before','last_commit_ex2_after',
                'last_commit_wait_hmma_sites','last_commit_wait_ex2_sites')):
            raise ValueError("Independent CUDA lowering differs from ready/mask certificate at "+event["id"])
        if pending_groups!=expected["issued_groups_after"]-expected["ready_groups_after"]:
            raise ValueError("Committed/completed copy-group certificate differs")
    high_water=max(c["offset_bytes"]+c["occupied_bytes"] for c in copies.values() if c["tier"]=="smem")
    arena_bytes=(high_water+127)//128*128
    if arena_bytes>typed.arena_slabs*SLAB:
        raise ValueError("Actual copy/workspace spans exceed the declared arena")
    if "residency_exchange_v78" in typed.provenance:
        arena_bytes = typed.arena_slabs * SLAB  # Match dynamic SMEM allocation across exchange controls.
    lines.extend(["}", "", "namespace {",
        f"constexpr int kArenaBytes={arena_bytes};",
        f"constexpr int kNonceElements={len(events) if opaque else 0};",
        f"constexpr bool kDiagnostic={str(completion_guard=='trap_nan').lower()};",
        "const void* selected(const at::Tensor& q,bool causal) {",
        "    if(q.scalar_type()==at::kHalf) return causal ? reinterpret_cast<const void*>(&solver_attention<cutlass::half_t,true>) : reinterpret_cast<const void*>(&solver_attention<cutlass::half_t,false>);",
        "    return causal ? reinterpret_cast<const void*>(&solver_attention<cutlass::bfloat16_t,true>) : reinterpret_cast<const void*>(&solver_attention<cutlass::bfloat16_t,false>);",
        "}",
        "void configure(const void* function) {",
        "    if(kArenaBytes>=48*1024) C10_CUDA_CHECK(cudaFuncSetAttribute(function,cudaFuncAttributeMaxDynamicSharedMemorySize,kArenaBytes));",
        "}", "}",
        "int plan_nonce_elements(){return kNonceElements;}",
        "bool plan_diagnostic_guard(){return kDiagnostic;}",
        "void plan_forward_cuda(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,at::Tensor& out,at::Tensor& lse,const at::Tensor& nonce,bool causal) {",
        "    const void* function=selected(q,causal);configure(function);",
        "    const void* qp=q.data_ptr();const void* kp=k.data_ptr();const void* vp=v.data_ptr();",
        "    void* op=out.data_ptr();void* lp=lse.data_ptr();const void* np=nonce.data_ptr();int h=static_cast<int>(q.size(2));",
        "    void* arguments[]={&qp,&kp,&vp,&op,&lp,&np,&h};",
        "    auto stream=c10::cuda::getCurrentCUDAStream(q.get_device());",
        "    C10_CUDA_CHECK(cudaLaunchKernel(function,dim3(1,q.size(0),q.size(2)),dim3(128),arguments,kArenaBytes,stream.stream()));",
        "    C10_CUDA_KERNEL_LAUNCH_CHECK();", "}",
        "std::vector<int64_t> plan_resources_cuda(const at::Tensor& q,bool causal) {",
        "    const void* function=selected(q,causal);configure(function);cudaFuncAttributes a{};",
        "    C10_CUDA_CHECK(cudaFuncGetAttributes(&a,function));int blocks=0;",
        "    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,function,128,kArenaBytes));",
        "    return {64,32,128,64,128,kArenaBytes,static_cast<int64_t>(a.sharedSizeBytes),a.numRegs,static_cast<int64_t>(a.localSizeBytes),a.binaryVersion,a.ptxVersion,blocks,kNonceElements,kDiagnostic};", "}",
        "} // namespace TORCH_EXTENSION_NAME", ""])
    source = "\n".join(lines)
    metadata = {"backend": "cuda-cute-v52-fourk-ready-mask-sm80-mma", "copy_fix_revision": "Exact4KiB physical spans, completed-copy cutoff and pending distributed-read union", "geometry": typed.provenance["shape"],
        "operator_domain": "Complete ordinary exact-N64 forward, BNHD contiguous, O+natural-log LSE",
        "source_order_only": True, "binary_completion_claim": False,
        "compiled_recovery_execution_claim": False, "physical_RF_release_claim": False,
        "required_shared_bytes":arena_bytes,"declared_solver_arena_bytes":typed.arena_slabs*SLAB,
        "shared_allocation_unit_bytes":SLAB,"v52_is_post_V5_development_amendment":True,
        "last_commit_wait_math_proxy_weights":site_weights,
        "overlap_proxy_boundary":"Source HMMA/EX2 sites only between last new commit and next genuine wait0; exact ordinal recurrence, actual compiled interposition must be audited; not measured overlap/cycle costs",
        "shared_address_high_water_bytes":high_water,"lowering_primitives":primitives,
        "async_copy_contract":"Each copy commits one group; loads wait0+CTA iff producer is newer than completed cutoff; ready loads retain pending shared-read union without CTA; overlapping writes release entire union; close exchange does not complete async groups",
        "source_ready_mask_certificate":r["lowering_trace"],"model_revision":typed.provenance["model_revision"],
        "value_copy_policy":"parent128" if typed.provenance["backend_spec"]["full_value_parent"] else "split64",
        "parent_view_contract":"Each N32xD128 V parent occupies8192 bytes until both64-column RF views load; each RF view remains4096 logical/16384 replicated bytes; no duplicate V global copy",
        "output_conversion":conversion,"output_cache_policy":output_cache,"cost_model_v42_eligible":False,
        "max_pending_copy_groups":max_pending_groups,"pending_copy_group_limit":8,
        "opaque_recovery": opaque_recovery, "required_nonce_elements": len(events) if opaque else 0,
        "nonce_contract": "Caller-owned immutable zero int32/uint32 vector, initialized before capture and explicitly attested; volatile load+XOR overhead is real",
        "opaque_events": [{"event_id": events[i]["id"], "nonce_slot": i,
            "extra_work": "volatile global u32 nonce read plus XOR of all Q raw16 bits" if events[i]["kind"] == "qk" else "volatile global u32 nonce read plus XOR of all S raw32 bits"} for i in opaque],
        "completion_guard": completion_guard, "guard_outputs": guard_arguments,
        "guard_scope": "trap_nan consumes every scalar then CTA barrier; bounded non-NaN diagnostics, overhead included, machine order unproved",
        "event_source_lines": event_lines, "events": events, "copies": list(copies.values()),
        "output_states": r["output_states"], "independent_replay": checked,
        "problem_sha256": hashlib.sha256(_canonical(p)).hexdigest(),
        "result_sha256": hashlib.sha256(_canonical(r)).hexdigest(),
        "emitter_sha256": sha(__file__),
        "backend_source_sha256": {name: sha(BACKEND/name) for name in ("backend.cuh", "binding.cpp","output_bits.cuh")},
        "resource_keys": ["block_m","block_n","head_dim","value_stripe_dim","threads",
            "dynamic_shared_bytes","static_shared_bytes","registers_per_thread","local_bytes_per_thread",
            "binary_version","ptx_version","theoretical_active_ctas_per_sm","required_nonce_elements","diagnostic_guard"],
        "load_contract": "torch.utils.cpp_extension.load(is_python_module=False); use torch.ops.<module_name>.forward/forward_out/resources; all host/kernel symbols namespaced by TORCH_EXTENSION_NAME; no Python development headers",
        "native_dependency": {"source_revision": "49774b31c8d1172a5b28f72e9aed0f322bf4ccba",
            "headers": ["csrc/flash_attn/src/utils.h", "csrc/cutlass/include"],
            "architecture": "Root must supply verified target/compiler; SM80 MMA requires compute capability >=80",
            "binary": "Separate extension; installed native FA2 is never replaced"}}
    return source, metadata


def emit(problem, result, outdir, *, opaque_recovery=True, completion_guard="none",conversion="legacy",output_cache="default"):
    out = Path(outdir)
    if out.exists():
        raise FileExistsError("Never overwrite a retained generated plan")
    source, metadata = source_for(problem, result, opaque_recovery=opaque_recovery, completion_guard=completion_guard,
                                  conversion=conversion,output_cache=output_cache)
    out.mkdir(parents=True, exist_ok=False)
    (out/"plan.cu").write_text(source)
    for name in ("backend.cuh", "binding.cpp","output_bits.cuh"):
        shutil.copyfile(BACKEND/name, out/name)
    metadata.update(state="emitted_not_compiled_or_executed", cuda_source_sha256=sha(out/"plan.cu"))
    (out/"manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem",type=Path,required=True)
    parser.add_argument("--result",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--no-opaque-recovery",action="store_true")
    parser.add_argument("--completion-guard",choices=("none","trap_nan"),default="none")
    parser.add_argument("--conversion",choices=("legacy","packed_rn"),default="legacy")
    parser.add_argument("--output-cache",choices=("default","streaming_cs"),default="default")
    args=parser.parse_args()
    manifest=emit(args.problem,args.result,args.out,opaque_recovery=not args.no_opaque_recovery,
                  completion_guard=args.completion_guard,conversion=args.conversion,output_cache=args.output_cache)
    print(json.dumps({"state":manifest["state"],"events":len(manifest["events"]),
        "required_shared_bytes":manifest["required_shared_bytes"],
        "required_nonce_elements":manifest["required_nonce_elements"],"out":str(args.out.resolve())}))


if __name__=="__main__":main()

"""Solve and emit CUDA/CuTe decode schedules for strided long-context Qwen3."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from native_model_v11.prepare import ABI, RESOURCE_KEYS


HERE = Path(__file__).resolve().parent
DEFAULT_VENDOR = HERE / 'vendor_spark_pin_v6' / 'vendor'
PARENT = HERE / 'native_model_v11' / 'emitted_v3' / 'decode_m16n128'
FA2_COMMIT = '49774b31c8d1172a5b28f72e9aed0f322bf4ccba'
VENDOR_MANIFEST_SHA = '919ac9c0333363d56fede38963da44266ea5d996f9ec178da4c7c55f02141e45'
KERNEL_HEADER_SHA = '765dd3ef217bc9d79c9c0494ba52ea63767099be737c14604bec748d85f0dde3'
PARENT_FILES = {
    'binding.cpp': '0482b02c3619fdeb62a183a42844b652a32f4d65fc9fb1b4eab6789c65d9ac31',
    'common.cuh': '2a3acac58065961eb603c60232b72bc14887a83c09a2d90ae1e89f83b6081f3c',
    'v7_traits.cuh': '74dc76a1f3d66452dcabd4224072c60351e4fa0f667ee818829992197e6a26d7',
}
BLOCK_N = (32, 64, 128)
WARPS = (1, 2, 4)
Q_RESIDENCY = ('reload', 'registers', 'registers_alias_qk')
SPLITS = (1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32)
SOURCE_FIELDS = ('block_n', 'warps', 'q_residency', 'cache_policy')
SHARED_LIMIT_BYTES = 99 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def geometry(block_n: int, warps: int, residency: int, cache: int) -> dict:
    block_m = 16 * warps
    q_bytes = block_m * 128 * 2
    kv_bytes = 2 * block_n * 128 * 2
    live_bytes = max(q_bytes, kv_bytes) if residency == 2 else q_bytes + kv_bytes
    family_code = 200 + (((BLOCK_N.index(block_n) * 3 + WARPS.index(warps)) * 3 + residency) * 2 + cache)
    return dict(
        phase='decode', family=f'abld_n{block_n}_w{warps}_q{residency}_c{cache}',
        family_code=family_code, block_m=block_m, block_n=block_n, warps=warps,
        q_residency=residency, cache_policy=cache,
        split_shared_bytes=max(live_bytes, 2 * q_bytes),
        direct_shared_bytes=max(live_bytes, q_bytes),
        q_register_bytes=q_bytes if residency != 0 else 0,
        tile_kv_bytes=kv_bytes,
    )


def enumerate_plans(batch: int, prompt: int) -> list[dict]:
    """Enumerate the finite legal source and launch assignments before timing."""
    if batch not in (1, 4) or prompt not in (8192, 32768):
        raise ValueError('Decode study requires B1/B4 and N8192/N32768')
    plans = []
    for block_n, warps, residency, cache in itertools.product(BLOCK_N, WARPS, range(3), range(2)):
        source = geometry(block_n, warps, residency, cache)
        tiles = [ceildiv(prompt + step, block_n) for step in range(1, 9)]
        for splits in SPLITS:
            shared = source['split_shared_bytes' if splits > 1 else 'direct_shared_bytes']
            if splits > min(tiles) or shared > SHARED_LIMIT_BYTES:
                continue
            max_tiles = ceildiv(max(tiles), splits)
            plans.append(dict(
                **source, candidate_id=f"{source['family']}_s{splits}", batch=batch,
                prompt=prompt, splits=splits, mapping=1, key_length_min=prompt + 1,
                key_length_max=prompt + 8, max_tiles_per_split=max_tiles,
                min_tiles_per_split=min(tiles) // splits,
                main_grid_ctas=batch * 8 * splits, dynamic_shared_bytes=shared,
                workspace_o_float_elements=batch * 32 * 128 * splits if splits > 1 else 0,
                workspace_lse_float_elements=batch * 32 * splits if splits > 1 else 0,
                q_smem_reads_per_cta=max_tiles if residency == 0 else 1,
                qk_shared_alias=int(residency == 2),
            ))
    return plans


def solve_plan(
    batch: int, prompt: int, *, fixed: dict | None = None,
    excluded: tuple[str, ...] = (), objective: dict[str, int] | None = None,
) -> dict:
    """Choose an executable plan by CP-SAT with optional integer A costs."""
    from ortools.sat.python import cp_model

    fixed = fixed or {}
    plans = enumerate_plans(batch, prompt)
    unknown = set(fixed) - set(plans[0])
    if unknown:
        raise ValueError(f'Unknown decode plan fields {sorted(unknown)}')
    excluded_ids = set(excluded)
    domain = [p for p in plans if p['candidate_id'] not in excluded_ids
              and all(p[key] == value for key, value in fixed.items())
              and (objective is None or p['candidate_id'] in objective)]
    if not domain:
        raise ValueError('No legal decode schedule satisfies the requested domain')

    model = cp_model.CpModel()
    selected = model.new_int_var(0, len(domain) - 1, 'selected_plan')
    keys = ('block_m', 'block_n', 'warps', 'q_residency', 'cache_policy',
            'splits', 'dynamic_shared_bytes', 'min_tiles_per_split', 'q_smem_reads_per_cta')
    variables = {key: model.new_int_var_from_domain(
        cp_model.Domain.from_values(sorted({p[key] for p in domain})), key) for key in keys}
    model.add_allowed_assignments([selected, *variables.values()], [
        [i, *(p[key] for key in keys)] for i, p in enumerate(domain)
    ])
    model.add(variables['block_m'] == 16 * variables['warps'])
    model.add(variables['dynamic_shared_bytes'] <= SHARED_LIMIT_BYTES)
    model.add(variables['min_tiles_per_split'] >= 1)

    # Seed objective is an analytical work proxy. A can replace it with a
    # complete mapping of integer costs over the plans it wants B to consider.
    def cost(plan: dict) -> int:
        if objective is not None:
            value = objective[plan['candidate_id']]
            if type(value) is not int or abs(value) >= 10**12:
                raise ValueError('A objective must contain bounded integer costs')
            return value
        waves = ceildiv(plan['main_grid_ctas'], 48)
        tile_work = plan['max_tiles_per_split'] * plan['block_n'] * plan['warps']
        return waves * (tile_work + 4 * plan['q_smem_reads_per_cta']) + 12 * plan['splits']

    costs = [cost(plan) for plan in domain]
    score = model.new_int_var(min(costs), max(costs), 'schedule_cost')
    model.add_element(selected, costs, score)
    model.minimize(score * len(domain) + selected)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 17023
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f'Decode schedule solver returned {solver.status_name(status)}')
    plan = dict(domain[solver.value(selected)])
    plan['solver'] = dict(
        status='OPTIMAL', allowed_assignments=len(domain), objective_value=solver.value(score),
        objective_source='hardware_feedback' if objective is not None else 'analytic_seed',
        model_sha256=sha(str(model.proto).encode()), seed=17023,
        assignment={name: solver.value(var) for name, var in variables.items()},
    )
    return plan


def substitute_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError('Pinned FA2 code anchor does not match')
    return text.replace(old, new, 1)


def emit_splitkv(header: bytes) -> bytes:
    """Isolate the pinned split-KV operation and add executable Q lifetimes."""
    if sha(header) != KERNEL_HEADER_SHA:
        raise ValueError('Unexpected pinned FA2 kernel header bytes')
    text = header.decode().replace('\r\n', '\n')
    start = text.index('template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>\ninline __device__ void compute_attn_1rowblock_splitkv')
    end = text.index('\n////////////////////////////////////////////////////////////////////////////////////////////////////', start)
    body = text[start:end].replace('compute_attn_1rowblock_splitkv(', 'compute_long_ab_splitkv(', 1)

    begin_partition = body.index('    const int n_blocks_per_split')
    end_partition = body.index('    if (n_block_min >= n_block_max)', begin_partition)
    body = body[:begin_partition] + '''    static_assert(!Is_causal && !Is_local && !Append_KV);
    // Distribute contiguous KV tile intervals with lengths differing by at most one.
    const int total_tiles = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    const int base_tiles = total_tiles / num_n_splits;
    const int extra_tiles = total_tiles % num_n_splits;
    const int n_block_min = n_split_idx * base_tiles + std::min(n_split_idx, extra_tiles);
    const int n_block_max = n_block_min + base_tiles + (n_split_idx < extra_tiles ? 1 : 0);
''' + body[end_partition:]

    body = substitute_once(body,
        'make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});',
        '''make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});''')
    body = substitute_once(body, '    int n_block = n_block_max - 1;\n', '''    // When Q and K alias, every Q shared read finishes before any K write.
    if constexpr (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }
    if constexpr (Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor q_register_copy = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, q_register_copy);
        __syncthreads();
    }
    int n_block = n_block_max - 1;
''')
    old = '''    // FLASH_NAMESPACE::cp_async_wait<0>();
    // __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tKsK); }
    // __syncthreads();'''
    body = substitute_once(body, old, '''    if constexpr (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        // Q and K are in separate async groups. K can continue transferring.
        FLASH_NAMESPACE::cp_async_wait<1>();
        __syncthreads();
        Tensor q_register_copy = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, q_register_copy);
    }''')
    if body.count('FLASH_NAMESPACE::gemm(\n') != 2:
        raise ValueError('Unexpected pinned split-KV QK call sites')
    body = body.replace('FLASH_NAMESPACE::gemm(\n',
                        'FLASH_NAMESPACE::gemm<Kernel_traits::Is_Q_in_regs>(\n')
    body = substitute_once(body, 'if constexpr (Split) { __syncthreads(); }',
                           'if constexpr (Split || Kernel_traits::Share_Q_K_smem) { __syncthreads(); }')
    return ('// Derived from flash_fwd_kernel.h, Copyright (c) 2024, Tri Dao.\n'
            '#pragma once\n\nnamespace FLASH_NAMESPACE {\n\n' + body +
            '\n}  // namespace FLASH_NAMESPACE\n').encode()


def emit_family(out: Path, plan: dict, vendor: Path = DEFAULT_VENDOR) -> dict:
    """Emit stable compiler inputs for the source tuple in a B assignment."""
    out, vendor = Path(out), Path(vendor)
    source = geometry(*(plan[field] for field in SOURCE_FIELDS))
    if source['family'] != plan['family']:
        raise ValueError('Decode source family differs from the requested plan')
    manifest_bytes = (vendor / 'manifest.json').read_bytes()
    if sha(manifest_bytes) != VENDOR_MANIFEST_SHA or json.loads(manifest_bytes)['commit'] != FA2_COMMIT:
        raise ValueError('Unexpected pinned FA2 vendor manifest')
    files = {}
    for name, expected in PARENT_FILES.items():
        files[name] = (PARENT / name).read_bytes()
        if sha(files[name]) != expected:
            raise ValueError(f'Pinned V11 adapter bytes differ for {name}')
    files['decode.cu'] = (HERE / 'ab_long_decode.cu').read_bytes()
    files['ab_long_splitkv.cuh'] = emit_splitkv(
        (vendor / 'csrc' / 'flash_attn' / 'src' / 'flash_fwd_kernel.h').read_bytes())
    files['config.cuh'] = (
        '#pragma once\n'
        f"#define MODEL_FAMILY_CODE {source['family_code']}\n"
        f"#define AB_DECODE_WARPS {source['warps']}\n"
        f"#define AB_DECODE_BLOCK_N {source['block_n']}\n"
        f"#define AB_DECODE_Q_RESIDENCY {source['q_residency']}\n"
        f"#define AB_DECODE_CACHE_ALWAYS {source['cache_policy']}\n"
        f"#define AB_DECODE_SPLIT_SHARED_BYTES {source['split_shared_bytes']}\n"
    ).encode()
    manifest = dict(
        schema='ab-long-qwen3-decode-family-v1', state='source_only', **source,
        upstream_commit=FA2_COMMIT, vendor_manifest_sha256=VENDOR_MANIFEST_SHA,
        parent_kernel_sha256=KERNEL_HEADER_SHA,
        source_sha256={name: sha(data) for name, data in files.items()},
        abi=ABI, resource_keys=RESOURCE_KEYS, candidate_mapping=1,
        q_lifetime=Q_RESIDENCY[source['q_residency']],
        partition='Balanced contiguous KV tile intervals. All visible keys and four GQA rows per KV head.',
        layout='Zero-copy V11 BNHD. HF DynamicCache update is already complete before attention.',
        dependency_order='Async Q completion precedes Q register preload. Q shared reads complete before aliased K writes. K/V use separate shared tiles. PV precedes output alias reuse.',
    )
    out.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (out / name).write_bytes(data)
    (out / 'manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode())
    return manifest


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--batch', type=int, required=True)
    cli.add_argument('--prompt', type=int, required=True)
    cli.add_argument('--fixed', type=json.loads)
    cli.add_argument('--out', type=Path)
    cli.add_argument('--vendor', type=Path, default=DEFAULT_VENDOR)
    args = cli.parse_args()
    plan = solve_plan(args.batch, args.prompt, fixed=args.fixed)
    if args.out:
        emit_family(args.out, plan, args.vendor)
    print(json.dumps(plan, indent=2))

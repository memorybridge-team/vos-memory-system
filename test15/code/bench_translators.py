"""Stage-by-stage translator timing on one switch-sized state: 16 records x 64x64x64 spatial (BF16 on CPU), 16x256 pointers."""
import os, sys, time, torch
which = sys.argv[1]
dev = torch.device('cuda')
torch.manual_seed(0)
x_cpu = torch.randn(1, 1, 16, 64, 64, 64).bfloat16()
p_cpu = torch.randn(1, 1, 16, 256)
if which == 'paper':
    from vos_memory_inspector.state_training.trainer import restore_model
    path = "/workspace/j700qtcgrg/CMMT-training-contracts/runs/r2-20261009T043718Z/r2-gpu-nccl-20261009T074116Z/training/delivery-final-validation-r2/best_validation_r2_weights.pth"
    models = {'paper_r2': restore_model('transformer', torch.load(path, map_location='cpu', weights_only=True)).to(dev).float().eval()}
else:
    from vos_memory_inspector.translator_artifact import load_learned_translator
    models = {n: load_learned_translator(f"/workspace/residual-vs-mlp-20261116/runs/{d}/train/paired_translators.pt", 'residual_mlp').to(dev).float().eval()
              for n, d in [('residual_mlp', 'residual_random'), ('direct_mlp', 'direct_mlp')]}
for name, m in models.items():
    params = sum(q.numel() for q in m.parameters())
    def h2d():
        return x_cpu.to(dev).float(), p_cpu.to(dev)
    def fwd(x, p):
        return m.translate_tensors(x, p)
    def d2h(a, b):
        return a.bfloat16().cpu(), b.cpu()
    with torch.inference_mode():
        for _ in range(5):
            a, b = fwd(*h2d()); d2h(a, b)
        torch.cuda.synchronize()
        res = {}
        x, p = h2d()
        for stage in ('h2d', 'forward', 'd2h', 'total_wall'):
            ts = []
            for _ in range(50):
                if stage == 'total_wall':
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    a, b = fwd(*h2d()); d2h(a, b); torch.cuda.synchronize()
                    ts.append(1000 * (time.perf_counter() - t0)); continue
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                if stage == 'h2d': h2d()
                elif stage == 'forward': a, b = fwd(x, p)
                else: d2h(a, b)
                e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
            ts.sort(); res[stage] = ts[len(ts) // 2]
        print(f"{name}: params {params:,}  " + "  ".join(f"{k} {v:.3f} ms" for k, v in res.items()), flush=True)

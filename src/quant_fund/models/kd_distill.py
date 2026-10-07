"""Knowledge distillation (Hinton et al. 2015).

Teacher (hidden=48) trained to convergence; a small student (h=8)
trained on soft targets at temperature T beats the same student
trained on hard labels alone — the dark-knowledge transfer.
"""

from __future__ import annotations

from quant_fund.models._compress_synth import acc_of, split


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("kd_distill needs the torch `nn` extra") from exc


def bench_kd_distill(
    seed: int = 353,
    n: int = 800,
    student_n: int = 150,
    teacher_h: int = 48,
    student_h: int = 8,
    temp: float = 4.0,
    iters: int = 300,
) -> dict[str, float]:
    torch = _torch()
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = split(seed, n)
    x_tr_t, y_tr_t = torch.tensor(x_tr).float(), torch.tensor(y_tr)
    x_te_t, y_te_t = torch.tensor(x_te).float(), torch.tensor(y_te)

    teacher = torch.nn.Sequential(
        torch.nn.Linear(8, teacher_h), torch.nn.ReLU(), torch.nn.Linear(teacher_h, 2)
    )
    opt = torch.optim.Adam(teacher.parameters(), lr=5e-3)
    for _i in range(iters):
        loss = torch.nn.functional.cross_entropy(teacher(x_tr_t), y_tr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    acc_t = acc_of(torch, teacher, x_te_t, y_te_t)
    with torch.no_grad():
        soft = torch.softmax(teacher(x_tr_t) / temp, -1)

    def student_train(use_kd):
        torch.manual_seed(seed)
        stu = torch.nn.Sequential(
            torch.nn.Linear(8, student_h), torch.nn.ReLU(), torch.nn.Linear(student_h, 2)
        )
        opt = torch.optim.Adam(stu.parameters(), lr=5e-3)
        xs = x_tr_t[:student_n]
        ys = y_tr_t[:student_n]
        soft_s = soft[:student_n]
        for _i in range(iters):
            if use_kd:
                logp = torch.log_softmax(stu(xs) / temp, -1)
                loss = -(soft_s * logp).sum(-1).mean() * temp * temp
            else:
                loss = torch.nn.functional.cross_entropy(stu(xs), ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return acc_of(torch, stu, x_te_t, y_te_t)

    acc_hard = student_train(False)
    acc_kd = student_train(True)
    return {
        "synthetic_kd_acc": acc_kd,
        "synthetic_kd_hard_acc": acc_hard,
        "synthetic_kd_teacher_acc": acc_t,
        "synthetic_kd_gain": acc_kd - acc_hard,
        "synthetic_torch_available": 1.0,
    }

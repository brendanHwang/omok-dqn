"""기보 기반 오프라인 DQN. 지도학습 정답 대신 Bellman 목표로 Q값을 학습한다."""
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Flatten())
        # Q값은 확률이 아니므로 sigmoid를 사용하지 않는다.
        nn.init.zeros_(self.layers[-2].weight)
        nn.init.zeros_(self.layers[-2].bias)

    def forward(self, x):
        return self.layers(x)


def targets(target, next_states, rewards, dones, gamma):
    with torch.no_grad():
        values = target(next_states).masked_fill(next_states.flatten(1) != 0, -torch.inf)
        future = values.max(dim=1).values
        # 종료 상태는 빈칸이 없어도 -inf * 0이 발생하지 않도록 먼저 0으로 치환한다.
        future = torch.where(dones, torch.zeros_like(future), future)
        return rewards + gamma * future


def read_data(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def batch(data, indices, device):
    return (torch.as_tensor(data['states'][indices, None], device=device, dtype=torch.float32),
            torch.as_tensor(data['actions'][indices], device=device),
            torch.as_tensor(data['rewards'][indices], device=device),
            torch.as_tensor(data['next_states'][indices, None], device=device, dtype=torch.float32),
            torch.as_tensor(data['dones'][indices], device=device))


def evaluate(model, target, data, device, gamma):
    losses, hits, total = 0., 0, 0
    terminal_error, terminal_count = 0., 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data['actions']), 512):
            s, a, r, ns, d = batch(data, np.arange(start, min(start+512, len(data['actions']))), device)
            q = model(s)
            selected = q.gather(1, a[:,None]).squeeze(1)
            y = targets(target, ns, r, d, gamma)
            losses += nn.functional.smooth_l1_loss(selected, y, reduction='sum').item()
            hits += (q.masked_fill(s.flatten(1) != 0, -torch.inf).argmax(1) == a).sum().item()
            total += len(a)
            terminal_error += ((selected-r).abs() * d).sum().item()
            terminal_count += d.sum().item()
    model.train()
    return {'bellman_huber':losses/total, 'recorded_move_accuracy':hits/total,
            'terminal_q_mae':terminal_error/max(terminal_count,1), 'transitions':total}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS GPU를 사용할 수 없습니다. CPU로 자동 대체하지 않습니다.')
    device = torch.device('mps')
    online = QNetwork().to(device)
    target = copy.deepcopy(online).eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)
    train_data = read_data(args.data / 'train.npz')
    val_data = read_data(args.data / 'validation.npz')
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'device': str(device), 'torch':str(torch.__version__), 'parameters':sum(p.numel() for p in online.parameters()),
                'steps':args.steps, 'batch_size':args.batch_size, 'gamma':args.gamma, 'learning_rate':args.lr,
                'target_sync':args.target_sync, 'seed':args.seed, 'algorithm':'offline DQN',
                'reward':{'win':1,'loss':-1,'ongoing_or_draw':0},
                'transition':'own turn -> own move -> opponent move -> next own turn',
                'data_manifest':json.loads((args.data/'manifest.json').read_text())}
    history=[]
    started=perf_counter()
    print(json.dumps({k:v for k,v in metadata.items() if k!='data_manifest'}), flush=True)
    for step in range(1, args.steps+1):
        indices = rng.integers(len(train_data['actions']), size=args.batch_size)
        s,a,r,ns,d = batch(train_data,indices,device)
        y = targets(target,ns,r,d,args.gamma)
        prediction = online(s).gather(1,a[:,None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(prediction,y)
        if not torch.isfinite(loss).item():
            raise RuntimeError('학습 손실이 유한하지 않습니다')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(),10.)
        optimizer.step()
        if step % args.target_sync == 0:
            target.load_state_dict(online.state_dict())
        if step == 1 or step % 250 == 0 or step == args.steps:
            torch.mps.synchronize()
            record={'step':step,'loss':loss.item(),'elapsed_seconds':perf_counter()-started}
            history.append(record); print(json.dumps(record),flush=True)
    metadata['elapsed_seconds']=perf_counter()-started
    metadata['validation']=evaluate(online,target,val_data,device,args.gamma)
    # CPU에서도 다시 불러올 수 있도록 가중치를 CPU로 저장한다.
    torch.save({'state_dict':{k:v.cpu() for k,v in online.state_dict().items()},
                'metadata':metadata},args.output/'gomoku_dqn.pt')
    (args.output/'training_report.json').write_text(json.dumps({'metadata':metadata,'history':history},indent=2))
    print(json.dumps(metadata['validation']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('dataset/dqn'))
    p.add_argument('--output',type=Path,default=Path('models/dqn-v1'))
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--target-sync',type=int,default=500)
    p.add_argument('--seed',type=int,default=2026)
    train(p.parse_args())

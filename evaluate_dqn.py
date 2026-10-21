"""저장된 DQN을 다시 불러와 고정 CNN 또는 무작위 상대와 대결한다."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from dqn_data import has_five

ROOT=Path(__file__).resolve().parent


def cnn_worker():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    model=tf.keras.models.load_model(ROOT/'models/20201213_202430.h5',compile=False)
    for line in sys.stdin:
        request=json.loads(line)
        board=np.asarray(request['board'],dtype=np.int8)
        player=request['player']
        s=np.where(board==player,1,np.where(board==0,0,-1)).astype(np.float32)
        q=model(s[None,:,:,None],training=False).numpy().reshape(20,20)
        q[board!=0]=-np.inf
        print(json.dumps({'action':int(q.argmax())}),flush=True)


def evaluate(args):
    import torch
    from train_dqn import QNetwork
    torch.set_num_threads(2)
    checkpoint=torch.load(args.model,map_location='cpu',weights_only=True)
    model=QNetwork().eval()
    model.load_state_dict(checkpoint['state_dict'])
    rng=np.random.default_rng(2026)
    worker=None
    if args.opponent=='cnn':
        worker=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--cnn-worker'],
                                stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    from tactical_dqn import TacticalDQN
    tactics=TacticalDQN(q_weight=0. if args.agent=='tactical-only' else .03)
    results=[]; latencies=[]
    try:
        for game in range(args.games):
            board=np.zeros((20,20),dtype=np.int8)
            # 각 흑·백 쌍은 같은 중앙 두 수로 시작하여 완전 결정론적 한 판 반복을 피한다.
            opening=np.random.default_rng(args.opening_seed+game//2).choice(25,size=2,replace=False)
            moves=[]
            for i,a in enumerate(opening):
                x,y=int(a%5+7),int(a//5+7)
                board[y,x]=i+1; moves.append(y*20+x)
            dqn_player=1+game%2; player=1; winner=0
            while len(moves)<400:
                if player==dqn_player:
                    started=perf_counter()
                    s=np.where(board==player,1,np.where(board==0,0,-1)).astype(np.float32)
                    with torch.no_grad():
                        q=model(torch.from_numpy(s[None,None])).numpy().ravel()
                    if args.agent=='tactical-only':
                        q.fill(0.)
                    q[board.ravel()!=0]=-np.inf
                    action=(int(q.argmax()) if args.agent=='dqn'
                            else tactics.choose(board,player,q))
                    latencies.append(perf_counter()-started)
                elif worker:
                    worker.stdin.write(json.dumps({'board':board.tolist(),'player':player})+'\n')
                    worker.stdin.flush()
                    line=worker.stdout.readline()
                    if not line: raise RuntimeError('CNN 평가 프로세스가 종료되었습니다')
                    action=json.loads(line)['action']
                else:
                    action=int(rng.choice(np.flatnonzero(board.ravel()==0)))
                y,x=divmod(action,20)
                if board[y,x]!=0: raise RuntimeError('평가 중 불법 착수가 발생했습니다')
                board[y,x]=player; moves.append(action)
                if has_five(board,x,y,player): winner=player; break
                player=3-player
            result={'game':game+1,'dqn_player':dqn_player,'winner':winner,
                    'outcome':'draw' if not winner else 'win' if winner==dqn_player else 'loss',
                    'moves':moves}
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k!='moves'}),flush=True)
    finally:
        if worker:
            worker.stdin.close(); worker.wait(timeout=30)
    report={'model':str(args.model),'opponent':args.opponent,'agent':args.agent,'opening_seed':args.opening_seed,'games':args.games,
            'wins':sum(r['outcome']=='win' for r in results),
            'losses':sum(r['outcome']=='loss' for r in results),
            'draws':sum(r['outcome']=='draw' for r in results),
            'mean_dqn_cpu_inference_ms':float(np.mean(latencies)*1000),
            'p95_agent_ms':float(np.percentile(latencies,95)*1000),
            'max_agent_ms':float(np.max(latencies)*1000),
            'search':None if args.agent=='dqn' else {'depth':tactics.depth,'width':tactics.width,
                                                   'replies':tactics.replies,'q_weight':tactics.q_weight},
            'protocol':'paired central two-stone openings, alternate black/white; fixed model; agent field specifies search',
            'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cnn-worker',action='store_true')
    p.add_argument('--model',type=Path,default=ROOT/'models/dqn-v1/gomoku_dqn.pt')
    p.add_argument('--opponent',choices=['cnn','random'],default='cnn')
    p.add_argument('--games',type=int,default=10)
    p.add_argument('--agent',choices=['dqn','tactical-dqn','tactical-only'],default='dqn')
    p.add_argument('--opening-seed',type=int,default=2026)
    p.add_argument('--output',type=Path,default=ROOT/'models/dqn-v1/evaluation_cnn.json')
    args=p.parse_args()
    if args.cnn_worker: cnn_worker()
    else: evaluate(args)

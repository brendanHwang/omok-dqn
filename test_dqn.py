"""보상·관점·합법 행동·종료 상태 목표값의 회귀 점검."""
import unittest
import numpy as np
import torch
from dqn_data import parse_game, transitions
from train_dqn import targets


class DQNTests(unittest.TestCase):
    def test_terminal_rewards_and_perspective(self):
        moves=[(1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(4,1),(4,2),(5,1)]
        text='Piskvorky 20x20, 0\n'+'\n'.join(f'{x},{y},0' for x,y in moves)
        s,a,w=parse_game(text); rows=transitions(s,a,w)
        self.assertEqual(w,1,'흑의 다섯 돌을 승리로 판정해야 합니다')
        self.assertEqual(rows[-1][2],1,'승리 착수의 보상은 +1입니다')
        self.assertEqual(rows[-2][2],-1,'상대 승리 직전 행동의 보상은 -1입니다')
        self.assertFalse(rows[0][-1],'진행 중 경험은 종료 상태가 아닙니다')
        self.assertEqual(rows[0][3][0,0],1,'다음 내 차례에도 내 돌은 +1입니다')
        self.assertEqual(rows[0][3][1,0],-1,'다음 내 차례에도 상대 돌은 -1입니다')
        self.assertEqual(rows[1][0][0,0],-1,'백의 입력에서는 흑이 상대 돌입니다')

    def test_incomplete_and_duplicate_moves(self):
        for moves in ('1,1,0','1,1,0\n1,1,0'):
            with self.assertRaises(ValueError,msg='불완전하거나 불법인 기보는 제외해야 합니다'):
                parse_game('Piskvorky 20x20, 0\n'+moves)

    def test_legal_mask_and_terminal_bootstrap(self):
        class Fixed(torch.nn.Module):
            def forward(self,x):
                q=torch.zeros((len(x),400),device=x.device); q[:,0]=99; q[:,1]=2
                return q
        ns=torch.zeros((2,1,20,20)); ns[0,0,0,0]=1; ns[1]=1
        y=targets(Fixed(),ns,torch.tensor([0.,-1.]),torch.tensor([False,True]),.9)
        torch.testing.assert_close(y,torch.tensor([1.8,-1.]),msg='불법 행동과 종료 상태의 미래 가치를 제외해야 합니다')


if __name__=='__main__':
    unittest.main()

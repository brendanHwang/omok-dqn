import unittest
import numpy as np
from tactical_dqn import TacticalDQN


class TacticalTests(unittest.TestCase):
    def setUp(self):
        self.agent=TacticalDQN()
        self.q=np.zeros(400)

    def test_win_before_block(self):
        b=np.zeros((20,20),dtype=np.int8)
        b[5,2:6]=1; b[9,2:6]=2
        action=self.agent.choose(b,1,self.q)
        self.assertIn(action,(5*20+1,5*20+6),'직접 승리할 수 있으면 방어보다 먼저 둬야 합니다')

    def test_block_and_keep_board(self):
        b=np.zeros((20,20),dtype=np.int8)
        b[9,2:6]=2; b[9,1]=1
        before=b.copy(); self.q[0]=999
        action=self.agent.choose(b,1,self.q)
        self.assertEqual(action,9*20+6,'상대의 유일한 즉시 승리점을 막아야 합니다')
        np.testing.assert_array_equal(before,b,err_msg='탐색 후 원본 바둑판이 복원되어야 합니다')

    def test_fork(self):
        b=np.zeros((20,20),dtype=np.int8); b[8,8:11]=1
        action=self.agent.choose(b,1,self.q)
        self.assertIn(action,(8*20+7,8*20+11),'양쪽 끝 승리점을 만드는 열린 사목을 우선해야 합니다')

    def test_no_illegal_move(self):
        b=np.zeros((20,20),dtype=np.int8); b[10,10]=1; b[11,11]=2
        self.q[210]=999
        a=self.agent.choose(b,2,self.q)
        self.assertEqual(b.ravel()[a],0,'점수가 높아도 이미 둔 칸은 선택하지 않아야 합니다')


if __name__=='__main__': unittest.main()

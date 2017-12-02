"""
Tests for bimatrix_generators.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_
from quantecon.gridtools import num_compositions

from quantecon.game_theory import blotto_game, ranking_game, sgc_game


class TestBlottoGame:
    def setUp(self):
        self.h, self.T = 4, 3
        rho = 0.5
        self.g = blotto_game(self.h, self.T, rho)

    def test_size(self):
        n = num_compositions(self.h, self.T)
        eq_(self.g.nums_actions, (n, n))

    def test_constant_diagonal(self):
        for i in range(self.g.N):
            diag = self.g.players[i].payoff_array.diagonal()
            ok_((diag == diag[0]).all())

    def test_seed(self):
        seed = 0
        h, T = 3, 4
        rho = -0.5
        g0 = blotto_game(h, T, rho, random_state=seed)
        g1 = blotto_game(h, T, rho, random_state=seed)
        for i in range(self.g.N):
            assert_array_equal(g0.players[i].payoff_array,
                               g1.players[i].payoff_array)


class TestRankingGame:
    def setUp(self):
        self.n = 100
        self.g = ranking_game(self.n)
        self.payoff0 = self.g.players[0].payoff_array
        self.payoff1 = self.g.players[1].payoff_array

    def test_size(self):
        eq_(self.g.nums_actions, (self.n, self.n))

    def test_weakly_decreasing_rowise_payoffs(self):
        ok_((self.payoff0[:, 1:-1] >= self.payoff0[:, 2:]).all())
        ok_((self.payoff1[:, 1:-1] >= self.payoff1[:, 2:]).all())

    def test_elements_first_row(self):
        ok_((self.payoff0[0, 0] + self.payoff1[0, 0]) == 1.)
        ok_(all([payoff in [0, 1, 0.5] for payoff in self.payoff0[0, :]]))
        ok_(all([payoff in [0, 1, 0.5] for payoff in self.payoff1[0, :]]))

    def test_seed(self):
        seed = 0
        n = 100
        g0 = ranking_game(n, random_state=seed)
        g1 = ranking_game(n, random_state=seed)
        for i in range(self.g.N):
            assert_array_equal(g0.players[i].payoff_array,
                               g1.players[i].payoff_array)


def test_sgc_game():
    k = 2
    s = """\
        0.750 0.750 1.000 0.500 0.500 1.000 0.000 0.500 0.000 0.500 0.000 0.500
        0.000 0.500 0.500 1.000 0.750 0.750 1.000 0.500 0.000 0.500 0.000 0.500
        0.000 0.500 0.000 0.500 1.000 0.500 0.500 1.000 0.750 0.750 0.000 0.500
        0.000 0.500 0.000 0.500 0.000 0.500 0.500 0.000 0.500 0.000 0.500 0.000
        0.750 0.000 0.000 0.750 0.000 0.000 0.000 0.000 0.500 0.000 0.500 0.000
        0.500 0.000 0.000 0.750 0.750 0.000 0.000 0.000 0.000 0.000 0.500 0.000
        0.500 0.000 0.500 0.000 0.000 0.000 0.000 0.000 0.750 0.000 0.000 0.750
        0.500 0.000 0.500 0.000 0.500 0.000 0.000 0.000 0.000 0.000 0.000 0.750
        0.750 0.000"""
    bimatrix = np.fromstring(s, sep=' ')
    bimatrix.shape = (4*k-1, 4*k-1, 2)
    bimatrix = bimatrix.swapaxes(0, 1)

    g = sgc_game(k)
    assert_array_equal(g.payoff_profile_array, bimatrix)


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)

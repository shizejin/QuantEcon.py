"""
This module contains functions that generate NormalFormGame instances of
the 2-player games studied by Fearnley, Igwe, and Savani (2015):

* Colonel Blotto Games (`blotto_game`): A non-zero sum extension of the
  Blotto game as studied by Hortala-Vallve and Llorente-Saguer (2012),
  where opposing parties have asymmetric and heterogeneous battlefield
  valuations.

* Ranking Games (`ranking_game`)

* SGC Games (`sgc_game`): These games were introduced by Sandholm,
  Gilpin, and Conitzer (2005) as a worst case scenario for support
  enumeration as it has a unique equilibrium where each player uses half
  of his actions in his support.

* Tournament Games (`tournament_game`)

* Unit vector Games (`unit_vector_game`)

Large part of the code here is based on the C code available at
https://github.com/bimatrix-games/bimatrix-generators distributed under
BSD 3-Clause License.

References
----------
* J. Fearnley, T. P. Igwe, R. Savani, "An Empirical Study of Finding
  Approximate Equilibria in Bimatrix Games," International Symposium on
  Experimental Algorithms (SEA), 2015.

* R. Hortala-Vallve and A. Llorente-Saguer, "Pure Strategy Nash
  Equilibria in Non-Zero Sum Colonel Blotto Games", International
  Journal of Game Theory, 2012.

* T. Sandholm, A. Gilpin, and V. Conitzer, "Mixed-Integer Programming
  Methods for Finding Nash Equilibria," AAAI, 2005.

"""

# BSD 3-Clause License
#
# Copyright (c) 2015, John Fearnley, Tobenna P. Igwe, Rahul Savani
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from numba import jit
from ..normal_form_game import Player, NormalFormGame
from ...util import check_random_state
from ...gridtools import simplex_grid


def blotto_game(h, T, rho, mu=0, random_state=None):
    """
    Return a NormalFormGame instance of a 2-player non-zero sum Colonel
    Blotto game (Hortala-Vallve and Llorente-Saguer, 2012), where the
    players have an equal number `T` of troops to assign to `h` hills
    (so that the number of actions for each player is equal to
    (T+h-1) choose (h-1) = (T+h-1)!/(T!*(h-1)!)). Each player has a
    value for each hill that he receives if he assigns strictly more
    troops to the hill than his opponent (ties are broken uniformly at
    random), where the values are drawn from a multivariate normal
    distribution with covariance `rho`. Each player’s payoff is the sum
    of the values of the hills won by that player.

    Parameters
    ----------
    h : scalar(int)
        Number of hills.
    T : scalar(int)
        Number of troops.
    rho : scalar(float)
        Covariance of values of each hill. Must be in [-1, 1].
    mu : scalar(float), optional(default=0)
        Mean of values of each hill.
    random_state : int or np.random.RandomState, optional
        Random seed (integer) or np.random.RandomState instance to set
        the initial state of the random number generator for
        reproducibility. If None, a randomly initialized RandomState is
        used.

    Returns
    -------
    g : NormalFormGame

    """
    actions = simplex_grid(h, T)
    n = actions.shape[0]
    payoff_arrays = tuple(np.empty((n, n)) for i in range(2))
    mean = np.array([mu, mu])
    cov = np.array([[1, rho], [rho, 1]])
    random_state = check_random_state(random_state)
    values = random_state.multivariate_normal(mean, cov, h)
    _populate_blotto_payoff_arrays(payoff_arrays, actions, values)
    g = NormalFormGame(
        [Player(payoff_array) for payoff_array in payoff_arrays]
    )
    return g


@jit(nopython=True)
def _populate_blotto_payoff_arrays(payoff_arrays, actions, values):
    """
    Populate the ndarrays in `payoff_arrays` with the payoff values of
    the Blotto game with h hills and T troops.

    Parameters
    ----------
    payoff_arrays : tuple(ndarray(float, ndim=2))
        Tuple of 2 ndarrays of shape (n, n), where n = (T+h-1)!/
        (T!*(h-1)!). Modified in place.
    actions : ndarray(int, ndim=2)
        ndarray of shape (n, h) containing all possible actions, i.e.,
        h-part compositions of T.
    values : ndarray(float, ndim=2)
        ndarray of shape (h, 2), where `values[k, :]` contains the
        players' values of hill `k`.

    """
    n, h = actions.shape
    payoffs = np.empty(2)
    for i in range(n):
        for j in range(n):
            payoffs[:] = 0
            for k in range(h):
                if actions[i, k] == actions[j, k]:
                    for p in range(2):
                        payoffs[p] += values[k, p] / 2
                else:
                    winner = np.int(actions[i, k] < actions[j, k])
                    payoffs[winner] += values[k, winner]
            payoff_arrays[0][i, j], payoff_arrays[1][j, i] = payoffs


def sgc_game(k):
    """
    Return a NormalFormGame instance of the 2-player game introduced by
    Sandholm, Gilpin, and Conitzer (2005), which has a unique Nash
    equilibrium, where each player plays half of the actions with
    positive probabilities. Payoffs are normalized so that the minimum
    and the maximum payoffs are 0 and 1, respectively.

    Parameters
    ----------
    k : scalar(int)
        Positive integer determining the number of actions. The returned
        game will have `4*k-1` actions for each player.

    Returns
    -------
    g : NormalFormGame

    Examples
    --------
    >>> g = sgc_game(2)
    >>> g.players[0]
    Player([[ 0.75,  0.5 ,  1.  ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 1.  ,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 0.5 ,  1.  ,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 0.  ,  0.  ,  0.  ,  0.75,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.75,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.75,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.75]])
    >>> g.players[1]
    Player([[ 0.75,  0.5 ,  1.  ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 1.  ,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 0.5 ,  1.  ,  0.75,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.75,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.75,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.75],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.75,  0.  ]])

    """
    payoff_arrays = tuple(np.empty((4*k-1, 4*k-1)) for i in range(2))
    _populate_sgc_payoff_arrays(payoff_arrays)
    g = NormalFormGame(
        [Player(payoff_array) for payoff_array in payoff_arrays]
    )
    return g


@jit(nopython=True)
def _populate_sgc_payoff_arrays(payoff_arrays):
    """
    Populate the ndarrays in `payoff_arrays` with the payoff values of
    the SGC game.

    Parameters
    ----------
    payoff_arrays : tuple(ndarray(float, ndim=2))
        Tuple of 2 ndarrays of shape (4*k-1, 4*k-1). Modified in place.

    """
    n = payoff_arrays[0].shape[0]  # 4*k-1
    m = (n+1)//2 - 1  # 2*k-1
    for payoff_array in payoff_arrays:
        for i in range(m):
            for j in range(m):
                payoff_array[i, j] = 0.75
            for j in range(m, n):
                payoff_array[i, j] = 0.5
        for i in range(m, n):
            for j in range(n):
                payoff_array[i, j] = 0

        payoff_array[0, m-1] = 1
        payoff_array[0, 1] = 0.5
        for i in range(1, m-1):
            payoff_array[i, i-1] = 1
            payoff_array[i, i+1] = 0.5
        payoff_array[m-1, m-2] = 1
        payoff_array[m-1, 0] = 0.5

    k = (m+1)//2
    for h in range(k):
        i, j = m + 2*h, m + 2*h
        payoff_arrays[0][i, j] = 0.75
        payoff_arrays[0][i+1, j+1] = 0.75
        payoff_arrays[1][j, i+1] = 0.75
        payoff_arrays[1][j+1, i] = 0.75

###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import qiskit
from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute

import math, numpy, scipy

from qiskit.opflow import I, X, Y

""" get_ring_xy_mixer(N)
    Mixer operator to use in QAOA ansatz (partitioned 1D XX+YY mixer that preserves Hamming weight)
    """


def get_ring_xy_mixer(N):
    XY = 0.5 * ((X ^ X) + (Y ^ Y))

    terms = []
    for idx_xy in range(N - 1):
        if idx_xy == 0:
            term = XY
            for idx in range(2, N):
                term = term ^ I
        else:
            term = I
            for idx in range(1, idx_xy):
                term = term ^ I
            term = term ^ XY
            for idx in range(idx_xy + 2, N):
                term = term ^ I
        terms.append(term)
    return sum(terms)


""" fracAngle(l,n) 
    is an angle theta, such that Ry(theta)|0⟩ = sqrt(l/n)|0⟩ + sqrt(1-l/n)|1⟩"""


def fracAngle(l, n):
    return 2 * numpy.arccos(numpy.sqrt(l / n))


""" shiftup(s) 
    consecutively swaps the first qubit upwards through the next s-1 qubits 
    which are in state |0⟩, using 2 CNOTs each"""


def shiftup(s):
    qr = QuantumRegister(s, "shiftup{}".format(s))
    circ = QuantumCircuit(qr)
    for i in range(s - 1):
        circ.cx(qr[i], qr[i + 1])
        circ.cx(qr[i + 1], qr[i])
    return circ


""" blocki(n,l)
    redistributes the excitation of qubit 1 across qubits 1,2 and swaps the two qubits:
    - |00⟩ ↦ |00⟩
    - |01⟩ ↦ sqrt(1-l/n) |01⟩ + sqrt(l/n) |10⟩
    - |11⟩ ↦ |11⟩
    using 2 CNOTs (or less if the input is a known classical state) """


def blocki(n, l=1, qin="xx"):
    qr = QuantumRegister(2, "blocki{}{}".format(n, l))
    circ = QuantumCircuit(qr)

    # empty block for trivial inputs '00' or '11'
    if qin == "00" or qin == "11":
        pass
    # simple fixed input '01'
    elif qin == "01":
        circ.x(qr[0:2])
        circ.ry(fracAngle(l, n), qr[0])
        circ.cx(qr[0], qr[1])
    # arbitrary input
    else:  # i.e (qin == 'xx' or qin == 'x1' or qin == '0x'):
        circ.ry(1.0 * (math.pi) / 2, qr[1])
        circ.cx(qr[1], qr[0])
        circ.ry(0.5 * fracAngle(n - l, n), qr[1])
        circ.ry(0.5 * fracAngle(n - l, n), qr[0])
        circ.cx(qr[1], qr[0])
        circ.ry(-1.0 * (math.pi) / 2, qr[1])

    return circ


""" blockii(n,l)
    redistributes the excitation of qubit 2 across qubits 2,3 (controlled on qubit 1)
    and swaps qubits 2,3 (independent of qubit 1)
    - |xy0⟩ ↦ |yx0⟩
    - |011⟩ ↦ sqrt(1-l/n) |011⟩ + sqrt(l/n) |101⟩
    - |xx1⟩ ↦ |xx1⟩
    using 5 CNOTs (or less if the input prefix is a known classical state)"""


def blockii(n, l, qin="xxx"):
    # print("blockii{}{} on SCS subinput {}".format(n,l,qin))
    qr = QuantumRegister(3, "blockii{}{}".format(n, l))
    circ = QuantumCircuit(qr)

    # if qubit 1 is guaranteed to be 1 (even after preceeding block),
    # pass on to blocki without the little-endian '1'
    if qin[2] == "1":
        blocki_circ = blocki(n, l, qin[0:2])
        circ.compose(blocki_circ, qubits=[1, 2], inplace=True)
    # on any other input, implement 5-CNOT circuit
    else:
        circ.cx(qr[1], qr[2])
        circ.ry(-0.25 * fracAngle(l, n), qr[1])
        circ.cx(qr[0], qr[1])
        circ.ry(0.25 * fracAngle(l, n), qr[1])
        circ.cx(qr[2], qr[1])
        circ.ry(-0.25 * fracAngle(l, n), qr[1])
        circ.cx(qr[0], qr[1])
        circ.ry(0.25 * fracAngle(l, n), qr[1])
        circ.cx(qr[1], qr[2])

    return circ


""" SCS(N,k,K)
* implements the Split & Cyclic Shift operation on unary encoding of k ≤ l ≤ K:
  |0^(n-l) 1^(l)⟩ ↦ sqrt(1-l/n) |x^(n-l) 1^(l)⟩ + sqrt(l/n) |1 0^(n-l) 1^(l-1)⟩

* topology = 'LNN' increases CNOT count by 2*(N-K-1) for shiftup swapping"""


def SCS(N, k=0, K=None, topology=None):
    # set up circuit
    if K is None:
        K = N
    assert 0 <= k and k <= K and K <= N, "argument mismatch 0≤{}≤{}≤{}".format(k, K, N)
    qr = QuantumRegister(N, "scs{}{}".format(N, k, K))
    circ = QuantumCircuit(qr)

    ## ground blocki
    qin = [
        "1" if k > 1 else "0" if K <= 1 else "x",
        "1" if k > 0 else "0" if K <= 0 else "x",
    ]
    qin = "".join(qin)  # fixed input
    t = N - 1 if 1 == K and K < N and topology != "LNN" else 1  # target qubit 1 or N-1
    blocki_circ = blocki(N, 1, qin)
    circ.compose(blocki_circ, qubits=[0, t], inplace=True)
    ## stair of blockii
    for l in range(2, min(N, K + 1)):
        qin = [
            "1" if k > l else "0" if K <= l else "x",
            "1" if k > l - 1 else "x",
            "1" if k > l - 1 else "x",
        ]
        qin = "".join(qin)  # fixed input
        t = N - 1 if l == K and K < N and topology != "LNN" else l  # target qubit 1 or N-1
        blockii_circ = blockii(N, l, qin)
        circ.compose(blockii_circ, qubits=[l - 2, l - 1, t], inplace=True)

    ## shiftup for LNN topologies
    if K < N and topology == "LNN":
        circ.compose(shiftup(N - K), qubits=range(K, N), inplace=True)
    return circ


""" U(N,k,K)
* implements the Dicke State preparation on unary encoding of k ≤ l ≤ K:
  |0^(n-l) 1^(l)⟩ ↦ |D^n_l⟩

* topology = 'LNN' increases CNOT counts"""


def U(N, k=0, K=None, topology=None):
    # set up circuit
    if K is None:
        K = N
    assert 0 <= k and k <= K and K <= N, "argument mismatch 0≤{}≤{}≤{}".format(k, K, N)
    qr = QuantumRegister(N, "u{}{}".format(N, k, K))
    circ = QuantumCircuit(qr)

    # inductive SCS construction
    for n in range(N, 1, -1):
        # reduce k by 1 for every SCS
        klow, khigh = max(0, k - (N - n)), min(K, n)
        SCS_circ = SCS(n, klow, khigh, topology)
        circ.compose(SCS_circ, inplace=True)
    return circ


""" dicke_simple(N,K)
*   Based on improved techniqes upon    "Deterministic Preparation of Dicke States", https://arxiv.org/abs/1904.07358
    with further improvements from      "On Actual Preparation of Dicke State on a Quantum Computer", https://arxiv.org/abs/2007.01681

*   Generates the Dicke State through the better of the following methods:
    - U(N,K) |0^(N-K) 1^(K)⟩                  for K < N/2
    - X^(⊗N) U(N,N-K) |0^(K) 1^(N-K)⟩ X^(⊗N)  for K >= N/2
"""


def dicke_simple(N, K, topology=None):
    # input string
    qr = QuantumRegister(N, "qr{}{}".format(N, K))
    circ = QuantumCircuit(qr)
    ## Prepare D_{N,K} with K < N/2 as D_{N,N-K}
    circ.x(qr[0 : max(K, N - K)])  # Initialize
    dicke_unitary = U(N, max(K, N - K), max(K, N - K), topology)  # Dicke state Unitary
    circ.compose(dicke_unitary, inplace=True)
    if K < N / 2:  # Bitflip
        circ.x(qr[0:N])
    return circ


""" dicke_divide_conquer(N,K)
*   Based on divide-conquer idea from   "A Divide-and-Conquer Approach to Dicke State Preparation", https://arxiv.org/abs/2112.12435

*   Generates the Dicke State by Distributing Hamming weight to two Dicke State Unitaries U(N1,K), U(N-N1,K) with N1=⌊N/2⌋
*   Also uses Bit-Flip Tricks depending on the size of K relative to N1,N-N1,N
"""


def dicke_divide_conquer(N, K):
    N1 = int(N / 2)
    N2 = N - N1
    # input string
    qr = QuantumRegister(N, "qr{}{}".format(N, K))
    circ = QuantumCircuit(qr)

    ## for K < N1, prepare D_N,K
    ## for K > N1, prepare D_N,N-K and flip
    flip = False
    if K > N1:
        flip, K = True, N - K

    ## State Prep Divide
    x = [scipy.special.comb(N1, i) * scipy.special.comb(N2, K - i) for i in range(0, K + 1)]
    circ.ry(fracAngle(x[0], sum(x[0 : K + 1])), qr[0])
    for i in range(1, K):
        theta = fracAngle(sum(x[i + 1 : K + 1]), sum(x[i : K + 1]))
        circ.ry(0.5 * theta, qr[i])
        circ.cx(qr[i - 1], qr[i])
        circ.ry(-0.5 * theta, qr[i])

    circ.cnot(qr[0:K], qr[N - K : N])
    ## State Prep Conquer
    ## dicke unitary on N1, flipping qubits
    circ.x(qr[0:N1])
    dicke_N1 = U(N1, N1 - K)
    circ.compose(dicke_N1, qubits=list(range(N1 - 1, -1, -1)), inplace=True)
    ## dicke unitary on N2, flipping qubits
    if K < N2:
        circ.x(qr[N1 : N - K])
    dicke_N2 = U(N2, N2 - K)
    circ.compose(dicke_N2, qubits=list(range(N1, N)), inplace=True)
    if not flip:
        circ.x(qr[0:N])

    return circ

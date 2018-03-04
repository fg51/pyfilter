#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pvtol_lqr.m - LQR design for vectored thrust aircraft
# RMM, 14 Jan 03


#このファイルは、Astrom and Mruray第5章の平面垂直離着陸（PVTOL）航空機の例を使用して、
#LQRベースの設計上の問題をPythonコントロールパッケージの基本機を使って処理します。


import numpy as np
from numpy import diag, cos, sin, matrix
from scipy import signal, linalg

from matplotlib import pyplot as plt


def main():
    #
    # System dynamics
    #
    # 状態空間形式のPVTOLシステムのダイナミクス
    #

    # parameter of system
    m = 4       # mass of aircraft
    J = 0.0475  # moment of pitch axis
    r = 0.25    # 力の中心までの距離
    g = 9.8
    c = 0.05    # dumping factor (predict)


    #  dynamicsの状態空間
    xe = [0, 0, 0, 0, 0, 0]     # 平衡点
    ue = [0, m * g]

    # Dynamics 行列 (*が動作するように行列型を使用する)
    A = matrix([
        [0, 0,    0,    1,    0,    0],
        [0, 0,    0,    0,    1,    0],
        [0, 0,    0,    0,    0,    1],
        [0, 0, (-ue[0] * sin(xe[2]) - ue[1] * cos(xe[2])) / m, -c / m, 0, 0],
        [0, 0, (ue[0] * cos(xe[2]) - ue[1] * sin(xe[2])) / m, 0, -c / m, 0],
        [0, 0,    0,    0,    0,    0],
    ])

    # Input 行列
    B = matrix([
        [0, 0],
        [0, 0],
        [0, 0],
        [cos(xe[2]) / m, -sin(xe[2]) / m],
        [sin(xe[2]) / m,  cos(xe[2]) / m],
        [r / J, 0],
    ])

    # Output 行列
    C = matrix([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ])
    D = matrix([
        [0, 0],
        [0, 0],
    ])


    #　xy位置のstepに対応する入力と出力を構築する
    #ベクトルxdおよびydは、システムの所望の平衡状態である状態に対応する。
    #行列CxおよびCyは、対応する出力である。
    #
    # これらのベクトルは、閉ループシステムのダイナミクスを次のように計算することで使用される。

    #
    #   xdot = Ax + B u     =>  xdot = (A-BK)x + K xd
    #         u = -K(x - xd)           y = Cx
    #
    # 閉ループ動特性は、入力ベクトルとしてK * xdを用いて「step」コマンドを使用してシミュレートすることができる
    # （「入力」は単位サイズであると仮定し、xdは所望の定常状態に対応する）。
    #

    xd = matrix([[1], [0], [0], [0], [0], [0]])
    yd = matrix([[0], [1], [0], [0], [0], [0]])

    #
    # 関連するダイナミクスを抽出してSISOライブラリで使用する
    #
    # 現在のpython-controlライブラリはSISO転送関数しかサポートしていないので、
    # 元のMATLABコードの一部を修正してSISOシステムを抽出する必要があります。
    # これを行うために、横（x）および縦（y）ダイナミクスに関連する状態からなるように、
    # 「lat」および「alt」インデックスベクトルを定義します。
    #

    # 状態変数
    lat = (0, 2, 3, 5)
    alt = (1, 4)

    #分離されたダイナミックス
    Ax = (A[lat, :])[:, lat]
    Bx = B[lat, 0]
    Cx = C[0, lat]
    Dx = D[0, 0]

    Ay = (A[alt, :])[:, alt]
    By = B[alt, 1]
    Cy = C[1, alt]
    Dy = D[1, 1]

    #  plotラベル
    plt.clf()
    plt.suptitle("LQR controllers for vectored thrust aircraft (pvtol-lqr)")

    #
    # LQR design
    #

    # 対角行列の重み付け
    Qx1 = diag([1, 1, 1, 1, 1, 1])
    Qu1a = diag([1, 1])
    K1a = matrix(lqr(A, B, Qx1, Qu1a)[0])

    # ループを閉じる: xdot = Ax - B K (x-xd)
    # Note: python-controlでは、この入力を一度に行う必要があります
    #　H1a = ss(A-B*K1a, B*K1a*concatenate((xd, yd), axis=1), C, D)　
    # (T, Y) = step(H1a, T=linspace(0,10,100))

    # 最初の入力に対するステップ応答
    H1ax = signal.StateSpace(
        Ax - Bx * K1a[0, lat],
        Bx * K1a[0, lat] * xd[lat, :],
        Cx,
        Dx)
    Tx, Yx = signal.step(H1ax, T=np.linspace(0, 10, 100))

    # 第2入力に対するステップ応答
    H1ay = signal.StateSpace(Ay - By * K1a[1, alt], By * K1a[1, alt] * yd[alt, :], Cy, Dy)
    Ty, Yy = signal.step(H1ay, T=np.linspace(0, 10, 100))

    ax1 = plt.subplot(221)
    ax1.set_title("Identity weights")
    ax1.plot(Tx.T, Yx.T, '-',  label="x")
    ax1.plot(Ty.T, Yy.T, '--', label="y")
    ax1.hlines([1], 0, 10, color="k")

    ax1.axis([0, 10, -0.1, 1.4])
    ax1.set_ylabel('position')
    ax1.legend(loc='lower right')


    # 異なる入力重みを見る
    Qu1a = diag([1, 1])
    K1a, _, _ = lqr(A, B, Qx1, Qu1a)
    H1ax = signal.StateSpace(
        Ax - Bx * K1a[0, lat],
        Bx * K1a[0, lat] * xd[lat, :],
        Cx,
        Dx)

    Qu1b = (40 ** 2) * diag([1, 1])
    K1b, _, _ = lqr(A, B, Qx1, Qu1b)
    H1bx = signal.StateSpace(
        Ax - Bx * K1b[0, lat],
        Bx * K1b[0, lat] * xd[lat, :],
        Cx,
        Dx)

    Qu1c = (200 ** 2) * diag([1, 1])
    K1c, _, _ = lqr(A, B, Qx1, Qu1c)
    H1cx = signal.StateSpace(
        Ax - Bx * K1c[0, lat],
        Bx * K1c[0, lat] * xd[lat, :],
        Cx,
        Dx)

    T1, Y1 = signal.step(H1ax, T=np.linspace(0, 10, 100))
    T2, Y2 = signal.step(H1bx, T=np.linspace(0, 10, 100))
    T3, Y3 = signal.step(H1cx, T=np.linspace(0, 10, 100))


    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("Effect of input weights")
    ax2.plot(T1.T, Y1.T, 'b-')
    ax2.plot(T2.T, Y2.T, 'b-')
    ax2.plot(T3.T, Y3.T, 'b-')
    ax2.hlines([1], 0, 10, color='k')

    ax2.axis([0, 10, -0.1, 1.4])

    # arcarrow([1.3, 0.8], [5, 0.45], -6)
    # text(5.3, 0.4, 'rho')

    # 出力重み付け - 出力を使用するようにQxを変更する
    Qx2 = C.T * C
    Qu2 = 0.1 * diag([1, 1])
    K2 = matrix(lqr(A, B, Qx2, Qu2)[0])

    H2x = signal.StateSpace(
        Ax - Bx * K2[0, lat],
        Bx * K2[0, lat] * xd[lat, :],
        Cx,
        Dx)
    H2y = signal.StateSpace(
        Ay - By * K2[1, alt],
        By * K2[1, alt] * yd[alt, :],
        Cy,
        Dy)

    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("Output weighting")
    T2x, Y2x = signal.step(H2x, T=np.linspace(0, 10, 100))
    T2y, Y2y = signal.step(H2y, T=np.linspace(0, 10, 100))
    ax3.plot(T2x.T, Y2x.T, label="x")
    ax3.plot(T2y.T, Y2y.T, label="y")
    ax3.set_ylabel('position')
    ax3.set_xlabel('time')
    ax3.legend(loc='lower right')


    #
    # 　物理的に動機付けされた重み付け
    #
    # xで1 cmの誤差、yで10 cmの誤差で決定する。
    # 角度を5度以下に調整して調整する。
    # 効率の低下により、サイドの力にはペナルティを課す。
    #

    Qx3 = diag([100, 10, 2 * np.pi / 5, 0, 0, 0])
    Qu3 = 0.1 * diag([1, 10])
    K3 = matrix(lqr(A, B, Qx3, Qu3)[0])

    H3x = signal.StateSpace(
        Ax - Bx * K3[0, lat],
        Bx * K3[0, lat] * xd[lat, :],
        Cx, Dx)
    H3y = signal.StateSpace(
        Ay - By * K3[1, alt],
        By * K3[1, alt] * yd[alt, :],
        Cy, Dy)

    ax4 = plt.subplot(224)
    T3x, Y3x = signal.step(H3x, T=np.linspace(0, 10, 100))
    T3y, Y3y = signal.step(H3y, T=np.linspace(0, 10, 100))
    ax4.plot(T3x.T, Y3x.T, label="x")
    ax4.plot(T3y.T, Y3y.T, label="y")
    ax4.set_title("Physically motivated weights")
    ax4.set_xlabel('time')
    ax4.legend(loc='lower right')


    plt.show()


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(linalg.solve_continuous_are(A, B, Q, R))  # pylint: disable=no-member

    #compute the LQR gain
    K = np.matrix(linalg.inv(R) * (B.T * X))

    eigVals, _ = linalg.eig(A - B * K)  # eigVals, eigVecs
    return K, X, eigVals


if __name__ == '__main__':
    main()

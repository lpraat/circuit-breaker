package com.lpraat.circuitbreaker.structtest

import com.lpraat.circuitbreaker.state.State
import com.lpraat.circuitbreaker.struct.Matrix
import org.scalatest.FunSuite


class MatrixTest extends FunSuite {

  test("Matrix") {
    val identity: Matrix[Double] = Matrix.replaceSeq(List((0,0,1.0), (1,1,1.0), (2,2,1.0))).exec(Matrix(3, 3))

    assert(identity(0)(0) == 1.0)
    assert(identity(1)(1) == 1.0)
    assert(identity(2)(2) == 1.0)

    val timesThreeCentralElement: State[Matrix[Double], Unit] =
      State(m => ((), Matrix.replace(1, 1)(m(1)(1) * 3).exec(m)))

    assert(timesThreeCentralElement.exec(identity)(1)(1) == 3.0)


    val fillWith10s: State[Matrix[Double], Unit] = for {
      _ <- Matrix.fill(Vector.fill(identity.rows * identity.columns)(10.0))
    } yield ()

    assert(!fillWith10s.exec(identity).m.flatten.exists(_ != 10))

  }




}

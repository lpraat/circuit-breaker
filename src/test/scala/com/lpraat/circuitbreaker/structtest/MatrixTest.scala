package com.lpraat.circuitbreaker.structtest

import com.lpraat.circuitbreaker.struct.Matrix
import org.scalatest.FunSuite


class MatrixTest extends FunSuite {

  test("Matrix") {
    val identity: Matrix = Matrix(3,3).replaceSeq(List((0,0,1.0), (1,1,1.0), (2,2,1.0)))

    assert(identity(0)(0) == 1.0)
    assert(identity(1)(1) == 1.0)
    assert(identity(2)(2) == 1.0)

    assert(identity.replace(1, 1)(10)(1)(1) == 10.0)

    val tenMatrix = identity.fill(Vector.fill(identity.rows * identity.columns)(10.0))
    assert(!tenMatrix.m.flatten.exists(_ != 10))

    assert(!(tenMatrix - tenMatrix).m.flatten.exists(_ != 0))

    val timesTwoIdentity = identity.map(el => el * 2)
    assert(timesTwoIdentity(0)(0) == 2.0)
    assert(timesTwoIdentity(1)(1) == 2.0)
    assert(timesTwoIdentity(2)(2) == 2.0)

  }


}

package com.lpraat.circuitbreaker.struct

import com.lpraat.circuitbreaker.state.State

import scala.annotation.tailrec


class Matrix[+A] private (val m: Vector[Vector[A]]) {

  def rows: Int = m.size
  def columns: Int = m(0).size

  override def toString: String = {

    @tailrec def loop(m: Vector[Vector[A]], s: String): String = m match {
      case head +: tail  => loop(tail, head.foldLeft(s)((s, elem) => s.concat(elem.toString + " ")).concat("\n"))
      case _ => s
    }

    loop(this.m, "")
  }

  def apply(row: Int): Vector[A] = {
    m(row)
  }

}

object Matrix extends {

  def apply(rows: Int, columns: Int): Matrix[Double] = {
    new Matrix(Vector.fill(rows)(Vector.fill(columns)(0.0)))
  }

  def apply[A](m: Vector[Vector[A]]): Matrix[A] = {
    new Matrix(m)
  }

  def fill[A](el: Vector[A]): State[Matrix[A], Unit] = {
    State(matrix => ((), Matrix(el.grouped(matrix.columns).toVector)))
  }

  def replace[A](row: Int, column: Int)(value: A): State[Matrix[A], Unit] = {
    State(matrix => {
      val newM = matrix.m.updated(row, matrix.m(row).updated(column, value))
      ((), Matrix(newM))
    })
  }

  def replaceSeq[A](l: List[(Int, Int, A)]): State[Matrix[A], Unit] = {

    @tailrec def loop(l: List[(Int, Int, A)], m: Matrix[A]): (Unit, Matrix[A]) =
      l match {
        case (row, col, v) :: tail => loop(tail, replace(row, col)(v).exec(m))
        case _ => ((), m)
      }

    State(matrix => loop(l, matrix))
  }


}




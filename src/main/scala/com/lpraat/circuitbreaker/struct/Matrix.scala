package com.lpraat.circuitbreaker.struct

import scala.annotation.tailrec


class Matrix (val m: Vector[Vector[Double]]) {

  def rows: Int = m.size
  def columns: Int = m(0).size

  override def toString: String = {

    @tailrec def loop(m: Vector[Vector[Double]], s: String): String = m match {
      case head +: tail  => loop(tail, head.foldLeft(s)((s, elem) => s.concat(elem.toString + " ")).concat("\n"))
      case _ => s
    }

    loop(this.m, "")
  }

  def apply(row: Int): Vector[Double] = {
    m(row)
  }

  def fill[B](el: Vector[Double]): Matrix = {
    Matrix(el.grouped(this.columns).toVector)
  }

  def replace(row: Int, column: Int)(value: Double): Matrix = {
    Matrix(this.m.updated(row, this.m(row).updated(column, value)))
  }

  def replaceSeq(l: List[(Int, Int, Double)]): Matrix = {

    @tailrec def loop(l: List[(Int, Int, Double)], m: Matrix): Matrix =
      l match {
        case (row, col, v) :: tail => loop(tail, m.replace(row, col)(v))
        case _ => m
      }

    loop(l, this)
  }


  def map(f: Double => Double): Matrix = {
    fill(this.m.flatten.map(el => f(el)))
  }

  def *(k: Double): Matrix = {
     this.map(el => el * k)
  }

  def op(f: (Double, Double) => Double, el: Matrix): Matrix = {
    Matrix(
      this.m.zipWithIndex.foldLeft(Vector[Vector[Double]]()) {
      case (acc, (v, i)) =>
        acc :+ v.zipWithIndex.foldLeft(Vector[Double]()) {
          case (acc1, (d, j)) =>
            acc1 :+ f(d, el(i)(j))
        }
    })
  }

  def -(el: Matrix): Matrix = {
    op((a, b) => a - b, el)
  }

  def +(el: Matrix): Matrix = {
    op((a, b) => a + b, el)
  }

}

object Matrix extends {

  def apply(rows: Int, columns: Int): Matrix = {
    new Matrix(Vector.fill(rows)(Vector.fill(columns)(0.0)))
  }

  def apply(m: Vector[Vector[Double]]): Matrix= {
    new Matrix(m)
  }




}




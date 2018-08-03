package com.lpraat.circuitbreaker.state


case class State[S,+A](run: S => (A,S)) {

  def flatMap[B](g: A => State[S, B]): State[S, B] =
    State(s => {
      val (a1, s1) = run(s)
      g(a1).run(s1)
    })

  def map[B](f: A => B): State[S, B] =
    flatMap(a => State.unit(f(a)))

  def map2[B,C](sb: State[S, B])(f: (A, B) => C): State[S, C] =
    flatMap(a => sb.map(b => f(a, b)))

  def exec(s: S): S =
    run(s)._2

}

object State {

  def unit[A,S](a: A): State[S,A] =
    State(s => (a, s))


  /*
  def get[S]: State[S, S] =
    State(s => (s, s))

  def set[S](s: S): State[S, Unit] =
    State(_ => ((), s))

  def modify[S](f: S => S): State[S, Unit] = for {
    s <- get
    _ <- set(f(s))
  } yield ()
  */
}
package goten

import "math"

func sin[T TensorElement](i T) T {
	return T(math.Sin(float64(i)))
}

func Sin[T TensorElement](t *Tensor[T]) *Tensor[T] {
	return Map[T, T](t, sin[T])
}

func (t *Tensor[T]) Sin() *Tensor[T] {
	return Sin(t)
}

func cos[T TensorElement](i T) T {
	return T(math.Cos(float64(i)))
}

func Cos[T TensorElement](t *Tensor[T]) *Tensor[T] {
	return Map[T, T](t, cos[T])
}

func (t *Tensor[T]) Cos() *Tensor[T] {
	return Cos(t)
}

func Tan[T TensorElement](t *Tensor[T]) *Tensor[float64] {
	return Map[T, float64](t, math.Tan)
}

func (t *Tensor[T]) Tan() *Tensor[float64] {
	return Tan(t)
}

func Asin[T TensorElement](t *Tensor[T]) *Tensor[float64] {
	return Map[T, float64](t, math.Asin)
}

func (t *Tensor[T]) Asin() *Tensor[float64] {
	return Asin(t)
}

func Acos[T TensorElement](t *Tensor[T]) *Tensor[float64] {
	return Map[T, float64](t, math.Acos)
}

func (t *Tensor[T]) Acos() *Tensor[float64] {
	return Acos(t)
}

func Atan[T TensorElement](t *Tensor[T]) *Tensor[float64] {
	return Map[T, float64](t, math.Atan)
}

func (t *Tensor[T]) Atan() *Tensor[float64] {
	return Atan(t)
}

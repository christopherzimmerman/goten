package goten

import "math"

func mathToGeneric[T TensorElement](fn func(float64) float64) func(T) T {
	return func(i T) T { return T(fn(float64(i))) }
}

func (t *Tensor[T]) Sin() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Sin))
}

func (t *Tensor[T]) Cos() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Cos))
}

func (t *Tensor[T]) Tan() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Tan))
}

func (t *Tensor[T]) Asin() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Asin))
}

func (t *Tensor[T]) Acos() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Acos))
}

func (t *Tensor[T]) Atan() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Atan))
}

func (t *Tensor[T]) Asinh() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Asinh))
}

func (t *Tensor[T]) Acosh() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Acosh))
}

func (t *Tensor[T]) Atanh() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Atanh))
}

func (t *Tensor[T]) Abs() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Abs))
}

func (t *Tensor[T]) Cbrt() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Cbrt))
}

func (t *Tensor[T]) Ceil() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Ceil))
}

func (t *Tensor[T]) Erf() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Erf))
}

func (t *Tensor[T]) Erfc() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Erfc))
}

func (t *Tensor[T]) Erfcinv() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Erfcinv))
}

func (t *Tensor[T]) Erfinv() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Erfinv))
}

func (t *Tensor[T]) Exp() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Exp))
}

func (t *Tensor[T]) Exp2() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Exp2))
}

func (t *Tensor[T]) Expm1() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Expm1))
}

func (t *Tensor[T]) Gamma() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Gamma))
}

func (t *Tensor[T]) J0() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.J0))
}

func (t *Tensor[T]) J1() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.J1))
}

func (t *Tensor[T]) Log() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Log))
}

func (t *Tensor[T]) Log1p() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Log1p))
}

func (t *Tensor[T]) Log10() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Log10))
}

func (t *Tensor[T]) Log2() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Log2))
}

func (t *Tensor[T]) Logb() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Logb))
}

func (t *Tensor[T]) Round() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Round))
}

func (t *Tensor[T]) RoundToEven() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.RoundToEven))
}

func (t *Tensor[T]) Sqrt() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Sqrt))
}

func (t *Tensor[T]) Trunc() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Trunc))
}

func (t *Tensor[T]) Y0() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Y0))
}

func (t *Tensor[T]) Y1() *Tensor[T] {
	return Map[T](t, mathToGeneric[T](math.Y1))
}

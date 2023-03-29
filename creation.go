package goten

func Full[T TensorElement](shape []int, value T) *Tensor[T] {
	result := New[T](shape)

	resultIter := result.Iter()
	for i := 0; i < result.size; i++ {
		resultIter.data.Set(resultIter.iterPosition, value)
		resultIter.Next()
	}

	return result
}

func Zeros[T TensorElement](shape []int) *Tensor[T] {
	return Full(shape, T(0))
}

func Ones[T TensorElement](shape []int) *Tensor[T] {
	return Full(shape, T(1))
}

func Arange[T TensorElement](start int, end int) *Tensor[T] {
	result := New[T]([]int{end - start})

	resultIter := result.Iter()
	for i := 0; i < result.size; i++ {
		resultIter.data.Set(resultIter.iterPosition, T(start+i))
		resultIter.Next()
	}

	return result
}

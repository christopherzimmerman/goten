package goten

import (
	"errors"
)

func concatShape[T TensorElement](tList []*Tensor[T], axis int, shape []int) ([]int, error) {
	rank := len(shape)

	for i := 0; i < len(tList); i++ {
		tListI := tList[i]
		if tListI.Rank() != rank {
			return []int{}, errors.New("all inputs must share the same rank")
		}

		for j := 0; j < rank; j++ {
			if j != axis && tListI.shape[j] != shape[j] {
				return []int{}, errors.New("all inputs must share a shape off-axis")
			}
		}
		shape[axis] += tListI.shape[axis]
	}
	return shape, nil
}

func Concat[T TensorElement](tList []*Tensor[T], axis int) (*Tensor[T], error) {
	shape := make([]int, tList[0].Rank())
	copy(shape, tList[0].shape)

	shape[axis] = 0
	newShape, err := concatShape(tList, axis, shape)

	if err != nil {
		return &Tensor[T]{}, nil
	}

	result := New[T](newShape)
	rangeBegin := make([]int, result.Rank())
	rangeEnd := make([]int, result.Rank())
	copy(rangeEnd, newShape)
	rangeEnd[axis] = 0

	for i := 0; i < len(tList); i++ {
		tListI := tList[i]
		if tListI.shape[axis] != 0 {
			rangeEnd[axis] += tListI.shape[axis]
			tmpRange := make([]interface{}, result.Rank())

			for j := 0; j < result.Rank(); j++ {
				tmpRange[j] = R(rangeBegin[j], rangeEnd[j])
			}

			err := result.Set(tmpRange, tListI)

			if err != nil {
				return &Tensor[T]{}, err
			}

			rangeBegin[axis] = rangeEnd[axis]
		}
	}
	return result, nil
}

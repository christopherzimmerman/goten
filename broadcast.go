package goten

import (
	"errors"
)

func broadcastEqual(a []int, b []int) bool {
	broadcast := true
	for i := 0; i < len(a); i++ {
		if !(a[i] == b[i] || a[i] == 1 || b[i] == 1) {
			broadcast = false
		}
	}
	return broadcast
}

func broadcastableShape(a []int, b []int) []int {
	newShape := make([]int, len(a))

	for i := 0; i < len(a); i++ {
		if a[i] > b[i] {
			newShape[i] = a[i]
		} else {
			newShape[i] = b[i]
		}
	}

	return newShape
}

func broadcastStrides(destShape []int, srcShape []int, destStrides []int, srcStrides []int) ([]int, error) {
	dims := len(destShape)
	start := dims - len(srcShape)
	result := make([]int, dims)

	for i := dims - 1; i >= start; i-- {
		sI := srcShape[i-start]
		switch sI {
		case 1:
			result[i] = 0
		case destShape[i]:
			result[i] = srcStrides[i-start]
		default:
			return []int{}, errors.New("cannot broadcast shapes")
		}
	}

	return result, nil
}

func stridesForBroadcast(shape []int, strides []int, outputShape []int) ([]int, error) {
	outputStrides := shapeToStrides(shape, RowMajor)
	return broadcastStrides(outputShape, shape, outputStrides, strides)
}

func (t *Tensor[T]) Broadcast(shape []int) (*Tensor[T], error) {
	strides, err := stridesForBroadcast(t.shape, t.strides, shape)

	if err != nil {
		return &Tensor[T]{}, err
	}

	flags := t.flags
	flags = flags.Clear(OwnData)
	flags = flags.Clear(Write)

	return NewFromComponents(t.data, shape, strides, t.offset, flags), nil
}

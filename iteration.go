package goten

import (
	"github.com/christopherzimmerman/goten/internal"
)

type Iterator[T TensorElement] struct {
	data         Storage[T]
	size         int
	offset       int
	rank         int
	shape        []int
	strides      []int
	coordinates  []int
	backStrides  []int
	iterPosition int
	nextFn       func(iterator *Iterator[T])
}

func (iter *Iterator[T]) Next() T {
	result := iter.data.At(iter.iterPosition)
	iter.nextFn(iter)
	return result
}

func advanceFlatIteration[T TensorElement](iterator *Iterator[T]) {
	iterator.iterPosition++
}

func advanceStridedIteration[T TensorElement](s *Iterator[T]) {
	for i := s.rank - 1; i >= 0; i-- {
		if s.coordinates[i] < s.shape[i]-1 {
			s.coordinates[i]++
			s.iterPosition += s.strides[i]
			break
		} else {
			s.coordinates[i] = 0
			s.iterPosition -= s.backStrides[i]
		}
	}
}

func (t *Tensor[T]) Iter() Iterator[T] {
	if t.isRowMajorContiguous() {
		return Iterator[T]{
			size:         t.size,
			data:         t.data,
			iterPosition: t.offset,
			nextFn:       advanceFlatIteration[T],
		}
	}

	coordinates := make([]int, t.Rank())
	backStrides := make([]int, t.Rank())
	offset := t.offset

	for i := 0; i < t.Rank(); i++ {
		backStrides[i] = t.strides[i] * (t.shape[i] - 1)
		if t.strides[i] < 0 {
			offset += (t.shape[i] - 1) * internal.IntAbs(t.strides[i])
		}
	}

	return Iterator[T]{
		data:         t.data,
		size:         t.size,
		iterPosition: offset,
		nextFn:       advanceStridedIteration[T],
		coordinates:  coordinates,
		shape:        t.shape,
		strides:      t.strides,
		backStrides:  backStrides,
		rank:         t.Rank(),
	}
}

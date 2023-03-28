package goten

import (
	"github.com/christopherzimmerman/goten/internal"
)

type Tensor[T TensorElement] struct {
	data    Storage[T]
	shape   []int
	strides []int
	offset  int
	size    int
	flags   TensorFlag
}

func New[T TensorElement](shape []int) *Tensor[T] {
	result := &Tensor[T]{}

	data := &CPU[T]{}
	data.AllocateFromShape(shape)
	result.data = data

	newShape := make([]int, len(shape))
	copy(newShape, shape)
	newStrides := shapeToStrides(shape, RowMajor)
	result.shape = newShape
	result.strides = newStrides
	result.offset = 0
	result.size = internal.SliceProduct(shape)
	result.flags = AllTensorFlags()
	result.updateFlags(AllTensorFlags())

	return result
}

func NewFromComponents[T TensorElement](data Storage[T], shape []int, strides []int, offset int, flags TensorFlag) *Tensor[T] {
	result := &Tensor[T]{}
	result.data = data
	result.shape = shape
	result.strides = strides
	result.offset = offset
	result.flags = flags.Clear(OwnData)
	result.size = internal.SliceProduct(result.shape)
	result.updateFlags(AllTensorFlags())

	return result
}

func (t *Tensor[T]) Shape() []int {
	return t.shape
}

func (t *Tensor[T]) Strides() []int {
	return t.strides
}

func (t *Tensor[T]) Offset() int {
	return t.offset
}

func (t *Tensor[T]) Size() int {
	return t.size
}

func (t *Tensor[T]) Flags() TensorFlag {
	return t.flags
}

func (t *Tensor[T]) Rank() int {
	return len(t.shape)
}

func (t *Tensor[T]) Data() Storage[T] {
	return t.data
}

func (t *Tensor[T]) isRowMajorContiguous() bool {
	switch t.Rank() {
	case 0:
		return true
	case 1:
		return t.shape[0] == 1 || t.strides[0] == 1
	default:
		strideTrack := 1
		for i := t.Rank() - 1; i >= 0; i-- {
			if t.shape[i] == 0 {
				return true
			} else if t.strides[i] != strideTrack {
				return false
			}
			strideTrack *= t.shape[i]
		}
		return true
	}
}

func (t *Tensor[T]) isColMajorContiguous() bool {
	switch t.Rank() {
	case 0:
		return true
	case 1:
		return t.shape[0] == 1 || t.strides[0] == 1
	default:
		strideTrack := 1
		for i := 0; i < t.Rank(); i++ {
			if t.shape[i] == 0 {
				return true
			} else if t.strides[i] != strideTrack {
				return false
			}
			strideTrack *= t.shape[i]
		}
		return true
	}
}

func (t *Tensor[T]) updateFlags(constrainedFlags TensorFlag) {
	if constrainedFlags.Has(Fortran) {
		if t.isColMajorContiguous() {
			t.flags = t.flags.Set(Fortran)
			if t.Rank() > 1 {
				t.flags = t.flags.Clear(Contiguous)
			}
		} else {
			t.flags = t.flags.Clear(Fortran)
		}
	}
	if constrainedFlags.Has(Contiguous) {
		if t.isRowMajorContiguous() {
			t.flags = t.flags.Set(Contiguous)
			if t.Rank() > 1 {
				t.flags = t.flags.Clear(Fortran)
			}
		} else {
			t.flags = t.flags.Clear(Contiguous)
		}
	}
}

func (t *Tensor[T]) Slice(args []interface{}) (*Tensor[T], error) {
	offset, newShape, newStrides, err := offsetForIndex(t.shape, t.strides, t.offset, args)
	return NewFromComponents(t.data, newShape, newStrides, offset, t.flags), err
}

func (t *Tensor[T]) SliceV(args ...interface{}) (*Tensor[T], error) {
	return t.Slice(args)
}

func (t *Tensor[T]) Set(args []interface{}, value *Tensor[T]) error {
	a, err := t.Slice(args)

	if err != nil {
		return err
	}

	if !internal.SliceEqual(a.shape, value.shape) {
		view, err := value.Broadcast(a.shape)
		value = view

		if err != nil {
			return err
		}
	}

	aIter := a.Iter()
	bIter := value.Iter()

	for i := 0; i < a.size; i++ {
		aIter.data.Set(aIter.iterPosition, bIter.Next())
		aIter.Next()
	}

	return nil
}

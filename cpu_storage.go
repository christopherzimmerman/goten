package goten

import (
	"github.com/christopherzimmerman/goten/internal"
	"log"
)

type CPU[T TensorElement] struct {
	data        []T
	initialized bool
}

func (*CPU[T]) UpdateMetadata(shape []int, strides []int) {}

func (c *CPU[T]) AllocateFromShape(shape []int) {
	if c.initialized {
		log.Fatal("tensor storage already initialized")
	}

	c.data = make([]T, internal.SliceProduct(shape))
	for i := 0; i < internal.SliceProduct(shape); i++ {
		c.data[i] = T(i)
	}

	c.initialized = true
}

func (c *CPU[T]) AllocateFromExisting(data []T) {
	c.data = data
	c.initialized = true
}

func (c *CPU[T]) At(i int) T {
	return c.data[i]
}

func (c *CPU[T]) Set(i int, value T) {
	c.data[i] = value
}

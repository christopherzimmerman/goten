package goten

type Storage[T TensorElement] interface {
	UpdateMetadata(shape []int, strides []int)
	AllocateFromShape(shape []int)
	AllocateFromExisting(data []T)
	At(i int) T
	Set(i int, value T)
}

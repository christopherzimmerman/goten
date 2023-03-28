package goten

// MemoryLayout defines a memory storage pattern
// for a Tensor
type MemoryLayout uint8

const (
	// RowMajor indicates C-style layout in a Tensor, where
	// the strides of the last dimension increase the fastest
	RowMajor MemoryLayout = iota

	// ColMajor indicates Fortran-style layout of a Tensor, where
	// the strides of the first dimension increase the fastest
	ColMajor
)

// TensorElement defines all acceptable data types that can be used as a generic
// Tensor.  Ideally, bool and complex types will be supported soon, but
// need to find a way to generically initialize / cast between types
type TensorElement interface {
	uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64 | float32 |
		float64 | int | uint
}

// TensorFlag stores metadata about the layout and properties of a Tensor.  It indicates
// MemoryLayout, as well as tracks the owner of the data of a Tensor, and if it is writeable
type TensorFlag uint8

const (
	// Contiguous means a Tensor is stored in Row-Major format
	Contiguous TensorFlag = 1 << iota

	// Fortran means a Tensor is stored in Col-Major format
	Fortran

	// OwnData indicates that a Tensor has control of its Storage, and its Storage
	// is not a view of another Tensor's data
	OwnData

	// Write indicates if a Tensor can write to its Storage.  Scenarios where this is
	// not allowed are mainly around broadcasting, where multiple elements point to
	// the same memory location
	Write
)

// Set returns a TensorFlag with the specified TensorFlag set to true
func (t TensorFlag) Set(flag TensorFlag) TensorFlag { return t | flag }

// Clear returns a TensorFlag with the specified TensorFlag set to false
func (t TensorFlag) Clear(flag TensorFlag) TensorFlag { return t &^ flag }

// Toggle returns a TensorFlag with the specified TensorFlag flipped from either true/false or false/true
func (t TensorFlag) Toggle(flag TensorFlag) TensorFlag { return t ^ flag }

// Has returns a bool indicating if a provided TensorFlag is set or not
func (t TensorFlag) Has(flag TensorFlag) bool { return t&flag != 0 }

// AllTensorFlags returns a TensorFlag with all possible values set to true
func AllTensorFlags() TensorFlag {
	return Contiguous | Fortran | OwnData | Write
}

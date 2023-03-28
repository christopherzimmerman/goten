package goten

type MemoryLayout uint8

const (
	RowMajor MemoryLayout = iota
	ColMajor
)

type TensorElement interface {
	uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64 | float32 |
		float64 | int | uint
}

type TensorFlag uint8

const (
	Contiguous TensorFlag = 1 << iota
	Fortran
	OwnData
	Write
)

func (t TensorFlag) Set(flag TensorFlag) TensorFlag    { return t | flag }
func (t TensorFlag) Clear(flag TensorFlag) TensorFlag  { return t &^ flag }
func (t TensorFlag) Toggle(flag TensorFlag) TensorFlag { return t ^ flag }
func (t TensorFlag) Has(flag TensorFlag) bool          { return t&flag != 0 }

func AllTensorFlags() TensorFlag {
	return Contiguous | Fortran | OwnData | Write
}

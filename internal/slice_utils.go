package internal

type IntegerType interface {
	uint8 | uint16 | uint32 | uint64 | int8 | int16 | int32 | int64 | int | uint
}

func SliceProduct[T IntegerType](s []T) T {
	result := T(1)

	for _, v := range s {
		result *= v
	}

	return result
}

func SliceEqual[T IntegerType](a []T, b []T) bool {
	if len(a) != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}

	return true
}

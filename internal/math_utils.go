package internal

func IntAbs[T IntegerType](i T) T {
	if i < 0 {
		return -i
	}
	return i
}

package goten

func shapeToStrides(shape []int, layout MemoryLayout) []int {
	strideTrack := 1
	newStrides := make([]int, len(shape))

	switch layout {
	case RowMajor:
		for i := len(shape) - 1; i >= 0; i-- {
			newStrides[i] = strideTrack
			strideTrack *= shape[i]
		}
	default:
		for i, sI := range shape {
			newStrides[i] = strideTrack
			strideTrack *= sI
		}
	}

	return newStrides
}

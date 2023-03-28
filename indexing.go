package goten

import (
	"errors"
	"github.com/christopherzimmerman/goten/internal"
)

type rangeProvider interface {
	Start() int
	End() int
}

type Range struct {
	start int
	end   int
}

func (r Range) Start() int {
	return r.start
}

func (r Range) End() int {
	return r.end
}

type SteppedRange struct {
	start int
	end   int
	step  int
}

func (sr SteppedRange) Start() int {
	return sr.start
}

func (sr SteppedRange) End() int {
	return sr.end
}

func (sr SteppedRange) Step() int {
	return sr.step
}

type PlaceholderRange struct{}

func R(start int, end int) Range {
	return Range{
		start: start,
		end:   end,
	}
}

func SR(start int, end int, step int) SteppedRange {
	return SteppedRange{
		start: start,
		end:   end,
		step:  step,
	}
}

func PR() PlaceholderRange {
	return PlaceholderRange{}
}

func rangeToIndexAndCount[T rangeProvider](r T, size int) (int, int, error) {
	startIndex := r.Start()
	if startIndex < 0 {
		startIndex += size
	}

	if startIndex < 0 {
		return 0, 0, errors.New("invalid range for tensor index")
	}

	endIndex := r.End()
	if endIndex < 0 {
		endIndex += size
	}
	endIndex -= 1
	count := endIndex - startIndex + 1
	if count < 0 {
		count = 0
	}

	return startIndex, count, nil
}

func normalizeInteger(shape []int, strides []int, arg int, index int) (int, int, int, error) {
	if arg < 0 {
		arg += shape[index]
	}

	if arg < 0 || arg >= shape[index] {
		return 0, 0, 0, errors.New("index out of range for tensor")
	}

	return 0, 0, arg, nil
}

func normalizeRange(shape []int, strides []int, arg Range, index int) (int, int, int, error) {
	end := arg.End()
	if end > shape[index] {
		arg = R(arg.Start(), shape[index])
	}

	offset, count, err := rangeToIndexAndCount(arg, shape[index])

	if err != nil {
		return 0, 0, 0, err
	}

	return count, strides[index], offset, nil
}

func normalizeSteppedRange(shape []int, strides []int, arg SteppedRange, index int) (int, int, int, error) {
	end := arg.End()
	if end > shape[index] {
		arg = SR(arg.Start(), shape[index], arg.Step())
	}

	start, offset, err := rangeToIndexAndCount(arg, shape[index])

	if err != nil {
		return 0, 0, 0, err
	}

	absStep := internal.IntAbs(arg.Step())

	return offset/absStep + offset%absStep, arg.Step() * strides[index], start, nil
}

func normalizePlaceHolderRange(shape []int, strides []int, arg PlaceholderRange, index int) (int, int, int, error) {
	return shape[index], strides[index], 0, nil
}

func normalizeArg(shape []int, strides []int, arg interface{}, index int) (int, int, int, error) {
	switch v := arg.(type) {
	case int:
		return normalizeInteger(shape, strides, v, index)
	case Range:
		return normalizeRange(shape, strides, v, index)
	case SteppedRange:
		return normalizeSteppedRange(shape, strides, v, index)
	case PlaceholderRange:
		return normalizePlaceHolderRange(shape, strides, v, index)
	default:
		return 0, 0, 0, errors.New("invalid type for indexing operation")
	}
}

func offsetForIndex(shape []int, strides []int, offset int, args []interface{}) (int, []int, []int, error) {
	newShape := make([]int, len(shape))
	copy(newShape, shape)
	newStrides := make([]int, len(shape))
	copy(newStrides, strides)
	newOffsets := make([]int, len(shape))

	trimmedShape := make([]int, 0)
	trimmedStrides := make([]int, 0)

	for i, arg := range args {
		shapeAtIndex, stridesAtIndex, offsetAtIndex, err := normalizeArg(shape, strides, arg, i)

		if err != nil {
			return 0, []int{}, []int{}, err
		}

		newShape[i] = shapeAtIndex
		newStrides[i] = stridesAtIndex
		newOffsets[i] = offsetAtIndex
	}

	for i, v := range newShape {
		if v != 0 {
			trimmedShape = append(trimmedShape, v)
			trimmedStrides = append(trimmedStrides, newStrides[i])
		}
	}

	for i := 0; i < len(shape); i++ {
		if strides[i] < 0 {
			offset += (shape[i] - 1) * internal.IntAbs(strides[i])
		}
		offset += newOffsets[i] * strides[i]
	}

	return offset, trimmedShape, trimmedStrides, nil
}

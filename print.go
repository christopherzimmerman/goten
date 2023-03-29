package goten

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

func format(value interface{}, pad int) string {
	result := ""
	switch v := value.(type) {
	case int:
		result = strconv.FormatInt(int64(v), 10)
	case int8:
		result = strconv.FormatInt(int64(v), 10)
	case int16:
		result = strconv.FormatInt(int64(v), 10)
	case int32:
		result = strconv.FormatInt(int64(v), 10)
	case int64:
		result = strconv.FormatInt(v, 10)
	case uint:
		result = strconv.FormatUint(uint64(v), 10)
	case uint8:
		result = strconv.FormatUint(uint64(v), 10)
	case uint16:
		result = strconv.FormatUint(uint64(v), 10)
	case uint32:
		result = strconv.FormatUint(uint64(v), 10)
	case uint64:
		result = strconv.FormatUint(v, 10)
	case float32:
		result = strconv.FormatFloat(float64(v), 'f', 2, 32)
	case float64:
		result = strconv.FormatFloat(v, 'f', 2, 32)
	}
	return fmt.Sprintf("%*s", pad, result)
}

func maxWidth[T TensorElement](iterator Iterator[T]) int {
	mx := 0
	for i := 0; i < iterator.size; i++ {
		value := format(iterator.Next(), 0)
		if len(value) > mx {
			mx = len(value)
		}
	}
	return mx
}

func leadingTrailing[T TensorElement](a *Tensor[T], edgeItems int, index []interface{}) (*Tensor[T], error) {
	axis := len(index)

	if axis == a.Rank() {
		return a.Slice(index)
	}

	if a.shape[axis] > 2*edgeItems {
		sliceOne, _ := leadingTrailing(a, edgeItems, append(index, R(0, edgeItems)))
		sliceTwo, _ := leadingTrailing(a, edgeItems, append(index, R(-1*edgeItems, a.shape[axis])))
		return Concat[T]([]*Tensor[T]{sliceOne, sliceTwo}, axis)
	} else {
		return leadingTrailing(a, edgeItems, append(index, R(0, a.shape[axis])))
	}
}

func extendLine(s string, line string, word string, lineWidth int, nextLinePrefix string) (string, string) {
	needsWrap := len(line)+len(word) > lineWidth

	if needsWrap {
		s += strings.TrimRightFunc(line, unicode.IsSpace) + "\n"
		line = nextLinePrefix
	}

	line += word
	return s, line
}

func recursor[T TensorElement](a *Tensor[T], index []interface{}, hangingIndent string, currentWidth int, summaryInsert string, edgeItems int, separator string, pad int) string {
	axis := len(index)
	axesLeft := a.Rank() - axis

	if axesLeft == 0 {
		view, _ := a.Slice(index)
		return format(a.Data().At(view.offset), pad)
	}

	nextHangingIndent := hangingIndent + " "
	nextWidth := currentWidth - 1

	axisLength := a.Shape()[axis]
	showSummary := (len(summaryInsert) > 0) && (2*edgeItems < axisLength)

	leadingItems := 0
	trailingItems := axisLength

	if showSummary {
		leadingItems = edgeItems
		trailingItems = edgeItems
	}

	s := ""

	if axesLeft == 1 {
		elemWidth := currentWidth - len(separator)
		line := hangingIndent

		for i := 0; i < leadingItems; i++ {
			word := recursor(a, append(index, i), nextHangingIndent, nextWidth, summaryInsert, edgeItems, separator, pad)
			s, line = extendLine(s, line, word, elemWidth, hangingIndent)
			line += separator
		}

		if showSummary {
			s, line = extendLine(s, line, summaryInsert, elemWidth, hangingIndent)
			line += separator
		}

		for i := trailingItems; i >= 2; i-- {
			word := recursor(a, append(index, -i), nextHangingIndent, nextWidth, summaryInsert, edgeItems, separator, pad)
			s, line = extendLine(s, line, word, elemWidth, hangingIndent)
			line += separator
		}

		word := recursor(a, append(index, -1), nextHangingIndent, nextWidth, summaryInsert, edgeItems, separator, pad)
		s, line = extendLine(s, line, word, elemWidth, hangingIndent)
		s += line
	} else {
		lineSep := separator + strings.Repeat("\n", axesLeft-1)

		for i := 0; i < leadingItems; i++ {
			nested := recursor(a, append(index, i), nextHangingIndent, nextWidth, summaryInsert, edgeItems, separator, pad)
			s += hangingIndent + nested + lineSep
		}

		if showSummary {
			s += hangingIndent + summaryInsert + "\n"
		}

		for i := trailingItems; i >= 2; i-- {
			nested := recursor(a, append(index, -i), nextHangingIndent, nextWidth, summaryInsert, edgeItems, separator, pad)
			s += hangingIndent + nested + lineSep
		}

		nested := recursor(a, append(index, -1), nextHangingIndent, nextWidth, summaryInsert, edgeItems, separator, pad)
		s += hangingIndent + nested
	}

	return "[" + s[len(hangingIndent):] + "]"
}

func formatArray[T TensorElement](a *Tensor[T], lineWidth int, nextLinePrefix string, separator string, edgeItems int, summaryInsert string, pad int) string {
	return recursor(a, make([]interface{}, 0), nextLinePrefix, lineWidth, summaryInsert, edgeItems, separator, pad)
}

func (t *Tensor[T]) String() string {
	data := t
	summaryInsert := ""
	if t.size > 1000 {
		data, _ = leadingTrailing(t, 3, []interface{}{})
		summaryInsert = "..."
	}
	pad := maxWidth(data.Iter())
	return formatArray(t, 75, " ", ", ", 3, summaryInsert, pad)
}

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
	leadingItems := 0
	trailingItems := axisLength

	s := ""

	if axesLeft == 1 {
		elemWidth := currentWidth - len(separator)
		line := hangingIndent

		for i := 0; i < leadingItems; i++ {
			word := recursor(a, append(index, i), nextHangingIndent, nextWidth, summaryInsert, edgeItems, separator, pad)
			s, line = extendLine(s, line, word, elemWidth, hangingIndent)
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
	pad := maxWidth(t.Iter())
	return formatArray(t, 75, " ", ",", 3, "...", pad)
}

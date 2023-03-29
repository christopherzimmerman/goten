Goten is a package that seeks to be a generic framework for scientific computing in Go.

It provides:

- An n-dimensional `Tensor` data structure
- Efficient `Map`, `Reduce` and `Accumulate` routines
- Linear algebra routines backed by `LAPACK` and `BLAS`

## Prerequisites

`Goten` aims to be a scientific computing library written in pure Go.
All standard operations and data structures are written in Go.  Certain
routines, primarily linear algebra routines, are instead provided by a
`BLAS` or `LAPACK` implementation.

## Just show me the code

The core data structure implemented by `Goten` is the `Tensor`, an N-dimensional
data structure.  A `Tensor` supports slicing, mutation, permutation, reduction,
and accumulation.  A `Tensor` can be a view of another `Tensor`, and can support
either C-style or Fortran-style storage.

### Creation

There are many ways to initialize a `Tensor`.

```go
package main

import (
	"github.com/christopherzimmerman/goten"
)

func main() {
	a := goten.New[uint8]([]int{3, 2, 2})
	//     [[[0, 0],
	//	 [0, 0]],
	//
	//	[[0, 0],
	//	 [0, 0]],
	//
	//	[[0, 0],
	//	 [0, 0]]]

	b := goten.Full([]int{3, 3}, 3.14159)
	// [[3.14, 3.14, 3.14],
	//  [3.14, 3.14, 3.14],
	//  [3.14, 3.14, 3.14]]

	c := goten.Ones[float32]([]int{2, 2, 2})
	//     [[[1.00, 1.00],
	//	 [1.00, 1.00]],
	//
	//	[[1.00, 1.00],
	//	 [1.00, 1.00]]]

	d := goten.Arange[uint8](0, 10)
	// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```
package helper

const (
	MaxInt = int(^uint(0) >> 1)
	MinInt = -MaxInt - 1
)

func Max(a int, b int) int {
	if a > b {
		return a
	}
	return b
}

func Min(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

func IndexToPosition(idx int) (int, int) {
	row := idx / 3
	col := idx % 3
	return row, col
}
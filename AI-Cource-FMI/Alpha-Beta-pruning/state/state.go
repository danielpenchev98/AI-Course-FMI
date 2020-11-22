package state


import (
	"strings"
	"fmt"
)

const maxDepth = 9


type Histogram struct {
	rows      []int
	cols      []int
	diags     []int
}

type State struct {
	freeTiles map[int]struct{}
	board     [][]int
	Histogram
}

var emptyStruct struct{}

func CreateState() *State{
	state := State{
		freeTiles: make(map[int]struct{}),
		board: [][]int{
			{-1, -1, -1},
			{-1, -1, -1},
			{-1, -1, -1},
		},
		Histogram: Histogram{
			rows:  []int{0, 0, 0},
			cols:  []int{0, 0, 0},
			diags: []int{0, 0},
		},
	}

	for i := 0; i < 9; i++ {
		state.freeTiles[i] = emptyStruct
	}

	return &state
}

func (s *State) PlayMove(playerNumber int, x int, y int) {
	s.board[y][x] = playerNumber
	delete(s.freeTiles, y*3+x)

	offset := 1
	if(playerNumber == 2){
		offset = -1
	}
	
	s.rows[y] = s.rows[y] + offset
	s.cols[x] = s.cols[x] + offset
	if x == y {
		s.diags[0] = s.diags[0] + offset
	}
	if x+y == 2 {
		s.diags[1] = s.diags[1] + offset
	}
}

func (s *State) UndoMove(playerNumber int, x int, y int) {
	s.board[y][x] = -1
	s.freeTiles[y*3+x] = emptyStruct

	offset := -1
	if playerNumber == 2{
		offset = 1
	}
	
	s.rows[y] = s.rows[y] + offset
	s.cols[x] = s.cols[x] + offset
	if x == y {
		s.diags[0] = s.diags[0] + offset
	} else if x+y == 2 {
		s.diags[1] = s.diags[1] + offset
	}
}

func (s *State) CalculateHeuristic() int {
	var currDepth int
	for i := 0; i < len(s.board); i++ {
		for j := 0; j < len(s.board); j++ {
			if s.board[i][j] != -1 {
				currDepth++
			}
		}
	}

	return maxDepth - currDepth
}

func (s *State) String() string {
	var sb strings.Builder
	for i := 0; i < len(s.board); i++ {
		for j := 0; j < len(s.board); j++ {
			sb.WriteString(fmt.Sprintf("%d ", s.board[i][j]))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func (s *State) IsWon() bool {
	for i := 0; i < len(s.board); i++ {
		if s.rows[i] == -3 || s.rows[i] == 3 || s.cols[i] == -3 || s.cols[i] == 3 {
			return true
		}
	}
	return s.diags[0] == -3 || s.diags[0] == 3 || s.diags[1] == -3 || s.diags[1] == 3
}

func (s *State) IsTerminal() bool {
	return len(s.freeTiles)==0 || s.IsWon()
}

func (s *State) GetNeightbours() []int{
	neightbours := make([]int,len(s.freeTiles))
	iter :=0
	for key,_ := range s.freeTiles {
		neightbours[iter]=key
		iter++
	}
	return neightbours
}
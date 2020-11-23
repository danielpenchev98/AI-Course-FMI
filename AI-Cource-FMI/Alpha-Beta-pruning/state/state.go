package state

import (
	"strings"
	"fmt"
	"math/rand"
	"time"
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

func (s *State) PlayMove(playerNumber int, y int, x int) {
	s.board[y][x] = playerNumber
	delete(s.freeTiles, y*3+x)

	offset := 1
	if playerNumber == 0 {
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

func (s *State) UndoMove(playerNumber int, y int, x int) {
	s.board[y][x] = -1
	s.freeTiles[y*3+x] = emptyStruct

	offset := -1
	if playerNumber == 0 {
		offset = 1
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

func (s *State) GetWinner() (int,bool) {
	for i := 0; i < len(s.board); i++ {
		if s.rows[i] == -3 || s.cols[i] == -3 {
			return 0, true
		} else if s.rows[i] == 3 || s.cols[i] == 3{
			return 1, true
		}
	}

	if s.diags[0] == -3 || s.diags[1] == -3 {
		return 0, true
	} else if s.diags[0] == 3 || s.diags[1] == 3 {
		return 1, true
	}

	return -1, false
}

func (s *State) IsTerminal() bool {
	_, ok := s.GetWinner()
	return len(s.freeTiles)==0 || ok
}

func (s *State) GetNeightbours() []int{
	neightbours := make([]int,len(s.freeTiles))
	iter :=0
	for key,_ := range s.freeTiles {
		neightbours[iter]=key
		iter++
	}
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(neightbours), func(i, j int) { neightbours[i], neightbours[j] = neightbours[j], neightbours[i] })
	return neightbours
}

func (s *State) PrintHistogram() {
	for i:=0;i<3;i++{
		fmt.Printf("s.col[%d]=%d\n",i,s.cols[i])
		fmt.Printf("s.row[%d]=%d\n",i,s.rows[i])
	}

	fmt.Printf("s.diags[%d]=%d\n",0,s.diags[0])
	fmt.Printf("s.diags[%d]=%d\n",1,s.diags[1])
}
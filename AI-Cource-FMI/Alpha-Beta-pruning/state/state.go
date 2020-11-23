package state

import (
	"strings"
	"fmt"
	"math/rand"
	"time"
)

const maxDepth = 9

type Histogram struct {
	rows      [][]int
	cols      [][]int
	diags     [][]int
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
			rows:  [][]int{
				{0, 0},
				{0, 0},
				{0, 0},
			},
			cols:  [][]int{
				{0, 0},
				{0, 0},
				{0, 0},
			},
			diags: [][]int{
				{0, 0},
				{0, 0},
				{0, 0},
			},
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
	
	s.rows[y][playerNumber]++
	s.cols[x][playerNumber]++
	if x == y {
		s.diags[0][playerNumber]++
	}
	if x+y == 2 {
		s.diags[1][playerNumber]++
	}
}

func (s *State) UndoMove(playerNumber int, y int, x int) {
	s.board[y][x] = -1
	s.freeTiles[y*3+x] = emptyStruct
	
	s.rows[y][playerNumber]--
	s.cols[x][playerNumber]--
	if x == y {
		s.diags[0][playerNumber]--
	}
	if x+y == 2 {
		s.diags[1][playerNumber]--
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

func (s *State) Utility() int{
	return s.getColsScore() + s.getRowsScore() + s.getDiagsScore()
}

func (s *State) getRowsScore() int {
	var score int
	for i:=0;i<len(s.board);i++{
		multiplier := -1
		for j:=0;j<2;j++{
			if s.rows[i][j] == 3 {
				score += 100 * multiplier
			} else if s.rows[i][j]==2 {
				score +=10 * multiplier
			}
			multiplier+=2
		}
		
		if s.rows[i][1] == 1 && s.rows[i][0] == 0{
			score += 1
		} else if s.rows[i][1] == 0 && s.rows[i][0] == 1 {
			score -=1
		}
	}
	return score
}

func (s *State) getColsScore() int {
	var score int
	for i:=0;i<len(s.board);i++{
		multiplier := -1
		for j:=0;j<2;j++{
			if s.cols[i][j] == 3 {
				score += 100 * multiplier
			} else if s.cols[i][j]==2 {
				score +=10 * multiplier
			}
			multiplier+=2
		}
		
		if s.cols[i][1] == 1 && s.cols[i][0] == 0{
			score += 1
		} else if s.cols[i][1] == 0 && s.cols[i][0] == 1 {
			score -=1
		}
	}
	return score
}

func (s *State) getDiagsScore() int {
	var score int
	for i:=0 ; i< 2;i++{
		multiplier := -1
		for j:=0;j<2;j++{
			if s.diags[i][j] == 3 {
				score += 100 * multiplier
			} else if s.diags[i][j]==2 {
				score +=10 * multiplier
			}
			multiplier+=2
		}

		if  s.diags[i][1]==1 && s.diags[i][0]==0{
			score +=1
		} else if s.diags[i][1]==0 && s.diags[i][0]==1 {
			score -=1
		}
	}
	return score
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
	score := s.Utility()
	if score*score >= 100*100{
		return score, true
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
		for j:=0;j<2;j++{
			fmt.Printf("s.col[%d][%d]=%d\n",i,j,s.cols[i][j])
			fmt.Printf("s.row[%d][%d]=%d\n",i,j,s.rows[i][j])
		}
	}

	fmt.Printf("s.diags[%d][%d]=%d\n",0,0,s.diags[0][0])
	fmt.Printf("s.diags[%d][%d]=%d\n",0,1,s.diags[0][1])
	fmt.Printf("s.diags[%d][%d]=%d\n",1,0,s.diags[1][0])
	fmt.Printf("s.diags[%d][%d]=%d\n",1,1,s.diags[1][1])
}
package state

import (
    "strings"
    "fmt"
  	"math/rand"
	"time"
	"errors"
)

type Histogram struct {
	rows      []int
	cols      []int
	diags     []int
}

type State struct {
    freeTiles map[int]struct{}
	board     [][]int
	winner int //-1 noone yet or draw, 0 second player, 1 first player
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
		winner: -1,
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

    s.rows[y]+=offset
    s.cols[x]+=offset
    if x == y {
        s.diags[0]+=offset
    }
    if x+y == 2 {
        s.diags[1]+=offset
	}
	
	if s.rows[y] == 3 * offset || s.cols[x] == 3 * offset || s.diags[0]==3 * offset || s.diags[1]==3 * offset {
		s.winner = playerNumber
	}
}

func (s *State) UndoMove(playerNumber int, y int, x int) {
    s.board[y][x] = -1
    s.freeTiles[y*3+x] = emptyStruct
	
	offset := -1
	if playerNumber == 0 {
		offset = 1
	}

	if s.rows[y] == -3 * offset || s.cols[x] == -3 * offset || s.diags[0] == -3 * offset || s.diags[1] == -3 * offset {
		s.winner = -1
	}

    s.rows[y]+=offset
    s.cols[x]+=offset
    if x == y {
        s.diags[0]+=offset
    }
    if x+y == 2 {
        s.diags[1]+=offset
    }
}

func (s *State) Utility() int{
    return len(s.freeTiles)
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



func (s *State) GetWinner() (int,error){
	if !s.HasWinner(){
		return -1, errors.New("The game doesnt have a winner")
	}
	return s.winner,nil
}

func (s *State) HasWinner() (bool) {
    return s.winner != -1
}

func (s *State) IsTerminal() bool {
    return len(s.freeTiles)==0 || s.HasWinner()
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